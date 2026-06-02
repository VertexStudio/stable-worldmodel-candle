#!/usr/bin/env python3
"""Benchmark official Python TD-MPC2 inference on CUDA."""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image


class DotDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stable-worldmodel-root",
        default=os.environ.get("STABLE_WORLDMODEL_ROOT")
        or os.environ.get("STABLE_WORLDMODEL_PY"),
        help="local stable-worldmodel source tree to prepend to PYTHONPATH",
    )
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--actor-trajs", type=int, default=None)
    parser.add_argument("--state-dim", type=int, default=12)
    parser.add_argument("--action-dim", type=int, default=10)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument(
        "--jpeg-input",
        type=Path,
        help="JPEG file used for encoded image ingestion benchmark",
    )
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def tdmpc2_cfg(args: argparse.Namespace) -> DotDict:
    return DotDict(
        action_dim=args.action_dim,
        image_size=64,
        extra_dims={"state": args.state_dim},
        wm=DotDict(
            tau=0.01,
            encoding={"state": 128},
            enc_dim=256,
            mlp_dim=384,
            simnorm_dim=8,
            num_bins=101,
            vmin=-6.0,
            vmax=2.0,
            discount=0.99,
            uncertainty_penalty=0.5,
            num_q=5,
        ),
    )


def main() -> None:
    args = parse_args()
    if args.iters <= 0:
        raise ValueError("--iters must be greater than zero")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.samples <= 0:
        raise ValueError("--samples must be greater than zero")
    if args.horizon <= 0:
        raise ValueError("--horizon must be greater than zero")
    if args.image_size <= 0:
        raise ValueError("--image-size must be greater than zero")

    if args.stable_worldmodel_root:
        sys.path.insert(0, str(Path(args.stable_worldmodel_root).resolve()))

    from stable_worldmodel.wm.tdmpc2 import TDMPC2
    from stable_worldmodel.wm.tdmpc2.module import log_std

    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    if not torch.cuda.is_available():
        raise RuntimeError("Python benchmark requires torch.cuda.is_available()")
    device = torch.device(args.device)
    jpeg_bytes = read_or_make_jpeg(args)

    actor_trajs = args.actor_trajs or args.samples
    model = TDMPC2(tdmpc2_cfg(args)).to(device).eval()
    state = torch.randn(args.batch_size, args.state_dim, dtype=torch.float32, device=device)
    action = torch.randn(args.batch_size, args.action_dim, dtype=torch.float32, device=device).clamp(
        -1.0, 1.0
    )
    action_candidates = torch.randn(
        args.batch_size,
        args.samples,
        args.horizon,
        args.action_dim,
        dtype=torch.float32,
        device=device,
    ).clamp(-1.0, 1.0)
    actor_noise = torch.randn(
        actor_trajs,
        args.batch_size,
        args.horizon,
        args.action_dim,
        dtype=torch.float32,
        device=device,
    )
    obs = {"state": state}
    z = model.encode(obs).contiguous()

    with torch.inference_mode():
        rows = [
            bench("media_jpeg", args, lambda: python_media_jpeg(jpeg_bytes, args.batch_size, device)),
            bench("encode", args, lambda: model.encode(obs)),
            bench("dynamics", args, lambda: model.forward(z, action)),
            bench("score", args, lambda: model.get_cost(obs, action_candidates)),
            bench(
                "full",
                args,
                lambda: full_step(model, obs, state, action, action_candidates),
            ),
            bench("policy_rollout", args, lambda: actor_mean_rollout(model, z, args.horizon)),
            bench(
                "policy_sample",
                args,
                lambda: actor_sample_rollout(model, z, actor_noise, log_std),
            ),
        ]

    payload = {
        "git_commit": git_commit(),
        "backend": "python-pytorch",
        "model": "TdMpc2",
        "device": str(device),
        "dtype": "f32",
        "batch_size": args.batch_size,
        "samples": args.samples,
        "horizon": args.horizon,
        "actor_trajs": actor_trajs,
        "image_size": args.image_size,
        "jpeg_input": str(args.jpeg_input) if args.jpeg_input else "generated-in-memory",
        "media_jpeg": "JPEG bytes -> Pillow RGB decode -> NumPy HWC -> CUDA F32 NCHW /255",
        "warmup": args.warmup,
        "iters": args.iters,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(device),
        "stats": rows,
    }

    text = json.dumps(payload, indent=2)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")
    print(text)


def read_or_make_jpeg(args: argparse.Namespace) -> bytes:
    if args.jpeg_input:
        return args.jpeg_input.read_bytes()

    rng = np.random.default_rng(args.seed)
    image = synthetic_rgb_image(args.image_size, rng)
    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(
        buffer,
        format="JPEG",
        quality=95,
        subsampling=0,
        optimize=False,
    )
    return buffer.getvalue()


def synthetic_rgb_image(size: int, rng: np.random.Generator) -> np.ndarray:
    axis = np.linspace(0, 255, size, dtype=np.uint8)
    xx, yy = np.meshgrid(axis, axis)
    noise = rng.integers(0, 16, size=(size, size), dtype=np.uint8)
    return np.stack(
        [
            xx,
            yy,
            ((xx.astype(np.uint16) + yy.astype(np.uint16)) // 2 + noise).astype(np.uint8),
        ],
        axis=-1,
    )


def python_media_jpeg(jpeg_bytes: bytes, batch_size: int, device: torch.device) -> torch.Tensor:
    frames = []
    for _ in range(batch_size):
        with Image.open(io.BytesIO(jpeg_bytes)) as image:
            frames.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
    hwc = np.stack(frames, axis=0)
    nchw = np.transpose(hwc, (0, 3, 1, 2)).copy()
    return torch.from_numpy(nchw).to(device=device, dtype=torch.float32).mul_(1.0 / 255.0)


def full_step(model, obs, state, action, action_candidates):
    z = model.encode(obs)
    _ = model.forward(z, action)
    return model.get_cost({"state": state}, action_candidates)


def actor_mean_rollout(model, z, horizon: int):
    curr_z = z
    actions = []
    rewards = []
    for _ in range(horizon):
        mean_raw, _ = model.pi(curr_z).chunk(2, dim=-1)
        action = torch.tanh(mean_raw)
        z_a = torch.cat([curr_z, action], dim=-1)
        rewards.append(model.reward(z_a))
        curr_z = model.dynamics(z_a)
        actions.append(action)
    return torch.stack(actions, dim=1), torch.stack(rewards, dim=1), curr_z


def actor_sample_rollout(model, z, noise, log_std_fn):
    num_trajs, batch, horizon, action_dim = noise.shape
    curr_z = z.unsqueeze(0).expand(num_trajs, batch, z.shape[-1]).reshape(
        num_trajs * batch, z.shape[-1]
    )
    actions = []
    for step_idx in range(horizon):
        mean_raw, log_std_raw = model.pi(curr_z).chunk(2, dim=-1)
        step_log_std = log_std_fn(log_std_raw, low=-10, dif=12)
        step_noise = noise[:, :, step_idx, :].reshape(num_trajs * batch, action_dim)
        action = torch.tanh(mean_raw + step_log_std.exp() * step_noise)
        curr_z = model.dynamics(torch.cat([curr_z, action], dim=-1))
        actions.append(action.reshape(num_trajs, batch, action_dim))
    return torch.stack(actions, dim=2).mean(0)


def bench(name: str, args: argparse.Namespace, op: Callable[[], object]) -> dict[str, float | str]:
    for _ in range(args.warmup):
        op()
    torch.cuda.synchronize()

    samples = []
    for _ in range(args.iters):
        torch.cuda.synchronize()
        started = time.perf_counter()
        op()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)

    samples.sort()
    return {
        "name": name,
        "mean_ms": sum(samples) / len(samples),
        "p50_ms": percentile(samples, 0.50),
        "p95_ms": percentile(samples, 0.95),
        "p99_ms": percentile(samples, 0.99),
    }


def percentile(samples: list[float], pct: float) -> float:
    idx = min(len(samples) - 1, int((len(samples) - 1) * pct + 0.999999))
    return samples[idx]


def git_commit() -> str:
    try:
        output = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
    except Exception:
        return "unknown"
    return output.strip() or "unknown"


if __name__ == "__main__":
    main()
