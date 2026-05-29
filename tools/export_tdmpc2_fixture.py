#!/usr/bin/env python3
"""Export deterministic TD-MPC2 fixtures from Python."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


class DotDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--weights-output", required=True, type=Path)
    parser.add_argument(
        "--stable-worldmodel-root",
        default=os.environ.get("STABLE_WORLDMODEL_PY"),
        help="optional local stable-worldmodel checkout to prepend to PYTHONPATH",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument(
        "--fixture-kind",
        choices=("state", "pixel", "both"),
        default="state",
        help="observation path to export",
    )
    parser.add_argument("--state-dim", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--pixel-dim", type=int, default=128)
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def cfg(args: argparse.Namespace) -> DotDict:
    encoding = {}
    extra_dims = {}
    if args.fixture_kind in ("pixel", "both"):
        encoding["pixels"] = args.pixel_dim
    if args.fixture_kind in ("state", "both"):
        encoding["state"] = 128
        extra_dims["state"] = args.state_dim

    return DotDict(
        action_dim=args.action_dim,
        image_size=args.image_size,
        extra_dims=extra_dims,
        wm=DotDict(
            tau=0.01,
            encoding=encoding,
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


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(tensor.detach().cpu().numpy().astype("float32"))


def main() -> None:
    args = parse_args()
    if args.stable_worldmodel_root:
        sys.path.insert(0, str(Path(args.stable_worldmodel_root).resolve()))

    from stable_worldmodel.wm.tdmpc2 import TDMPC2

    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")
    device = torch.device(args.device)

    model = TDMPC2(cfg(args)).eval()
    args.weights_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.weights_output)
    model = model.to(device)

    with torch.no_grad():
        obs_cpu = {}
        obs_device = {}
        if args.fixture_kind in ("state", "both"):
            state = torch.randn(args.batch_size, args.state_dim, dtype=torch.float32)
            obs_cpu["state"] = state
            obs_device["state"] = state.to(device)
        if args.fixture_kind in ("pixel", "both"):
            pixels = torch.randn(
                args.batch_size,
                3,
                args.image_size,
                args.image_size,
                dtype=torch.float32,
            )
            obs_cpu["pixels"] = pixels
            obs_device["pixels"] = pixels.to(device)

        action = torch.randn(args.batch_size, args.action_dim, dtype=torch.float32).clamp(
            -1.0, 1.0
        )
        action_candidates = torch.randn(
            args.batch_size,
            args.samples,
            args.horizon,
            args.action_dim,
            dtype=torch.float32,
        ).clamp(-1.0, 1.0)

        action_device = action.to(device)
        action_candidates_device = action_candidates.to(device)

        z = model.encode(obs_device).contiguous()
        next_z, reward_logits = model.forward(z, action_device)
        actor_mean = torch.tanh(model.pi(z).chunk(2, dim=-1)[0]).contiguous()
        cost = model.get_cost(obs_device, action_candidates_device).contiguous()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(
        action=tensor_to_numpy(action),
        action_candidates=tensor_to_numpy(action_candidates),
        z=tensor_to_numpy(z),
        next_z=tensor_to_numpy(next_z),
        reward_logits=tensor_to_numpy(reward_logits),
        actor_mean=tensor_to_numpy(actor_mean),
        cost=tensor_to_numpy(cost),
    )
    for name, tensor in obs_cpu.items():
        arrays[name] = tensor_to_numpy(tensor)
    np.savez(args.output, **arrays)

    print(f"fixture={args.output}")
    print(f"weights={args.weights_output}")
    print(f"device={device}")
    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    if device.type == "cuda":
        print(f"cuda_device={torch.cuda.get_device_name(device)}")
    print(f"batch_size={args.batch_size}")
    print(f"samples={args.samples}")
    print(f"horizon={args.horizon}")
    print(f"fixture_kind={args.fixture_kind}")
    print(f"state_dim={args.state_dim}")
    print(f"image_size={args.image_size}")
    print(f"pixel_dim={args.pixel_dim}")
    print(f"action_dim={args.action_dim}")


if __name__ == "__main__":
    main()
