# stable-worldmodel-candle

Rust/Candle inference runtime for `stable-worldmodel` checkpoints.

![stable-worldmodel Candle runtime overview](docs/stable-worldmodel-runtime.png)

Model implementations live under `src/models/`. Shared checkpoint and config
helpers live at the crate root, and CLIs select a backend explicitly.

## Current Scope

- Top-level modules: `checkpoint`, `config`, and `models`.
- `models::lewm`: ViT-Tiny encoder, projector, action encoder, conditional predictor, latent rollout, and goal MSE cost.
- `models::tdmpc2`: state/vector observation encoder, latent dynamics, reward/Q heads, actor mean action, and candidate cost scoring.
- Loading from PyTorch `.pt` state dicts via `VarBuilder::from_pth`, or from `.safetensors`.
- Optional Hugging Face Hub checkpoint download support behind `--features hub`.
- Rust 2024 edition with published Candle crates.
- Backend-specific shape smoke-test CLIs:

```bash
cargo run --bin lewm-inspect -- --action-dim 2
cargo run --bin tdmpc2-inspect -- --state-dim 12 --action-dim 4
```

With a checkpoint:

```bash
cargo run --release --bin lewm-inspect -- --weights /path/to/weights_epoch_100.pt --action-dim 2
cargo run --release --bin tdmpc2-inspect -- --weights /path/to/weights_epoch_250.pt --state-dim 12 --action-dim 4
```

## Checkpoints and Parity

The Python `stable_worldmodel.wm.utils.load_pretrained` path resolves model repos
from Hugging Face by downloading:

```text
config.json
weights.pt
```

Official LeWM mirrors currently use this layout, for example
`quentinll/lewm-pusht`, `quentinll/lewm-reacher`, and
`quentinll/lewm-tworooms`.

To export a deterministic Python fixture from the original implementation:

```bash
# From a checkout where stable-worldmodel and stable-worldmodel-candle are siblings.
cd stable-worldmodel
uv run --python 3.12 --no-dev --extra train \
  --with imageio --with 'transformers<5' \
  python ../stable-worldmodel-candle/tools/export_lewm_fixture.py \
  --stable-worldmodel-root . \
  --model quentinll/lewm-pusht \
  --device cpu \
  --output ../stable-worldmodel-candle/target/lewm-pusht-fixture.npz
```

The `transformers<5` pin matters for the current public LeWM checkpoints: the
weights use the Hugging Face ViT 4.x key layout (`encoder.encoder.layer.*`).

Then compare Candle outputs against the Python fixture:

```bash
cd ../stable-worldmodel-candle
cargo run --bin lewm-compare-fixture -- \
  --fixture target/lewm-pusht-fixture.npz \
  --weights ~/.stable_worldmodel/checkpoints/models--quentinll--lewm-pusht/weights.pt \
  --config ~/.stable_worldmodel/checkpoints/models--quentinll--lewm-pusht/config.json
```

Or let Rust download the same HF files through Candle-style hub support:

```bash
cargo run --features hub --bin lewm-compare-fixture -- \
  --fixture target/lewm-pusht-fixture.npz \
  --hf-repo quentinll/lewm-pusht
```

The current verified PushT fixture covers pixel encoding, action embedding,
single-step prediction, latent rollout, and goal cost.

For backend parity, generate CPU and CUDA Python fixtures from identical CPU
input tensors, then compare them before comparing Candle:

```bash
cd stable-worldmodel
uv run --python 3.12 --no-dev --extra train \
  --with imageio --with 'transformers<5' \
  python ../stable-worldmodel-candle/tools/export_lewm_fixture.py \
  --stable-worldmodel-root . \
  --model quentinll/lewm-pusht \
  --device cpu \
  --output ../stable-worldmodel-candle/target/lewm-pusht-python-cpu.npz

uv run --python 3.12 --no-dev --extra train \
  --with imageio --with 'transformers<5' \
  python ../stable-worldmodel-candle/tools/export_lewm_fixture.py \
  --stable-worldmodel-root . \
  --model quentinll/lewm-pusht \
  --device cuda \
  --output ../stable-worldmodel-candle/target/lewm-pusht-python-cuda.npz

uv run --python 3.12 --no-dev --extra train \
  python ../stable-worldmodel-candle/tools/compare_npz.py \
  ../stable-worldmodel-candle/target/lewm-pusht-python-cpu.npz \
  ../stable-worldmodel-candle/target/lewm-pusht-python-cuda.npz \
  --left-label python-cpu \
  --right-label python-cuda
```

The fixture exporter disables TF32 matmul/cuDNN paths, disables cuDNN
benchmarking, runs with gradients off, and exports model outputs after
`model.eval()`.

## Platform Builds

Default CPU build, portable on macOS and Linux:

```bash
cargo check --all-targets
```

macOS Accelerate:

```bash
cargo check --features accelerate --all-targets
```

macOS Metal:

```bash
cargo check --features metal --all-targets
```

Linux CUDA:

```bash
cargo check --features cuda --all-targets
cargo run --release --features cuda --bin lewm-inspect -- \
  --device cuda \
  --weights /path/to/weights_epoch_100.pt \
  --action-dim 2
```

cuDNN is available as an additive feature:

```bash
cargo check --features cudnn --all-targets
```

Full LeWM CUDA parity matrix:

```bash
tools/cuda_parity.sh
```

The matrix runs environment sanity checks, Rust CUDA build/tests, optional
cuDNN checks when detected, Python CPU-vs-CUDA fixture diffs, Candle CPU vs
Python CPU, Candle CUDA vs Python CUDA, and Candle CUDA vs Python CPU. Set
`STABLE_WORLDMODEL_ROOT`, `MODEL`, `CPU_FIXTURE`, `CUDA_FIXTURE`,
`PYTHON_VERSION`, `RUN_CUDNN`, or `CARGO_LOCKED=0` to override defaults.

Default parity tolerances are per-output: `act_emb=1e-5`, `emb=1e-3`,
`pred=1e-3`, `rollout=2e-3`, and `cost=1e-2`. The Python and Rust comparators
also reject NaNs/Infs and require cost argmin/top-candidate stability.

Latest local CUDA parity result, run on 2026-05-29:

- Host: NVIDIA GeForce RTX 4090, driver `580.159.03`, `nvidia-smi` CUDA
  `13.0`, `nvcc 13.0.88`.
- Python fixture env: PyTorch `2.10.0+cu128`, `torch.cuda.is_available() ==
  True`, `torch.version.cuda == 12.8`.
- Rust checks: `cargo check --locked --features cuda --all-targets`,
  `cargo test --locked --features cuda`, and
  `cargo check --locked --features cudnn --all-targets` all passed.

| Comparison | `emb` max abs | `act_emb` max abs | `pred` max abs | `rollout` max abs | `cost` max abs | Cost argmin |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Python CPU vs Python CUDA | `2.622604e-06` | `9.536743e-07` | `2.563000e-06` | `2.622604e-06` | `4.196167e-05` | stable |
| Candle CPU vs Python CPU | `2.178848e-04` | `1.192093e-06` | `4.816353e-04` | `6.887764e-04` | `4.620552e-03` | stable |
| Candle CUDA vs Python CUDA | `2.174266e-04` | `4.768372e-07` | `4.823357e-04` | `6.892309e-04` | `4.647255e-03` | stable |
| Candle CUDA vs Python CPU | `2.185255e-04` | `9.536743e-07` | `4.818588e-04` | `6.889254e-04` | `4.620552e-03` | stable |

For the Python CPU/CUDA fixture comparison, `pixels`, `actions`, and
`action_candidates` were byte-identical because inputs are generated on CPU
before being copied to the selected backend.

## Source Layout

```text
src/
├── checkpoint.rs        # weight-loading helpers
├── config.rs            # top-level model selection config
├── models/
│   ├── mod.rs
│   └── lewm/            # LeWM backend
│   └── tdmpc2/          # state/vector TD-MPC2 backend
└── bin/
    └── lewm-inspect.rs  # LeWM smoke-test CLI
    └── tdmpc2-inspect.rs
```

Future stable-worldmodel backends can be added as sibling modules, for example
`models::pldm` or `models::prejepa`. Crate-root APIs should stay focused on
shared loading and configuration utilities.

## Alignment Notes

The Python repo state-dict path saves checkpoints as:

```text
config.json
weights_epoch_N.pt
```

The Rust model intentionally uses the same module names where possible:

- `encoder.embeddings.*`
- `encoder.encoder.layer.*`
- `encoder.layernorm.*`
- `projector.net.*`
- `action_encoder.patch_embed.*`
- `predictor.transformer.layers.*`
- `pred_proj.net.*`

That means raw LeWM `model.state_dict()` checkpoints should be loadable without renaming, assuming the same LeWM config and action dimension.

TD-MPC2 object checkpoints (`*_object.ckpt`) are serialized Python objects and
are not directly Candle-loadable. For Candle, export a state dict or safetensors
checkpoint plus config.

## Remaining Work

- Promote the fixture comparison into a CI-friendly integration test once small test weights are available.
- Add image preprocessing utilities matching `stable_pretraining.data.transforms.ToImage` plus ImageNet normalization and resize.
- Add TD-MPC2 pixel CNN support and policy rollout sampling.
- Port CEM/iCEM/MPPI planner loops in Rust, keeping candidate evaluation on the selected Candle device.
- Add optional safetensors export guidance for deployments that prefer mmap loading.
- Add additional sibling model backends starting from the simplest production inference path for each model.
