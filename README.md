# stable-worldmodel-rs

Rust/Candle inference runtime for `stable-worldmodel`.

The crate is structured like Candle model support: model families are peers under
`src/models/`, examples and CLIs opt into a specific architecture, and the
top-level API stays neutral.

LeWM is the first implemented backend because its inference graph is compact and
maps directly to Candle primitives:

```text
pixels -> Hugging Face ViTModel encoder -> projector
actions -> action embedder
embeddings + action embeddings -> AdaLN predictor -> rollout/cost
```

## Current Scope

- Neutral top-level modules: `checkpoint`, `config`, and `models`.
- First model backend: `models::lewm`.
- Second model backend: `models::tdmpc2` for state/vector observations.
- LeWM ViT-Tiny encoder matching `stable_pretraining.backbone.utils.vit_hf(size="tiny", patch_size=14, image_size=224, pretrained=false, use_mask_token=false)`.
- LeWM projector, action encoder, conditional predictor, latent rollout, and goal MSE cost.
- Loading from PyTorch `.pt` state dicts via `VarBuilder::from_pth`, or from `.safetensors`.
- Rust 2024 edition with local Candle path dependencies from `../candle`.
- LeWM and TD-MPC2 shape smoke-test CLIs:

```bash
cargo run --bin lewm-inspect -- --action-dim 2
cargo run --bin tdmpc2-inspect -- --state-dim 12 --action-dim 4
```

With a checkpoint:

```bash
cargo run --release --bin lewm-inspect -- --weights /path/to/weights_epoch_100.pt --action-dim 2
cargo run --release --bin tdmpc2-inspect -- --weights /path/to/weights_epoch_250.pt --state-dim 12 --action-dim 4
```

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

## Source Layout

```text
src/
├── checkpoint.rs        # neutral weight-loading helpers
├── config.rs            # top-level model selection config
├── models/
│   ├── mod.rs
│   └── lewm/            # first supported model backend
│   └── tdmpc2/          # state/vector TD-MPC2 backend
└── bin/
    └── lewm-inspect.rs  # LeWM smoke-test CLI
    └── tdmpc2-inspect.rs
```

Future stable-worldmodel backends should be added as sibling modules, for example
`models::pldm` or `models::prejepa`, rather than expanding model-specific APIs at
the crate root.

## Alignment Notes

The Python repo saves checkpoints as:

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

## Remaining Work

- Parse the Python `config.json` directly instead of using `LeWmConfig::tiny_patch14_224(action_dim)`.
- Add a Python/Rust parity test that exports fixed tensors from PyTorch and checks max absolute error against Candle.
- Add image preprocessing utilities matching `stable_pretraining.data.transforms.ToImage` plus ImageNet normalization and resize.
- Add TD-MPC2 pixel CNN support and policy rollout sampling.
- Port CEM/iCEM/MPPI planner loops in Rust, keeping candidate evaluation on the selected Candle device.
- Add optional safetensors export guidance for deployments that prefer mmap loading.
- Add sibling model backends after LeWM, starting from the simplest production inference path for each model.
