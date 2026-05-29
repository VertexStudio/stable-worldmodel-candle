# stable-worldmodel-rs

Rust/Candle inference port for the `stable-worldmodel` LeWM path.

This crate follows the production-facing subset of the Python model:

```text
pixels -> Hugging Face ViTModel encoder -> projector
actions -> action embedder
embeddings + action embeddings -> AdaLN predictor -> rollout/cost
```

## Current Scope

- ViT-Tiny encoder matching `stable_pretraining.backbone.utils.vit_hf(size="tiny", patch_size=14, image_size=224, pretrained=false, use_mask_token=false)`.
- LeWM projector, action encoder, conditional predictor, latent rollout, and goal MSE cost.
- Loading from PyTorch `.pt` state dicts via `VarBuilder::from_pth`, or from `.safetensors`.
- Rust 2024 edition with local Candle path dependencies from `../candle`.
- A shape smoke-test CLI:

```bash
cargo run --bin lewm-inspect -- --action-dim 2
```

With a checkpoint:

```bash
cargo run --release --bin lewm-inspect -- --weights /path/to/weights_epoch_100.pt --action-dim 2
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

That means raw `model.state_dict()` checkpoints should be loadable without renaming, assuming the same LeWM config and action dimension.

## Remaining Work

- Parse the Python `config.json` directly instead of using `LeWmConfig::tiny_patch14_224(action_dim)`.
- Add a Python/Rust parity test that exports fixed tensors from PyTorch and checks max absolute error against Candle.
- Add image preprocessing utilities matching `stable_pretraining.data.transforms.ToImage` plus ImageNet normalization and resize.
- Port CEM/iCEM planner loops in Rust, keeping candidate evaluation on the selected Candle device.
- Add optional safetensors export guidance for deployments that prefer mmap loading.
