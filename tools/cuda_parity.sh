#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWM_ROOT="${STABLE_WORLDMODEL_ROOT:-}"
MODEL="${MODEL:-quentinll/lewm-pusht}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
CUDA_FIXTURE="${CUDA_FIXTURE:-"$ROOT/target/lewm-pusht-python-cuda.npz"}"
CARGO_LOCKED="${CARGO_LOCKED:-1}"

section() {
  printf '\n== %s ==\n' "$1"
}

cargo_locked_args=()
if [[ "$CARGO_LOCKED" != "0" ]]; then
  cargo_locked_args+=(--locked)
fi

uv_args=(
  uv run
  --project "$ROOT"
  --python "$PYTHON_VERSION"
  --no-dev
)

swm_root_args=()
if [[ -n "$SWM_ROOT" ]]; then
  swm_root_args+=(--stable-worldmodel-root "$SWM_ROOT")
fi

run_python() {
  (
    cd "$ROOT"
    "${uv_args[@]}" python "$@"
  )
}

section "Environment sanity"
nvidia-smi
nvcc --version || true
(
  cd "$ROOT"
  "${uv_args[@]}" python - <<'PY'
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
)

section "Rust CUDA/cuDNN build"
(
  cd "$ROOT"
  cargo check "${cargo_locked_args[@]}" --all-targets
  cargo test "${cargo_locked_args[@]}"
)

section "Python LeWM CUDA fixture"
run_python "$ROOT/tools/export_lewm_fixture.py" \
  "${swm_root_args[@]}" \
  --model "$MODEL" \
  --device cuda \
  --output "$CUDA_FIXTURE"

section "Candle CUDA vs Python CUDA"
(
  cd "$ROOT"
  cargo run --release "${cargo_locked_args[@]}" --features hub \
    --bin lewm-compare-fixture -- \
    --device cuda \
    --fixture "$CUDA_FIXTURE" \
    --hf-repo "$MODEL"
)
