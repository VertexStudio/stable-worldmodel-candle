#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWM_ROOT="${STABLE_WORLDMODEL_ROOT:-}"
MODEL="${MODEL:-quentinll/lewm-pusht}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
CUDA_FIXTURE="${CUDA_FIXTURE:-"$ROOT/target/lewm-pusht-python-cuda.npz"}"
RUN_CUDNN="${RUN_CUDNN:-auto}"
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

cudnn_available() {
  if [[ -n "${CUDNN_LIB_DIR:-}" || -n "${CUDNN_INCLUDE_DIR:-}" ]]; then
    return 0
  fi
  if command -v ldconfig >/dev/null 2>&1; then
    ldconfig -p 2>/dev/null | grep -q 'libcudnn'
    return
  fi
  return 1
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

section "Rust CUDA build"
(
  cd "$ROOT"
  cargo check "${cargo_locked_args[@]}" --features cuda --all-targets
  cargo test "${cargo_locked_args[@]}" --features cuda
)

case "$RUN_CUDNN" in
  1|true|yes)
    section "Rust cuDNN build"
    (cd "$ROOT" && cargo check "${cargo_locked_args[@]}" --features cudnn --all-targets)
    ;;
  auto)
    if cudnn_available; then
      section "Rust cuDNN build"
      (cd "$ROOT" && cargo check "${cargo_locked_args[@]}" --features cudnn --all-targets)
    else
      echo "Skipping cuDNN build check; set RUN_CUDNN=1 to force it."
    fi
    ;;
  0|false|no)
    echo "Skipping cuDNN build check."
    ;;
  *)
    echo "unsupported RUN_CUDNN=$RUN_CUDNN" >&2
    exit 1
    ;;
esac

section "Python LeWM CUDA fixture"
run_python "$ROOT/tools/export_lewm_fixture.py" \
  "${swm_root_args[@]}" \
  --model "$MODEL" \
  --device cuda \
  --output "$CUDA_FIXTURE"

section "Candle CUDA vs Python CUDA"
(
  cd "$ROOT"
  cargo run --release "${cargo_locked_args[@]}" --features 'cuda hub' \
    --bin lewm-compare-fixture -- \
    --device cuda \
    --fixture "$CUDA_FIXTURE" \
    --hf-repo "$MODEL"
)
