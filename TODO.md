# TODO: Realtime Inference Roadmap

This roadmap is scoped to deployed inference for `stable-worldmodel`
checkpoints. The main goals are low latency, predictable p95/p99 timing,
minimal runtime dependencies, backend parity, and clean integration into
robotics, simulation, and control systems.

## 1. Standardize A Portable Checkpoint Package

**What**

- Define a deployment artifact layout:
  - `config.json`
  - `model.safetensors` or `weights.pt`
  - preprocessing metadata
  - action normalization metadata
  - model family/version
  - expected observation/action schema

**Why**

Realtime inference should not depend on Python object checkpoints, Hydra
instantiation, Lightning modules, or package-version-sensitive model loading.
A stable artifact format makes startup predictable and makes deployments
repeatable across CPU, CUDA, Metal, and embedded targets.

**Performance benefit**

- Faster and safer cold starts.
- Fewer runtime branches and fewer failure modes at model load time.
- Enables memory-mapped safetensors for lower startup memory pressure.

**Done when**

- A model can be loaded from one directory without Python.
- The loader validates model type, tensor names, input schema, dtype, and device.
- LeWM and TD-MPC2 have documented artifact examples.

## 2. Build A CPU/CUDA/Python Parity Matrix

**What**

- Extend fixture export tools with `--device cpu|cuda`.
- Compare:
  - Python CPU vs Python CUDA
  - Candle CPU vs Python CPU
  - Candle CUDA vs Python CUDA
  - Candle CUDA vs Python CPU
- Track per-output tolerances for embeddings, dynamics, rollouts, and costs.

**Why**

Before optimizing CUDA inference, we need to know whether differences come from
PyTorch backend drift, Candle backend drift, precision, preprocessing, or model
implementation mistakes.

**Performance benefit**

- Prevents performance work from hiding correctness regressions.
- Gives confidence that CUDA speedups preserve planner decisions.
- Catches backend-specific issues before deployment.

**Done when**

- LeWM parity can run on Linux CUDA from one documented command.
- TD-MPC2 state/vector parity can run from generated fixtures.
- Reports include max/mean error, cost error, and top-candidate agreement.

## 3. Implement Rust Preprocessing

**What**

- Add Rust preprocessing for:
  - image resize
  - channel layout conversion
  - ImageNet normalization
  - dtype/device transfer
  - action scaling, clamping, and history stacking

**Why**

If input preparation still requires Python, the runtime is not truly deployable
as a lightweight Rust/Candle backend. Preprocessing is also a common source of
silent parity errors.

**Performance benefit**

- Removes Python from the inference path.
- Reduces unnecessary copies and layout conversions.
- Makes end-to-end latency measurable inside the Rust runtime.

**Done when**

- A raw observation can be converted into model-ready tensors in Rust.
- Preprocessing parity is tested against the original Python transforms.
- The CLIs can run from raw image/state/action inputs, not only synthetic tensors.

## 4. Add A Realtime Session API

**What**

- Introduce a runtime API that owns cached model state:

```rust
session.reset(initial_observation)?;
session.update_observation(observation)?;
let action = session.plan_next_action(deadline)?;
```

- Keep observation history, action history, cached latents, and warm-started
  candidate sequences inside the session.

**Why**

Realtime control loops should not rebuild histories, reallocate tensors, or
re-encode unchanged context every step. A session object gives users a stable
runtime abstraction instead of requiring them to manually wire model internals.

**Performance benefit**

- Lower per-step latency.
- Lower jitter from repeated setup work.
- Easier integration into simulators, robot controllers, and services.

**Done when**

- LeWM and TD-MPC2 can be driven through a shared session-style API.
- The API supports reset, step/update, planning, and device selection.
- Cached-state behavior is covered by tests.

## 5. Remove Allocations From The Hot Path

**What**

- Add reusable workspaces for rollout/planning tensors.
- Preallocate candidate actions, scores, latent buffers, and temporary tensors.
- Avoid shape-dependent allocation in repeated control steps where practical.

**Why**

Average latency is not enough for realtime systems. Allocation spikes can break
control deadlines even when the model is small.

**Performance benefit**

- More predictable p95/p99 latency.
- Lower allocator pressure under high-frequency planning.
- Cleaner profiling because model compute is separated from setup overhead.

**Done when**

- A repeated planning loop can run with stable allocation behavior.
- Benchmarks report allocation counts or an equivalent no-growth check.
- Hot-path APIs reuse buffers across calls.

## 6. Port Device-Resident Planning Solvers

**What**

- Implement Candle-native planners:
  - CEM
  - iCEM
  - MPPI
- Keep candidate generation, rollout, scoring, elite selection, and action
  updates on the selected device.

**Why**

For MPC-style inference, the planner is often the real runtime bottleneck, not
one model forward pass. Calling back into Python or moving candidates between
CPU and GPU defeats the purpose of a Candle deployment runtime.

**Performance benefit**

- Large speedup for thousands of candidate trajectories.
- Fewer CPU/GPU synchronizations.
- Better CUDA utilization during latent rollout and cost evaluation.

**Done when**

- At least one solver can run end-to-end with LeWM and TD-MPC2.
- Candidate tensors stay device-resident during planning.
- Planner outputs match Python solver behavior within documented tolerances.

## 7. Add Deadline-Aware Planning

**What**

- Support planning with a hard time budget.
- Degrade gracefully by reducing iterations, sample count, or horizon.
- Return a valid fallback action from the previous plan when a deadline is hit.

**Why**

Realtime control systems need bounded behavior. A slower but bounded planner is
often more useful than a better plan that sometimes misses the control deadline.

**Performance benefit**

- Predictable control-loop timing.
- Safer behavior under load.
- Easier deployment in robotics and simulation services with fixed tick rates.

**Done when**

- Planning APIs accept a deadline or max-duration setting.
- Timeout behavior is deterministic and tested.
- Benchmarks report success/fallback rates under constrained budgets.

## 8. Add Latency And Throughput Benchmarks

**What**

- Benchmark:
  - preprocessing
  - encoder latency
  - one-step dynamics latency
  - rollout latency by horizon/sample count
  - full planning loop latency
  - memory use
  - p50/p95/p99 timing

**Why**

Deployment work needs numbers. Without stable benchmarks, it is hard to know
whether CUDA, Metal, batching, precision changes, or planner changes actually
improve realtime performance.

**Performance benefit**

- Makes regressions visible.
- Guides optimization priorities.
- Helps users choose device, dtype, horizon, and sample count.

**Done when**

- Benchmarks run on CPU, Metal, and CUDA where available.
- Results are printed in a compact, comparable format.
- CI or release notes can include benchmark snapshots.

## 9. Support Mixed Precision And Quantization Safely

**What**

- Add controlled support for:
  - F32
  - BF16/F16
  - INT8 or other quantized paths where accuracy is acceptable
- Compare final costs and selected actions, not just raw tensor error.

**Why**

Precision changes are attractive for latency and memory, but planning can be
sensitive to small cost changes. The runtime needs accuracy gates before using
lower precision by default.

**Performance benefit**

- Lower memory bandwidth.
- Better GPU throughput.
- Smaller deployment artifacts.

**Done when**

- Precision modes are explicit runtime/config choices.
- Parity tests include cost ranking and selected-action agreement.
- Quantized artifacts have documented accuracy and latency tradeoffs.

## 10. Complete TD-MPC2 Production Inference Coverage

**What**

- Add the missing TD-MPC2 inference pieces:
  - pixel CNN encoder path
  - stochastic policy rollout where needed
  - fixture export from Python TD-MPC2
  - real checkpoint/state-dict loading path

**Why**

TD-MPC2 is a strong fit for realtime model-predictive control. The current
state/vector path is useful, but production parity requires the same inference
surfaces users train and evaluate in Python.

**Performance benefit**

- Enables fast latent planning on compact TD-MPC2 models.
- Gives a realtime-oriented backend beyond image JEPA rollouts.
- Exercises planner APIs against a model family designed for MPC.

**Done when**

- TD-MPC2 Python fixtures compare against Candle on CPU and CUDA.
- Pixel and state/vector encoders are supported.
- `get_cost` and actor rollout behavior are covered by parity tests.

## 11. Add Deployment Interfaces

**What**

- Keep the Rust library API as the primary interface.
- Add optional integration layers as needed:
  - C ABI
  - ROS2 wrapper
  - gRPC service
  - WASM build where model/backend support allows it

**Why**

The runtime should be easy to embed in control software, simulators, and
services without forcing users into a specific application framework.

**Performance benefit**

- Avoids Python service overhead.
- Allows direct integration into latency-sensitive systems.
- Enables different deployment models without changing core inference code.

**Done when**

- At least one non-Rust integration path can run a loaded checkpoint.
- Integration layers call the same tested runtime/session APIs.
- Latency overhead of each wrapper is measured.
