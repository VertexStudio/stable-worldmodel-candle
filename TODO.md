# TODO: Realtime Inference Runtime Roadmap

This roadmap is for turning `stable-worldmodel-candle` into a deployable
Rust/Candle runtime for world-model control. The deployment loop we are building
toward is:

```text
image/video/state observation -> latent -> candidate rollout -> cost -> action
```

The important property is that this loop stays inside one Rust runtime and, as
much as Candle permits, on one selected device. That is what makes latency
predictable and deployment practical.

## Working Protocol

- Work on `main` unless explicitly told otherwise.
- Commit each completed capability as its own chunk.
- Push after each completed commit or small group of tightly related commits.
- Run the relevant checks before committing the chunk.
- Keep commits capability-based, for example:
  - `rewrite realtime roadmap`
  - `add runtime device specs`
  - `add runtime benchmark harness`
  - `add deployment artifact loader`
  - `add image preprocessing parity`
  - `add tdmpc2 pixel encoder`
  - `add cem planner`
  - `add c abi runtime entrypoints`

## Current State

- LeWM image-model implementation exists.
- LeWM Python/Candle CPU/CUDA parity exists through `tools/cuda_parity.sh`.
- TD-MPC2 state/vector inference exists.
- TD-MPC2 fixture parity is not complete.
- Rust preprocessing for raw image/video/state inputs is not complete.
- There is no benchmark harness for p50/p95/p99 runtime measurements.
- There is no runtime session API.
- There are no Rust-native planning solvers.
- Deployment interfaces are currently the Rust library plus CLI tools.

## Phase 1: Baseline Parity And Benchmarks

**Goal**

Make current behavior measurable before optimizing it.

**Build**

- Keep `tools/cuda_parity.sh` as the LeWM CUDA validation path.
- Add TD-MPC2 state/vector fixture parity:
  - Python CPU vs Python CUDA.
  - Candle CPU vs Python CPU.
  - Candle CUDA vs Python CUDA.
  - Candle CUDA vs Python CPU.
- Add shared runtime parsing for device and dtype choices:
  - CPU.
  - CUDA device index when built with `cuda`.
  - Metal device index when built with `metal`.
  - F32 first; BF16/F16 only when explicitly requested and supported.
- Add `runtime-bench`:
  - benchmark preprocessing, encode, dynamics, rollout, candidate scoring, and
    full planning once planners exist;
  - call `Device::synchronize()` before and after timed CUDA/Metal sections;
  - report p50, p95, p99, mean, warmup count, iteration count, model, device,
    dtype, batch size, frame/history count, candidate samples, horizon, and git
    commit;
  - support compact text output and JSON output.

**Done When**

- LeWM parity still passes on CPU and CUDA.
- TD-MPC2 state/vector parity passes on CPU and CUDA.
- `runtime-bench` can run against synthetic LeWM and TD-MPC2 inputs.
- Benchmark output is reproducible enough to compare before/after changes.

## Phase 2: Portable Artifacts And Schemas

**Goal**

Load deployments from stable artifact directories without Python object
checkpoints, Hydra instantiation, or Lightning modules.

**Build**

- Define a deployment package layout:
  - `config.json`
  - preferred `model.safetensors`
  - optional compatibility `weights.pt`
  - `preprocess.json`
  - `schema.json`
- Add a package loader that validates:
  - model family and config version;
  - expected tensor names;
  - observation schema;
  - action schema;
  - dtype;
  - selected Candle device.
- Keep `.pt` loading as an import/compatibility path, not the preferred runtime
  package format.

**Done When**

- LeWM and TD-MPC2 can load from a directory package.
- Invalid package metadata fails with clear errors.
- README documents the package layout and conversion expectations.

## Phase 3: Image, Video, State, And Action Preprocessing

**Goal**

Make image/video/state inputs first-class Rust runtime inputs.

**Build**

- Add Rust preprocessing for decoded RGB frames:
  - resize;
  - crop/pad policy;
  - channel layout conversion;
  - ImageNet normalization where required;
  - frame/history stacking;
  - dtype and device transfer.
- Add state/action preprocessing:
  - schema validation;
  - action scaling and clamping;
  - history stacking.
- Keep heavy decode/resize dependencies optional.
- Keep the core runtime able to accept already-decoded frame buffers or tensors.
- Extend LeWM parity from synthetic tensors to raw image/frame inputs.

**Done When**

- Raw decoded image/video frames can become model-ready tensors in Rust.
- State and action inputs are validated and normalized in Rust.
- Rust preprocessing matches Python transforms within documented tolerances.
- CLIs can run from raw frame/state/action inputs, not only synthetic tensors.

## Phase 4: TD-MPC2 Pixel And Fixture Coverage

**Goal**

Make TD-MPC2 usable for both state/vector and pixel observation deployments.

**Build**

- Add TD-MPC2 pixel CNN encoder support.
- Add Python fixture export for TD-MPC2:
  - deterministic state/vector inputs;
  - deterministic image inputs;
  - action candidates;
  - state dict or safetensors export;
  - encode, forward, actor mean action, rollout where applicable, and get_cost.
- Add Rust fixture comparison for the same surfaces.

**Done When**

- TD-MPC2 state/vector parity passes on CPU and CUDA.
- TD-MPC2 pixel parity passes on CPU and CUDA.
- `get_cost` and actor behavior are covered by parity tests.
- Pixel and state/vector encoders are both supported from deployment packages.

## Phase 5: Runtime Session API

**Goal**

Avoid rebuilding runtime state every control step.

**Build**

- Add family-specific sessions first:
  - `LeWmSession`
  - `TdMpc2Session`
- Each session owns:
  - model;
  - device;
  - dtype;
  - observation history;
  - action history;
  - frame buffers;
  - cached latents;
  - optional warm-started candidate actions.
- Initial API:
  - `reset(initial_observation)`;
  - `update_observation(observation)`;
  - `encode_current()`;
  - `score_candidates(action_candidates)`;
  - `plan_next_action(...)` after solvers are available.
- Do not force a shared trait until LeWM and TD-MPC2 prove the common surface.

**Done When**

- LeWM and TD-MPC2 can be driven through session objects.
- Sessions preserve history and cached state correctly across repeated steps.
- Device placement is stable and explicit.

## Phase 6: Device-Resident Planning Solvers

**Goal**

Move the real MPC workload into Rust/Candle.

A model forward pass answers: "what happens if I take this action?" A planner
answers: "which action should I take now?" For MPC-style control, each runtime
step generates many candidate action sequences, rolls the world model forward,
scores trajectories, picks the best first action, and repeats at the next
timestep. If this stays in Python, Rust/Candle only removes part of the
overhead.

**Build**

- Add common planning types:
  - `PlanConfig` with horizon, samples, iterations, elite count/fraction,
    action bounds, temperature/noise, optional seed, and optional deadline;
  - `PlanResult` with first action, full action sequence, scores, best index,
    iterations completed, elapsed time, and fallback status.
- Add a scorer interface around candidate tensors shaped
  `[batch, samples, horizon, action_dim]`.
- Implement solvers in this order:
  - CEM first as the reference planner;
  - MPPI second;
  - iCEM third after session warm-start behavior exists.
- Keep candidate generation, rollout, scoring, elite/update logic, and
  best-action selection on the selected Candle device.
- Avoid Python loops, host/device round trips, and repeated allocation wherever
  Candle allows.

**Done When**

- CEM can plan end-to-end with LeWM and TD-MPC2 sessions.
- MPPI and iCEM have matching runtime APIs.
- Solver output is finite, bounded, deterministic under fixed seeds where
  expected, and stable in top-candidate ranking.
- Candidate tensors remain device-resident except explicit final result
  extraction.

## Phase 7: Deadline-Aware Planning And Hot-Path Optimization

**Goal**

Make control-loop timing predictable.

**Build**

- Add deadline or max-duration planning support.
- Deadline behavior:
  - return the best completed iteration if at least one iteration finished;
  - otherwise return the previous warm-start action if available;
  - otherwise return a configured fallback action.
- Surface timeout/fallback behavior in `PlanResult`.
- Optimize after benchmarks exist:
  - reduce avoidable tensor clones;
  - reduce host/device transfers;
  - reuse candidate, score, latent, and rollout buffers where practical;
  - remove repeated history construction from per-step paths.

**Done When**

- Deadline behavior is deterministic and tested.
- Benchmarks report full planning latency and fallback rates.
- Repeated planning loops show lower p95/p99 latency after optimization.

## Phase 8: Precision Modes

**Goal**

Use lower precision only when it preserves planning decisions.

**Build**

- Keep F32 as the baseline.
- Add BF16/F16 per backend only after parity passes.
- Compare cost ranking and selected action, not only raw tensor error.
- Defer INT8 and other quantized paths until F32/BF16/F16 runtime behavior is
  stable.

**Done When**

- Precision mode is an explicit runtime/config choice.
- Lower precision has documented accuracy and latency tradeoffs.
- Selected-action agreement is tracked for planner outputs.

## Phase 9: Deployment Interfaces

**Goal**

Expose the stable runtime without forcing a Python service.

**Build**

- Keep the Rust API as the primary interface.
- Add C ABI after the Rust session and planner APIs stabilize.
- Defer ROS2, gRPC, and WASM to future integration work.

**Done When**

- Rust users can load a deployment package, create a session, and plan actions.
- C callers can load a package, submit observations, and receive actions through
  a small stable ABI.
- C ABI overhead is measured separately from core runtime latency.

## Standard Checks

- `cargo check --locked --all-targets`
- `cargo test --locked`
- `cargo check --locked --features cuda --all-targets`
- `cargo test --locked --features cuda`
- `cargo check --locked --features cudnn --all-targets` when cuDNN is installed
- `tools/cuda_parity.sh` when parity behavior changes

Each chunk should add narrower tests for the behavior it implements.
