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
- LeWM Python/Candle CUDA parity exists through `tools/cuda_parity.sh`.
- TD-MPC2 state/vector inference and CUDA fixture parity exist.
- Python parity tooling now runs from this repo's `pyproject.toml`/`uv.lock`
  and depends on the official `stable-worldmodel[train]` package.
- `tools/convert_state_dict_safetensors.py` converts PyTorch tensor state dicts
  into deployment-preferred `model.safetensors` files.
- TD-MPC2 pixel CNN inference exists for NCHW/NHWC image tensors with CUDA
  fixture parity for pixel-only and mixed pixel+state.
- Rust preprocessing exists for decoded RGB frame stacks, latest-frame pixel
  tensors, normalized state arrays, and clamped action arrays.
- `media` ingestion now decodes JPEG bytes through nvJPEG into Candle CUDA
  U8 RGB tensors and preprocesses packed U8 RGB/BGR/RGBA/BGRA CUDA tensors into
  normalized F32 NCHW or NTCHW Candle tensors.
- CUDA NV12 preprocessing now converts CUDA-resident Y and UV planes through
  fused BT.601/BT.709 color conversion, resize, normalization, and NCHW/history
  writes for video-surface ingestion.
- NVDECODE capability probing now binds the Candle CUDA context and queries
  `libnvcuvid` for codec/chroma/bit-depth support; Rust and C ABI entrypoints
  cover 4:2:0 H.264/HEVC/AV1/VP9 probes.
- NVDECODE decoder lifecycle now creates and destroys an 8-bit 4:2:0 CUVID
  decoder with NV12 output on the Candle CUDA context; Rust and C ABI tests
  cover H.264 decoder allocation.
- The C ABI now exposes Rust-owned CUDA packed-image and NV12 media buffers,
  device pointer queries, and TD-MPC2/LeWM reset calls that preprocess those
  buffers before session reset.
- `runtime-bench` reports p50/p95/p99 runtime measurements for synthetic LeWM
  and TD-MPC2 paths, including TD-MPC2 CEM/MPPI/iCEM planning latency.
- Family-specific runtime session APIs exist for LeWM and TD-MPC2.
- TD-MPC2 actor-mean and stochastic sampled policy rollouts run through Candle
  CUDA tensors and are exposed through the Rust model API, session API,
  benchmark harness, Python CUDA parity fixtures, and C ABI.
- CEM exists as the first Rust-native planning solver. It keeps candidate
  generation, rollout/scoring, and elite selection in Candle tensors on the
  selected device.
- MPPI exists and keeps its softmax-weighted control update in Candle tensors
  on the selected device.
- iCEM exists with device-side elite selection, elite carryover between
  iterations, and a shifted warm-start sequence between `plan` calls.
- Deployment interfaces are currently the Rust library, CLI tools, and an
  initial C ABI for TD-MPC2 state/vector, pixel, and mixed state+pixel
  CEM/MPPI/iCEM planning plus LeWM image-history goal planning.

## Phase 1: Baseline Parity And Benchmarks

**Goal**

Make current behavior measurable before optimizing it.

**Build**

- Keep `tools/cuda_parity.sh` as the LeWM CUDA validation path.
- Add TD-MPC2 state/vector fixture parity:
  - Candle CUDA vs Python CUDA.
- Add shared runtime parsing for device and dtype choices:
  - CUDA device index.
  - F32 first; BF16/F16 only when explicitly requested and supported.
- Add `runtime-bench`:
  - benchmark preprocessing, encode, dynamics, rollout, candidate scoring, and
    full planning once planners exist;
  - call `Device::synchronize()` before and after timed CUDA sections;
  - report p50, p95, p99, mean, warmup count, iteration count, model, device,
    dtype, batch size, frame/history count, candidate samples, horizon, and git
    commit;
  - support compact text output and JSON output.

**Done When**

- LeWM parity passes on CUDA.
- TD-MPC2 state/vector parity passes on CUDA.
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
  - legacy `weights.pt`
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
- Add NVIDIA media ingestion:
  - nvJPEG decode from encoded JPEG bytes into Candle CUDA RGB tensors;
  - reusable CUDA RGB decode buffers;
  - fused Candle CUDA resize/channel/normalize kernels;
  - history-slot writes for LeWM/video frame windows;
  - NV12 Y/UV CUDA surface preprocessing for video frames;
  - C ABI CUDA media buffer allocation and pointer queries;
  - NVDECODE capability probing through `libnvcuvid`;
  - NVDECODE decoder lifecycle on the Candle CUDA context;
  - NVDECODE parser callbacks and frame mapping into CUDA NV12 buffers;
  - NPP or fused CUDA color conversion for additional YUV surface formats.
- Add state/action preprocessing:
  - schema validation;
  - action scaling and clamping;
  - history stacking.
- Keep the core runtime able to accept decoded frame buffers or tensors.
- Extend LeWM parity from synthetic tensors to raw image/frame inputs.

**Done When**

- Raw encoded image/video inputs and decoded image/video frames can become
  model-ready tensors in Rust.
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

**Status**

- TD-MPC2 pixel CNN support is implemented with the upstream conv layout and
  `pixel_encoder` projection.
- Pixel-only and combined pixel+state encoding are covered by Rust tests.
- Pixel-only fixture export/comparison passes on CUDA.
- Mixed pixel+state fixture export/comparison passes on CUDA.

**Done When**

- TD-MPC2 state/vector parity passes on CUDA.
- TD-MPC2 pixel parity passes on CUDA.
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
  - warm-started candidate actions when enabled.
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
    action bounds, temperature/noise, seed, and deadline;
  - `PlanResult` with first action, full action sequence, scores, best index,
    iterations completed, elapsed time, and result source.
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

**Status**

- CEM is implemented through `planner::CemPlanner`.
- MPPI is implemented through `planner::MppiPlanner`.
- iCEM is implemented through `planner::IcemPlanner`.
- `TdMpc2Session` implements the planner scorer interface directly.
- `LeWmGoalScorer` adapts `LeWmSession` plus a goal embedding to the same
  scorer interface.
- CEM/iCEM elite selection uses Candle sort/gather ops on the selected device.
  `PlanResult::used_host_elite_selection` is false for the built-in planners.
- `runtime-bench --model td-mpc2` reports CEM, MPPI, and iCEM planner latency
  using the same session/scorer path as deployment code.
- `runtime-bench --model td-mpc2` reports representative TD-MPC2 C ABI rows for
  actor mean action, actor policy rollout, sampled policy rollout, and CEM
  planning.
- `runtime-bench --model le-wm` reports representative LeWM C ABI planner rows
  for CEM, MPPI, and iCEM.
- TD-MPC2 sampled actor rollout uses explicit CUDA noise tensors for parity and
  generated Candle CUDA noise for deployment runs.
- Planner seeded sampling now uses planner-owned cuRAND generators on the Candle
  CUDA stream, reserves non-overlapping offset ranges per `plan` call, and keeps
  candidate noise generation inside CUDA tensors.
- Deadline handling is implemented for zero-completed-iteration cases: CEM and
  MPPI use a configured action, while iCEM prefers its warm-start sequence
  before using the configured action. `PlanResult` reports which path was used.
- Planner configs expose a seed for deterministic CUDA RNG sampling; fresh
  planners replay from offset zero, persistent planners advance across control
  steps, and `reset_rng_sequence()` replays from the beginning.

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
  - otherwise return a configured action.
- Surface timeout behavior in `PlanResult`.
- Optimize after benchmarks exist:
  - reduce avoidable tensor clones;
  - reduce host/device transfers;
  - reuse candidate, score, latent, and rollout buffers where practical;
  - remove repeated history construction from per-step paths.

**Done When**

- Deadline behavior is validated across CEM/MPPI/iCEM.
- Deadline behavior is deterministic and tested.
- Benchmarks report full planning latency. Deadline-source reporting remains
  pending.
- Buffer reuse and allocation reduction are benchmarked and implemented where
  they reduce steady-state planner latency.
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

**Status**

- The crate builds both `rlib` and `cdylib`.
- `ffi` exposes TD-MPC2 state/vector, pixel, and mixed state+pixel artifact
  loading, observation reset, dimension accessors, CEM planning, MPPI planning,
  iCEM planning with persistent warm-start state, actor mean action, actor-mean
  and sampled policy rollout, handle cleanup, and thread-local error reporting.
- `ffi` exposes LeWM artifact loading, image-history reset, goal pixel setup,
  CEM/MPPI/iCEM planning, handle cleanup, and thread-local error reporting.

**Done When**

- Rust users can load a deployment package, create a session, and plan actions.
- C callers can load a package, submit observations, and receive actions through
  a small stable ABI.
- TD-MPC2 and LeWM C ABI overhead are measured separately from core runtime
  latency.

## Standard Checks

- `cargo check --locked --all-targets`
- `cargo test --locked`
- `tools/cuda_parity.sh` when parity behavior changes

Each chunk should add narrower tests for the behavior it implements.
