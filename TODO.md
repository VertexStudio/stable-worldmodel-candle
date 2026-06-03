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

## Implemented Capabilities

- LeWM image-model implementation exists.
- LeWM Python/Candle CUDA parity exists through `tools/cuda_parity.sh`.
- `lewm-plan-fixture` runs CEM/MPPI/iCEM against checkpoint-backed LeWM fixture
  inputs and public checkpoints; the PushT validation improves over the fixture
  candidate baseline with all three solvers.
- `lewm-plan-images` runs a LeWM checkpoint from JPEG current/goal images
  through nvJPEG, Candle CUDA preprocessing, LeWM encode/rollout/scoring, and a
  Rust planner, then emits HTML plus JSON.
- LeWM session scoring uses the cached input history length, so image-history
  runs such as PushT `history_size=1` are scored with the same rollout history
  semantics as the Python comparison.
- `tools/run_pusht_lewm_rust_demo.py` runs `swm/PushT-v1` from
  a `pusht_expert_train.h5` start/goal sample, executes Rust-planned LeWM
  actions, and emits an HTML/JSON/GIF rollout report. Validation snapshot:
  dataset row `209214`, two iCEM replans, `47` executed actions, success
  `true`.
- `tools/benchmark_lewm_plan_images_python.py` compares the same image-input
  LeWM planning workload against official Python/PyTorch inference.
- Upstream `stable-worldmodel` support is tracked in
  `docs/upstream-stable-worldmodel.md`; the audited commit is
  `40dff37fc983c5276ada65eb1c7873cefbcccd8a`.
- TD-MPC2 state/vector inference and CUDA fixture parity exist.
- Python parity tooling runs from this repo's `pyproject.toml`/`uv.lock`
  and depends on the official `stable-worldmodel[train]` package.
- `tools/convert_state_dict_safetensors.py` converts PyTorch tensor state dicts
  into deployment-preferred `model.safetensors` files.
- TD-MPC2 pixel CNN inference exists for NCHW/NHWC image tensors with CUDA
  fixture parity for pixel-only and mixed pixel+state.
- Rust preprocessing exists for decoded RGB frame stacks, single-frame pixel
  tensors, normalized state arrays, and clamped action arrays.
- `media` ingestion decodes JPEG bytes through nvJPEG into Candle CUDA
  U8 RGB tensors and preprocesses packed U8 RGB/BGR/RGBA/BGRA CUDA tensors into
  normalized F32 NCHW or NTCHW Candle tensors.
- CUDA NV12 preprocessing converts CUDA-resident Y and UV planes through
  fused BT.601/BT.709 color conversion, resize, normalization, and NCHW/history
  writes for video-surface ingestion.
- NVDECODE capability probing binds the Candle CUDA context and queries
  `libnvcuvid` for codec/chroma/bit-depth support; Rust and C ABI entrypoints
  cover 4:2:0 H.264/HEVC/AV1/VP9 probes.
- NVDECODE decoder lifecycle creates and destroys an 8-bit 4:2:0 CUVID
  decoder with NV12 output on the Candle CUDA context; Rust and C ABI tests
  cover H.264 decoder allocation.
- NVDECODE parser sessions accept Annex B packets, decode pictures, map
  display frames, and launch a CUDA copy into Rust-owned CUDA NV12 buffers; Rust
  and C ABI entrypoints cover parser lifecycle and packet validation, with an
  opt-in H.264 packet validation through `SWM_NVDEC_TEST_PACKET`.
- The C ABI exposes Rust-owned CUDA packed-image and NV12 media buffers,
  device pointer queries, and TD-MPC2/LeWM reset calls that preprocess those
  buffers before session reset. The CUDA media reset paths cache packed-image
  and NV12 preprocessor output tensors inside the runtime handle across
  matching calls.
- `runtime-bench` reports p50/p95/p99 runtime measurements for synthetic LeWM
  and TD-MPC2 paths, including packed-image/NV12 CUDA preprocessing and
  TD-MPC2 CEM/MPPI/iCEM planning latency.
- Python-vs-Rust TD-MPC2 CUDA runtime benchmarking includes encoded JPEG
  ingestion (`media_jpeg`), common model sections, and split fixed/generated
  sampled actor rollout rows, with an SVG comparison graph in `docs/`.
- Python-vs-Rust LeWM image planning benchmarking reports synchronized CUDA
  p50/p95/p99 stats for repeated runs and regenerates the README graph from
  p50 latency.
- LeWM PLDM, VCReg, and temporal-straightening training losses are implemented
  in Rust/Candle and validated against the official Python CUDA loss modules.
- LeWM batch-loss training API accepts CUDA pixel/action tensors, runs model
  forward loss computation, and is covered by a CUDA forward/backward/AdamW
  smoke test that saves updated safetensors and reloads them through the runtime
  checkpoint loader.
- `lewm-train-batch` trains from fixed NPZ pixel/action batches on CUDA and
  writes updated safetensors.
- `tools/export_pusht_lewm_training_batch.py` exports PushT H5 image/action
  histories into the NPZ contract consumed by `lewm-train-batch`.
- Family-specific runtime session APIs exist for LeWM and TD-MPC2.
- TD-MPC2 actor-mean and stochastic sampled policy rollouts run through Candle
  CUDA tensors and are exposed through the Rust model API, session API,
  benchmark harness, Python CUDA parity fixtures, and C ABI.
- CEM exists as the first Rust-native planning solver. It keeps candidate
  generation, rollout/scoring, and elite selection in Candle tensors on the
  selected device.
- MPPI exists and keeps its softmax-weighted control update in Candle tensors
  on the selected device.
- iCEM exists with CUDA lowest-k elite selection, elite carryover between
  iterations, and a shifted warm-start sequence between `plan` calls.
- Deployment interfaces are currently the Rust library, CLI tools, and an
  initial C ABI for TD-MPC2 state/vector, pixel, and mixed state+pixel
  CEM/MPPI/iCEM planning plus LeWM image-history goal planning.

## Phase 1: Baseline Parity And Benchmarks

**Goal**

Make baseline behavior measurable before optimizing it.

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

Move the MPC runtime workload into Rust/Candle.

A model forward pass answers: "what happens if I take this action?" A planner
answers: "which action should I take at this step?" For MPC-style control, each runtime
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
- CEM/iCEM elite selection uses the CUDA lowest-k path plus Candle gather on the
  selected device. `PlanResult::used_host_elite_selection` is false for the
  built-in planners.
- `runtime-bench --model td-mpc2` reports CEM, MPPI, and iCEM planner latency
  using the same session/scorer path as deployment code.
- `runtime-bench --model td-mpc2` reports representative TD-MPC2 C ABI rows for
  actor mean action, actor policy rollout, sampled policy rollout, and
  CEM/MPPI/iCEM planning.
- `runtime-bench --model le-wm` reports representative LeWM C ABI planner rows
  for CEM, MPPI, and iCEM.
- `lewm-plan-fixture` validates checkpoint-backed LeWM goal planning with
  planner-sampled candidates, scored through `LeWmGoalScorer` on Candle CUDA.
- `runtime-bench` reports `media_jpeg`, `media_packed`, and `media_nv12` rows
  so encoded image ingestion and image/video preprocessing kernels are tracked
  with the same p50/p95/p99 harness as model and planner work.
- TD-MPC2 sampled actor rollout uses explicit CUDA noise tensors for parity and
  Candle CUDA RNG noise for deployment runs.
- Planner seeded sampling uses planner-owned cuRAND generators on the Candle
  CUDA stream, reserves non-overlapping offset ranges per `plan` call, and keeps
  candidate noise generation inside CUDA tensors.
- CEM/MPPI/iCEM planners cache reusable action-bound tensors and initial
  mean/std tensors per CUDA device location, dtype, shape, and scalar value.
- Deadline handling is implemented for zero-completed-iteration cases: CEM and
  MPPI use a configured action, while iCEM prefers its warm-start sequence
  before using the configured action. `PlanResult` reports which path was used.
- Planner configs expose a seed for deterministic CUDA RNG sampling; new
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
- Larger candidate, score, latent, and rollout buffer reuse is benchmarked and
  implemented where it reduces steady-state planner latency.
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

## Phase 10: LeWM Training Surfaces

**Goal**

Make the model/loss/optimizer step possible in Rust while keeping the runtime
focus on NVIDIA CUDA.

**Build**

- Keep official CUDA parity for PLDM, VCReg, and temporal-straightening losses.
- Add trainable LeWM construction through `candle_nn::VarMap` so gradients can
  flow through the same model modules used for inference.
- Add a batch loss API that accepts CUDA pixel/action tensors, encodes the
  observation history, predicts latent transitions, and returns named loss
  tensors.
- Add an AdamW training-step harness over fixed mini-batches before adding
  streaming dataset ingestion.
- Add safetensors checkpoint save/load for the trained Rust weights.

**Status**

- `models::lewm::loss` implements PLDM inverse-dynamics MSE, temporal-alignment
  MSE, VCReg variance/covariance terms, and temporal-straightening loss.
- `tools/export_lewm_training_loss_fixture.py` exports official Python CUDA loss
  outputs.
- `lewm-compare-training-loss` validates Rust/Candle CUDA loss outputs against
  the Python export. Validation snapshot on 2026-06-03: max abs
  `1.192093e-7` across the tracked scalar loss outputs.
- `models::lewm::training::batch_loss` computes weighted prediction, PLDM/VCReg,
  and temporal-straightening terms from CUDA pixel/action batches.
- `lewm_training_step_updates_and_reloads_cuda_weights` builds a tiny trainable
  LeWM with `candle_nn::VarMap`, runs backward, applies AdamW, verifies CUDA
  variables update, saves safetensors, and reloads them through the runtime
  checkpoint loader.
- `lewm-train-batch` is an executable Rust training entrypoint for fixed
  mini-batches. A 2026-06-03 tiny CUDA run moved total loss from
  `4.54091215e0` to `4.52249146e0` over two AdamW steps and saved
  `target/lewm-train-tiny-output.safetensors`.
- `tools/export_pusht_lewm_training_batch.py` exported PushT rows `1459998` and
  `2206878` into model-ready pixels `(2,3,3,224,224)` and normalized action
  blocks `(2,3,10)`. `lewm-train-batch` ran one full LeWM CUDA AdamW step on
  that batch, moving total loss from `6.78525972e0` to `6.72897959e0` and saving
  `target/pusht-lewm-trained-smoke.safetensors`.
- A longer random-initialized PushT run moved total loss from `6.78525972e0` to
  `6.23445511e0` over ten AdamW steps. A checkpoint-initialized PushT run used
  `target/lewm-pusht-model.safetensors` converted from the public
  `quentinll/lewm-pusht` `weights.pt`, moved total loss from `2.20985317e0` to
  `2.19493628e0` over three AdamW steps, and saved
  `target/pusht-lewm-checkpoint-trained-smoke.safetensors`.

**Done When**

- Loss terms match official Python CUDA on fixed input batches before the update.
- Streaming PushT batch iteration runs through the Rust model/loss/optimizer
  step and writes periodic safetensors checkpoints.

## Standard Checks

- `cargo check --locked --all-targets`
- `cargo test --locked`
- `tools/cuda_parity.sh` when parity behavior changes

Each chunk should add narrower tests for the behavior it implements.
