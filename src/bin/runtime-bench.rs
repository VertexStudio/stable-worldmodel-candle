use std::{
    ffi::CStr,
    fs,
    path::PathBuf,
    process::Command,
    time::{Duration, Instant},
};

use candle::{IndexOp, Tensor};
use clap::{Parser, ValueEnum};
use serde_json::json;
use stable_worldmodel_candle::media::nvjpeg::NvJpegDecoder;
use stable_worldmodel_candle::{
    checkpoint,
    ffi::{
        SwmCemPlanConfig, SwmIcemPlanConfig, SwmLeWm, SwmMppiPlanConfig, SwmStatus, SwmTdMpc2,
        swm_last_error_message, swm_lewm_plan_cem, swm_lewm_plan_icem, swm_lewm_plan_mppi,
        swm_tdmpc2_actor_mean_action, swm_tdmpc2_plan_cem, swm_tdmpc2_plan_icem,
        swm_tdmpc2_plan_mppi, swm_tdmpc2_rollout_actor_mean, swm_tdmpc2_rollout_actor_sample,
    },
    media::{
        ImagePreprocess, ImagePreprocessor, Nv12ColorSpace, Nv12ImageShape, Nv12Preprocessor,
        PackedImageFormat, PackedImageShape, nv12_tensors, packed_u8_tensor,
    },
    models::{
        lewm::{LeWm, LeWmConfig},
        tdmpc2::{TdMpc2, TdMpc2Config},
    },
    planner::{CemConfig, CemPlanner, IcemConfig, IcemPlanner, MppiConfig, MppiPlanner},
    runtime::{DTypeSpec, DeviceSpec},
    session::TdMpc2Session,
};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ModelArg {
    LeWm,
    TdMpc2,
}

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, value_enum, default_value_t = ModelArg::LeWm)]
    model: ModelArg,

    #[arg(long, default_value_t = DeviceSpec::Cuda(0))]
    device: DeviceSpec,

    #[arg(long, default_value_t = DTypeSpec::F32)]
    dtype: DTypeSpec,

    #[arg(long, default_value_t = 5)]
    warmup: usize,

    #[arg(long, default_value_t = 20)]
    iters: usize,

    #[arg(long, default_value_t = 1)]
    batch_size: usize,

    #[arg(long, default_value_t = 2)]
    samples: usize,

    #[arg(long, default_value_t = 5)]
    horizon: usize,

    #[arg(long, default_value_t = 2)]
    planner_iterations: usize,

    #[arg(long)]
    elites: Option<usize>,

    #[arg(long, default_value_t = false)]
    json: bool,

    #[arg(long, default_value_t = 12)]
    state_dim: usize,

    #[arg(long, default_value_t = 10)]
    action_dim: usize,

    #[arg(long)]
    jpeg_input: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct BenchStats {
    name: &'static str,
    mean_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if args.iters == 0 {
        anyhow::bail!("--iters must be greater than zero");
    }
    let device = args.device.resolve()?;
    let dtype = args.dtype.dtype();
    let stats = match args.model {
        ModelArg::LeWm => bench_lewm(&args, &device, dtype)?,
        ModelArg::TdMpc2 => bench_tdmpc2(&args, &device, dtype)?,
    };

    if args.json {
        let rows = stats
            .iter()
            .map(|stat| {
                json!({
                    "name": stat.name,
                    "mean_ms": stat.mean_ms,
                    "p50_ms": stat.p50_ms,
                    "p95_ms": stat.p95_ms,
                    "p99_ms": stat.p99_ms,
                })
            })
            .collect::<Vec<_>>();
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "git_commit": git_commit(),
                "model": format!("{:?}", args.model),
                "device": args.device.to_string(),
                "dtype": args.dtype.to_string(),
                "batch_size": args.batch_size,
                "samples": args.samples,
                "horizon": args.horizon,
                "planner_iterations": args.planner_iterations,
                "elites": elite_count(&args),
                "warmup": args.warmup,
                "iters": args.iters,
                "jpeg_input": args.jpeg_input.as_ref().map(|path| path.display().to_string()),
                "stats": rows,
            }))?
        );
    } else {
        println!(
            "runtime-bench git={} model={:?} device={} dtype={} batch={} samples={} horizon={} planner_iterations={} elites={} warmup={} iters={}",
            git_commit(),
            args.model,
            args.device,
            args.dtype,
            args.batch_size,
            args.samples,
            args.horizon,
            args.planner_iterations,
            elite_count(&args),
            args.warmup,
            args.iters
        );
        for stat in stats {
            println!(
                "{:<18} mean={:>9.3}ms p50={:>9.3}ms p95={:>9.3}ms p99={:>9.3}ms",
                stat.name, stat.mean_ms, stat.p50_ms, stat.p95_ms, stat.p99_ms
            );
        }
    }

    Ok(())
}

fn bench_lewm(
    args: &Args,
    device: &candle::Device,
    dtype: candle::DType,
) -> anyhow::Result<Vec<BenchStats>> {
    let cfg = LeWmConfig::tiny_patch14_224(args.action_dim);
    let history = cfg.history_size;
    let vb = checkpoint::empty_var_builder(dtype, device);
    let model = LeWm::new(cfg, vb)?;

    let pixels = Tensor::randn(0f32, 1f32, (args.batch_size, history, 3, 224, 224), device)?
        .to_dtype(dtype)?;
    let actions = Tensor::randn(
        0f32,
        1f32,
        (args.batch_size, args.samples, args.horizon, args.action_dim),
        device,
    )?
    .to_dtype(dtype)?;
    let media_config = ImagePreprocess::imagenet_224();
    let media_frames = args.batch_size * history;
    let packed_shape = PackedImageShape::new(media_frames, 224, 224, PackedImageFormat::Rgb);
    let packed_pixels = packed_u8_tensor(packed_shape, device)?;
    let mut packed_preprocessor = ImagePreprocessor::new(device, packed_shape, media_config)?;
    let nv12_shape = Nv12ImageShape::new(media_frames, 224, 224);
    let (nv12_y, nv12_uv) = nv12_tensors(nv12_shape, device)?;
    let mut nv12_preprocessor =
        Nv12Preprocessor::new(device, nv12_shape, Nv12ColorSpace::Bt709Video, media_config)?;

    let emb = model.encode_pixels(&pixels)?;
    let emb_init =
        emb.unsqueeze(1)?
            .broadcast_as((args.batch_size, args.samples, history, emb.dim(2)?))?;
    let rollout = model.rollout_embeddings(&emb_init, &actions)?;
    let goal = emb.i((.., emb.dim(1)? - 1, ..))?;

    if args.samples < 2 {
        anyhow::bail!("LeWM planner benchmarks require --samples >= 2");
    }
    if args.planner_iterations == 0 {
        anyhow::bail!("--planner-iterations must be greater than zero");
    }
    let elites = elite_count(args);
    if elites < 2 {
        anyhow::bail!("LeWM planner benchmarks require --elites >= 2");
    }
    if elites > args.samples {
        anyhow::bail!("--elites cannot exceed --samples");
    }

    let mut ffi_handle = SwmLeWm::synthetic_for_bench(
        LeWmConfig::tiny_patch14_224(args.action_dim),
        dtype,
        device,
        &pixels,
        &goal,
    )?;
    let mut ffi_action = vec![0f32; args.batch_size * args.action_dim];
    let mut ffi_sequence = vec![0f32; args.batch_size * args.horizon * args.action_dim];
    let mut ffi_cost = vec![0f32; args.batch_size];
    let ffi_cem_cfg = SwmCemPlanConfig {
        horizon: args.horizon,
        samples: args.samples,
        elites,
        iterations: args.planner_iterations,
        init_std: 1.0,
        min_std: 1e-3,
    };
    let ffi_mppi_cfg = SwmMppiPlanConfig {
        horizon: args.horizon,
        samples: args.samples,
        iterations: args.planner_iterations,
        noise_std: 1.0,
        temperature: 1.0,
    };
    let ffi_icem_cfg = SwmIcemPlanConfig {
        horizon: args.horizon,
        samples: args.samples,
        elites,
        keep_elites: elites.min(args.samples),
        iterations: args.planner_iterations,
        init_std: 1.0,
        min_std: 1e-3,
    };
    Ok(vec![
        bench("media_packed", args, device, || {
            packed_preprocessor.preprocess_packed_u8(&packed_pixels)?;
            Ok(())
        })?,
        bench("media_nv12", args, device, || {
            nv12_preprocessor.preprocess_nv12(&nv12_y, &nv12_uv)?;
            Ok(())
        })?,
        bench("encode", args, device, || {
            model.encode_pixels(&pixels)?;
            Ok(())
        })?,
        bench("rollout", args, device, || {
            model.rollout_embeddings(&emb_init, &actions)?;
            Ok(())
        })?,
        bench("cost", args, device, || {
            model.goal_cost(&rollout, &goal)?;
            Ok(())
        })?,
        bench("full", args, device, || {
            let emb = model.encode_pixels(&pixels)?;
            let emb_init = emb.unsqueeze(1)?.broadcast_as((
                args.batch_size,
                args.samples,
                history,
                emb.dim(2)?,
            ))?;
            let rollout = model.rollout_embeddings(&emb_init, &actions)?;
            let goal = emb.i((.., emb.dim(1)? - 1, ..))?;
            model.goal_cost(&rollout, &goal)?;
            Ok(())
        })?,
        bench("ffi_plan_cem", args, device, || {
            let status = unsafe {
                swm_lewm_plan_cem(
                    &mut ffi_handle,
                    ffi_cem_cfg,
                    ffi_action.as_mut_ptr(),
                    ffi_sequence.as_mut_ptr(),
                    ffi_cost.as_mut_ptr(),
                )
            };
            ensure_ffi_status(status)
        })?,
        bench("ffi_plan_mppi", args, device, || {
            let status = unsafe {
                swm_lewm_plan_mppi(
                    &mut ffi_handle,
                    ffi_mppi_cfg,
                    ffi_action.as_mut_ptr(),
                    ffi_sequence.as_mut_ptr(),
                    ffi_cost.as_mut_ptr(),
                )
            };
            ensure_ffi_status(status)
        })?,
        bench("ffi_plan_icem", args, device, || {
            let status = unsafe {
                swm_lewm_plan_icem(
                    &mut ffi_handle,
                    ffi_icem_cfg,
                    ffi_action.as_mut_ptr(),
                    ffi_sequence.as_mut_ptr(),
                    ffi_cost.as_mut_ptr(),
                )
            };
            ensure_ffi_status(status)
        })?,
    ])
}

fn bench_tdmpc2(
    args: &Args,
    device: &candle::Device,
    dtype: candle::DType,
) -> anyhow::Result<Vec<BenchStats>> {
    let cfg = TdMpc2Config::state_only(args.state_dim, args.action_dim);
    let vb = checkpoint::empty_var_builder(dtype, device);
    let model = TdMpc2::new(cfg, vb)?;

    let state =
        Tensor::randn(0f32, 1f32, (args.batch_size, args.state_dim), device)?.to_dtype(dtype)?;
    let z = model.encode_state(&state)?;
    let action =
        Tensor::randn(0f32, 1f32, (args.batch_size, args.action_dim), device)?.to_dtype(dtype)?;
    let action_candidates = Tensor::randn(
        0f32,
        1f32,
        (args.batch_size, args.samples, args.horizon, args.action_dim),
        device,
    )?
    .to_dtype(dtype)?;
    let actor_noise = Tensor::randn(
        0f32,
        1f32,
        (args.samples, args.batch_size, args.horizon, args.action_dim),
        device,
    )?
    .to_dtype(dtype)?;
    let media_size = 64usize;
    let media_config = ImagePreprocess {
        output_height: media_size,
        output_width: media_size,
        mean: [0.0, 0.0, 0.0],
        std: [1.0, 1.0, 1.0],
    };
    let packed_shape = PackedImageShape::new(
        args.batch_size,
        media_size,
        media_size,
        PackedImageFormat::Rgb,
    );
    let packed_pixels = packed_u8_tensor(packed_shape, device)?;
    let mut packed_preprocessor = ImagePreprocessor::new(device, packed_shape, media_config)?;
    let nv12_shape = Nv12ImageShape::new(args.batch_size, media_size, media_size);
    let (nv12_y, nv12_uv) = nv12_tensors(nv12_shape, device)?;
    let mut nv12_preprocessor =
        Nv12Preprocessor::new(device, nv12_shape, Nv12ColorSpace::Bt709Video, media_config)?;
    let mut jpeg_media = JpegMediaBench::new(args, device, media_config)?;

    let session_model = TdMpc2::new(
        TdMpc2Config::state_only(args.state_dim, args.action_dim),
        checkpoint::empty_var_builder(dtype, device),
    )?;
    let mut session = TdMpc2Session::new(session_model, device.clone(), dtype);
    session.reset_state(&state)?;

    let mut ffi_handle = SwmTdMpc2::synthetic_state_for_bench(
        args.state_dim,
        args.action_dim,
        dtype,
        device,
        &state,
    )?;
    let mut ffi_action = vec![0f32; args.batch_size * args.action_dim];
    let mut ffi_policy_actions = vec![0f32; args.batch_size * args.horizon * args.action_dim];
    let mut ffi_policy_rewards = vec![0f32; args.batch_size * args.horizon];
    let mut ffi_plan_sequence = vec![0f32; args.batch_size * args.horizon * args.action_dim];
    let mut ffi_plan_cost = vec![0f32; args.batch_size];

    if args.samples < 2 {
        anyhow::bail!("TD-MPC2 planning benchmarks require --samples >= 2");
    }
    if args.planner_iterations == 0 {
        anyhow::bail!("--planner-iterations must be greater than zero");
    }
    let elites = elite_count(args);
    if elites < 2 {
        anyhow::bail!("TD-MPC2 planning benchmarks require --elites >= 2");
    }
    if elites > args.samples {
        anyhow::bail!("--elites cannot exceed --samples");
    }

    let mut cem_cfg = CemConfig::new(args.horizon, args.samples, elites, args.action_dim);
    cem_cfg.iterations = args.planner_iterations;
    let cem = CemPlanner::new(cem_cfg);

    let mut mppi_cfg = MppiConfig::new(args.horizon, args.samples, args.action_dim);
    mppi_cfg.iterations = args.planner_iterations;
    let mppi = MppiPlanner::new(mppi_cfg);

    let mut icem_cfg = IcemConfig::new(args.horizon, args.samples, elites, args.action_dim);
    icem_cfg.iterations = args.planner_iterations;
    icem_cfg.keep_elites = elites.min(args.samples);
    let mut icem = IcemPlanner::new(icem_cfg);
    let ffi_cem_cfg = SwmCemPlanConfig {
        horizon: args.horizon,
        samples: args.samples,
        elites,
        iterations: args.planner_iterations,
        init_std: 1.0,
        min_std: 1e-3,
    };
    let ffi_mppi_cfg = SwmMppiPlanConfig {
        horizon: args.horizon,
        samples: args.samples,
        iterations: args.planner_iterations,
        noise_std: 1.0,
        temperature: 1.0,
    };
    let ffi_icem_cfg = SwmIcemPlanConfig {
        horizon: args.horizon,
        samples: args.samples,
        elites,
        keep_elites: elites.min(args.samples),
        iterations: args.planner_iterations,
        init_std: 1.0,
        min_std: 1e-3,
    };

    let mut rows = Vec::new();
    if let Some(jpeg_media) = jpeg_media.as_mut() {
        rows.push(bench("media_jpeg", args, device, || jpeg_media.run())?);
    }
    rows.push(bench("media_packed", args, device, || {
        packed_preprocessor.preprocess_packed_u8(&packed_pixels)?;
        Ok(())
    })?);
    rows.push(bench("media_nv12", args, device, || {
        nv12_preprocessor.preprocess_nv12(&nv12_y, &nv12_uv)?;
        Ok(())
    })?);
    rows.push(bench("encode", args, device, || {
        model.encode_state(&state)?;
        Ok(())
    })?);
    rows.push(bench("dynamics", args, device, || {
        model.forward(&z, &action)?;
        Ok(())
    })?);
    rows.push(bench("score", args, device, || {
        model.get_cost_state(&state, &action_candidates)?;
        Ok(())
    })?);
    rows.push(bench("full", args, device, || {
        let z = model.encode_state(&state)?;
        let _ = model.forward(&z, &action)?;
        model.get_cost_state(&state, &action_candidates)?;
        Ok(())
    })?);
    rows.push(bench("policy_rollout", args, device, || {
        model.rollout_actor_mean_logits(&z, args.horizon)?;
        Ok(())
    })?);
    rows.push(bench("policy_sample_fixed", args, device, || {
        model.rollout_actor_sampled_with_noise(&z, &actor_noise)?;
        Ok(())
    })?);
    rows.push(bench("policy_sample_generated", args, device, || {
        model.rollout_actor_sampled(&z, args.horizon, args.samples)?;
        Ok(())
    })?);
    rows.push(bench("ffi_actor_mean", args, device, || {
        let status =
            unsafe { swm_tdmpc2_actor_mean_action(&mut ffi_handle, ffi_action.as_mut_ptr()) };
        ensure_ffi_status(status)
    })?);
    rows.push(bench("ffi_policy_roll", args, device, || {
        let status = unsafe {
            swm_tdmpc2_rollout_actor_mean(
                &mut ffi_handle,
                args.horizon,
                ffi_policy_actions.as_mut_ptr(),
                ffi_policy_rewards.as_mut_ptr(),
            )
        };
        ensure_ffi_status(status)
    })?);
    rows.push(bench("ffi_policy_samp", args, device, || {
        let status = unsafe {
            swm_tdmpc2_rollout_actor_sample(
                &mut ffi_handle,
                args.horizon,
                args.samples,
                ffi_policy_actions.as_mut_ptr(),
            )
        };
        ensure_ffi_status(status)
    })?);
    rows.push(bench("plan_cem", args, device, || {
        cem.plan(&session)?;
        Ok(())
    })?);
    rows.push(bench("ffi_plan_cem", args, device, || {
        let status = unsafe {
            swm_tdmpc2_plan_cem(
                &mut ffi_handle,
                ffi_cem_cfg,
                ffi_action.as_mut_ptr(),
                ffi_plan_sequence.as_mut_ptr(),
                ffi_plan_cost.as_mut_ptr(),
            )
        };
        ensure_ffi_status(status)
    })?);
    rows.push(bench("ffi_plan_mppi", args, device, || {
        let status = unsafe {
            swm_tdmpc2_plan_mppi(
                &mut ffi_handle,
                ffi_mppi_cfg,
                ffi_action.as_mut_ptr(),
                ffi_plan_sequence.as_mut_ptr(),
                ffi_plan_cost.as_mut_ptr(),
            )
        };
        ensure_ffi_status(status)
    })?);
    rows.push(bench("ffi_plan_icem", args, device, || {
        let status = unsafe {
            swm_tdmpc2_plan_icem(
                &mut ffi_handle,
                ffi_icem_cfg,
                ffi_action.as_mut_ptr(),
                ffi_plan_sequence.as_mut_ptr(),
                ffi_plan_cost.as_mut_ptr(),
            )
        };
        ensure_ffi_status(status)
    })?);
    rows.push(bench("plan_mppi", args, device, || {
        mppi.plan(&session)?;
        Ok(())
    })?);
    rows.push(bench("plan_icem", args, device, || {
        icem.plan(&session)?;
        Ok(())
    })?);
    Ok(rows)
}

struct JpegMediaBench {
    encoded: Vec<u8>,
    decoder: NvJpegDecoder,
    rgb_output: Tensor,
    preprocessor: ImagePreprocessor,
}

impl JpegMediaBench {
    fn new(
        args: &Args,
        device: &candle::Device,
        config: ImagePreprocess,
    ) -> anyhow::Result<Option<Self>> {
        let Some(path) = args.jpeg_input.as_ref() else {
            return Ok(None);
        };
        if args.batch_size != 1 {
            anyhow::bail!("--jpeg-input benchmark row currently requires --batch-size 1");
        }

        let encoded = fs::read(path).map_err(|err| {
            anyhow::anyhow!("failed to read JPEG input {}: {err}", path.display())
        })?;
        let decoder = NvJpegDecoder::new(device)?;
        let info = decoder.image_info(&encoded)?;
        let rgb_output = decoder.alloc_rgb_interleaved(info)?;
        let preprocessor = ImagePreprocessor::new(device, info.packed_rgb_shape(), config)?;
        Ok(Some(Self {
            encoded,
            decoder,
            rgb_output,
            preprocessor,
        }))
    }

    fn run(&mut self) -> anyhow::Result<()> {
        self.decoder.decode_preprocessed_nchw_into(
            &self.encoded,
            &self.rgb_output,
            &mut self.preprocessor,
        )?;
        Ok(())
    }
}

fn ensure_ffi_status(status: SwmStatus) -> anyhow::Result<()> {
    if status == SwmStatus::Ok {
        return Ok(());
    }
    let message = unsafe {
        let ptr = swm_last_error_message();
        if ptr.is_null() {
            "no C ABI error message".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };
    anyhow::bail!("C ABI call failed with {status:?}: {message}")
}

fn elite_count(args: &Args) -> usize {
    args.elites.unwrap_or_else(|| {
        if args.samples < 2 {
            args.samples
        } else {
            (args.samples / 4).clamp(2, args.samples)
        }
    })
}

fn bench<F>(
    name: &'static str,
    args: &Args,
    device: &candle::Device,
    mut op: F,
) -> anyhow::Result<BenchStats>
where
    F: FnMut() -> anyhow::Result<()>,
{
    for _ in 0..args.warmup {
        op()?;
    }
    device.synchronize()?;

    let mut samples = Vec::with_capacity(args.iters);
    for _ in 0..args.iters {
        device.synchronize()?;
        let started = Instant::now();
        op()?;
        device.synchronize()?;
        samples.push(started.elapsed());
    }
    Ok(stats(name, samples))
}

fn stats(name: &'static str, mut samples: Vec<Duration>) -> BenchStats {
    samples.sort_unstable();
    let total_ms = samples.iter().map(duration_ms).sum::<f64>();
    let mean_ms = total_ms / samples.len() as f64;
    BenchStats {
        name,
        mean_ms,
        p50_ms: duration_ms(&samples[percentile_index(samples.len(), 0.50)]),
        p95_ms: duration_ms(&samples[percentile_index(samples.len(), 0.95)]),
        p99_ms: duration_ms(&samples[percentile_index(samples.len(), 0.99)]),
    }
}

fn percentile_index(len: usize, percentile: f64) -> usize {
    let idx = ((len.saturating_sub(1)) as f64 * percentile).ceil() as usize;
    idx.min(len.saturating_sub(1))
}

fn duration_ms(duration: &Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn git_commit() -> String {
    Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
            } else {
                None
            }
        })
        .filter(|commit| !commit.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}
