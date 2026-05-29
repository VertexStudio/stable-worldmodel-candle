#[cfg(feature = "accelerate")]
extern crate accelerate_src;
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use std::{
    process::Command,
    time::{Duration, Instant},
};

use candle::{IndexOp, Tensor};
use clap::{Parser, ValueEnum};
use serde_json::json;
use stable_worldmodel_candle::{
    checkpoint,
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

    #[arg(long, default_value_t = DeviceSpec::Cpu)]
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

    let emb = model.encode_pixels(&pixels)?;
    let emb_init =
        emb.unsqueeze(1)?
            .broadcast_as((args.batch_size, args.samples, history, emb.dim(2)?))?;
    let rollout = model.rollout_embeddings(&emb_init, &actions)?;
    let goal = emb.i((.., emb.dim(1)? - 1, ..))?;

    Ok(vec![
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

    let session_model = TdMpc2::new(
        TdMpc2Config::state_only(args.state_dim, args.action_dim),
        checkpoint::empty_var_builder(dtype, device),
    )?;
    let mut session = TdMpc2Session::new(session_model, device.clone(), dtype);
    session.reset_state(&state)?;

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

    Ok(vec![
        bench("encode", args, device, || {
            model.encode_state(&state)?;
            Ok(())
        })?,
        bench("dynamics", args, device, || {
            model.forward(&z, &action)?;
            Ok(())
        })?,
        bench("score", args, device, || {
            model.get_cost_state(&state, &action_candidates)?;
            Ok(())
        })?,
        bench("full", args, device, || {
            let z = model.encode_state(&state)?;
            let _ = model.forward(&z, &action)?;
            model.get_cost_state(&state, &action_candidates)?;
            Ok(())
        })?,
        bench("plan_cem", args, device, || {
            cem.plan(&session)?;
            Ok(())
        })?,
        bench("plan_mppi", args, device, || {
            mppi.plan(&session)?;
            Ok(())
        })?,
        bench("plan_icem", args, device, || {
            icem.plan(&session)?;
            Ok(())
        })?,
    ])
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
