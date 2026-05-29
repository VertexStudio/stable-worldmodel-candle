#[cfg(feature = "accelerate")]
extern crate accelerate_src;
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use std::path::PathBuf;

use anyhow::Context;
use candle::{DType, Device, Tensor};
use clap::{Parser, ValueEnum};
use stable_worldmodel_rs::{
    checkpoint,
    models::lewm::{LeWm, LeWmConfig},
};

#[cfg(feature = "hub")]
use stable_worldmodel_rs::hub;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DeviceArg {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(feature = "metal")]
    Metal,
}

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    fixture: PathBuf,

    #[arg(long)]
    weights: Option<PathBuf>,

    #[arg(long)]
    config: Option<PathBuf>,

    #[arg(long)]
    hf_repo: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    #[arg(long, default_value_t = 10)]
    action_dim: usize,

    #[arg(long, value_enum, default_value_t = DeviceArg::Cpu)]
    device: DeviceArg,

    #[arg(long, default_value_t = 1e-4)]
    tolerance: f32,
}

fn device(arg: DeviceArg) -> candle::Result<Device> {
    match arg {
        DeviceArg::Cpu => Ok(Device::Cpu),
        #[cfg(feature = "cuda")]
        DeviceArg::Cuda => Device::new_cuda(0),
        #[cfg(feature = "metal")]
        DeviceArg::Metal => Device::new_metal(0),
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = device(args.device)?;
    let (weights, config) = resolve_files(&args)?;

    let cfg = match config.as_ref() {
        Some(path) => LeWmConfig::from_stable_worldmodel_json_file(path)?,
        None => LeWmConfig::tiny_patch14_224(args.action_dim),
    };
    let vb = checkpoint::var_builder_from_path(&weights, DType::F32, &device)?;
    let model = LeWm::new(cfg, vb)?;

    let arrays = Tensor::read_npz_by_name(
        &args.fixture,
        &[
            "pixels",
            "actions",
            "action_candidates",
            "goal_emb",
            "emb",
            "act_emb",
            "pred",
            "rollout",
            "cost",
        ],
    )?;
    let pixels = arrays[0].to_device(&device)?.to_dtype(DType::F32)?;
    let actions = arrays[1].to_device(&device)?.to_dtype(DType::F32)?;
    let action_candidates = arrays[2].to_device(&device)?.to_dtype(DType::F32)?;
    let goal_emb = arrays[3].to_device(&device)?.to_dtype(DType::F32)?;

    let emb = model.encode_pixels(&pixels)?;
    compare("emb", &emb, &arrays[4], args.tolerance)?;

    let act_emb = model.encode_actions(&actions)?;
    compare("act_emb", &act_emb, &arrays[5], args.tolerance)?;

    let pred = model.predict_from_action_embeddings(&emb, &act_emb)?;
    compare("pred", &pred, &arrays[6], args.tolerance)?;

    let (b, s, _, _) = action_candidates.dims4()?;
    let (_, h, d) = emb.dims3()?;
    let emb_init = emb.unsqueeze(1)?.broadcast_as((b, s, h, d))?;
    let rollout = model.rollout_embeddings(&emb_init, &action_candidates)?;
    compare("rollout", &rollout, &arrays[7], args.tolerance)?;

    let cost = model.goal_cost(&rollout, &goal_emb)?;
    compare("cost", &cost, &arrays[8], args.tolerance)?;

    Ok(())
}

fn resolve_files(args: &Args) -> anyhow::Result<(PathBuf, Option<PathBuf>)> {
    if let Some(repo_id) = args.hf_repo.as_ref() {
        #[cfg(feature = "hub")]
        {
            let files =
                hub::download_stable_worldmodel_checkpoint(repo_id, args.revision.as_deref())?;
            return Ok((files.weights, Some(files.config)));
        }

        #[cfg(not(feature = "hub"))]
        {
            let _ = repo_id;
            anyhow::bail!("--hf-repo requires building with --features hub");
        }
    }

    let weights = args
        .weights
        .clone()
        .context("provide --weights or --hf-repo")?;
    Ok((weights, args.config.clone()))
}

fn compare(name: &str, actual: &Tensor, expected: &Tensor, tolerance: f32) -> anyhow::Result<()> {
    if actual.shape() != expected.shape() {
        anyhow::bail!(
            "{name} shape mismatch: Candle {:?}, Python {:?}",
            actual.shape(),
            expected.shape()
        );
    }
    let expected = expected.to_device(actual.device())?.to_dtype(DType::F32)?;
    let actual = actual.to_dtype(DType::F32)?;
    let shape = actual.shape().clone();
    let diff = (actual - expected)?.abs()?;
    let max_abs = diff.max_all()?.to_scalar::<f32>()?;
    let mean_abs = diff.mean_all()?.to_scalar::<f32>()?;
    println!(
        "{name}: shape={:?} max_abs={max_abs:.6e} mean_abs={mean_abs:.6e}",
        shape
    );
    if max_abs > tolerance {
        anyhow::bail!("{name} max_abs {max_abs:.6e} exceeds tolerance {tolerance:.6e}");
    }
    Ok(())
}
