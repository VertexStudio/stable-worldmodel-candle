#[cfg(feature = "accelerate")]
extern crate accelerate_src;
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use std::path::PathBuf;

use candle::{DType, Tensor};
use clap::Parser;
use stable_worldmodel_candle::{
    checkpoint,
    models::tdmpc2::{TdMpc2, TdMpc2Config},
    runtime::DeviceSpec,
};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    fixture: PathBuf,

    #[arg(long)]
    weights: PathBuf,

    #[arg(long, default_value_t = 12)]
    state_dim: usize,

    #[arg(long, default_value_t = 4)]
    action_dim: usize,

    #[arg(long, default_value_t = DeviceSpec::Cpu)]
    device: DeviceSpec,

    #[arg(long, default_value_t = 1e-4)]
    tolerance: f32,

    #[arg(long, default_value_t = 1e-2)]
    cost_tolerance: f32,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = args.device.resolve()?;
    let cfg = TdMpc2Config::state_only(args.state_dim, args.action_dim);
    let vb = checkpoint::var_builder_from_path(&args.weights, DType::F32, &device)?;
    let model = TdMpc2::new(cfg, vb)?;

    let arrays = Tensor::read_npz_by_name(
        &args.fixture,
        &[
            "state",
            "action",
            "action_candidates",
            "z",
            "next_z",
            "reward_logits",
            "actor_mean",
            "cost",
        ],
    )?;
    let state = arrays[0].to_device(&device)?.to_dtype(DType::F32)?;
    let action = arrays[1].to_device(&device)?.to_dtype(DType::F32)?;
    let action_candidates = arrays[2].to_device(&device)?.to_dtype(DType::F32)?;

    let z = model.encode_state(&state)?;
    compare("z", &z, &arrays[3], args.tolerance)?;

    let (next_z, reward_logits) = model.forward(&z, &action)?;
    compare("next_z", &next_z, &arrays[4], args.tolerance)?;
    compare("reward_logits", &reward_logits, &arrays[5], args.tolerance)?;

    let actor_mean = model.actor_mean_action(&z)?;
    compare("actor_mean", &actor_mean, &arrays[6], args.tolerance)?;

    let cost = model.get_cost_state(&state, &action_candidates)?;
    compare("cost", &cost, &arrays[7], args.cost_tolerance)?;
    compare_cost_ordering(&cost, &arrays[7])?;

    Ok(())
}

fn compare(name: &str, actual: &Tensor, expected: &Tensor, tolerance: f32) -> anyhow::Result<()> {
    if actual.shape() != expected.shape() {
        anyhow::bail!(
            "{name} shape mismatch: Candle {:?}, Python {:?}",
            actual.shape(),
            expected.shape()
        );
    }
    ensure_finite(&format!("{name} Candle"), actual)?;
    ensure_finite(&format!("{name} Python"), expected)?;
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

fn ensure_finite(name: &str, tensor: &Tensor) -> anyhow::Result<()> {
    let values = tensor
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if let Some((idx, value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        anyhow::bail!("{name} contains non-finite value {value} at flat index {idx}");
    }
    Ok(())
}

fn compare_cost_ordering(actual: &Tensor, expected: &Tensor) -> anyhow::Result<()> {
    if actual.shape() != expected.shape() {
        return Ok(());
    }
    let actual = actual.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let expected = expected.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let mut mismatches = Vec::new();
    for (batch, (actual_row, expected_row)) in actual.iter().zip(expected.iter()).enumerate() {
        let actual_best = argmin(actual_row);
        let expected_best = argmin(expected_row);
        if actual_best != expected_best {
            mismatches.push((batch, actual_best, expected_best));
        }
    }
    if !mismatches.is_empty() {
        anyhow::bail!("cost argmin mismatch by batch: {mismatches:?}");
    }
    println!("cost argmin: stable for {} batch item(s)", actual.len());
    Ok(())
}

fn argmin(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .min_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}
