use std::path::PathBuf;

use candle::{DType, IndexOp, Tensor};
use clap::Parser;
use stable_worldmodel_candle::{
    checkpoint,
    models::tdmpc2::{EncodingConfig, TdMpc2, TdMpc2Config},
    runtime::DeviceSpec,
};

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum FixtureKind {
    State,
    Pixel,
    Both,
}

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    fixture: PathBuf,

    #[arg(long)]
    weights: PathBuf,

    #[arg(long, value_enum, default_value_t = FixtureKind::State)]
    fixture_kind: FixtureKind,

    #[arg(long, default_value_t = 12)]
    state_dim: usize,

    #[arg(long, default_value_t = 64)]
    image_size: usize,

    #[arg(long, default_value_t = 128)]
    pixel_dim: usize,

    #[arg(long, default_value_t = 4)]
    action_dim: usize,

    #[arg(long, default_value_t = DeviceSpec::Cuda(0))]
    device: DeviceSpec,

    #[arg(long, default_value_t = 1e-4)]
    tolerance: f32,

    #[arg(long, default_value_t = 1e-2)]
    cost_tolerance: f32,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = args.device.resolve()?;
    let cfg = tdmpc2_config(&args);
    let vb = checkpoint::var_builder_from_path(&args.weights, DType::F32, &device)?;
    let model = TdMpc2::new(cfg, vb)?;

    let mut names = Vec::new();
    if matches!(args.fixture_kind, FixtureKind::State | FixtureKind::Both) {
        names.push("state");
    }
    if matches!(args.fixture_kind, FixtureKind::Pixel | FixtureKind::Both) {
        names.push("pixels");
    }
    names.extend([
        "action",
        "action_candidates",
        "z",
        "next_z",
        "reward_logits",
        "actor_noise",
        "actor_log_std",
        "actor_mean",
        "actor_sample",
        "actor_sample_rollout",
        "cost",
    ]);
    let arrays = Tensor::read_npz_by_name(&args.fixture, &names)?;
    let mut idx = 0;
    let state = if matches!(args.fixture_kind, FixtureKind::State | FixtureKind::Both) {
        let tensor = arrays[idx].to_device(&device)?.to_dtype(DType::F32)?;
        idx += 1;
        Some(tensor)
    } else {
        None
    };
    let pixels = if matches!(args.fixture_kind, FixtureKind::Pixel | FixtureKind::Both) {
        let tensor = arrays[idx].to_device(&device)?.to_dtype(DType::F32)?;
        idx += 1;
        Some(tensor)
    } else {
        None
    };
    let action = arrays[idx].to_device(&device)?.to_dtype(DType::F32)?;
    idx += 1;
    let action_candidates = arrays[idx].to_device(&device)?.to_dtype(DType::F32)?;
    idx += 1;
    let expected_z = &arrays[idx];
    idx += 1;
    let expected_next_z = &arrays[idx];
    idx += 1;
    let expected_reward_logits = &arrays[idx];
    idx += 1;
    let actor_noise = arrays[idx].to_device(&device)?.to_dtype(DType::F32)?;
    idx += 1;
    let expected_actor_log_std = &arrays[idx];
    idx += 1;
    let expected_actor_mean = &arrays[idx];
    idx += 1;
    let expected_actor_sample = &arrays[idx];
    idx += 1;
    let expected_actor_sample_rollout = &arrays[idx];
    idx += 1;
    let expected_cost = &arrays[idx];

    let observations = observations(state.as_ref(), pixels.as_ref());
    let z = model.encode(&observations)?;
    compare("z", &z, expected_z, args.tolerance)?;

    let (next_z, reward_logits) = model.forward(&z, &action)?;
    compare("next_z", &next_z, expected_next_z, args.tolerance)?;
    compare(
        "reward_logits",
        &reward_logits,
        expected_reward_logits,
        args.tolerance,
    )?;

    let (_, actor_log_std) = model.actor_mean_log_std(&z)?;
    compare(
        "actor_log_std",
        &actor_log_std,
        expected_actor_log_std,
        args.tolerance,
    )?;

    let actor_mean = model.actor_mean_action(&z)?;
    compare(
        "actor_mean",
        &actor_mean,
        expected_actor_mean,
        args.tolerance,
    )?;

    let actor_sample_noise = actor_noise.i((0, .., 0, ..))?;
    let actor_sample = model.actor_sample_action(&z, &actor_sample_noise)?;
    compare(
        "actor_sample",
        &actor_sample,
        expected_actor_sample,
        args.tolerance,
    )?;

    let actor_sample_rollout = model.rollout_actor_sampled_with_noise(&z, &actor_noise)?;
    compare(
        "actor_sample_rollout",
        &actor_sample_rollout,
        expected_actor_sample_rollout,
        args.tolerance,
    )?;

    let cost = model.get_cost(&observations, &action_candidates)?;
    compare("cost", &cost, expected_cost, args.cost_tolerance)?;
    compare_cost_ordering(&cost, expected_cost)?;

    Ok(())
}

fn tdmpc2_config(args: &Args) -> TdMpc2Config {
    match args.fixture_kind {
        FixtureKind::State => TdMpc2Config::state_only(args.state_dim, args.action_dim),
        FixtureKind::Pixel => {
            TdMpc2Config::pixel_only(args.image_size, args.action_dim, args.pixel_dim)
        }
        FixtureKind::Both => {
            let mut cfg =
                TdMpc2Config::pixel_only(args.image_size, args.action_dim, args.pixel_dim);
            cfg.encodings
                .push(EncodingConfig::new("state", args.state_dim, 128));
            cfg
        }
    }
}

fn observations<'a>(
    state: Option<&'a Tensor>,
    pixels: Option<&'a Tensor>,
) -> Vec<(&'static str, &'a Tensor)> {
    let mut observations = Vec::new();
    if let Some(pixels) = pixels {
        observations.push(("pixels", pixels));
    }
    if let Some(state) = state {
        observations.push(("state", state));
    }
    observations
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
