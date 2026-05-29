#[cfg(feature = "accelerate")]
extern crate accelerate_src;
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use std::path::PathBuf;

use candle::Tensor;
use clap::Parser;
use stable_worldmodel_candle::{
    checkpoint,
    models::tdmpc2::{TdMpc2, TdMpc2Config},
    runtime::{DTypeSpec, DeviceSpec},
};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    weights: Option<PathBuf>,

    #[arg(long, default_value_t = 12)]
    state_dim: usize,

    #[arg(long, default_value_t = 4)]
    action_dim: usize,

    #[arg(long, default_value_t = 2)]
    batch_size: usize,

    #[arg(long, default_value_t = 5)]
    num_samples: usize,

    #[arg(long, default_value_t = 3)]
    horizon: usize,

    #[arg(long, default_value_t = DeviceSpec::Cpu)]
    device: DeviceSpec,

    #[arg(long, default_value_t = DTypeSpec::F32)]
    dtype: DTypeSpec,

    /// Deprecated; use --dtype bf16.
    #[arg(long, default_value_t = false)]
    bf16: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = args.device.resolve()?;
    let dtype = if args.bf16 {
        DTypeSpec::Bf16
    } else {
        args.dtype
    }
    .dtype();
    let cfg = TdMpc2Config::state_only(args.state_dim, args.action_dim);

    let vb = match args.weights.as_ref() {
        Some(path) => checkpoint::var_builder_from_path(path, dtype, &device)?,
        None => checkpoint::empty_var_builder(dtype, &device),
    };

    let model = TdMpc2::new(cfg, vb)?;
    println!("model built");

    let state =
        Tensor::randn(0f32, 1f32, (args.batch_size, args.state_dim), &device)?.to_dtype(dtype)?;
    let z = model.encode_state(&state)?;
    println!("encoded state: {:?}", z.shape());

    let actions = Tensor::randn(
        0f32,
        1f32,
        (
            args.batch_size,
            args.num_samples,
            args.horizon,
            args.action_dim,
        ),
        &device,
    )?
    .to_dtype(dtype)?;
    let cost = model.get_cost_state(&state, &actions)?;
    println!("candidate costs: {:?}", cost.shape());

    Ok(())
}
