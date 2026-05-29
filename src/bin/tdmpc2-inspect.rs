#[cfg(feature = "accelerate")]
extern crate accelerate_src;
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use std::path::PathBuf;

use candle::{DType, Device, Tensor};
use clap::{Parser, ValueEnum};
use stable_worldmodel_rs::{
    checkpoint,
    models::tdmpc2::{TdMpc2, TdMpc2Config},
};

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

    #[arg(long, value_enum, default_value_t = DeviceArg::Cpu)]
    device: DeviceArg,

    #[arg(long, default_value_t = false)]
    bf16: bool,
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
    let dtype = if args.bf16 { DType::BF16 } else { DType::F32 };
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
