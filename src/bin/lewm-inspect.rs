#[cfg(feature = "accelerate")]
extern crate accelerate_src;
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use std::path::PathBuf;

use candle::{DType, Device, IndexOp, Tensor};
use clap::{Parser, ValueEnum};
use stable_worldmodel_candle::{
    checkpoint,
    models::lewm::{LeWm, LeWmConfig},
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

    #[arg(long, default_value_t = 2)]
    action_dim: usize,

    #[arg(long, default_value_t = 1)]
    batch_size: usize,

    #[arg(long, default_value_t = 3)]
    history: usize,

    #[arg(long, default_value_t = 8)]
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
    let cfg = LeWmConfig::tiny_patch14_224(args.action_dim);

    let vb = match args.weights.as_ref() {
        Some(path) => checkpoint::var_builder_from_path(path, dtype, &device)?,
        None => checkpoint::empty_var_builder(dtype, &device),
    };

    let model = LeWm::new(cfg, vb)?;
    println!("model built");

    let b = args.batch_size;
    let h = args.history;
    let image = Tensor::randn(0f32, 1f32, (b, h, 3, 224, 224), &device)?.to_dtype(dtype)?;
    let emb = model.encode_pixels(&image)?;
    println!("encoded pixels: {:?}", emb.shape());

    let emb_init = emb.unsqueeze(1)?;
    let actions = Tensor::randn(0f32, 1f32, (b, 1, args.horizon, args.action_dim), &device)?
        .to_dtype(dtype)?;
    let rollout = model.rollout_embeddings(&emb_init, &actions)?;
    println!("rollout embeddings: {:?}", rollout.shape());

    let cost = model.goal_cost(&rollout, &emb.i((.., emb.dim(1)? - 1, ..))?)?;
    println!("goal cost: {:?}", cost.shape());
    Ok(())
}
