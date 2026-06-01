//! Candle inference components for stable-worldmodel.

#[cfg(not(target_os = "linux"))]
compile_error!("stable-worldmodel-candle is Linux/NVIDIA CUDA only.");

#[cfg(not(feature = "cudnn"))]
compile_error!("stable-worldmodel-candle requires the CUDA/cuDNN feature stack.");

pub mod artifact;
pub mod checkpoint;
pub mod config;
pub mod ffi;
#[cfg(feature = "hub")]
pub mod hub;
#[cfg(feature = "cuda")]
pub mod media;
pub mod models;
pub mod planner;
pub mod preprocess;
pub mod runtime;
pub mod session;
