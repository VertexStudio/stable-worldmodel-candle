//! Candle inference components for stable-worldmodel LeWM.
//!
//! This crate intentionally mirrors the Python inference path:
//! `ViTModel -> projector -> action embedder -> conditional predictor -> rollout/cost`.

pub mod config;
pub mod lewm;
pub mod modules;
pub mod vit;

pub use config::{
    ActionEmbedderConfig, LeWmConfig, MlpConfig, NormKind, PredictorConfig, VitEncoderConfig,
};
pub use lewm::LeWm;
