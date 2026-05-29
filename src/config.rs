use serde::{Deserialize, Serialize};

use crate::models::lewm::LeWmConfig;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    LeWm,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "model_type", rename_all = "snake_case")]
pub enum ModelConfig {
    LeWm(LeWmConfig),
}

impl ModelConfig {
    pub fn kind(&self) -> ModelKind {
        match self {
            Self::LeWm(_) => ModelKind::LeWm,
        }
    }

    pub fn lewm_tiny_patch14_224(action_dim: usize) -> Self {
        Self::LeWm(LeWmConfig::tiny_patch14_224(action_dim))
    }
}
