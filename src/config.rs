use serde::{Deserialize, Serialize};

use crate::models::{lewm::LeWmConfig, tdmpc2::TdMpc2Config};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    LeWm,
    TdMpc2,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "model_type", rename_all = "snake_case")]
pub enum ModelConfig {
    LeWm(LeWmConfig),
    #[serde(rename = "tdmpc2")]
    TdMpc2(TdMpc2Config),
}

impl ModelConfig {
    pub fn kind(&self) -> ModelKind {
        match self {
            Self::LeWm(_) => ModelKind::LeWm,
            Self::TdMpc2(_) => ModelKind::TdMpc2,
        }
    }

    pub fn lewm_tiny_patch14_224(action_dim: usize) -> Self {
        Self::LeWm(LeWmConfig::tiny_patch14_224(action_dim))
    }

    pub fn tdmpc2_state_only(state_dim: usize, action_dim: usize) -> Self {
        Self::TdMpc2(TdMpc2Config::state_only(state_dim, action_dim))
    }
}
