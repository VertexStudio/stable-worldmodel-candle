use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NormKind {
    BatchNorm1d,
    LayerNorm,
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VitEncoderConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub layer_norm_eps: f64,
    pub num_channels: usize,
    pub qkv_bias: bool,
}

impl VitEncoderConfig {
    pub fn tiny_patch14_224() -> Self {
        Self {
            image_size: 224,
            patch_size: 14,
            hidden_size: 192,
            num_hidden_layers: 12,
            num_attention_heads: 3,
            intermediate_size: 768,
            layer_norm_eps: 1e-12,
            num_channels: 3,
            qkv_bias: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictorConfig {
    pub num_frames: usize,
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    pub depth: usize,
    pub heads: usize,
    pub dim_head: usize,
    pub mlp_dim: usize,
}

impl PredictorConfig {
    pub fn lewm_tiny(embed_dim: usize, history_size: usize) -> Self {
        Self {
            num_frames: history_size,
            input_dim: embed_dim,
            hidden_dim: embed_dim,
            output_dim: embed_dim,
            depth: 6,
            heads: 16,
            dim_head: 64,
            mlp_dim: 2048,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionEmbedderConfig {
    pub input_dim: usize,
    pub smoothed_dim: usize,
    pub emb_dim: usize,
    pub mlp_scale: usize,
}

impl ActionEmbedderConfig {
    pub fn new(input_dim: usize, emb_dim: usize) -> Self {
        Self {
            input_dim,
            smoothed_dim: input_dim,
            emb_dim,
            mlp_scale: 4,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlpConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    pub norm: NormKind,
}

impl MlpConfig {
    pub fn projector(embed_dim: usize) -> Self {
        Self {
            input_dim: embed_dim,
            hidden_dim: 2048,
            output_dim: embed_dim,
            norm: NormKind::BatchNorm1d,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeWmConfig {
    pub encoder: VitEncoderConfig,
    pub predictor: PredictorConfig,
    pub action_encoder: ActionEmbedderConfig,
    pub projector: MlpConfig,
    pub pred_proj: MlpConfig,
    pub history_size: usize,
}

impl LeWmConfig {
    pub fn tiny_patch14_224(action_dim: usize) -> Self {
        let embed_dim = 192;
        let history_size = 3;
        Self {
            encoder: VitEncoderConfig::tiny_patch14_224(),
            predictor: PredictorConfig::lewm_tiny(embed_dim, history_size),
            action_encoder: ActionEmbedderConfig::new(action_dim, embed_dim),
            projector: MlpConfig::projector(embed_dim),
            pred_proj: MlpConfig::projector(embed_dim),
            history_size,
        }
    }
}
