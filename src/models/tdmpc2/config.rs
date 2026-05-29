use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EncodingConfig {
    pub name: String,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl EncodingConfig {
    pub fn new(name: impl Into<String>, input_dim: usize, output_dim: usize) -> Self {
        Self {
            name: name.into(),
            input_dim,
            output_dim,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TdMpc2Config {
    pub action_dim: usize,
    #[serde(default)]
    pub image_size: Option<usize>,
    pub enc_dim: usize,
    pub mlp_dim: usize,
    pub simnorm_dim: usize,
    pub num_q: usize,
    pub num_bins: usize,
    pub vmin: f64,
    pub vmax: f64,
    pub discount: f64,
    pub uncertainty_penalty: f64,
    pub encodings: Vec<EncodingConfig>,
}

impl TdMpc2Config {
    pub fn state_only(state_dim: usize, action_dim: usize) -> Self {
        Self {
            action_dim,
            image_size: None,
            enc_dim: 256,
            mlp_dim: 384,
            simnorm_dim: 8,
            num_q: 5,
            num_bins: 101,
            vmin: -6.0,
            vmax: 2.0,
            discount: 0.99,
            uncertainty_penalty: 0.5,
            encodings: vec![EncodingConfig::new("state", state_dim, 128)],
        }
    }

    pub fn latent_dim(&self) -> usize {
        self.encodings
            .iter()
            .map(|encoding| encoding.output_dim)
            .sum()
    }

    pub fn pixel_only(image_size: usize, action_dim: usize, pixel_dim: usize) -> Self {
        Self {
            action_dim,
            image_size: Some(image_size),
            enc_dim: 256,
            mlp_dim: 384,
            simnorm_dim: 8,
            num_q: 5,
            num_bins: 101,
            vmin: -6.0,
            vmax: 2.0,
            discount: 0.99,
            uncertainty_penalty: 0.5,
            encodings: vec![EncodingConfig::new("pixels", image_size, pixel_dim)],
        }
    }
}
