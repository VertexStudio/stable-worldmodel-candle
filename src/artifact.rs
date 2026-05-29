use std::{
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use crate::config::ModelConfig;

#[derive(Debug, Clone)]
pub struct DeploymentArtifact {
    pub root: PathBuf,
    pub config: ModelConfig,
    pub weights: PathBuf,
    pub preprocess: PreprocessConfig,
    pub schema: RuntimeSchema,
}

impl DeploymentArtifact {
    pub fn from_dir(root: impl AsRef<Path>) -> anyhow::Result<Self> {
        let root = root.as_ref();
        let config_path = required_file(root, "config.json")?;
        let preprocess_path = required_file(root, "preprocess.json")?;
        let schema_path = required_file(root, "schema.json")?;
        let weights = weights_file(root)?;

        let config = read_json(&config_path)?;
        let preprocess = read_json(&preprocess_path)?;
        let schema: RuntimeSchema = read_json(&schema_path)?;
        schema.validate()?;

        Ok(Self {
            root: root.to_path_buf(),
            config,
            weights,
            preprocess,
            schema,
        })
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PreprocessConfig {
    #[serde(default)]
    pub image_size: Option<usize>,
    #[serde(default)]
    pub image_mean: Option<[f32; 3]>,
    #[serde(default)]
    pub image_std: Option<[f32; 3]>,
    #[serde(default)]
    pub action_min: Option<Vec<f32>>,
    #[serde(default)]
    pub action_max: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeSchema {
    pub observations: Vec<ObservationSpec>,
    pub action: ActionSpec,
}

impl RuntimeSchema {
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.observations.is_empty() {
            anyhow::bail!("schema must define at least one observation");
        }
        for observation in &self.observations {
            observation.validate()?;
        }
        self.action.validate()?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationSpec {
    pub name: String,
    pub kind: ObservationKind,
    pub shape: Vec<usize>,
    #[serde(default)]
    pub dtype: Option<String>,
}

impl ObservationSpec {
    fn validate(&self) -> anyhow::Result<()> {
        if self.name.trim().is_empty() {
            anyhow::bail!("observation name cannot be empty");
        }
        if self.shape.is_empty() {
            anyhow::bail!("observation '{}' shape cannot be empty", self.name);
        }
        if self.shape.contains(&0) {
            anyhow::bail!("observation '{}' shape cannot contain zero", self.name);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationKind {
    State,
    Image,
    Video,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionSpec {
    pub dim: usize,
    #[serde(default)]
    pub min: Option<Vec<f32>>,
    #[serde(default)]
    pub max: Option<Vec<f32>>,
}

impl ActionSpec {
    fn validate(&self) -> anyhow::Result<()> {
        if self.dim == 0 {
            anyhow::bail!("action dim must be greater than zero");
        }
        if let Some(min) = &self.min {
            if min.len() != self.dim {
                anyhow::bail!(
                    "action min length {} does not match dim {}",
                    min.len(),
                    self.dim
                );
            }
        }
        if let Some(max) = &self.max {
            if max.len() != self.dim {
                anyhow::bail!(
                    "action max length {} does not match dim {}",
                    max.len(),
                    self.dim
                );
            }
        }
        Ok(())
    }
}

fn required_file(root: &Path, name: &str) -> anyhow::Result<PathBuf> {
    let path = root.join(name);
    if !path.is_file() {
        anyhow::bail!(
            "deployment artifact is missing required file {}",
            path.display()
        );
    }
    Ok(path)
}

fn weights_file(root: &Path) -> anyhow::Result<PathBuf> {
    let safetensors = root.join("model.safetensors");
    if safetensors.is_file() {
        return Ok(safetensors);
    }
    let pt = root.join("weights.pt");
    if pt.is_file() {
        return Ok(pt);
    }
    anyhow::bail!(
        "deployment artifact must contain model.safetensors or weights.pt in {}",
        root.display()
    )
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> anyhow::Result<T> {
    let text = fs::read_to_string(path)?;
    serde_json::from_str(&text).map_err(Into::into)
}
