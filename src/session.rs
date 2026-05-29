use candle::{DType, Device, Result, Tensor};

use crate::models::{
    lewm::LeWm,
    tdmpc2::{TdMpc2, TdMpc2Config},
};

#[derive(Debug)]
pub struct LeWmSession {
    model: LeWm,
    device: Device,
    dtype: DType,
    emb: Option<Tensor>,
}

impl LeWmSession {
    pub fn new(model: LeWm, device: Device, dtype: DType) -> Self {
        Self {
            model,
            device,
            dtype,
            emb: None,
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn reset_pixels(&mut self, pixels: &Tensor) -> Result<Tensor> {
        let pixels = pixels.to_device(&self.device)?.to_dtype(self.dtype)?;
        let emb = self.model.encode_pixels(&pixels)?;
        self.emb = Some(emb.clone());
        Ok(emb)
    }

    pub fn cached_embedding(&self) -> Option<&Tensor> {
        self.emb.as_ref()
    }

    pub fn score_candidates(
        &self,
        action_candidates: &Tensor,
        goal_emb: &Tensor,
    ) -> Result<Tensor> {
        let emb = self.emb.as_ref().ok_or_else(|| {
            candle::Error::Msg("LeWmSession must be reset before scoring candidates".to_string())
        })?;
        let action_candidates = action_candidates
            .to_device(&self.device)?
            .to_dtype(self.dtype)?;
        let goal_emb = goal_emb.to_device(&self.device)?.to_dtype(self.dtype)?;
        let (b, s, _, _) = action_candidates.dims4()?;
        let (_, h, d) = emb.dims3()?;
        let emb_init = emb.unsqueeze(1)?.broadcast_as((b, s, h, d))?;
        let rollout = self
            .model
            .rollout_embeddings(&emb_init, &action_candidates)?;
        self.model.goal_cost(&rollout, &goal_emb)
    }
}

#[derive(Debug)]
pub struct TdMpc2Session {
    model: TdMpc2,
    device: Device,
    dtype: DType,
    observations: Option<Vec<(String, Tensor)>>,
    z: Option<Tensor>,
}

impl TdMpc2Session {
    pub fn new(model: TdMpc2, device: Device, dtype: DType) -> Self {
        Self {
            model,
            device,
            dtype,
            observations: None,
            z: None,
        }
    }

    pub fn config(&self) -> &TdMpc2Config {
        self.model.config()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn reset_state(&mut self, state: &Tensor) -> Result<Tensor> {
        self.reset_observations(&[("state", state)])
    }

    pub fn reset_pixels(&mut self, pixels: &Tensor) -> Result<Tensor> {
        self.reset_observations(&[("pixels", pixels)])
    }

    pub fn reset_observations(&mut self, observations: &[(&str, &Tensor)]) -> Result<Tensor> {
        let observations = observations
            .iter()
            .map(|(name, tensor)| {
                Ok((
                    (*name).to_string(),
                    tensor.to_device(&self.device)?.to_dtype(self.dtype)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let refs = observations
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect::<Vec<_>>();
        let z = self.model.encode(&refs)?;
        self.observations = Some(observations);
        self.z = Some(z.clone());
        Ok(z)
    }

    pub fn cached_latent(&self) -> Option<&Tensor> {
        self.z.as_ref()
    }

    pub fn actor_mean_action(&self) -> Result<Tensor> {
        let z = self.z.as_ref().ok_or_else(|| {
            candle::Error::Msg("TdMpc2Session must be reset before actor action".to_string())
        })?;
        self.model.actor_mean_action(z)
    }

    pub fn score_candidates(&self, action_candidates: &Tensor) -> Result<Tensor> {
        let observations = self.observations.as_ref().ok_or_else(|| {
            candle::Error::Msg("TdMpc2Session must be reset before scoring candidates".to_string())
        })?;
        let refs = observations
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect::<Vec<_>>();
        let action_candidates = action_candidates
            .to_device(&self.device)?
            .to_dtype(self.dtype)?;
        self.model.get_cost(&refs, &action_candidates)
    }
}
