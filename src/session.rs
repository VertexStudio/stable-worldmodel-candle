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
        let emb = self.encode_pixels(pixels)?;
        self.emb = Some(emb.clone());
        Ok(emb)
    }

    pub fn encode_pixels(&self, pixels: &Tensor) -> Result<Tensor> {
        let pixels = pixels.to_device(&self.device)?.to_dtype(self.dtype)?;
        self.model.encode_pixels(&pixels)
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
        let rollout =
            self.model
                .rollout_embeddings_with_history(&emb_init, &action_candidates, h)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{checkpoint, models::lewm::LeWmConfig};

    #[test]
    fn lewm_session_scores_with_cached_input_history() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let dtype = DType::F32;
        let cfg = LeWmConfig::tiny_patch14_224(2);
        assert_eq!(cfg.history_size, 3);
        let model = LeWm::new(cfg, checkpoint::empty_var_builder(dtype, &device))?;
        let direct_model = model.clone();
        let mut session = LeWmSession::new(model, device.clone(), dtype);

        let pixels = Tensor::randn(0f32, 1f32, (1, 1, 3, 224, 224), &device)?;
        let goal_pixels = Tensor::randn(0f32, 1f32, (1, 1, 3, 224, 224), &device)?;
        let actions = Tensor::randn(0f32, 1f32, (1, 2, 5, 2), &device)?;

        let emb = session.reset_pixels(&pixels)?;
        let goal_emb = session.encode_pixels(&goal_pixels)?;
        let session_cost = session.score_candidates(&actions, &goal_emb)?;

        let emb_init = emb.unsqueeze(1)?.broadcast_as((1, 2, 1, emb.dim(2)?))?;
        let rollout = direct_model.rollout_embeddings_with_history(&emb_init, &actions, 1)?;
        let direct_cost = direct_model.goal_cost(&rollout, &goal_emb)?;
        let max_abs = (session_cost - direct_cost)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;

        assert!(max_abs <= 1e-6, "max abs diff {max_abs}");
        Ok(())
    }
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

    pub fn actor_sample_action(&self, noise: &Tensor) -> Result<Tensor> {
        let z = self.z.as_ref().ok_or_else(|| {
            candle::Error::Msg("TdMpc2Session must be reset before actor action".to_string())
        })?;
        let noise = noise.to_device(&self.device)?.to_dtype(self.dtype)?;
        self.model.actor_sample_action(z, &noise)
    }

    pub fn rollout_actor_mean(&self, horizon: usize) -> Result<(Tensor, Tensor, Tensor)> {
        let z = self.z.as_ref().ok_or_else(|| {
            candle::Error::Msg("TdMpc2Session must be reset before actor rollout".to_string())
        })?;
        self.model.rollout_actor_mean(z, horizon)
    }

    pub fn rollout_actor_sampled(&self, horizon: usize, num_trajs: usize) -> Result<Tensor> {
        let z = self.z.as_ref().ok_or_else(|| {
            candle::Error::Msg("TdMpc2Session must be reset before actor rollout".to_string())
        })?;
        self.model.rollout_actor_sampled(z, horizon, num_trajs)
    }

    pub fn rollout_actor_sampled_with_noise(&self, noise: &Tensor) -> Result<Tensor> {
        let z = self.z.as_ref().ok_or_else(|| {
            candle::Error::Msg("TdMpc2Session must be reset before actor rollout".to_string())
        })?;
        let noise = noise.to_device(&self.device)?.to_dtype(self.dtype)?;
        self.model.rollout_actor_sampled_with_noise(z, &noise)
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
