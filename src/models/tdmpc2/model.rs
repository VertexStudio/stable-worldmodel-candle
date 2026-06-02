use candle::{D, IndexOp, Module, Result, Tensor};
use candle_nn::VarBuilder;

use super::{
    config::TdMpc2Config,
    modules::{PixelEncoder, TdMpc2Mlp, VectorEncoder, cat_last, two_hot_inv},
};

#[derive(Debug, Clone)]
pub struct TdMpc2 {
    cfg: TdMpc2Config,
    pixel_encoder: Option<PixelEncoder>,
    encoders: Vec<VectorEncoder>,
    dynamics: TdMpc2Mlp,
    reward: TdMpc2Mlp,
    pi: TdMpc2Mlp,
    qs: Vec<TdMpc2Mlp>,
}

impl TdMpc2 {
    pub fn new(cfg: TdMpc2Config, vb: VarBuilder) -> Result<Self> {
        let latent_dim = cfg.latent_dim();
        if latent_dim == 0 {
            candle::bail!("TD-MPC2 requires at least one observation encoding")
        }

        let pixel_cfg = cfg
            .encodings
            .iter()
            .find(|encoding| encoding.name == "pixels")
            .cloned();
        let pixel_encoder = match pixel_cfg {
            Some(encoding) => {
                let image_size = cfg.image_size.ok_or_else(|| {
                    candle::Error::Msg(
                        "TD-MPC2 pixel encoding requires config.image_size".to_string(),
                    )
                })?;
                Some(PixelEncoder::new(
                    image_size,
                    encoding.output_dim,
                    vb.clone(),
                )?)
            }
            None => None,
        };

        let encoders = cfg
            .encodings
            .iter()
            .cloned()
            .filter(|encoding| encoding.name != "pixels")
            .map(|encoding| {
                VectorEncoder::new(
                    encoding.clone(),
                    cfg.enc_dim,
                    vb.pp("extra_encoders").pp(&encoding.name),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let dynamics = TdMpc2Mlp::simnorm_output(
            latent_dim + cfg.action_dim,
            cfg.mlp_dim,
            latent_dim,
            cfg.simnorm_dim,
            vb.pp("dynamics"),
        )?;
        let reward = TdMpc2Mlp::linear_output(
            latent_dim + cfg.action_dim,
            cfg.mlp_dim,
            cfg.num_bins,
            vb.pp("reward"),
        )?;
        let pi =
            TdMpc2Mlp::linear_output(latent_dim, cfg.mlp_dim, 2 * cfg.action_dim, vb.pp("pi"))?;
        let qs = (0..cfg.num_q)
            .map(|idx| {
                TdMpc2Mlp::linear_output(
                    latent_dim + cfg.action_dim,
                    cfg.mlp_dim,
                    cfg.num_bins,
                    vb.pp("qs").pp(idx),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            cfg,
            pixel_encoder,
            encoders,
            dynamics,
            reward,
            pi,
            qs,
        })
    }

    pub fn config(&self) -> &TdMpc2Config {
        &self.cfg
    }

    pub fn encode(&self, observations: &[(&str, &Tensor)]) -> Result<Tensor> {
        let mut embeddings = Vec::with_capacity(self.encoders.len());
        if let Some(pixel_encoder) = &self.pixel_encoder {
            let (_, xs) = observations
                .iter()
                .find(|(name, _)| *name == "pixels")
                .ok_or_else(|| candle::Error::Msg("missing observation 'pixels'".to_string()))?;
            embeddings.push(pixel_encoder.forward(xs)?);
        }
        for encoder in &self.encoders {
            let (_, xs) = observations
                .iter()
                .find(|(name, _)| *name == encoder.name())
                .ok_or_else(|| {
                    candle::Error::Msg(format!("missing observation '{}'", encoder.name()))
                })?;
            embeddings.push(encoder.forward(xs)?);
        }
        let z = cat_last(&embeddings)?;
        self.sim_norm(&z)
    }

    pub fn encode_state(&self, state: &Tensor) -> Result<Tensor> {
        self.encode(&[("state", state)])
    }

    pub fn encode_pixels(&self, pixels: &Tensor) -> Result<Tensor> {
        self.encode(&[("pixels", pixels)])
    }

    pub fn forward(&self, z: &Tensor, action: &Tensor) -> Result<(Tensor, Tensor)> {
        let z_a = cat_last(&[z.clone(), action.clone()])?;
        let next_z = self.dynamics.forward(&z_a)?;
        let reward_logits = self.reward.forward(&z_a)?;
        Ok((next_z, reward_logits))
    }

    pub fn actor_mean_log_std(&self, z: &Tensor) -> Result<(Tensor, Tensor)> {
        let chunks = self.pi.forward(z)?.chunk(2, D::Minus1)?;
        let mean_raw = chunks[0].clone();
        let log_std = chunks[1].tanh()?.affine(6.0, -4.0)?;
        Ok((mean_raw, log_std))
    }

    pub fn actor_mean_raw(&self, z: &Tensor) -> Result<Tensor> {
        let chunks = self.pi.forward(z)?.chunk(2, D::Minus1)?;
        Ok(chunks[0].clone())
    }

    pub fn actor_mean_action(&self, z: &Tensor) -> Result<Tensor> {
        self.actor_mean_raw(z)?.tanh()
    }

    pub fn actor_sample_action(&self, z: &Tensor, noise: &Tensor) -> Result<Tensor> {
        let (mean_raw, log_std) = self.actor_mean_log_std(z)?;
        if noise.shape() != mean_raw.shape() {
            candle::bail!(
                "TD-MPC2 actor noise shape {:?} must match actor mean shape {:?}",
                noise.shape(),
                mean_raw.shape()
            );
        }
        let noise = noise
            .to_device(mean_raw.device())?
            .to_dtype(mean_raw.dtype())?;
        let std = log_std.exp()?;
        mean_raw.broadcast_add(&std.broadcast_mul(&noise)?)?.tanh()
    }

    pub fn rollout_actor_mean(
        &self,
        z: &Tensor,
        horizon: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (actions, reward_logits, z) = self.rollout_actor_mean_logits(z, horizon)?;
        let rewards = two_hot_inv(
            &reward_logits,
            self.cfg.vmin,
            self.cfg.vmax,
            self.cfg.num_bins,
        )?;
        Ok((actions, rewards, z))
    }

    pub fn rollout_actor_mean_logits(
        &self,
        z: &Tensor,
        horizon: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        if horizon == 0 {
            candle::bail!("TD-MPC2 actor rollout horizon must be greater than zero");
        }

        let mut z = z.clone();
        let mut actions = Vec::with_capacity(horizon);
        let mut reward_logits = Vec::with_capacity(horizon);
        for _ in 0..horizon {
            let action = self.actor_mean_action(&z)?;
            let z_a = cat_last(&[z.clone(), action.clone()])?;
            let reward = self.reward.forward(&z_a)?;
            z = self.dynamics.forward(&z_a)?;
            actions.push(action);
            reward_logits.push(reward);
        }

        let action_refs = actions.iter().collect::<Vec<_>>();
        let reward_refs = reward_logits.iter().collect::<Vec<_>>();
        Ok((
            Tensor::stack(&action_refs, 1)?,
            Tensor::stack(&reward_refs, 1)?,
            z,
        ))
    }

    pub fn rollout_actor_sampled(
        &self,
        z: &Tensor,
        horizon: usize,
        num_trajs: usize,
    ) -> Result<Tensor> {
        if horizon == 0 {
            candle::bail!("TD-MPC2 sampled actor rollout horizon must be greater than zero");
        }
        if num_trajs == 0 {
            candle::bail!("TD-MPC2 sampled actor rollout requires at least one trajectory");
        }
        let batch = z.dim(0)?;
        let noise = Tensor::randn(
            0f32,
            1f32,
            (num_trajs, batch, horizon, self.cfg.action_dim),
            z.device(),
        )?
        .to_dtype(z.dtype())?;
        self.rollout_actor_sampled_with_noise(z, &noise)
    }

    pub fn rollout_actor_sampled_with_noise(&self, z: &Tensor, noise: &Tensor) -> Result<Tensor> {
        let (num_trajs, batch, horizon, action_dim) = noise.dims4()?;
        if num_trajs == 0 {
            candle::bail!("TD-MPC2 sampled actor rollout requires at least one trajectory");
        }
        if horizon == 0 {
            candle::bail!("TD-MPC2 sampled actor rollout horizon must be greater than zero");
        }
        if action_dim != self.cfg.action_dim {
            candle::bail!(
                "TD-MPC2 sampled actor noise action dim {action_dim} does not match action_dim {}",
                self.cfg.action_dim
            );
        }
        let (z_batch, latent_dim) = z.dims2()?;
        if z_batch != batch {
            candle::bail!(
                "TD-MPC2 sampled actor noise batch {batch} does not match latent batch {z_batch}"
            );
        }
        let noise = noise.to_device(z.device())?.to_dtype(z.dtype())?;
        let mut z = z
            .unsqueeze(0)?
            .broadcast_as((num_trajs, batch, latent_dim))?
            .reshape((num_trajs * batch, latent_dim))?;
        let mut actions = Vec::with_capacity(horizon);
        for t in 0..horizon {
            let noise_t = noise
                .i((.., .., t, ..))?
                .reshape((num_trajs * batch, action_dim))?;
            let action = self.actor_sample_action(&z, &noise_t)?;
            let z_a = cat_last(&[z, action.clone()])?;
            z = self.dynamics.forward(&z_a)?;
            actions.push(action.reshape((num_trajs, batch, action_dim))?);
        }
        let action_refs = actions.iter().collect::<Vec<_>>();
        Tensor::stack(&action_refs, 2)?.mean(0)
    }

    pub fn get_cost_state(&self, state: &Tensor, action_candidates: &Tensor) -> Result<Tensor> {
        self.get_cost(&[("state", state)], action_candidates)
    }

    pub fn get_cost(
        &self,
        observations: &[(&str, &Tensor)],
        action_candidates: &Tensor,
    ) -> Result<Tensor> {
        let (b, n, horizon, action_dim) = action_candidates.dims4()?;
        if action_dim != self.cfg.action_dim {
            candle::bail!(
                "action candidates last dim {action_dim} does not match TD-MPC2 action_dim {}",
                self.cfg.action_dim
            );
        }

        let mut z = self.encode(observations)?;
        let latent_dim = self.cfg.latent_dim();
        z = match z.dims() {
            [zb, zd] if *zb == b && *zd == latent_dim => z
                .unsqueeze(1)?
                .broadcast_as((b, n, latent_dim))?
                .reshape((b * n, latent_dim))?,
            [zb, zn, zd] if *zb == b && *zn == n && *zd == latent_dim => {
                z.reshape((b * n, latent_dim))?
            }
            [zbn, zd] if *zbn == b * n && *zd == latent_dim => z,
            other => candle::bail!("unexpected encoded observation shape {other:?}"),
        };

        let actions = action_candidates.reshape((b * n, horizon, action_dim))?;
        let mut total_return = Tensor::zeros((b * n, 1), z.dtype(), z.device())?;
        let mut discount = 1.0;

        for t in 0..horizon {
            let action = actions.i((.., t, ..))?;
            let z_a = cat_last(&[z.clone(), action])?;
            let reward = two_hot_inv(
                &self.reward.forward(&z_a)?,
                self.cfg.vmin,
                self.cfg.vmax,
                self.cfg.num_bins,
            )?;
            z = self.dynamics.forward(&z_a)?;
            total_return = (total_return + (reward * discount)?)?;
            discount *= self.cfg.discount;
        }

        let terminal_action = self.actor_mean_action(&z)?;
        let z_a = cat_last(&[z, terminal_action])?;
        let q_values = self
            .qs
            .iter()
            .map(|q| {
                let logits = q.forward(&z_a)?;
                two_hot_inv(&logits, self.cfg.vmin, self.cfg.vmax, self.cfg.num_bins)
            })
            .collect::<Result<Vec<_>>>()?;
        let q_refs = q_values.iter().collect::<Vec<_>>();
        let q_values = Tensor::stack(&q_refs, 0)?;
        let q_mean = q_values.mean(0)?;
        let q_std = q_values.var(0)?.sqrt()?;
        let penalty = ((q_mean.abs()? * q_std)? * self.cfg.uncertainty_penalty)?;
        let conservative_q = (q_mean - penalty)?;
        let total_return = (total_return + (conservative_q * discount)?)?;
        total_return.neg()?.reshape((b, n))
    }

    fn sim_norm(&self, xs: &Tensor) -> Result<Tensor> {
        super::modules::SimNorm::new(self.cfg.simnorm_dim).forward(xs)
    }
}
