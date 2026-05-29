use candle::{D, IndexOp, Module, Result, Tensor};
use candle_nn::VarBuilder;

use super::{
    config::TdMpc2Config,
    modules::{TdMpc2Mlp, VectorEncoder, cat_last, two_hot_inv},
};

#[derive(Debug, Clone)]
pub struct TdMpc2 {
    cfg: TdMpc2Config,
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
            candle::bail!("TD-MPC2 requires at least one vector encoding")
        }

        let encoders = cfg
            .encodings
            .iter()
            .cloned()
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

    pub fn forward(&self, z: &Tensor, action: &Tensor) -> Result<(Tensor, Tensor)> {
        let z_a = cat_last(&[z.clone(), action.clone()])?;
        let next_z = self.dynamics.forward(&z_a)?;
        let reward_logits = self.reward.forward(&z_a)?;
        Ok((next_z, reward_logits))
    }

    pub fn actor_mean_action(&self, z: &Tensor) -> Result<Tensor> {
        let chunks = self.pi.forward(z)?.chunk(2, D::Minus1)?;
        chunks[0].tanh()
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
