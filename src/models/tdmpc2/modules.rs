use candle::{D, Module, Result, Tensor};
use candle_nn::{Activation, LayerNorm, Linear, VarBuilder, layer_norm, linear};

use super::config::EncodingConfig;

#[derive(Debug, Clone)]
pub struct SimNorm {
    simplex_dim: usize,
}

impl SimNorm {
    pub fn new(simplex_dim: usize) -> Self {
        Self { simplex_dim }
    }
}

impl Module for SimNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let shape = xs.dims().to_vec();
        let last_dim = *shape.last().ok_or_else(|| {
            candle::Error::Msg("SimNorm expects at least one dimension".to_string())
        })?;
        if last_dim % self.simplex_dim != 0 {
            candle::bail!(
                "SimNorm last dim {last_dim} must be divisible by simplex dim {}",
                self.simplex_dim
            );
        }

        let mut grouped_shape = shape[..shape.len() - 1].to_vec();
        grouped_shape.push(last_dim / self.simplex_dim);
        grouped_shape.push(self.simplex_dim);

        let xs = xs.reshape(grouped_shape)?;
        let xs = candle_nn::ops::softmax_last_dim(&xs)?;
        xs.reshape(shape)
    }
}

#[derive(Debug, Clone)]
enum NormedActivation {
    Mish,
    SimNorm(SimNorm),
}

impl NormedActivation {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Mish => Activation::Mish.forward(xs),
            Self::SimNorm(sim_norm) => sim_norm.forward(xs),
        }
    }
}

#[derive(Debug, Clone)]
pub struct NormedLinear {
    linear: Linear,
    ln: LayerNorm,
    act: NormedActivation,
}

impl NormedLinear {
    pub fn mish(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        Self::new(in_dim, out_dim, NormedActivation::Mish, vb)
    }

    pub fn sim_norm(
        in_dim: usize,
        out_dim: usize,
        simplex_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Self::new(
            in_dim,
            out_dim,
            NormedActivation::SimNorm(SimNorm::new(simplex_dim)),
            vb,
        )
    }

    fn new(in_dim: usize, out_dim: usize, act: NormedActivation, vb: VarBuilder) -> Result<Self> {
        let linear = linear(in_dim, out_dim, vb.clone())?;
        let ln = layer_norm(out_dim, 1e-5, vb.pp("ln"))?;
        Ok(Self { linear, ln, act })
    }
}

impl Module for NormedLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.linear.forward(xs)?;
        let xs = self.ln.forward(&xs)?;
        self.act.forward(&xs)
    }
}

#[derive(Debug, Clone)]
pub struct VectorEncoder {
    cfg: EncodingConfig,
    fc1: NormedLinear,
    fc2: Linear,
    ln: LayerNorm,
}

impl VectorEncoder {
    pub fn new(cfg: EncodingConfig, enc_dim: usize, vb: VarBuilder) -> Result<Self> {
        let fc1 = NormedLinear::mish(cfg.input_dim, enc_dim, vb.pp("0"))?;
        let fc2 = linear(enc_dim, cfg.output_dim, vb.pp("1"))?;
        let ln = layer_norm(cfg.output_dim, 1e-5, vb.pp("2"))?;
        Ok(Self { cfg, fc1, fc2, ln })
    }

    pub fn name(&self) -> &str {
        &self.cfg.name
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dims = xs.dims();
        if dims.is_empty() || *dims.last().unwrap() != self.cfg.input_dim {
            candle::bail!(
                "encoder {} expects last dim {}, got {:?}",
                self.cfg.name,
                self.cfg.input_dim,
                xs.shape()
            );
        }

        let leading = &dims[..dims.len() - 1];
        let batch = leading.iter().product::<usize>();
        let xs = xs.reshape((batch, self.cfg.input_dim))?;
        let xs = self.fc1.forward(&xs)?;
        let xs = self.fc2.forward(&xs)?;
        let xs = self.ln.forward(&xs)?;

        let mut out_shape = leading.to_vec();
        out_shape.push(self.cfg.output_dim);
        xs.reshape(out_shape)
    }
}

#[derive(Debug, Clone)]
enum MlpOutput {
    Linear(Linear),
    NormedSim(NormedLinear),
}

impl Module for MlpOutput {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Linear(linear) => linear.forward(xs),
            Self::NormedSim(normed) => normed.forward(xs),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TdMpc2Mlp {
    fc1: NormedLinear,
    fc2: NormedLinear,
    out: MlpOutput,
}

impl TdMpc2Mlp {
    pub fn linear_output(
        in_dim: usize,
        hidden_dim: usize,
        out_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let fc1 = NormedLinear::mish(in_dim, hidden_dim, vb.pp("0"))?;
        let fc2 = NormedLinear::mish(hidden_dim, hidden_dim, vb.pp("1"))?;
        let out = MlpOutput::Linear(linear(hidden_dim, out_dim, vb.pp("2"))?);
        Ok(Self { fc1, fc2, out })
    }

    pub fn simnorm_output(
        in_dim: usize,
        hidden_dim: usize,
        out_dim: usize,
        simplex_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let fc1 = NormedLinear::mish(in_dim, hidden_dim, vb.pp("0"))?;
        let fc2 = NormedLinear::mish(hidden_dim, hidden_dim, vb.pp("1"))?;
        let out = MlpOutput::NormedSim(NormedLinear::sim_norm(
            hidden_dim,
            out_dim,
            simplex_dim,
            vb.pp("2"),
        )?);
        Ok(Self { fc1, fc2, out })
    }
}

impl Module for TdMpc2Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?;
        let xs = self.fc2.forward(&xs)?;
        self.out.forward(&xs)
    }
}

pub fn cat_last(xs: &[Tensor]) -> Result<Tensor> {
    if xs.is_empty() {
        candle::bail!("cannot concatenate an empty tensor list")
    }
    let rank = xs[0].rank();
    let refs = xs.iter().collect::<Vec<_>>();
    Tensor::cat(&refs, rank - 1)
}

pub fn two_hot_inv(logits: &Tensor, vmin: f64, vmax: f64, num_bins: usize) -> Result<Tensor> {
    let device = logits.device();
    let dtype = logits.dtype();
    let bin_size = (vmax - vmin) / (num_bins.saturating_sub(1) as f64);
    let values = (0..num_bins)
        .map(|idx| vmin as f32 + (idx as f32) * bin_size as f32)
        .collect::<Vec<_>>();
    let bin_values = Tensor::from_vec(values, (1, num_bins), device)?.to_dtype(dtype)?;
    let probs = candle_nn::ops::softmax_last_dim(logits)?;
    let symlog_value = probs.broadcast_mul(&bin_values)?.sum_keepdim(D::Minus1)?;
    symexp(&symlog_value)
}

fn symexp(xs: &Tensor) -> Result<Tensor> {
    Ok((xs.sign()? * (xs.abs()?.exp()? - 1.0)?)?)
}
