use std::time::{Duration, Instant};

use candle::{DType, Device, IndexOp, Result, Tensor};

use crate::session::{LeWmSession, TdMpc2Session};

pub trait CandidateScorer {
    fn device(&self) -> &Device;
    fn dtype(&self) -> DType;
    fn batch_size(&self) -> Option<usize> {
        None
    }
    fn score_candidates(&self, action_candidates: &Tensor) -> Result<Tensor>;
}

impl CandidateScorer for TdMpc2Session {
    fn device(&self) -> &Device {
        self.device()
    }

    fn dtype(&self) -> DType {
        self.dtype()
    }

    fn batch_size(&self) -> Option<usize> {
        self.cached_latent()
            .and_then(|latent| latent.dims().first().copied())
    }

    fn score_candidates(&self, action_candidates: &Tensor) -> Result<Tensor> {
        TdMpc2Session::score_candidates(self, action_candidates)
    }
}

pub struct LeWmGoalScorer<'a> {
    session: &'a LeWmSession,
    goal_emb: &'a Tensor,
}

impl<'a> LeWmGoalScorer<'a> {
    pub fn new(session: &'a LeWmSession, goal_emb: &'a Tensor) -> Self {
        Self { session, goal_emb }
    }
}

impl CandidateScorer for LeWmGoalScorer<'_> {
    fn device(&self) -> &Device {
        self.session.device()
    }

    fn dtype(&self) -> DType {
        self.session.dtype()
    }

    fn batch_size(&self) -> Option<usize> {
        self.session
            .cached_embedding()
            .and_then(|emb| emb.dims().first().copied())
    }

    fn score_candidates(&self, action_candidates: &Tensor) -> Result<Tensor> {
        self.session
            .score_candidates(action_candidates, self.goal_emb)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ActionBounds {
    pub low: Vec<f32>,
    pub high: Vec<f32>,
}

impl ActionBounds {
    pub fn symmetric(action_dim: usize, limit: f32) -> Self {
        Self {
            low: vec![-limit; action_dim],
            high: vec![limit; action_dim],
        }
    }

    pub fn scalar(action_dim: usize, low: f32, high: f32) -> Self {
        Self {
            low: vec![low; action_dim],
            high: vec![high; action_dim],
        }
    }

    fn validate(&self, action_dim: usize) -> Result<()> {
        if self.low.len() != action_dim || self.high.len() != action_dim {
            candle::bail!(
                "action bounds must match action_dim {action_dim}, got low={} high={}",
                self.low.len(),
                self.high.len()
            );
        }
        for (idx, (&low, &high)) in self.low.iter().zip(self.high.iter()).enumerate() {
            if !low.is_finite() || !high.is_finite() {
                candle::bail!("action bound {idx} is not finite");
            }
            if low > high {
                candle::bail!("action bound {idx} has low {low} greater than high {high}");
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct CemConfig {
    pub horizon: usize,
    pub samples: usize,
    pub elites: usize,
    pub iterations: usize,
    pub action_dim: usize,
    pub action_bounds: ActionBounds,
    pub init_std: f32,
    pub min_std: f32,
    pub deadline: Option<Duration>,
}

impl CemConfig {
    pub fn new(horizon: usize, samples: usize, elites: usize, action_dim: usize) -> Self {
        Self {
            horizon,
            samples,
            elites,
            iterations: 4,
            action_dim,
            action_bounds: ActionBounds::symmetric(action_dim, 1.0),
            init_std: 1.0,
            min_std: 1e-3,
            deadline: None,
        }
    }

    fn validate(&self) -> Result<()> {
        if self.horizon == 0 {
            candle::bail!("CEM horizon must be greater than zero");
        }
        if self.samples == 0 {
            candle::bail!("CEM samples must be greater than zero");
        }
        if self.elites < 2 {
            candle::bail!("CEM elites must be at least two");
        }
        if self.elites > self.samples {
            candle::bail!(
                "CEM elites {} cannot exceed samples {}",
                self.elites,
                self.samples
            );
        }
        if self.iterations == 0 {
            candle::bail!("CEM iterations must be greater than zero");
        }
        if self.action_dim == 0 {
            candle::bail!("CEM action_dim must be greater than zero");
        }
        if !self.init_std.is_finite() || self.init_std <= 0.0 {
            candle::bail!("CEM init_std must be finite and greater than zero");
        }
        if !self.min_std.is_finite() || self.min_std < 0.0 {
            candle::bail!("CEM min_std must be finite and non-negative");
        }
        self.action_bounds.validate(self.action_dim)
    }
}

#[derive(Debug)]
pub struct PlanResult {
    pub first_action: Tensor,
    pub sequence: Tensor,
    pub scores: Tensor,
    pub best_indices: Vec<usize>,
    pub iterations_completed: usize,
    pub elapsed: Duration,
    pub deadline_reached: bool,
    pub used_host_elite_selection: bool,
}

#[derive(Debug, Clone)]
pub struct CemPlanner {
    config: CemConfig,
}

impl CemPlanner {
    pub fn new(config: CemConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &CemConfig {
        &self.config
    }

    pub fn plan<S: CandidateScorer>(&self, scorer: &S) -> Result<PlanResult> {
        self.config.validate()?;
        let start = Instant::now();
        let device = scorer.device();
        let dtype = scorer.dtype();
        let cfg = &self.config;
        let batch = scorer.batch_size().unwrap_or(1);

        let mut mean = Tensor::zeros((batch, cfg.horizon, cfg.action_dim), dtype, device)?;
        let mut std = Tensor::ones((batch, cfg.horizon, cfg.action_dim), dtype, device)?
            .affine(cfg.init_std as f64, 0.0)?;
        let mut last_candidates = None;
        let mut last_scores = None;
        let mut iterations_completed = 0;
        let mut deadline_reached = false;

        for iter_idx in 0..cfg.iterations {
            if iter_idx > 0 && deadline_elapsed(start, cfg.deadline) {
                deadline_reached = true;
                break;
            }

            let candidates = sample_candidates(&mean, &std, cfg, dtype, device)?;
            let scores = scorer.score_candidates(&candidates)?;
            validate_scores_shape(&scores, batch, cfg.samples)?;
            let (elites, _) = select_elites(&candidates, &scores, cfg.elites)?;
            mean = elites.mean(1)?;
            std = enforce_min_std(&elites.var(1)?.sqrt()?, cfg.min_std)?;

            last_candidates = Some(candidates);
            last_scores = Some(scores);
            iterations_completed += 1;
        }

        let candidates = last_candidates
            .ok_or_else(|| candle::Error::Msg("CEM did not complete any iteration".to_string()))?;
        let scores = last_scores
            .ok_or_else(|| candle::Error::Msg("CEM did not produce scores".to_string()))?;
        let best_indices = best_indices_from_scores(&scores)?;
        let sequence = gather_best_sequences(&candidates, &best_indices)?;
        let first_action = sequence.i((.., 0, ..))?;
        let elapsed = start.elapsed();

        Ok(PlanResult {
            first_action,
            sequence,
            scores,
            best_indices,
            iterations_completed,
            elapsed,
            deadline_reached,
            used_host_elite_selection: true,
        })
    }
}

fn deadline_elapsed(start: Instant, deadline: Option<Duration>) -> bool {
    deadline.is_some_and(|deadline| start.elapsed() >= deadline)
}

fn sample_candidates(
    mean: &Tensor,
    std: &Tensor,
    cfg: &CemConfig,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let batch = mean.dim(0)?;
    let shape = (batch, cfg.samples, cfg.horizon, cfg.action_dim);
    let noise = Tensor::randn(0f32, 1f32, shape, device)?.to_dtype(dtype)?;
    let mean = mean.unsqueeze(1)?.broadcast_as(shape)?;
    let std = std.unsqueeze(1)?.broadcast_as(shape)?;
    let candidates = mean.broadcast_add(&noise.broadcast_mul(&std)?)?;
    clamp_actions(&candidates, &cfg.action_bounds, dtype, device)
}

fn clamp_actions(
    candidates: &Tensor,
    bounds: &ActionBounds,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let (_, _, _, action_dim) = candidates.dims4()?;
    let low = Tensor::from_vec(bounds.low.clone(), (action_dim,), device)?
        .to_dtype(dtype)?
        .reshape((1, 1, 1, action_dim))?;
    let high = Tensor::from_vec(bounds.high.clone(), (action_dim,), device)?
        .to_dtype(dtype)?
        .reshape((1, 1, 1, action_dim))?;
    candidates.broadcast_maximum(&low)?.broadcast_minimum(&high)
}

fn enforce_min_std(std: &Tensor, min_std: f32) -> Result<Tensor> {
    if min_std == 0.0 {
        return Ok(std.clone());
    }
    let floor = Tensor::new(min_std, std.device())?
        .to_dtype(std.dtype())?
        .broadcast_as(std.shape())?;
    std.broadcast_maximum(&floor)
}

fn validate_scores_shape(scores: &Tensor, batch: usize, samples: usize) -> Result<()> {
    match scores.dims() {
        [b, n] if *b == batch && *n == samples => Ok(()),
        other => {
            candle::bail!("candidate scorer must return [{batch}, {samples}] scores, got {other:?}")
        }
    }
}

fn select_elites(
    candidates: &Tensor,
    scores: &Tensor,
    elite_count: usize,
) -> Result<(Tensor, Vec<usize>)> {
    let (_, samples, _, _) = candidates.dims4()?;
    if elite_count > samples {
        candle::bail!("elite_count {elite_count} cannot exceed samples {samples}");
    }

    let ranked = ranked_indices_from_scores(scores)?;
    let mut elite_tensors = Vec::with_capacity(ranked.len());
    let mut best_indices = Vec::with_capacity(ranked.len());

    for (batch_idx, order) in ranked.iter().enumerate() {
        best_indices.push(order[0]);
        let elite_indices = order
            .iter()
            .take(elite_count)
            .map(|&idx| idx as i64)
            .collect::<Vec<_>>();
        let elite_indices = Tensor::from_vec(elite_indices, (elite_count,), candidates.device())?;
        let batch_candidates = candidates.i((batch_idx, .., .., ..))?;
        elite_tensors.push(batch_candidates.index_select(&elite_indices, 0)?);
    }

    let elite_refs = elite_tensors.iter().collect::<Vec<_>>();
    Ok((Tensor::stack(&elite_refs, 0)?, best_indices))
}

fn gather_best_sequences(candidates: &Tensor, best_indices: &[usize]) -> Result<Tensor> {
    let (batch, samples, _, _) = candidates.dims4()?;
    if best_indices.len() != batch {
        candle::bail!(
            "best index count {} does not match candidate batch {batch}",
            best_indices.len()
        );
    }

    let mut sequences = Vec::with_capacity(batch);
    for (batch_idx, &best_idx) in best_indices.iter().enumerate() {
        if best_idx >= samples {
            candle::bail!("best index {best_idx} out of range for {samples} samples");
        }
        sequences.push(candidates.i((batch_idx, best_idx, .., ..))?);
    }

    let sequence_refs = sequences.iter().collect::<Vec<_>>();
    Tensor::stack(&sequence_refs, 0)
}

fn best_indices_from_scores(scores: &Tensor) -> Result<Vec<usize>> {
    ranked_indices_from_scores(scores)
        .map(|ranked| ranked.into_iter().map(|order| order[0]).collect::<Vec<_>>())
}

fn ranked_indices_from_scores(scores: &Tensor) -> Result<Vec<Vec<usize>>> {
    let rows = scores.to_vec2::<f32>()?;
    let mut ranked = Vec::with_capacity(rows.len());

    for (batch_idx, row) in rows.iter().enumerate() {
        if row.is_empty() {
            candle::bail!("score row {batch_idx} is empty");
        }
        for (sample_idx, &value) in row.iter().enumerate() {
            if !value.is_finite() {
                candle::bail!(
                    "score at batch {batch_idx} sample {sample_idx} is not finite: {value}"
                );
            }
        }
        let mut order = (0..row.len()).collect::<Vec<_>>();
        order.sort_by(|&left, &right| row[left].total_cmp(&row[right]));
        ranked.push(order);
    }

    Ok(ranked)
}
