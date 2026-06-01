use std::{
    sync::{
        Mutex, MutexGuard,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use candle::{
    CudaStorage, DType, Device, DeviceLocation, IndexOp, Result, Storage, Tensor,
    cuda_backend::cudarc, op::BackpropOp,
};
use candle_nn::ops;

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

#[derive(Debug, Clone, PartialEq)]
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
    pub fallback_action: Option<Vec<f32>>,
    pub seed: Option<u64>,
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
            fallback_action: None,
            seed: None,
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
        self.action_bounds.validate(self.action_dim)?;
        validate_fallback_action(
            self.fallback_action.as_deref(),
            self.action_dim,
            &self.action_bounds,
            "CEM",
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MppiConfig {
    pub horizon: usize,
    pub samples: usize,
    pub iterations: usize,
    pub action_dim: usize,
    pub action_bounds: ActionBounds,
    pub noise_std: f32,
    pub temperature: f32,
    pub deadline: Option<Duration>,
    pub fallback_action: Option<Vec<f32>>,
    pub seed: Option<u64>,
}

impl MppiConfig {
    pub fn new(horizon: usize, samples: usize, action_dim: usize) -> Self {
        Self {
            horizon,
            samples,
            iterations: 1,
            action_dim,
            action_bounds: ActionBounds::symmetric(action_dim, 1.0),
            noise_std: 1.0,
            temperature: 1.0,
            deadline: None,
            fallback_action: None,
            seed: None,
        }
    }

    fn validate(&self) -> Result<()> {
        if self.horizon == 0 {
            candle::bail!("MPPI horizon must be greater than zero");
        }
        if self.samples == 0 {
            candle::bail!("MPPI samples must be greater than zero");
        }
        if self.iterations == 0 {
            candle::bail!("MPPI iterations must be greater than zero");
        }
        if self.action_dim == 0 {
            candle::bail!("MPPI action_dim must be greater than zero");
        }
        if !self.noise_std.is_finite() || self.noise_std <= 0.0 {
            candle::bail!("MPPI noise_std must be finite and greater than zero");
        }
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            candle::bail!("MPPI temperature must be finite and greater than zero");
        }
        self.action_bounds.validate(self.action_dim)?;
        validate_fallback_action(
            self.fallback_action.as_deref(),
            self.action_dim,
            &self.action_bounds,
            "MPPI",
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct IcemConfig {
    pub horizon: usize,
    pub samples: usize,
    pub elites: usize,
    pub keep_elites: usize,
    pub iterations: usize,
    pub action_dim: usize,
    pub action_bounds: ActionBounds,
    pub init_std: f32,
    pub min_std: f32,
    pub deadline: Option<Duration>,
    pub fallback_action: Option<Vec<f32>>,
    pub seed: Option<u64>,
}

impl IcemConfig {
    pub fn new(horizon: usize, samples: usize, elites: usize, action_dim: usize) -> Self {
        Self {
            horizon,
            samples,
            elites,
            keep_elites: elites,
            iterations: 4,
            action_dim,
            action_bounds: ActionBounds::symmetric(action_dim, 1.0),
            init_std: 1.0,
            min_std: 1e-3,
            deadline: None,
            fallback_action: None,
            seed: None,
        }
    }

    fn validate(&self) -> Result<()> {
        if self.horizon == 0 {
            candle::bail!("iCEM horizon must be greater than zero");
        }
        if self.samples == 0 {
            candle::bail!("iCEM samples must be greater than zero");
        }
        if self.elites < 2 {
            candle::bail!("iCEM elites must be at least two");
        }
        if self.elites > self.samples {
            candle::bail!(
                "iCEM elites {} cannot exceed samples {} on the first iteration",
                self.elites,
                self.samples
            );
        }
        if self.keep_elites > self.elites {
            candle::bail!(
                "iCEM keep_elites {} cannot exceed elites {}",
                self.keep_elites,
                self.elites
            );
        }
        if self.iterations == 0 {
            candle::bail!("iCEM iterations must be greater than zero");
        }
        if self.action_dim == 0 {
            candle::bail!("iCEM action_dim must be greater than zero");
        }
        if !self.init_std.is_finite() || self.init_std <= 0.0 {
            candle::bail!("iCEM init_std must be finite and greater than zero");
        }
        if !self.min_std.is_finite() || self.min_std < 0.0 {
            candle::bail!("iCEM min_std must be finite and non-negative");
        }
        self.action_bounds.validate(self.action_dim)?;
        validate_fallback_action(
            self.fallback_action.as_deref(),
            self.action_dim,
            &self.action_bounds,
            "iCEM",
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanFallback {
    None,
    WarmStart,
    ConfiguredAction,
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
    pub fallback: PlanFallback,
    pub used_host_elite_selection: bool,
}

#[derive(Debug, Clone)]
pub struct CemPlanner {
    config: CemConfig,
    rng: PlannerRng,
    workspace: PlannerWorkspace,
}

impl CemPlanner {
    pub fn new(config: CemConfig) -> Self {
        Self {
            config,
            rng: PlannerRng::new(),
            workspace: PlannerWorkspace::new(),
        }
    }

    pub fn config(&self) -> &CemConfig {
        &self.config
    }

    pub fn reset_rng_sequence(&self) {
        self.rng.reset();
    }

    pub fn rng_offset(&self) -> u64 {
        self.rng.offset()
    }

    pub fn plan<S: CandidateScorer>(&self, scorer: &S) -> Result<PlanResult> {
        self.config.validate()?;
        let start = Instant::now();
        let device = scorer.device();
        let dtype = scorer.dtype();
        let cfg = &self.config;
        let batch = scorer.batch_size().unwrap_or(1);
        let mut sampler = self.rng.begin_plan(
            device,
            cfg.seed,
            normal_draw_reservation(
                batch,
                cfg.samples,
                cfg.horizon,
                cfg.action_dim,
                cfg.iterations,
            )?,
        )?;

        let mut mean =
            self.workspace
                .sequence(batch, cfg.horizon, cfg.action_dim, dtype, device, 0.0)?;
        let mut std = self.workspace.sequence(
            batch,
            cfg.horizon,
            cfg.action_dim,
            dtype,
            device,
            cfg.init_std,
        )?;
        let (low, high) = self.workspace.bounds(&cfg.action_bounds, dtype, device)?;
        let mut last_candidates = None;
        let mut last_scores = None;
        let mut iterations_completed = 0;
        let mut deadline_reached = false;

        for iter_idx in 0..cfg.iterations {
            if deadline_elapsed(start, cfg.deadline) {
                deadline_reached = true;
                if iter_idx == 0 {
                    return configured_fallback_result(
                        cfg.fallback_action.as_deref(),
                        batch,
                        cfg.horizon,
                        cfg.action_dim,
                        dtype,
                        device,
                        start,
                        "CEM",
                    );
                }
                break;
            }

            let candidates = sample_candidates(
                &mean,
                &std,
                cfg.samples,
                &low,
                &high,
                dtype,
                device,
                &mut sampler,
            )?;
            let scores = scorer.score_candidates(&candidates)?;
            validate_scores_shape(&scores, batch, cfg.samples)?;
            let elites = select_elites(&candidates, &scores, cfg.elites)?;
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
        let sorted_indices = sorted_score_indices(&scores)?;
        let best_index_tensor = sorted_indices.narrow(1, 0, 1)?;
        let best_indices = best_indices_from_tensor(&best_index_tensor)?;
        let sequence = gather_candidate_sequences(&candidates, &best_index_tensor)?.squeeze(1)?;
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
            fallback: PlanFallback::None,
            used_host_elite_selection: false,
        })
    }
}

#[derive(Debug, Clone)]
pub struct MppiPlanner {
    config: MppiConfig,
    rng: PlannerRng,
    workspace: PlannerWorkspace,
}

impl MppiPlanner {
    pub fn new(config: MppiConfig) -> Self {
        Self {
            config,
            rng: PlannerRng::new(),
            workspace: PlannerWorkspace::new(),
        }
    }

    pub fn config(&self) -> &MppiConfig {
        &self.config
    }

    pub fn reset_rng_sequence(&self) {
        self.rng.reset();
    }

    pub fn rng_offset(&self) -> u64 {
        self.rng.offset()
    }

    pub fn plan<S: CandidateScorer>(&self, scorer: &S) -> Result<PlanResult> {
        self.config.validate()?;
        let start = Instant::now();
        let device = scorer.device();
        let dtype = scorer.dtype();
        let cfg = &self.config;
        let batch = scorer.batch_size().unwrap_or(1);
        let mut sampler = self.rng.begin_plan(
            device,
            cfg.seed,
            normal_draw_reservation(
                batch,
                cfg.samples,
                cfg.horizon,
                cfg.action_dim,
                cfg.iterations,
            )?,
        )?;

        let mut mean =
            self.workspace
                .sequence(batch, cfg.horizon, cfg.action_dim, dtype, device, 0.0)?;
        let std = self.workspace.sequence(
            batch,
            cfg.horizon,
            cfg.action_dim,
            dtype,
            device,
            cfg.noise_std,
        )?;
        let (low, high) = self.workspace.bounds(&cfg.action_bounds, dtype, device)?;
        let mut last_scores = None;
        let mut iterations_completed = 0;
        let mut deadline_reached = false;

        for iter_idx in 0..cfg.iterations {
            if deadline_elapsed(start, cfg.deadline) {
                deadline_reached = true;
                if iter_idx == 0 {
                    return configured_fallback_result(
                        cfg.fallback_action.as_deref(),
                        batch,
                        cfg.horizon,
                        cfg.action_dim,
                        dtype,
                        device,
                        start,
                        "MPPI",
                    );
                }
                break;
            }

            let candidates = sample_candidates(
                &mean,
                &std,
                cfg.samples,
                &low,
                &high,
                dtype,
                device,
                &mut sampler,
            )?;
            let scores = scorer.score_candidates(&candidates)?;
            validate_scores_shape(&scores, batch, cfg.samples)?;
            mean = mppi_weighted_sequence(&candidates, &scores, cfg.temperature)?;

            last_scores = Some(scores);
            iterations_completed += 1;
        }

        let scores = last_scores
            .ok_or_else(|| candle::Error::Msg("MPPI did not produce scores".to_string()))?;
        let sorted_indices = sorted_score_indices(&scores)?;
        let best_index_tensor = sorted_indices.narrow(1, 0, 1)?;
        let best_indices = best_indices_from_tensor(&best_index_tensor)?;
        let sequence = mean;
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
            fallback: PlanFallback::None,
            used_host_elite_selection: false,
        })
    }
}

#[derive(Debug, Clone)]
pub struct IcemPlanner {
    config: IcemConfig,
    warm_start: Option<Tensor>,
    rng: PlannerRng,
    workspace: PlannerWorkspace,
}

impl IcemPlanner {
    pub fn new(config: IcemConfig) -> Self {
        Self {
            config,
            warm_start: None,
            rng: PlannerRng::new(),
            workspace: PlannerWorkspace::new(),
        }
    }

    pub fn config(&self) -> &IcemConfig {
        &self.config
    }

    pub fn warm_start_sequence(&self) -> Option<&Tensor> {
        self.warm_start.as_ref()
    }

    pub fn clear_warm_start(&mut self) {
        self.warm_start = None;
    }

    pub fn reset_rng_sequence(&self) {
        self.rng.reset();
    }

    pub fn rng_offset(&self) -> u64 {
        self.rng.offset()
    }

    pub fn set_warm_start_sequence(&mut self, sequence: Tensor) {
        self.warm_start = Some(sequence);
    }

    pub fn plan<S: CandidateScorer>(&mut self, scorer: &S) -> Result<PlanResult> {
        self.config.validate()?;
        let start = Instant::now();
        let device = scorer.device();
        let dtype = scorer.dtype();
        let cfg = &self.config;
        let batch = scorer.batch_size().unwrap_or(1);
        let mut sampler = self.rng.begin_plan(
            device,
            cfg.seed,
            normal_draw_reservation(
                batch,
                cfg.samples,
                cfg.horizon,
                cfg.action_dim,
                cfg.iterations,
            )?,
        )?;

        let mut mean = self.initial_mean(batch, dtype, device)?;
        let mut std = self.workspace.sequence(
            batch,
            cfg.horizon,
            cfg.action_dim,
            dtype,
            device,
            cfg.init_std,
        )?;
        let (low, high) = self.workspace.bounds(&cfg.action_bounds, dtype, device)?;
        let mut carried_elites = None;
        let mut last_candidates = None;
        let mut last_scores = None;
        let mut iterations_completed = 0;
        let mut deadline_reached = false;

        for iter_idx in 0..cfg.iterations {
            if deadline_elapsed(start, cfg.deadline) {
                deadline_reached = true;
                if iter_idx == 0 {
                    if let Some(sequence) = self.fallback_warm_start(batch, dtype, device)? {
                        return fallback_plan_result(
                            sequence,
                            dtype,
                            device,
                            start,
                            PlanFallback::WarmStart,
                        );
                    }
                    return configured_fallback_result(
                        cfg.fallback_action.as_deref(),
                        batch,
                        cfg.horizon,
                        cfg.action_dim,
                        dtype,
                        device,
                        start,
                        "iCEM",
                    );
                }
                break;
            }

            let sampled = sample_candidates(
                &mean,
                &std,
                cfg.samples,
                &low,
                &high,
                dtype,
                device,
                &mut sampler,
            )?;
            let candidates = match carried_elites.as_ref() {
                Some(elites) => Tensor::cat(&[&sampled, elites], 1)?,
                None => sampled,
            };
            let candidate_count = candidates.dim(1)?;
            let scores = scorer.score_candidates(&candidates)?;
            validate_scores_shape(&scores, batch, candidate_count)?;

            let elites = select_elites(&candidates, &scores, cfg.elites)?;
            mean = elites.mean(1)?;
            std = enforce_min_std(&elites.var(1)?.sqrt()?, cfg.min_std)?;
            carried_elites = if cfg.keep_elites == 0 {
                None
            } else {
                Some(elites.narrow(1, 0, cfg.keep_elites)?)
            };

            last_candidates = Some(candidates);
            last_scores = Some(scores);
            iterations_completed += 1;
        }

        let candidates = last_candidates
            .ok_or_else(|| candle::Error::Msg("iCEM did not complete any iteration".to_string()))?;
        let scores = last_scores
            .ok_or_else(|| candle::Error::Msg("iCEM did not produce scores".to_string()))?;
        let sorted_indices = sorted_score_indices(&scores)?;
        let best_index_tensor = sorted_indices.narrow(1, 0, 1)?;
        let best_indices = best_indices_from_tensor(&best_index_tensor)?;
        let sequence = gather_candidate_sequences(&candidates, &best_index_tensor)?.squeeze(1)?;
        self.warm_start = Some(shift_sequence_for_warm_start(&sequence)?);
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
            fallback: PlanFallback::None,
            used_host_elite_selection: false,
        })
    }

    fn initial_mean(&self, batch: usize, dtype: DType, device: &Device) -> Result<Tensor> {
        let cfg = &self.config;
        let shape = (batch, cfg.horizon, cfg.action_dim);
        match self.warm_start.as_ref() {
            Some(sequence) if sequence.dims() == [batch, cfg.horizon, cfg.action_dim] => {
                sequence.to_device(device)?.to_dtype(dtype)?.reshape(shape)
            }
            Some(sequence) => candle::bail!(
                "iCEM warm-start shape {:?} does not match expected {:?}",
                sequence.dims(),
                [batch, cfg.horizon, cfg.action_dim]
            ),
            None => self
                .workspace
                .sequence(batch, cfg.horizon, cfg.action_dim, dtype, device, 0.0),
        }
    }

    fn fallback_warm_start(
        &self,
        batch: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Option<Tensor>> {
        let cfg = &self.config;
        match self.warm_start.as_ref() {
            Some(sequence) if sequence.dims() == [batch, cfg.horizon, cfg.action_dim] => {
                Ok(Some(sequence.to_device(device)?.to_dtype(dtype)?))
            }
            Some(sequence) => candle::bail!(
                "iCEM warm-start shape {:?} does not match expected {:?}",
                sequence.dims(),
                [batch, cfg.horizon, cfg.action_dim]
            ),
            None => Ok(None),
        }
    }
}

fn deadline_elapsed(start: Instant, deadline: Option<Duration>) -> bool {
    deadline.is_some_and(|deadline| start.elapsed() >= deadline)
}

fn validate_fallback_action(
    fallback_action: Option<&[f32]>,
    action_dim: usize,
    bounds: &ActionBounds,
    planner_name: &str,
) -> Result<()> {
    let Some(action) = fallback_action else {
        return Ok(());
    };
    if action.len() != action_dim {
        candle::bail!(
            "{planner_name} fallback_action length {} must match action_dim {action_dim}",
            action.len()
        );
    }
    for (idx, (&value, (&low, &high))) in action
        .iter()
        .zip(bounds.low.iter().zip(bounds.high.iter()))
        .enumerate()
    {
        if !value.is_finite() {
            candle::bail!("{planner_name} fallback_action[{idx}] is not finite");
        }
        if value < low || value > high {
            candle::bail!(
                "{planner_name} fallback_action[{idx}]={value} is outside [{low}, {high}]"
            );
        }
    }
    Ok(())
}

fn configured_fallback_result(
    fallback_action: Option<&[f32]>,
    batch: usize,
    horizon: usize,
    action_dim: usize,
    dtype: DType,
    device: &Device,
    start: Instant,
    planner_name: &str,
) -> Result<PlanResult> {
    let Some(action) = fallback_action else {
        candle::bail!(
            "{planner_name} deadline reached before any iteration completed and no fallback_action is configured"
        );
    };
    let sequence =
        fallback_sequence_from_action(action, batch, horizon, action_dim, dtype, device)?;
    fallback_plan_result(
        sequence,
        dtype,
        device,
        start,
        PlanFallback::ConfiguredAction,
    )
}

fn fallback_sequence_from_action(
    action: &[f32],
    batch: usize,
    horizon: usize,
    action_dim: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    Tensor::from_vec(action.to_vec(), (1, 1, action_dim), device)?
        .to_dtype(dtype)?
        .broadcast_as((batch, horizon, action_dim))
}

fn fallback_plan_result(
    sequence: Tensor,
    dtype: DType,
    device: &Device,
    start: Instant,
    fallback: PlanFallback,
) -> Result<PlanResult> {
    let batch = sequence.dim(0)?;
    let first_action = sequence.i((.., 0, ..))?;
    Ok(PlanResult {
        first_action,
        sequence,
        scores: Tensor::zeros((batch, 1), dtype, device)?,
        best_indices: vec![0; batch],
        iterations_completed: 0,
        elapsed: start.elapsed(),
        deadline_reached: true,
        fallback,
        used_host_elite_selection: false,
    })
}

fn sample_candidates(
    mean: &Tensor,
    std: &Tensor,
    samples: usize,
    low: &Tensor,
    high: &Tensor,
    dtype: DType,
    device: &Device,
    sampler: &mut PlanSampler,
) -> Result<Tensor> {
    let batch = mean.dim(0)?;
    let (_, horizon, action_dim) = mean.dims3()?;
    let shape = (batch, samples, horizon, action_dim);
    let noise = sampler.standard_normal(shape, dtype, device)?;
    let mean = mean.unsqueeze(1)?.broadcast_as(shape)?;
    let std = std.unsqueeze(1)?.broadcast_as(shape)?;
    let candidates = mean.broadcast_add(&noise.broadcast_mul(&std)?)?;
    clamp_actions(&candidates, low, high)
}

#[derive(Debug)]
struct PlannerWorkspace {
    bounds: Mutex<Option<CachedBounds>>,
    sequence: Mutex<Option<CachedSequence>>,
}

impl PlannerWorkspace {
    fn new() -> Self {
        Self {
            bounds: Mutex::new(None),
            sequence: Mutex::new(None),
        }
    }

    fn bounds(
        &self,
        bounds: &ActionBounds,
        dtype: DType,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let location = device.location();
        let mut cache = lock_workspace(&self.bounds)?;
        if let Some(cached) = cache.as_ref()
            && cached.matches(bounds, dtype, location)
        {
            return Ok((cached.low.clone(), cached.high.clone()));
        }

        let action_dim = bounds.low.len();
        let low = Tensor::from_vec(bounds.low.clone(), (action_dim,), device)?
            .to_dtype(dtype)?
            .reshape((1, 1, 1, action_dim))?;
        let high = Tensor::from_vec(bounds.high.clone(), (action_dim,), device)?
            .to_dtype(dtype)?
            .reshape((1, 1, 1, action_dim))?;
        *cache = Some(CachedBounds {
            location,
            dtype,
            low_values: bounds.low.clone(),
            high_values: bounds.high.clone(),
            low: low.clone(),
            high: high.clone(),
        });
        Ok((low, high))
    }

    fn sequence(
        &self,
        batch: usize,
        horizon: usize,
        action_dim: usize,
        dtype: DType,
        device: &Device,
        value: f32,
    ) -> Result<Tensor> {
        let location = device.location();
        let mut cache = lock_workspace(&self.sequence)?;
        if let Some(cached) = cache.as_ref()
            && cached.matches(batch, horizon, action_dim, dtype, location, value)
        {
            return Ok(cached.tensor.clone());
        }

        let shape = (batch, horizon, action_dim);
        let tensor = if value == 0.0 {
            Tensor::zeros(shape, dtype, device)?
        } else {
            Tensor::ones(shape, dtype, device)?.affine(value as f64, 0.0)?
        };
        *cache = Some(CachedSequence {
            location,
            dtype,
            batch,
            horizon,
            action_dim,
            value_bits: value.to_bits(),
            tensor: tensor.clone(),
        });
        Ok(tensor)
    }
}

impl Clone for PlannerWorkspace {
    fn clone(&self) -> Self {
        Self::new()
    }
}

#[derive(Debug)]
struct CachedBounds {
    location: DeviceLocation,
    dtype: DType,
    low_values: Vec<f32>,
    high_values: Vec<f32>,
    low: Tensor,
    high: Tensor,
}

impl CachedBounds {
    fn matches(&self, bounds: &ActionBounds, dtype: DType, location: DeviceLocation) -> bool {
        self.location == location
            && self.dtype == dtype
            && self.low_values == bounds.low
            && self.high_values == bounds.high
    }
}

#[derive(Debug)]
struct CachedSequence {
    location: DeviceLocation,
    dtype: DType,
    batch: usize,
    horizon: usize,
    action_dim: usize,
    value_bits: u32,
    tensor: Tensor,
}

impl CachedSequence {
    fn matches(
        &self,
        batch: usize,
        horizon: usize,
        action_dim: usize,
        dtype: DType,
        location: DeviceLocation,
        value: f32,
    ) -> bool {
        self.location == location
            && self.dtype == dtype
            && self.batch == batch
            && self.horizon == horizon
            && self.action_dim == action_dim
            && self.value_bits == value.to_bits()
    }
}

fn lock_workspace<T>(mutex: &Mutex<T>) -> Result<MutexGuard<'_, T>> {
    mutex
        .lock()
        .map_err(|_| candle::Error::Msg("planner workspace mutex poisoned".to_string()))
}

#[derive(Debug)]
struct PlannerRng {
    next_offset: AtomicU64,
}

impl PlannerRng {
    fn new() -> Self {
        Self {
            next_offset: AtomicU64::new(0),
        }
    }

    fn reset(&self) {
        self.next_offset.store(0, Ordering::SeqCst);
    }

    fn offset(&self) -> u64 {
        self.next_offset.load(Ordering::SeqCst)
    }

    fn begin_plan(
        &self,
        device: &Device,
        seed: Option<u64>,
        reserved_draws: u64,
    ) -> Result<PlanSampler> {
        let Some(seed) = seed else {
            return Ok(PlanSampler::Device);
        };
        let offset = self.reserve_offset(reserved_draws)?;
        Ok(PlanSampler::Cuda(CudaNormalSampler::new(
            seed, offset, device,
        )?))
    }

    fn reserve_offset(&self, reserved_draws: u64) -> Result<u64> {
        let reserved_draws = reserved_draws.max(1);
        self.next_offset
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                current.checked_add(reserved_draws)
            })
            .map_err(|_| candle::Error::Msg("planner CUDA RNG offset overflowed".to_string()))
    }
}

impl Clone for PlannerRng {
    fn clone(&self) -> Self {
        Self {
            next_offset: AtomicU64::new(self.offset()),
        }
    }
}

enum PlanSampler {
    Device,
    Cuda(CudaNormalSampler),
}

impl PlanSampler {
    fn standard_normal(
        &mut self,
        shape: (usize, usize, usize, usize),
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        match self {
            Self::Device => Tensor::randn(0f32, 1f32, shape, device)?.to_dtype(dtype),
            Self::Cuda(sampler) => sampler.standard_normal(shape, dtype),
        }
    }
}

struct CudaNormalSampler {
    rng: cudarc::curand::CudaRng,
    device: candle::CudaDevice,
}

impl CudaNormalSampler {
    fn new(seed: u64, offset: u64, device: &Device) -> Result<Self> {
        let cuda = device.as_cuda_device()?.clone();
        let mut rng =
            cudarc::curand::CudaRng::new(seed, cuda.cuda_stream()).map_err(candle::Error::wrap)?;
        rng.set_offset(offset).map_err(candle::Error::wrap)?;
        Ok(Self { rng, device: cuda })
    }

    fn standard_normal(
        &mut self,
        shape: (usize, usize, usize, usize),
        dtype: DType,
    ) -> Result<Tensor> {
        let elem_count = shape
            .0
            .checked_mul(shape.1)
            .and_then(|v| v.checked_mul(shape.2))
            .and_then(|v| v.checked_mul(shape.3))
            .ok_or_else(|| candle::Error::Msg("planner CUDA RNG shape overflowed".to_string()))?;
        let elem_count = round_curand_normal_count(elem_count)?;
        let mut data = unsafe { self.device.alloc::<f32>(elem_count)? };
        self.rng
            .fill_with_normal(&mut data, 0f32, 1f32)
            .map_err(candle::Error::wrap)?;
        let storage = CudaStorage::wrap_cuda_slice(data, self.device.clone());
        Tensor::from_storage(Storage::Cuda(storage), shape, BackpropOp::none(), false)
            .to_dtype(dtype)
    }
}

fn normal_draw_reservation(
    batch: usize,
    samples: usize,
    horizon: usize,
    action_dim: usize,
    iterations: usize,
) -> Result<u64> {
    let per_iteration = batch
        .checked_mul(samples)
        .and_then(|v| v.checked_mul(horizon))
        .and_then(|v| v.checked_mul(action_dim))
        .ok_or_else(|| candle::Error::Msg("planner CUDA RNG shape overflowed".to_string()))?;
    let per_iteration = round_curand_normal_count(per_iteration)? as u64;
    per_iteration
        .checked_mul(iterations as u64)
        .ok_or_else(|| candle::Error::Msg("planner CUDA RNG offset overflowed".to_string()))
}

fn round_curand_normal_count(count: usize) -> Result<usize> {
    if count % 2 == 0 {
        Ok(count)
    } else {
        count
            .checked_add(1)
            .ok_or_else(|| candle::Error::Msg("planner CUDA RNG shape overflowed".to_string()))
    }
}

fn mppi_weighted_sequence(
    candidates: &Tensor,
    scores: &Tensor,
    temperature: f32,
) -> Result<Tensor> {
    let (batch, samples, horizon, action_dim) = candidates.dims4()?;
    let min_score = scores.min_keepdim(1)?;
    let logits = scores
        .broadcast_sub(&min_score)?
        .affine(-(1.0 / temperature as f64), 0.0)?;
    let weights = ops::softmax(&logits, 1)?
        .reshape((batch, samples, 1, 1))?
        .broadcast_as((batch, samples, horizon, action_dim))?;
    candidates.broadcast_mul(&weights)?.sum(1)
}

fn shift_sequence_for_warm_start(sequence: &Tensor) -> Result<Tensor> {
    let (_, horizon, _) = sequence.dims3()?;
    if horizon == 1 {
        return Ok(sequence.clone());
    }
    let tail = sequence.narrow(1, 1, horizon - 1)?;
    let last = sequence.narrow(1, horizon - 1, 1)?;
    Tensor::cat(&[&tail, &last], 1)
}

fn clamp_actions(candidates: &Tensor, low: &Tensor, high: &Tensor) -> Result<Tensor> {
    candidates.broadcast_maximum(low)?.broadcast_minimum(high)
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

fn validate_scores_values(scores: &Tensor) -> Result<()> {
    let scores = scores.to_dtype(DType::F32)?;
    let min = scores.min_all()?.to_scalar::<f32>()?;
    let max = scores.max_all()?.to_scalar::<f32>()?;
    if !min.is_finite() || !max.is_finite() {
        candle::bail!("scores contain non-finite values: min={min} max={max}");
    }
    Ok(())
}

fn sorted_score_indices(scores: &Tensor) -> Result<Tensor> {
    validate_scores_values(scores)?;
    scores.arg_sort_last_dim(true)
}

fn select_elites(candidates: &Tensor, scores: &Tensor, elite_count: usize) -> Result<Tensor> {
    let (_, samples, _, _) = candidates.dims4()?;
    if elite_count > samples {
        candle::bail!("elite_count {elite_count} cannot exceed samples {samples}");
    }

    let elite_indices = sorted_score_indices(scores)?.narrow(1, 0, elite_count)?;
    gather_candidate_sequences(candidates, &elite_indices)
}

fn gather_candidate_sequences(candidates: &Tensor, indices: &Tensor) -> Result<Tensor> {
    let (batch, _, horizon, action_dim) = candidates.dims4()?;
    let selected = match indices.dims() {
        [b, selected] if *b == batch => *selected,
        other => {
            candle::bail!("candidate indices must have shape [{batch}, selected], got {other:?}")
        }
    };

    let gather_indices = indices
        .reshape((batch, selected, 1, 1))?
        .broadcast_as((batch, selected, horizon, action_dim))?
        .contiguous()?;
    candidates.contiguous()?.gather(&gather_indices, 1)
}

fn best_indices_from_tensor(indices: &Tensor) -> Result<Vec<usize>> {
    let rows = indices.to_vec2::<u32>()?;
    let mut best_indices = Vec::with_capacity(rows.len());
    for (batch_idx, row) in rows.iter().enumerate() {
        let Some(&best_idx) = row.first() else {
            candle::bail!("best index row {batch_idx} is empty");
        };
        best_indices.push(best_idx as usize);
    }
    Ok(best_indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_elites_uses_device_sort_and_gather_per_batch() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let candidates = Tensor::arange(0f32, 8f32, &device)?.reshape((2, 4, 1, 1))?;
        let scores = Tensor::new(&[[3f32, 1., 4., 0.5], [9., -1., 2., -2.]], &device)?;

        let elites = select_elites(&candidates, &scores, 2)?;
        assert_eq!(
            elites.reshape((2, 2))?.to_vec2::<f32>()?,
            &[[3., 1.], [7., 5.]]
        );

        let sorted_indices = sorted_score_indices(&scores)?;
        let best_indices = sorted_indices.narrow(1, 0, 1)?;
        let sequences = gather_candidate_sequences(&candidates, &best_indices)?.squeeze(1)?;

        assert_eq!(best_indices_from_tensor(&best_indices)?, &[3, 3]);
        assert_eq!(sequences.reshape((2,))?.to_vec1::<f32>()?, &[3., 7.]);
        Ok(())
    }

    #[test]
    fn sorted_score_indices_rejects_non_finite_scores() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let scores = Tensor::new(&[[0f32, f32::INFINITY]], &device)?;
        let err = sorted_score_indices(&scores).unwrap_err();

        assert!(err.to_string().contains("non-finite"));
        Ok(())
    }

    #[test]
    fn planner_workspace_reuses_cached_tensors() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let workspace = PlannerWorkspace::new();
        let bounds = ActionBounds::symmetric(4, 0.5);

        let (low_a, high_a) = workspace.bounds(&bounds, DType::F32, &device)?;
        let (low_b, high_b) = workspace.bounds(&bounds, DType::F32, &device)?;
        let zeros_a = workspace.sequence(2, 3, 4, DType::F32, &device, 0.0)?;
        let zeros_b = workspace.sequence(2, 3, 4, DType::F32, &device, 0.0)?;
        let std_a = workspace.sequence(2, 3, 4, DType::F32, &device, 0.75)?;
        let std_b = workspace.sequence(2, 3, 4, DType::F32, &device, 0.75)?;

        assert_eq!(low_a.id(), low_b.id());
        assert_eq!(high_a.id(), high_b.id());
        assert_eq!(zeros_a.id(), zeros_b.id());
        assert_eq!(std_a.id(), std_b.id());
        Ok(())
    }
}
