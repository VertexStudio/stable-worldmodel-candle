use std::time::Duration;

use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use stable_worldmodel_candle::{
    checkpoint,
    models::tdmpc2::{TdMpc2, TdMpc2Config},
    planner::{
        ActionBounds, CemConfig, CemPlanner, IcemConfig, IcemPlanner, MppiConfig, MppiPlanner,
        PlanFallback,
    },
    session::TdMpc2Session,
};

#[test]
fn cem_plans_tdmpc2_action_sequence() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;
    let dtype = DType::F32;
    let state_dim = 12;
    let action_dim = 4;
    let cfg = TdMpc2Config::state_only(state_dim, action_dim);
    let model = TdMpc2::new(cfg, empty_vb(dtype, &device))?;
    let mut session = TdMpc2Session::new(model, device.clone(), dtype);

    let batch = 2;
    let state = Tensor::randn(0f32, 1f32, (batch, state_dim), &device)?;
    session.reset_state(&state)?;

    let mut cem_cfg = CemConfig::new(3, 8, 3, action_dim);
    cem_cfg.iterations = 2;
    cem_cfg.init_std = 0.5;
    cem_cfg.action_bounds = ActionBounds::symmetric(action_dim, 0.75);

    let result = CemPlanner::new(cem_cfg).plan(&session)?;

    assert_eq!(result.first_action.dims(), &[batch, action_dim]);
    assert_eq!(result.sequence.dims(), &[batch, 3, action_dim]);
    assert_eq!(result.scores.dims(), &[batch, 8]);
    assert_eq!(result.best_indices.len(), batch);
    assert_eq!(result.iterations_completed, 2);
    assert_eq!(result.fallback, PlanFallback::None);
    assert!(!result.used_host_elite_selection);

    for action in result.first_action.to_vec2::<f32>()? {
        for value in action {
            assert!((-0.75..=0.75).contains(&value));
        }
    }
    for row in result.scores.to_vec2::<f32>()? {
        for value in row {
            assert!(value.is_finite());
        }
    }

    Ok(())
}

#[test]
fn cem_propagates_scorer_reset_error() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;
    let dtype = DType::F32;
    let action_dim = 4;
    let cfg = TdMpc2Config::state_only(12, action_dim);
    let model = TdMpc2::new(cfg, empty_vb(dtype, &device))?;
    let session = TdMpc2Session::new(model, device, dtype);

    let planner = CemPlanner::new(CemConfig::new(3, 8, 3, action_dim));
    let err = planner.plan(&session).unwrap_err();

    assert!(err.to_string().contains("must be reset"));
    Ok(())
}

#[test]
fn mppi_plans_tdmpc2_action_sequence() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;
    let dtype = DType::F32;
    let state_dim = 12;
    let action_dim = 4;
    let model = TdMpc2::new(
        TdMpc2Config::state_only(state_dim, action_dim),
        empty_vb(dtype, &device),
    )?;
    let mut session = TdMpc2Session::new(model, device.clone(), dtype);

    let batch = 2;
    let state = Tensor::randn(0f32, 1f32, (batch, state_dim), &device)?;
    session.reset_state(&state)?;

    let mut mppi_cfg = MppiConfig::new(3, 8, action_dim);
    mppi_cfg.iterations = 2;
    mppi_cfg.noise_std = 0.5;
    mppi_cfg.temperature = 0.75;
    mppi_cfg.action_bounds = ActionBounds::symmetric(action_dim, 0.75);

    let result = MppiPlanner::new(mppi_cfg).plan(&session)?;

    assert_eq!(result.first_action.dims(), &[batch, action_dim]);
    assert_eq!(result.sequence.dims(), &[batch, 3, action_dim]);
    assert_eq!(result.scores.dims(), &[batch, 8]);
    assert_eq!(result.best_indices.len(), batch);
    assert_eq!(result.iterations_completed, 2);
    assert_eq!(result.fallback, PlanFallback::None);
    assert!(!result.used_host_elite_selection);

    for action in result.first_action.to_vec2::<f32>()? {
        for value in action {
            assert!((-0.75..=0.75).contains(&value));
        }
    }
    for row in result.scores.to_vec2::<f32>()? {
        for value in row {
            assert!(value.is_finite());
        }
    }

    Ok(())
}

#[test]
fn icem_keeps_shifted_warm_start_between_plans() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;
    let dtype = DType::F32;
    let state_dim = 12;
    let action_dim = 4;
    let model = TdMpc2::new(
        TdMpc2Config::state_only(state_dim, action_dim),
        empty_vb(dtype, &device),
    )?;
    let mut session = TdMpc2Session::new(model, device.clone(), dtype);

    let batch = 2;
    let state = Tensor::randn(0f32, 1f32, (batch, state_dim), &device)?;
    session.reset_state(&state)?;

    let mut icem_cfg = IcemConfig::new(3, 8, 3, action_dim);
    icem_cfg.keep_elites = 2;
    icem_cfg.iterations = 2;
    icem_cfg.init_std = 0.5;
    icem_cfg.action_bounds = ActionBounds::symmetric(action_dim, 0.75);

    let mut planner = IcemPlanner::new(icem_cfg);
    let first = planner.plan(&session)?;
    let warm_start = planner
        .warm_start_sequence()
        .expect("iCEM should retain a warm-start sequence")
        .clone();
    let second = planner.plan(&session)?;

    assert_eq!(first.first_action.dims(), &[batch, action_dim]);
    assert_eq!(first.sequence.dims(), &[batch, 3, action_dim]);
    assert_eq!(first.scores.dims(), &[batch, 10]);
    assert_eq!(first.best_indices.len(), batch);
    assert_eq!(first.iterations_completed, 2);
    assert_eq!(first.fallback, PlanFallback::None);
    assert!(!first.used_host_elite_selection);
    assert_eq!(warm_start.dims(), &[batch, 3, action_dim]);
    assert_eq!(second.first_action.dims(), &[batch, action_dim]);
    assert!(planner.warm_start_sequence().is_some());

    for action in first.first_action.to_vec2::<f32>()? {
        for value in action {
            assert!((-0.75..=0.75).contains(&value));
        }
    }

    Ok(())
}

#[test]
fn cem_returns_configured_fallback_when_deadline_prevents_iteration() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;
    let dtype = DType::F32;
    let action_dim = 4;
    let model = TdMpc2::new(
        TdMpc2Config::state_only(12, action_dim),
        empty_vb(dtype, &device),
    )?;
    let session = TdMpc2Session::new(model, device, dtype);

    let mut cem_cfg = CemConfig::new(3, 8, 3, action_dim);
    cem_cfg.deadline = Some(Duration::ZERO);
    cem_cfg.fallback_action = Some(vec![0.25, -0.25, 0.5, -0.5]);

    let result = CemPlanner::new(cem_cfg).plan(&session)?;

    assert_eq!(result.iterations_completed, 0);
    assert!(result.deadline_reached);
    assert_eq!(result.fallback, PlanFallback::ConfiguredAction);
    assert_eq!(
        result.first_action.to_vec2::<f32>()?,
        &[[0.25, -0.25, 0.5, -0.5]]
    );
    assert_eq!(result.sequence.dims(), &[1, 3, action_dim]);
    Ok(())
}

#[test]
fn cem_seed_replays_same_candidate_sequence() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;
    let dtype = DType::F32;
    let state_dim = 12;
    let action_dim = 4;
    let model = TdMpc2::new(
        TdMpc2Config::state_only(state_dim, action_dim),
        empty_vb(dtype, &device),
    )?;
    let mut session = TdMpc2Session::new(model, device.clone(), dtype);
    let state = Tensor::randn(0f32, 1f32, (2, state_dim), &device)?;
    session.reset_state(&state)?;

    let mut cem_cfg = CemConfig::new(3, 8, 3, action_dim);
    cem_cfg.iterations = 2;
    cem_cfg.seed = Some(42);

    let first = CemPlanner::new(cem_cfg.clone()).plan(&session)?;
    let second = CemPlanner::new(cem_cfg).plan(&session)?;

    assert_eq!(
        first.sequence.to_vec3::<f32>()?,
        second.sequence.to_vec3::<f32>()?
    );
    assert_eq!(first.best_indices, second.best_indices);
    Ok(())
}

#[test]
fn mppi_returns_configured_fallback_when_deadline_prevents_iteration() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;
    let dtype = DType::F32;
    let action_dim = 4;
    let model = TdMpc2::new(
        TdMpc2Config::state_only(12, action_dim),
        empty_vb(dtype, &device),
    )?;
    let session = TdMpc2Session::new(model, device, dtype);

    let mut mppi_cfg = MppiConfig::new(3, 8, action_dim);
    mppi_cfg.deadline = Some(Duration::ZERO);
    mppi_cfg.fallback_action = Some(vec![-0.2, 0.2, -0.4, 0.4]);

    let result = MppiPlanner::new(mppi_cfg).plan(&session)?;

    assert_eq!(result.iterations_completed, 0);
    assert!(result.deadline_reached);
    assert_eq!(result.fallback, PlanFallback::ConfiguredAction);
    assert_eq!(
        result.first_action.to_vec2::<f32>()?,
        &[[-0.2, 0.2, -0.4, 0.4]]
    );
    assert_eq!(result.sequence.dims(), &[1, 3, action_dim]);
    Ok(())
}

#[test]
fn icem_returns_warm_start_when_deadline_prevents_iteration() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;
    let dtype = DType::F32;
    let action_dim = 4;
    let model = TdMpc2::new(
        TdMpc2Config::state_only(12, action_dim),
        empty_vb(dtype, &device),
    )?;
    let session = TdMpc2Session::new(model, device.clone(), dtype);

    let mut icem_cfg = IcemConfig::new(3, 8, 3, action_dim);
    icem_cfg.deadline = Some(Duration::ZERO);
    icem_cfg.fallback_action = Some(vec![0.1; action_dim]);
    let mut planner = IcemPlanner::new(icem_cfg);
    let warm_start = Tensor::new(
        &[[
            [0.2f32, 0.1, 0.0, -0.1],
            [0.3, 0.2, 0.1, 0.0],
            [0.4, 0.3, 0.2, 0.1],
        ]],
        &device,
    )?;
    planner.set_warm_start_sequence(warm_start.clone());

    let result = planner.plan(&session)?;

    assert_eq!(result.iterations_completed, 0);
    assert!(result.deadline_reached);
    assert_eq!(result.fallback, PlanFallback::WarmStart);
    assert_eq!(
        result.sequence.to_vec3::<f32>()?,
        warm_start.to_vec3::<f32>()?
    );
    Ok(())
}

fn empty_vb(dtype: DType, device: &Device) -> VarBuilder<'static> {
    checkpoint::empty_var_builder(dtype, device)
}
