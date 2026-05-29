use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use stable_worldmodel_candle::{
    checkpoint,
    models::tdmpc2::{TdMpc2, TdMpc2Config},
    planner::{ActionBounds, CemConfig, CemPlanner},
    session::TdMpc2Session,
};

#[test]
fn cem_plans_tdmpc2_action_sequence() -> anyhow::Result<()> {
    let device = Device::Cpu;
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
    assert!(result.used_host_elite_selection);

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
    let device = Device::Cpu;
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

fn empty_vb(dtype: DType, device: &Device) -> VarBuilder<'static> {
    checkpoint::empty_var_builder(dtype, device)
}
