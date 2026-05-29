use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use stable_worldmodel_candle::{
    checkpoint,
    models::tdmpc2::{TdMpc2, TdMpc2Config},
    session::TdMpc2Session,
};

#[test]
fn tdmpc2_session_scores_candidates_after_reset() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let dtype = DType::F32;
    let cfg = TdMpc2Config::state_only(12, 4);
    let model = TdMpc2::new(cfg, empty_vb(dtype, &device))?;
    let mut session = TdMpc2Session::new(model, device.clone(), dtype);

    let state = Tensor::randn(0f32, 1f32, (2, 12), &device)?;
    let z = session.reset_state(&state)?;
    assert_eq!(z.shape().dims(), &[2, 128]);
    assert!(session.cached_latent().is_some());

    let action = session.actor_mean_action()?;
    assert_eq!(action.shape().dims(), &[2, 4]);

    let candidates = Tensor::randn(0f32, 1f32, (2, 5, 3, 4), &device)?;
    let cost = session.score_candidates(&candidates)?;
    assert_eq!(cost.shape().dims(), &[2, 5]);
    Ok(())
}

#[test]
fn tdmpc2_session_requires_reset_before_scoring() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let dtype = DType::F32;
    let cfg = TdMpc2Config::state_only(12, 4);
    let model = TdMpc2::new(cfg, empty_vb(dtype, &device))?;
    let session = TdMpc2Session::new(model, device.clone(), dtype);

    let candidates = Tensor::randn(0f32, 1f32, (2, 5, 3, 4), &device)?;
    let err = session.score_candidates(&candidates).unwrap_err();

    assert!(err.to_string().contains("must be reset"));
    Ok(())
}

#[test]
fn tdmpc2_session_scores_pixel_candidates_after_reset() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let dtype = DType::F32;
    let cfg = TdMpc2Config::pixel_only(64, 4, 128);
    let model = TdMpc2::new(cfg, empty_vb(dtype, &device))?;
    let mut session = TdMpc2Session::new(model, device.clone(), dtype);

    let pixels = Tensor::randn(0f32, 1f32, (2, 3, 64, 64), &device)?;
    let z = session.reset_pixels(&pixels)?;
    assert_eq!(z.shape().dims(), &[2, 128]);

    let candidates = Tensor::randn(0f32, 1f32, (2, 5, 3, 4), &device)?;
    let cost = session.score_candidates(&candidates)?;
    assert_eq!(cost.shape().dims(), &[2, 5]);
    Ok(())
}

fn empty_vb(dtype: DType, device: &Device) -> VarBuilder<'static> {
    checkpoint::empty_var_builder(dtype, device)
}
