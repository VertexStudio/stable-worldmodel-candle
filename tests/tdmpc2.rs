use candle::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use stable_worldmodel_candle::models::tdmpc2::{TdMpc2, TdMpc2Config};

fn model(state_dim: usize, action_dim: usize) -> candle::Result<TdMpc2> {
    let device = Device::Cpu;
    let vars = VarMap::new();
    let vb = VarBuilder::from_varmap(&vars, DType::F32, &device);
    TdMpc2::new(TdMpc2Config::state_only(state_dim, action_dim), vb)
}

#[test]
fn encodes_state_observations() -> candle::Result<()> {
    let model = model(12, 4)?;
    let state = Tensor::randn(0f32, 1f32, (2, 12), &Device::Cpu)?;

    let z = model.encode_state(&state)?;

    assert_eq!(z.dims(), &[2, 128]);
    Ok(())
}

#[test]
fn forward_predicts_next_latent_and_reward_logits() -> candle::Result<()> {
    let model = model(12, 4)?;
    let z = Tensor::randn(0f32, 1f32, (2, 128), &Device::Cpu)?;
    let action = Tensor::randn(0f32, 1f32, (2, 4), &Device::Cpu)?;

    let (next_z, reward_logits) = model.forward(&z, &action)?;

    assert_eq!(next_z.dims(), &[2, 128]);
    assert_eq!(reward_logits.dims(), &[2, 101]);
    Ok(())
}

#[test]
fn scores_action_candidates_from_state_batch() -> candle::Result<()> {
    let model = model(12, 4)?;
    let state = Tensor::randn(0f32, 1f32, (2, 12), &Device::Cpu)?;
    let actions = Tensor::randn(0f32, 1f32, (2, 5, 3, 4), &Device::Cpu)?;

    let cost = model.get_cost_state(&state, &actions)?;

    assert_eq!(cost.dims(), &[2, 5]);
    for row in cost.to_vec2::<f32>()? {
        for value in row {
            assert!(value.is_finite());
        }
    }
    Ok(())
}

#[test]
fn scores_action_candidates_from_expanded_state_batch() -> candle::Result<()> {
    let model = model(12, 4)?;
    let state = Tensor::randn(0f32, 1f32, (2, 5, 12), &Device::Cpu)?;
    let actions = Tensor::randn(0f32, 1f32, (2, 5, 3, 4), &Device::Cpu)?;

    let cost = model.get_cost_state(&state, &actions)?;

    assert_eq!(cost.dims(), &[2, 5]);
    Ok(())
}
