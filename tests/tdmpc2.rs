use candle::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use stable_worldmodel_candle::models::tdmpc2::{EncodingConfig, TdMpc2, TdMpc2Config};

fn model(state_dim: usize, action_dim: usize, device: &Device) -> candle::Result<TdMpc2> {
    let vars = VarMap::new();
    let vb = VarBuilder::from_varmap(&vars, DType::F32, device);
    TdMpc2::new(TdMpc2Config::state_only(state_dim, action_dim), vb)
}

#[test]
fn encodes_state_observations() -> candle::Result<()> {
    let device = Device::new_cuda(0)?;
    let model = model(12, 4, &device)?;
    let state = Tensor::randn(0f32, 1f32, (2, 12), &device)?;

    let z = model.encode_state(&state)?;

    assert_eq!(z.dims(), &[2, 128]);
    Ok(())
}

#[test]
fn forward_predicts_next_latent_and_reward_logits() -> candle::Result<()> {
    let device = Device::new_cuda(0)?;
    let model = model(12, 4, &device)?;
    let z = Tensor::randn(0f32, 1f32, (2, 128), &device)?;
    let action = Tensor::randn(0f32, 1f32, (2, 4), &device)?;

    let (next_z, reward_logits) = model.forward(&z, &action)?;

    assert_eq!(next_z.dims(), &[2, 128]);
    assert_eq!(reward_logits.dims(), &[2, 101]);
    Ok(())
}

#[test]
fn rolls_out_actor_mean_policy() -> candle::Result<()> {
    let device = Device::new_cuda(0)?;
    let model = model(12, 4, &device)?;
    let z = Tensor::randn(0f32, 1f32, (2, 128), &device)?;

    let (actions, rewards, final_z) = model.rollout_actor_mean(&z, 3)?;

    assert_eq!(actions.dims(), &[2, 3, 4]);
    assert_eq!(rewards.dims(), &[2, 3, 1]);
    assert_eq!(final_z.dims(), &[2, 128]);
    for row in actions.flatten_all()?.to_vec1::<f32>()? {
        assert!(row.is_finite());
    }
    for value in rewards.flatten_all()?.to_vec1::<f32>()? {
        assert!(value.is_finite());
    }
    Ok(())
}

#[test]
fn scores_action_candidates_from_state_batch() -> candle::Result<()> {
    let device = Device::new_cuda(0)?;
    let model = model(12, 4, &device)?;
    let state = Tensor::randn(0f32, 1f32, (2, 12), &device)?;
    let actions = Tensor::randn(0f32, 1f32, (2, 5, 3, 4), &device)?;

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
    let device = Device::new_cuda(0)?;
    let model = model(12, 4, &device)?;
    let state = Tensor::randn(0f32, 1f32, (2, 5, 12), &device)?;
    let actions = Tensor::randn(0f32, 1f32, (2, 5, 3, 4), &device)?;

    let cost = model.get_cost_state(&state, &actions)?;

    assert_eq!(cost.dims(), &[2, 5]);
    Ok(())
}

#[test]
fn encodes_pixel_observations_from_nchw_and_nhwc() -> candle::Result<()> {
    let device = Device::new_cuda(0)?;
    let model = model_pixels(64, 4, 128, &device)?;
    let nchw = Tensor::randn(0f32, 1f32, (2, 3, 64, 64), &device)?;
    let nhwc = Tensor::randn(0f32, 1f32, (2, 64, 64, 3), &device)?;

    let z_nchw = model.encode_pixels(&nchw)?;
    let z_nhwc = model.encode_pixels(&nhwc)?;

    assert_eq!(z_nchw.dims(), &[2, 128]);
    assert_eq!(z_nhwc.dims(), &[2, 128]);
    Ok(())
}

#[test]
fn scores_action_candidates_from_pixel_batch() -> candle::Result<()> {
    let device = Device::new_cuda(0)?;
    let model = model_pixels(64, 4, 128, &device)?;
    let pixels = Tensor::randn(0f32, 1f32, (2, 3, 64, 64), &device)?;
    let actions = Tensor::randn(0f32, 1f32, (2, 5, 3, 4), &device)?;

    let cost = model.get_cost(&[("pixels", &pixels)], &actions)?;

    assert_eq!(cost.dims(), &[2, 5]);
    Ok(())
}

#[test]
fn encodes_combined_pixel_and_state_observations() -> candle::Result<()> {
    let device = Device::new_cuda(0)?;
    let mut cfg = TdMpc2Config::pixel_only(64, 4, 128);
    cfg.encodings.push(EncodingConfig::new("state", 12, 128));
    let vars = VarMap::new();
    let vb = VarBuilder::from_varmap(&vars, DType::F32, &device);
    let model = TdMpc2::new(cfg, vb)?;
    let pixels = Tensor::randn(0f32, 1f32, (2, 3, 64, 64), &device)?;
    let state = Tensor::randn(0f32, 1f32, (2, 12), &device)?;

    let z = model.encode(&[("pixels", &pixels), ("state", &state)])?;

    assert_eq!(z.dims(), &[2, 256]);
    Ok(())
}

fn model_pixels(
    image_size: usize,
    action_dim: usize,
    pixel_dim: usize,
    device: &Device,
) -> candle::Result<TdMpc2> {
    let vars = VarMap::new();
    let vb = VarBuilder::from_varmap(&vars, DType::F32, device);
    TdMpc2::new(
        TdMpc2Config::pixel_only(image_size, action_dim, pixel_dim),
        vb,
    )
}
