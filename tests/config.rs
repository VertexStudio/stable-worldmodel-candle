use stable_worldmodel_rs::{LeWmConfig, NormKind};

#[test]
fn tiny_patch14_defaults_match_python_config() {
    let cfg = LeWmConfig::tiny_patch14_224(2);

    assert_eq!(cfg.encoder.image_size, 224);
    assert_eq!(cfg.encoder.patch_size, 14);
    assert_eq!(cfg.encoder.hidden_size, 192);
    assert_eq!(cfg.encoder.num_hidden_layers, 12);
    assert_eq!(cfg.encoder.num_attention_heads, 3);

    assert_eq!(cfg.predictor.num_frames, 3);
    assert_eq!(cfg.predictor.depth, 6);
    assert_eq!(cfg.predictor.heads, 16);
    assert_eq!(cfg.predictor.dim_head, 64);
    assert_eq!(cfg.predictor.mlp_dim, 2048);

    assert_eq!(cfg.action_encoder.input_dim, 2);
    assert_eq!(cfg.projector.norm, NormKind::BatchNorm1d);
    assert_eq!(cfg.pred_proj.norm, NormKind::BatchNorm1d);
}

#[test]
fn config_round_trips_through_json() {
    let cfg = LeWmConfig::tiny_patch14_224(4);
    let json = serde_json::to_string(&cfg).unwrap();
    let decoded: LeWmConfig = serde_json::from_str(&json).unwrap();

    assert_eq!(decoded.action_encoder.input_dim, 4);
    assert_eq!(decoded.encoder.patch_size, cfg.encoder.patch_size);
    assert_eq!(decoded.predictor.hidden_dim, cfg.predictor.hidden_dim);
}
