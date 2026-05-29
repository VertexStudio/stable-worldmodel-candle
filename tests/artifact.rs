use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use stable_worldmodel_candle::{
    artifact::DeploymentArtifact,
    config::{ModelConfig, ModelKind},
};

#[test]
fn loads_deployment_artifact_with_safetensors() {
    let dir = temp_artifact_dir("safetensors");
    write_base_artifact(&dir, "model.safetensors");

    let artifact = DeploymentArtifact::from_dir(&dir).unwrap();

    assert_eq!(artifact.config.kind(), ModelKind::TdMpc2);
    assert_eq!(artifact.weights, dir.join("model.safetensors"));
    assert_eq!(artifact.schema.observations[0].name, "state");
    assert_eq!(artifact.schema.action.dim, 4);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn falls_back_to_pt_weights() {
    let dir = temp_artifact_dir("pt");
    write_base_artifact(&dir, "weights.pt");

    let artifact = DeploymentArtifact::from_dir(&dir).unwrap();

    assert_eq!(artifact.weights, dir.join("weights.pt"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn rejects_missing_schema() {
    let dir = temp_artifact_dir("missing-schema");
    fs::create_dir_all(&dir).unwrap();
    write_config(&dir);
    fs::write(dir.join("preprocess.json"), "{}").unwrap();
    fs::write(dir.join("model.safetensors"), "").unwrap();

    let err = DeploymentArtifact::from_dir(&dir).unwrap_err();

    assert!(err.to_string().contains("schema.json"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn rejects_invalid_action_bounds() {
    let dir = temp_artifact_dir("invalid-action");
    fs::create_dir_all(&dir).unwrap();
    write_config(&dir);
    fs::write(dir.join("preprocess.json"), "{}").unwrap();
    fs::write(dir.join("model.safetensors"), "").unwrap();
    fs::write(
        dir.join("schema.json"),
        r#"{
            "observations": [{"name": "state", "kind": "state", "shape": [12]}],
            "action": {"dim": 4, "min": [-1.0], "max": [1.0, 1.0, 1.0, 1.0]}
        }"#,
    )
    .unwrap();

    let err = DeploymentArtifact::from_dir(&dir).unwrap_err();

    assert!(err.to_string().contains("action min length"));

    fs::remove_dir_all(dir).unwrap();
}

fn write_base_artifact(dir: &Path, weights_name: &str) {
    fs::create_dir_all(dir).unwrap();
    write_config(dir);
    fs::write(
        dir.join("preprocess.json"),
        r#"{
            "action_min": [-1.0, -1.0, -1.0, -1.0],
            "action_max": [1.0, 1.0, 1.0, 1.0]
        }"#,
    )
    .unwrap();
    fs::write(
        dir.join("schema.json"),
        r#"{
            "observations": [{"name": "state", "kind": "state", "shape": [12]}],
            "action": {
                "dim": 4,
                "min": [-1.0, -1.0, -1.0, -1.0],
                "max": [1.0, 1.0, 1.0, 1.0]
            }
        }"#,
    )
    .unwrap();
    fs::write(dir.join(weights_name), "").unwrap();
}

fn write_config(dir: &Path) {
    let config = ModelConfig::tdmpc2_state_only(12, 4);
    fs::write(
        dir.join("config.json"),
        serde_json::to_string_pretty(&config).unwrap(),
    )
    .unwrap();
}

fn temp_artifact_dir(label: &str) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "stable-worldmodel-candle-artifact-{label}-{}-{stamp}",
        std::process::id()
    ))
}
