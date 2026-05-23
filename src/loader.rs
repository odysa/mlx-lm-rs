use serde::Deserialize;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

use hf_hub::api::sync::Api;

use crate::error::{Error, Result};

#[derive(Debug, Deserialize)]
struct WeightIndex {
    weight_map: std::collections::HashMap<String, String>,
}

/// Iterate the safetensors shard files referenced by `model.safetensors.index.json`.
/// Falls back to a single `model.safetensors` if the index is absent.
pub fn list_weight_files(model_dir: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
    let dir = model_dir.as_ref();
    match std::fs::read_to_string(dir.join("model.safetensors.index.json")) {
        Ok(s) => {
            let idx: WeightIndex = serde_json::from_str(&s)?;
            let mut shards: HashSet<&String> = idx.weight_map.values().collect();
            let mut paths: Vec<PathBuf> = shards.drain().map(|n| dir.join(n)).collect();
            paths.sort();
            Ok(paths)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let single = dir.join("model.safetensors");
            std::fs::metadata(&single).map_err(|_| {
                Error::Config(format!("no safetensors found under {}", dir.display()))
            })?;
            Ok(vec![single])
        }
        Err(e) => Err(e.into()),
    }
}

/// Resolve `--model` to a local directory: existing path, otherwise `org/repo`
/// downloaded via hf-hub.
pub fn resolve_model_dir(spec: &str) -> Result<PathBuf> {
    let p = Path::new(spec);
    if p.is_dir() {
        return Ok(p.to_path_buf());
    }
    if !spec.contains('/') {
        return Err(Error::Config(format!(
            "model must be a local dir or 'org/repo', got: {spec}"
        )));
    }
    download_repo(spec)
}

fn download_repo(repo_id: &str) -> Result<PathBuf> {
    let api = Api::new().map_err(|e| Error::HfHub(e.to_string()))?;
    let repo = api.model(repo_id.to_string());

    for f in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
        repo.get(f).map_err(|e| Error::HfHub(e.to_string()))?;
    }

    let snapshot_dir = match repo.get("model.safetensors.index.json") {
        Ok(idx_path) => {
            let s = std::fs::read_to_string(&idx_path)?;
            let idx: WeightIndex = serde_json::from_str(&s)?;
            for shard in idx.weight_map.values().collect::<HashSet<_>>() {
                repo.get(shard).map_err(|e| Error::HfHub(e.to_string()))?;
            }
            idx_path
        }
        Err(_) => repo
            .get("model.safetensors")
            .map_err(|e| Error::HfHub(e.to_string()))?,
    };

    snapshot_dir
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| Error::HfHub("hf cache path has no parent".into()))
}
