use std::path::Path;
use tokenizers::Tokenizer;

use crate::error::Result;

pub fn load_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer> {
    let path = model_dir.as_ref().join("tokenizer.json");
    Ok(Tokenizer::from_file(&path)?)
}
