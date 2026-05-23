use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("json: {0}")]
    Json(#[from] serde_json::Error),

    #[error("mlx: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),

    #[error("mlx io: {0}")]
    MlxIo(#[from] mlx_rs::error::IoError),

    #[error("tokenizer: {0}")]
    Tokenizer(String),

    #[error("hf-hub: {0}")]
    HfHub(String),

    #[error("template: {0}")]
    Template(#[from] minijinja::Error),

    #[error("missing weight: {0}")]
    MissingWeight(String),

    #[error("config: {0}")]
    Config(String),
}

impl From<tokenizers::Error> for Error {
    fn from(e: tokenizers::Error) -> Self {
        Error::Tokenizer(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;
