use minijinja::{context, Environment, Template, Value};
use serde::Deserialize;
use std::path::Path;

use crate::error::{Error, Result};

#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    chat_template: Option<String>,
    #[serde(default)]
    bos_token: Option<TokenSpec>,
    #[serde(default)]
    eos_token: Option<TokenSpec>,
    #[serde(default)]
    pad_token: Option<TokenSpec>,
    #[serde(default)]
    unk_token: Option<TokenSpec>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum TokenSpec {
    Str(String),
    Obj { content: String },
}

impl TokenSpec {
    fn into_string(self) -> String {
        match self {
            Self::Str(s) => s,
            Self::Obj { content } => content,
        }
    }
}

fn token_str(t: Option<TokenSpec>) -> String {
    t.map(TokenSpec::into_string).unwrap_or_default()
}

pub struct ChatTemplate {
    env: Environment<'static>,
    bos: String,
    eos: String,
    pad: String,
    unk: String,
}

impl ChatTemplate {
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Option<Self>> {
        let path = model_dir.as_ref().join("tokenizer_config.json");
        let s = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        let cfg: TokenizerConfig = serde_json::from_str(&s)?;
        let Some(template_src) = cfg.chat_template else {
            return Ok(None);
        };

        let mut env = Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_template_owned("chat", template_src)
            .map_err(Error::from)?;

        Ok(Some(Self {
            env,
            bos: token_str(cfg.bos_token),
            eos: token_str(cfg.eos_token),
            pad: token_str(cfg.pad_token),
            unk: token_str(cfg.unk_token),
        }))
    }

    pub fn render(&self, user_prompt: &str, add_generation_prompt: bool) -> Result<String> {
        self.render_messages(
            &[ChatTemplateMessage {
                role: "user".to_string(),
                content: user_prompt.to_string(),
            }],
            add_generation_prompt,
        )
    }

    pub fn render_messages(
        &self,
        messages: &[ChatTemplateMessage],
        add_generation_prompt: bool,
    ) -> Result<String> {
        let tmpl: Template<'_, '_> = self.env.get_template("chat")?;
        let messages: Vec<Value> = messages
            .iter()
            .map(|m| context! { role => m.role.as_str(), content => m.content.as_str() })
            .collect();
        Ok(tmpl.render(context! {
            messages => messages,
            add_generation_prompt => add_generation_prompt,
            bos_token => self.bos,
            eos_token => self.eos,
            pad_token => self.pad,
            unk_token => self.unk,
        })?)
    }
}

#[derive(Debug, Clone)]
pub struct ChatTemplateMessage {
    pub role: String,
    pub content: String,
}
