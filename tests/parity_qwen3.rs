//! Greedy parity check vs python `mlx_lm.generate` on the same Qwen3 model.
//!
//! This test is `#[ignore]` by default — it requires:
//!   * `pip install mlx-lm` available on PATH (`python3 -m mlx_lm.generate`)
//!   * `mlx-community/Qwen3-0.6B-bf16` already downloaded into the local
//!     HF cache (running our binary once with the same model triggers download)
//!
//! Run with:
//!   `cargo test --release --test parity_qwen3 -- --ignored --nocapture`

use std::process::Command;

const MODEL: &str = "mlx-community/Qwen3-0.6B-bf16";
const PROMPT: &str = "The capital of France is";
const MAX_TOKENS: usize = 16;

fn run_python_greedy() -> String {
    let out = Command::new("python3")
        .args([
            "-m",
            "mlx_lm",
            "generate",
            "--model",
            MODEL,
            "--prompt",
            PROMPT,
            "--max-tokens",
            &MAX_TOKENS.to_string(),
            "--temp",
            "0",
        ])
        .output()
        .expect("python3 -m mlx_lm.generate failed to launch (is `pip install mlx-lm` done?)");
    assert!(
        out.status.success(),
        "python mlx_lm exited non-zero: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn run_rust_greedy() -> String {
    let bin = env!("CARGO_BIN_EXE_mlx-lm-rs");
    let out = Command::new(bin)
        .args([
            "generate",
            "--model",
            MODEL,
            "--prompt",
            PROMPT,
            "--max-tokens",
            &MAX_TOKENS.to_string(),
            "--temp",
            "0",
        ])
        .output()
        .expect("our binary failed to launch");
    assert!(out.status.success(), "rust mlx-lm-rs exited non-zero");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Extract just the generated text. Python wraps it in `==========` lines and
/// then prints stats; ours streams to stdout, prefixed with `[mlx-lm-rs]` log
/// lines on stderr (so stdout is purely the generation).
fn extract_generated(stream: &str, is_python: bool) -> String {
    if is_python {
        let mut in_block = false;
        let mut out = String::new();
        for l in stream.lines() {
            if l.starts_with("==========") {
                if in_block {
                    break;
                }
                in_block = true;
                continue;
            }
            if in_block {
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(l);
            }
        }
        out.trim().to_string()
    } else {
        // Our binary writes log lines to stderr, generation to stdout.
        stream.trim().to_string()
    }
}

#[test]
#[ignore = "requires python mlx_lm and a downloaded Qwen3-0.6B-bf16"]
fn greedy_matches_python() {
    let py = run_python_greedy();
    let rs = run_rust_greedy();
    let py_n = extract_generated(&py, true);
    let rs_n = extract_generated(&rs, false);
    eprintln!("python: {py_n:?}");
    eprintln!("rust:   {rs_n:?}");
    assert_eq!(
        py_n, rs_n,
        "rust greedy output should match python mlx_lm.generate token-for-token"
    );
}
