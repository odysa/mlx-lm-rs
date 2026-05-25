# mlx-lm-rs

Greenfield Rust port of [Apple's `mlx-lm`](https://github.com/ml-explore/mlx-lm),
scoped to **Qwen3 dense inference** on Apple Silicon via [`mlx-rs`](https://github.com/oxideai/mlx-rs).

Token-for-token parity with Python `mlx_lm.generate` at temp=0 (covered by
`tests/parity_qwen3.rs`).

## Status

Early. Single architecture supported (Qwen3 dense, bf16). Not a drop-in
replacement for `mlx-lm` — explicitly out of scope right now:

- Quantized weights (4-bit / 8-bit)
- Sliding-window / `RotatingKVCache` (needed for Gemma2/Mistral)
- Samplers beyond greedy + temperature (top-k, top-p, min-p, repetition penalty)
- Speculative decoding, batched generation
- Full OpenAI API parity beyond basic chat completions
- LoRA / fine-tuning, GGUF, AWQ/GPTQ, distributed inference
- Tool calling / structured output
- Any model architecture other than Qwen3 dense

## Requirements

- macOS 14+ on Apple Silicon
- Rust 1.82+
- Full Xcode + the Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`)
  — the Command Line Tools alone don't include the `metal` shader compiler that
  MLX's cmake build needs.

## Build

```sh
cargo build --release
```

First build compiles MLX C++ via cmake (~5–15 min). Subsequent builds are fast.

## Use

```sh
# Defaults to mlx-community/Qwen3-4B-bf16; pass --model for anything else
cargo run --release -- generate \
    --model mlx-community/Qwen3-0.6B-bf16 \
    --prompt "Write a haiku about Rust." \
    --max-tokens 128
```

A local path also works: `--model /path/to/Qwen3-0.6B-bf16-snapshot`.

Useful flags:
- `--temp <f32>` — sampling temperature (`0.0` = greedy)
- `--max-tokens <usize>` — generation cap (default 256)
- `--no-chat-template` — skip Jinja chat template, use raw prompt
- `--no-stream` — don't flush stdout per token (cheaper for non-interactive use)
- `--no-stats` — suppress trailing prompt/gen tps + peak memory lines

## OpenAI-compatible server

```sh
cargo run --release -- serve \
    --model mlx-community/Qwen3-0.6B-bf16 \
    --host 127.0.0.1 \
    --port 8000
```

Supported endpoints:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

Basic chat completion:

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Qwen3-0.6B-bf16",
    "messages": [{"role": "user", "content": "Write a haiku about Rust."}],
    "max_tokens": 128,
    "temperature": 0
  }'
```

Streaming uses server-sent events:

```sh
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Qwen3-0.6B-bf16",
    "messages": [{"role": "user", "content": "Write a haiku about Rust."}],
    "max_tokens": 128,
    "stream": true
  }'
```

The server handles one generation at a time. It supports text messages,
`max_tokens` / `max_completion_tokens`, `temperature`, `stream`, and `n=1`.
Requests above the server `--max-tokens` cap are rejected. Tool calling,
JSON/schema-constrained output, stop sequences, and multimodal message parts are
rejected explicitly for now.

## Bench

```sh
cargo bench --bench benchmark                                   # full sweep
cargo bench --bench benchmark -- decode/short                   # one bench
BENCH_MODEL=mlx-community/Qwen3-4B-bf16 cargo bench --bench benchmark
```

In-process [criterion](https://github.com/bheisler/criterion.rs) bench: model is
loaded once via `OnceLock`, each iteration calls `Generator` directly. Reports
tokens/sec with 95% confidence intervals.

HTML report at `target/criterion/report/index.html` after a run.

## Parity test vs Python `mlx-lm`

Greedy decoding should produce bit-identical output to upstream. The test is
`#[ignore]` by default — it requires a local Python venv with `mlx-lm` and a
cached model.

```sh
python3 -m venv .venv-parity
.venv-parity/bin/pip install mlx-lm
PATH=$PWD/.venv-parity/bin:$PATH cargo test --release \
    --test parity_qwen3 -- --ignored
```

## How it's put together

| File | Role |
|---|---|
| `src/main.rs` | clap CLI; orchestrates download → load → generate → stream |
| `src/config.rs` | `Qwen3Config` (serde) |
| `src/loader.rs` | safetensors shard listing + hf-hub download |
| `src/tokenizer.rs` | thin wrapper around the HF `tokenizers` crate |
| `src/chat_template.rs` | Jinja chat template (minijinja + pycompat) |
| `src/cache.rs` | chunked KV cache (256-token blocks, in-place writes) |
| `src/sample.rs` | greedy + temperature sampler |
| `src/generate.rs` | `Generator` iterator with async_eval lookahead |
| `src/models/qwen3.rs` | Attention (q_norm/k_norm/RoPE/GQA), SwiGLU MLP, transformer block, lm_head |
| `src/stats.rs` | mlx-sys wrappers for peak memory + memory cache clear |
| `benches/benchmark.rs` | in-process criterion bench |
| `tests/parity_qwen3.rs` | greedy parity test vs Python `mlx-lm` |

## License

Apache-2.0. See [LICENSE](LICENSE).

This is an independent Rust implementation; not affiliated with Apple's
`mlx-lm`, `oxideai/mlx-rs`, or HuggingFace.
