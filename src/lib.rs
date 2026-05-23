pub mod cache;
pub mod chat_template;
pub mod config;
pub mod error;
pub mod generate;
pub mod loader;
pub mod models;
pub mod sample;
pub mod stats;
pub mod tokenizer;

pub use error::{Error, Result};

#[cfg(test)]
mod smoke {
    use mlx_rs::{ops, Array, Dtype};

    // Forces an MLX eval, which can abort the whole test process with a
    // foreign-exception "Rust cannot catch foreign exceptions" if MLX faults
    // for any reason (e.g. parallel test harness racing on Metal init). The
    // real coverage is `tests/parity_qwen3.rs`; keep this gated behind
    // `--ignored` so `cargo test` doesn't abort.
    #[test]
    #[ignore = "aborts via foreign exception under cargo test parallelism"]
    fn mlx_matmul_smoke() {
        let a = Array::ones::<f32>(&[2, 3]).unwrap();
        let b = Array::ones::<f32>(&[3, 4]).unwrap();
        let c = ops::matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 4]);
        assert_eq!(c.dtype(), Dtype::Float32);
        let v: &[f32] = c.as_slice();
        assert!(v.iter().all(|x| (*x - 3.0).abs() < 1e-6));
    }
}
