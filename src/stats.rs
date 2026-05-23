/// Peak GPU/unified memory used by MLX since the last reset, in bytes.
pub fn peak_memory_bytes() -> usize {
    let mut out: usize = 0;
    // SAFETY: out-pointer is valid; the function only writes a usize.
    unsafe {
        mlx_sys::mlx_get_peak_memory(&mut out as *mut usize);
    }
    out
}

pub fn reset_peak_memory() {
    // SAFETY: trivial — no inputs, no allocation, no aliasing.
    unsafe {
        mlx_sys::mlx_reset_peak_memory();
    }
}

/// Release MLX's allocator-side memory cache back to the OS.
///
/// MLX retains freed allocations in a per-process pool for reuse. Calling
/// this between phases (e.g. after each prefill chunk, or every N decode
/// tokens during long generations) keeps steady-state memory bounded — the
/// upstream Python `mlx_lm` does this for the same reason.
pub fn clear_cache() {
    // SAFETY: no inputs, no allocation aliasing concerns.
    unsafe {
        mlx_sys::mlx_clear_cache();
    }
}

/// `Prompt: N tokens, T tok/s` style line, matching python mlx-lm output.
pub fn fmt_throughput(label: &str, tokens: usize, secs: f64) -> String {
    let tps = if secs > 0.0 {
        tokens as f64 / secs
    } else {
        0.0
    };
    format!("{label}: {tokens} tokens, {tps:.3} tokens-per-sec")
}

pub fn fmt_peak_memory(bytes: usize) -> String {
    let gb = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    format!("Peak memory: {gb:.3} GB")
}
