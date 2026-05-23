use std::num::NonZeroUsize;

use mlx_rs::{
    ops::indexing::IndexOp,
    transforms::{async_eval, eval},
    Array,
};

use crate::cache::KvCache;
use crate::error::{Error, Result};
use crate::models::qwen3::Model;
use crate::sample::sample;
use crate::stats::clear_cache;

/// Release MLX's memory cache every N decode steps (matches upstream Python).
const CLEAR_CACHE_EVERY: usize = 256;

pub struct Generator<'a> {
    model: &'a mut Model,
    cache: Vec<KvCache>,
    temp: f32,
    eos_ids: Vec<u32>,
    max_tokens: usize,
    produced: usize,
    /// Token whose computation has been kicked off via `async_eval`. Holding
    /// it as an unmaterialized `Array` (not a `u32`) lets the *next* decode
    /// step's host-graph build run while the GPU is still computing this one.
    pending: Option<Array>,
}

impl<'a> Generator<'a> {
    /// `prompt_ids` is shape `[T]` (single batch). Prefill runs in
    /// `prefill_step_size` chunks; the last prompt token feeds the first
    /// decode step (whose result we kick off async here).
    pub fn new(
        model: &'a mut Model,
        prompt_ids: &[u32],
        max_tokens: usize,
        temp: f32,
        eos_ids: Vec<u32>,
        prefill_step_size: NonZeroUsize,
    ) -> Result<Self> {
        if prompt_ids.is_empty() {
            return Err(Error::Config("empty prompt".into()));
        }
        let mut cache = model.make_cache();

        let split = prompt_ids.len() - 1;
        let step = prefill_step_size.get();
        let mut consumed = 0;
        let mut active_buf: Vec<Array> = Vec::with_capacity(2 * cache.len());
        while consumed < split {
            let end = (consumed + step).min(split);
            let chunk = &prompt_ids[consumed..end];
            let chunk_arr = Array::from_slice(chunk, &[1, chunk.len() as i32]);
            model.forward(&chunk_arr, &mut cache)?;
            // Single eval over the active K/V slices of every layer. Avoids
            // materializing the chunked-cache padding and avoids the second
            // sync point a separate values eval would add.
            active_buf.clear();
            for c in cache.iter() {
                if let Some((k, v)) = c.active() {
                    active_buf.push(k);
                    active_buf.push(v);
                }
            }
            eval(active_buf.iter())?;
            clear_cache();
            consumed = end;
        }

        // Seed the first decode step. Skip when max_tokens == 0 — building
        // and scheduling it would be wasted work.
        let pending = if max_tokens == 0 {
            None
        } else {
            let last = prompt_ids[prompt_ids.len() - 1];
            let input = Array::from_slice(&[last], &[1, 1]);
            let p = step_decode(model, &mut cache, &input, temp)?;
            async_eval(std::iter::once(&p))?;
            Some(p)
        };

        Ok(Self {
            model,
            cache,
            temp,
            eos_ids,
            max_tokens,
            produced: 0,
            pending,
        })
    }
}

/// One decode pass: model forward + sample. Returns the (unmaterialized)
/// sampled-token Array.
fn step_decode(
    model: &mut Model,
    cache: &mut [KvCache],
    input: &Array,
    temp: f32,
) -> Result<Array> {
    let logits = model.forward(input, cache)?;
    let last = logits.index((.., -1, ..));
    sample(&last, temp)
}

impl Iterator for Generator<'_> {
    type Item = Result<u32>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.produced >= self.max_tokens {
            return None;
        }
        let cur = self.pending.take()?;

        // While the GPU finishes computing `cur`, build the *next* step's
        // graph on the host using `cur` as an unmaterialized input. Then
        // kick off its async eval. By the time we call `cur.item()` below,
        // both steps are scheduled and the GPU has been busy for two ticks.
        let next_pending = if self.produced + 1 < self.max_tokens {
            // `cur` is shape [1] from argmax/categorical; the model wants [1, 1].
            let input = match cur.reshape(&[1, 1]) {
                Ok(x) => x,
                Err(e) => return Some(Err(e.into())),
            };
            let next = match step_decode(self.model, &mut self.cache, &input, self.temp) {
                Ok(t) => t,
                Err(e) => return Some(Err(e)),
            };
            if let Err(e) = async_eval(std::iter::once(&next)) {
                return Some(Err(e.into()));
            }
            Some(next)
        } else {
            None
        };

        // `.item()` blocks until materialization. Most of that wait already
        // happened during the next-step graph build above.
        let tok: u32 = cur.item();
        self.produced += 1;

        if self.produced.is_multiple_of(CLEAR_CACHE_EVERY) {
            clear_cache();
        }

        if self.eos_ids.contains(&tok) {
            return None;
        }
        self.pending = next_pending;
        Some(Ok(tok))
    }
}
