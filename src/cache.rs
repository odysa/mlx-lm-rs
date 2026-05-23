use mlx_rs::{
    ops::{
        concatenate_axis,
        indexing::{Ellipsis, IndexOp, TryIndexMutOp},
        zeros_dtype,
    },
    Array,
};

use crate::error::Result;

/// Chunk size: K/V are pre-allocated in `STEP`-token blocks so that decode
/// steps within a block are O(1) in-place writes instead of O(N) concats.
const STEP: i32 = 256;

/// Per-layer KV cache, ported from upstream Python `mlx_lm.models.cache.KVCache`.
///
/// Layout: `[B, H, S, D]`. `offset` is the active sequence length; the
/// underlying buffer is rounded up to the next `STEP`-token boundary. When
/// the active region exceeds the buffer, we extend by `ceil(n_new / STEP)`
/// chunks (the existing buffer is `concatenate`d with a fresh zeros block).
#[derive(Debug, Default)]
pub struct KvCache {
    keys: Option<Array>,
    values: Option<Array>,
    offset: i32,
}

impl KvCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn offset(&self) -> i32 {
        self.offset
    }

    /// Active K/V slice views (shape `[B, H, offset, D]`) — i.e. the meaningful
    /// region of the buffer, with any unused padding past `offset` excluded.
    /// Returns `None` before the first update.
    pub fn active(&self) -> Option<(Array, Array)> {
        let k = self.keys.as_ref()?;
        let v = self.values.as_ref()?;
        Some((
            k.index((Ellipsis, 0..self.offset, ..)),
            v.index((Ellipsis, 0..self.offset, ..)),
        ))
    }

    pub fn update_and_fetch(&mut self, k: Array, v: Array) -> Result<(Array, Array)> {
        let kshape = k.shape();
        let s_axis = kshape.len() - 2;
        let n_new = kshape[s_axis];
        let prev = self.offset;
        let new_offset = prev + n_new;

        let cur_cap = self.keys.as_ref().map(|a| a.shape()[s_axis]).unwrap_or(0);
        if new_offset > cur_cap {
            self.grow(&k, &v, prev)?;
        }

        let keys = self.keys.as_mut().expect("keys allocated by grow");
        let values = self.values.as_mut().expect("values allocated by grow");
        keys.try_index_mut((Ellipsis, prev..new_offset, ..), &k)?;
        values.try_index_mut((Ellipsis, prev..new_offset, ..), &v)?;

        // Commit only after both writes succeed — failure leaves `offset` unchanged.
        self.offset = new_offset;

        Ok((
            keys.index((Ellipsis, 0..new_offset, ..)),
            values.index((Ellipsis, 0..new_offset, ..)),
        ))
    }

    /// Build the new buffers fully before mutating `self`, so a partial
    /// failure cannot leave keys/values out of sync.
    fn grow(&mut self, k: &Array, v: &Array, prev: i32) -> Result<()> {
        let kshape = k.shape();
        let vshape = v.shape();
        let s_axis = kshape.len() - 2;
        let n_new = kshape[s_axis];
        let extra = ((n_new + STEP - 1) / STEP) * STEP;

        let mut pad_kshape = kshape.to_vec();
        pad_kshape[s_axis] = extra;
        let mut pad_vshape = vshape.to_vec();
        pad_vshape[s_axis] = extra;
        let pad_k = zeros_dtype(&pad_kshape, k.dtype())?;
        let pad_v = zeros_dtype(&pad_vshape, v.dtype())?;

        // If `prev` doesn't sit on a STEP boundary, the existing buffer has
        // unused tail past the active region — drop it before extending.
        let trim_tail = prev % STEP != 0;
        let new_keys = extend(self.keys.as_ref(), pad_k, prev, trim_tail)?;
        let new_values = extend(self.values.as_ref(), pad_v, prev, trim_tail)?;

        self.keys = Some(new_keys);
        self.values = Some(new_values);
        Ok(())
    }
}

fn extend(existing: Option<&Array>, pad: Array, prev: i32, trim_tail: bool) -> Result<Array> {
    match existing {
        None => Ok(pad),
        Some(buf) => {
            let head = if trim_tail {
                buf.index((Ellipsis, 0..prev, ..))
            } else {
                buf.clone()
            };
            Ok(concatenate_axis(&[head, pad], -2)?)
        }
    }
}
