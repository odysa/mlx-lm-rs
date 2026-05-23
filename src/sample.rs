use mlx_rs::{ops::indexing::argmax_axis, random, Array};

use crate::error::{Error, Result};

/// Sample one token id per batch row from the final-step logits.
///
/// `logits` is shape `[B, V]` (already last-step). At temp == 0.0 we take
/// argmax (greedy). Otherwise we sample categorically from logits/T.
///
/// `temp` must be `0.0` or a finite positive value; negatives invert
/// preference and `NaN` is undefined.
pub fn sample(logits: &Array, temp: f32) -> Result<Array> {
    if temp == 0.0 {
        argmax_axis(logits, -1, None).map_err(Into::into)
    } else if temp.is_finite() && temp > 0.0 {
        let scaled = logits.divide(Array::from_f32(temp))?;
        random::categorical(scaled, -1, None, None).map_err(Into::into)
    } else {
        Err(Error::Config(format!(
            "temperature must be 0.0 or a finite positive value, got {temp}"
        )))
    }
}
