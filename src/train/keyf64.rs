//! This is an utility mod to enable using Floats in HashMaps.
//!
//! Default float does not implement Hash trait due to its nature.
//! This struct transform the float to its components that are
//! individually hashable. The transformation in both directions should be lossless.
//!
//! Example:
//! ```
//! fn main() {
//!     let x: f64 = 123.123456789;
//!     let x_key: KeyF64 = x.into();
//!     let y: f64 = x_key.into();
//!     assert_eq!(x, y);
//! }
//! ```
use std::fmt;

#[derive(Hash, PartialEq, Eq)]
pub struct KeyF64 {
    sign: i8,
    exponent: i16,
    mantissa: u64,
}

impl fmt::Debug for KeyF64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let num: f64 = self.into();
        f.debug_struct("KeyF64").field("f64", &num).finish()
    }
}

impl From<f64> for KeyF64 {
    fn from(value: f64) -> Self {
        const STORED_MANTISSA_DIGITS: u32 = f64::MANTISSA_DIGITS - 1;
        const STORED_MANTISSA_MASK: u64 = (1 << STORED_MANTISSA_DIGITS) - 1;
        const MANTISSA_MSB: u64 = 1 << STORED_MANTISSA_DIGITS;

        const EXPONENT_BITS: u32 = 64 - 1 - STORED_MANTISSA_DIGITS;
        const EXPONENT_MASK: u32 = (1 << EXPONENT_BITS) - 1;

        let bits = value.to_bits();
        let sign: i8 = if bits >> 63 == 0 { 1 } else { -1 };
        let mut exponent = ((bits >> STORED_MANTISSA_DIGITS) as u32 & EXPONENT_MASK) as i16;

        let mantissa = if exponent == 0 {
            (bits & STORED_MANTISSA_MASK) << 1
        } else {
            (bits & STORED_MANTISSA_MASK) | MANTISSA_MSB
        };
        exponent -= 1023 + 52;
        Self {
            sign,
            exponent,
            mantissa,
        }
    }
}

impl From<KeyF64> for f64 {
    fn from(value: KeyF64) -> f64 {
        (value.sign as f64) * (value.mantissa as f64) * (2f64.powf(value.exponent as f64))
    }
}

impl From<&KeyF64> for f64 {
    fn from(value: &KeyF64) -> f64 {
        (value.sign as f64) * (value.mantissa as f64) * (2f64.powf(value.exponent as f64))
    }
}

#[cfg(test)]
mod test {
    use super::KeyF64;
    use rand::Rng;
    #[test]
    fn basic() {
        let mut rng = rand::thread_rng();
        for _ in 0..10000 {
            let x: f64 = rng.gen();
            let xkey: KeyF64 = x.into();
            let y: f64 = xkey.into();
            assert_eq!(x, y);
        }
    }
}
