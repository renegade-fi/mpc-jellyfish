//! Defines commonly used gadgets directly on the circuit type
//!
//! These gadgets are largely copied from the single-prover implementation
use core::cmp::Ordering;

use mpc_relation::errors::CircuitError;

pub mod arithmetic;

// Helper function to find the next multiple of `divisor` for `current` value
pub(crate) fn next_multiple(current: usize, divisor: usize) -> Result<usize, CircuitError> {
    if divisor == 0 || divisor == 1 {
        return Err(CircuitError::InternalError(
            "can only be a multiple of divisor >= 2".to_string(),
        ));
    }

    match current.cmp(&divisor) {
        Ordering::Equal => Ok(current),
        Ordering::Less => Ok(divisor),
        Ordering::Greater => Ok((current / divisor + 1) * divisor),
    }
}

/// A shorthand notation for wrapping a `CurveGroup`'s scalar field in
/// an MPC fabric `Scalar`
macro_rules! scalar {
    ($x:expr) => {
        Scalar::new($x)
    };
}
pub(self) use scalar;
