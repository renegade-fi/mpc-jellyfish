//! Defines commonly used gadgets directly on the circuit type
//!
//! These gadgets are largely copied from the single-prover implementation

pub mod arithmetic;

/// A shorthand notation for wrapping a `CurveGroup`'s scalar field in
/// an MPC fabric `Scalar`
macro_rules! scalar {
    ($x:expr) => {
        Scalar::new($x)
    };
}
pub(crate) use scalar;
