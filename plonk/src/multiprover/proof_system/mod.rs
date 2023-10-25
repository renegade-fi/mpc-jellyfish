//! The `proof_system` module defines all the multi-prover components necessary
//! to construct an arithmetization and proofs of satisfaction for this
//! arithmetization

mod constraint_system;
mod prover;
mod snark;

pub use constraint_system::*;
pub(crate) use prover::*;
pub use snark::*;
