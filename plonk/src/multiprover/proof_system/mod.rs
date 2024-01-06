//! The `proof_system` module defines all the multi-prover components necessary
//! to construct an arithmetization and proofs of satisfaction for this
//! arithmetization

mod constraint_system;
pub mod proof_linking;
mod prover;
mod snark;
mod structs;

pub use constraint_system::*;
pub(crate) use prover::*;
pub use snark::*;
pub use structs::*;
