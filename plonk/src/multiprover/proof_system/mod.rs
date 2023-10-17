//! The `proof_system` module defines all the multi-prover components necessary
//! to construct an arithmetization and proofs of satisfaction for this
//! arithmetization

mod constraint_system;
mod error;
mod snark;

pub use constraint_system::*;
pub use error::*;
pub use snark::*;
