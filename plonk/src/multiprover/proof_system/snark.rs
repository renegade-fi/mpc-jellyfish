//! Defines the multiprover analog to the `PlonkKzgSnark` defined in the
//! `proof_system` module of this crate
//!
//! The implementation is designed to closely match the singleprover
//! implementation in structure

use core::marker::PhantomData;

use ark_ec::pairing::Pairing;

/// A multiprover Plonk instantiated with KZG as the underlying polynomial
/// commitment scheme
#[derive(Default)]
pub struct MultiproverPlonkKzgSnark<E: Pairing>(PhantomData<E>);

impl<E: Pairing> MultiproverPlonkKzgSnark<E> {
    /// Constructor
    pub fn new() -> Self {
        Self(PhantomData)
    }
}
