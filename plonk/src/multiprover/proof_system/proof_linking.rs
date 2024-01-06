//! Proof linking primitives for multiprover Plonk proofs
//!
//! We only implement the case in which a singleprover proof is linked into a
//! multiprover proof. Linking two multiprover proofs is left for future work

use core::{
    pin::Pin,
    task::{Context, Poll},
};

use ark_ec::{
    pairing::Pairing,
    short_weierstrass::{Affine, SWCurveConfig},
};
use futures::{ready, Future, FutureExt};

use crate::{
    errors::PlonkError,
    multiprover::primitives::{
        MultiproverKzgCommitment, MultiproverKzgCommitmentOpening, MultiproverKzgProof,
        MultiproverKzgProofOpening,
    },
    proof_system::proof_linking::LinkingProof,
};

use super::MultiproverPlonkKzgSnark;

// ---------------------------
// | Proof and Opening Types |
// ---------------------------

/// A multiprover proof that two circuits are linked on a given domain
#[derive(Clone)]
pub struct MultiproverLinkingProof<E: Pairing> {
    /// The commitment to the linking quotient polynomial
    pub quotient_commitment: MultiproverKzgCommitment<E>,
    /// The proof of opening of the linking polynomial identity
    pub opening_proof: MultiproverKzgProof<E>,
}

impl<E: Pairing> MultiproverLinkingProof<E> {
    /// Open a multiprover linking proof to a singleprover linking proof
    pub fn open_authenticated(&self) -> MultiproverLinkingProofOpening<E> {
        MultiproverLinkingProofOpening {
            quotient_commitment: self.quotient_commitment.open_authenticated(),
            opening_proof: self.opening_proof.open_authenticated(),
        }
    }
}

/// The result of opening a linking proof in an MPC
///
/// Wrapping the type in this way allows us to implement `Future` and resolve
/// this opening to a standard, single-prover linking proof
#[derive(Clone)]
pub struct MultiproverLinkingProofOpening<E: Pairing> {
    /// The commitment to the linking quotient polynomial
    pub quotient_commitment: MultiproverKzgCommitmentOpening<E>,
    /// The proof of opening of the linking polynomial identity
    pub opening_proof: MultiproverKzgProofOpening<E>,
}

impl<E: Pairing> Future for MultiproverLinkingProofOpening<E>
where
    E::ScalarField: Unpin,
{
    type Output = Result<LinkingProof<E>, PlonkError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let quotient_commitment = ready!(self.quotient_commitment.poll_unpin(cx));
        let opening_proof = ready!(self.opening_proof.poll_unpin(cx));

        match (quotient_commitment, opening_proof) {
            // Either field errors
            (Err(e), _) | (_, Err(e)) => Poll::Ready(Err(PlonkError::PCSError(e))),

            // Both were successfully opened
            (Ok(quotient_commitment), Ok(opening_proof)) => {
                Poll::Ready(Ok(LinkingProof { quotient_commitment, opening_proof }))
            },
        }
    }
}

// --------------------------------------
// | Collaborative Proof Linking Prover |
// --------------------------------------

impl<P: SWCurveConfig<BaseField = E::BaseField>, E: Pairing<G1Affine = Affine<P>>>
    MultiproverPlonkKzgSnark<E>
{
    /// Link a singleprover proof into a multiprover proof
    pub fn link_singleprover_proof() -> MultiproverLinkingProof<E> {
        todo!()
    }
}
