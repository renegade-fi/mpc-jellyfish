//! Proof linking primitives for multiprover Plonk proofs
//!
//! See the `proof_linking` module in the `relation` crate for more info

use core::{
    pin::Pin,
    task::{Context, Poll},
};

use ark_ec::{
    pairing::Pairing,
    short_weierstrass::{Affine, SWCurveConfig},
};
use ark_ff::{Field, One};
use ark_mpc::{
    algebra::{AuthenticatedDensePoly, Scalar, ScalarResult},
    MpcFabric,
};
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial};
use futures::{ready, Future, FutureExt};
use itertools::Itertools;
use mpc_relation::proof_linking::GroupLayout;

use crate::{
    errors::PlonkError,
    multiprover::primitives::{
        MpcTranscript, MultiproverKZG, MultiproverKzgCommitment, MultiproverKzgCommitmentOpening,
        MultiproverKzgProof, MultiproverKzgProofOpening,
    },
    proof_system::{proof_linking::LinkingProof, structs::CommitKey},
};

use super::{MpcLinkingHint, MultiproverPlonkKzgSnark};

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
    pub fn link_singleprover_proof(
        lhs_link_hint: &MpcLinkingHint<E>,
        rhs_link_hint: &MpcLinkingHint<E>,
        group_layout: &GroupLayout,
        commit_key: &CommitKey<E>,
        fabric: &MpcFabric<E::G1>,
    ) -> Result<MultiproverLinkingProof<E>, PlonkError> {
        let a1 = &lhs_link_hint.linking_wire_poly;
        let a2 = &rhs_link_hint.linking_wire_poly;

        // Compute the quotient polynomial
        let quotient = Self::compute_linking_quotient(a1, a2, group_layout);
        let quotient_commitment =
            MultiproverKZG::commit(commit_key, &quotient).map_err(PlonkError::PCSError)?;

        // Squeeze a challenge for the quotient opening proof
        let a1_comm = &lhs_link_hint.linking_wire_comm;
        let a2_comm = &rhs_link_hint.linking_wire_comm;
        let eta = Self::compute_quotient_challenge(a1_comm, a2_comm, &quotient_commitment, fabric);
        let opening_proof =
            Self::compute_identity_opening(a1, a2, &quotient, eta, group_layout, commit_key)?;

        Ok(MultiproverLinkingProof { quotient_commitment, opening_proof })
    }

    /// Compute the quotient polynomial for the linking proof
    ///
    /// Let the LHS proof's a(x) wiring polynomial be a_1(x) and the RHS proof's
    /// a(x) wiring polynomial be a_2(x). Then the quotient polynomial is:
    ///     q(x) = (a_1(x) - a_2(x)) / Z_D(x)
    /// Where Z_D(x) is the vanishing polynomial for the linking domain D
    fn compute_linking_quotient(
        a1: &AuthenticatedDensePoly<E::G1>,
        a2: &AuthenticatedDensePoly<E::G1>,
        group_layout: &GroupLayout,
    ) -> AuthenticatedDensePoly<E::G1> {
        // Divide the difference by the vanishing polynomial
        let vanishing_poly = Self::compute_vanishing_polynomial(group_layout);
        let diff = a1 - a2;

        &diff / &vanishing_poly
    }

    /// Compute the vanishing polynomial for the layout of a given group
    fn compute_vanishing_polynomial(layout: &GroupLayout) -> DensePolynomial<E::ScalarField> {
        // Get the generator of the group
        let gen = layout.get_domain_generator::<E::ScalarField>();

        // Compute the vanishing poly:
        //    Z(x) = (x - g^{offset})(x - g^{offset})...(x - g^{offset + size - 1})
        let mut curr_root = gen.pow([layout.offset as u64]);
        let mut vanishing_poly =
            DensePolynomial::from_coefficients_vec(vec![E::ScalarField::one()]);

        for _ in 0..layout.size {
            let monomial =
                DensePolynomial::from_coefficients_vec(vec![-curr_root, E::ScalarField::one()]);
            vanishing_poly = &vanishing_poly * &monomial;

            curr_root *= &gen;
        }

        vanishing_poly
    }

    /// Compute the evaluation of the domain zero polynomial at the challenge
    /// point
    fn compute_vanishing_poly_eval(
        challenge: &ScalarResult<E::G1>,
        layout: &GroupLayout,
    ) -> ScalarResult<E::G1> {
        // Use batch ops to more efficiently compute the monomials
        let base = layout.get_domain_generator::<E::ScalarField>();
        let layout_roots = (layout.offset..layout.offset + layout.size)
            .map(|i| base.pow([i as u64]))
            .map(Scalar::new)
            .collect_vec();
        let monomials =
            ScalarResult::batch_sub_constant(&vec![challenge.clone(); layout.size], &layout_roots);

        monomials.into_iter().product()
    }

    /// Squeeze a challenge for the quotient opening proof
    ///
    /// This challenge is squeezed from a transcript that absorbs the wire
    /// polynomials that encode proof linking gates
    /// challenge of each proof, thereby branching off the transcripts of
    /// the proofs _after_ they have committed to the wiring polynomials
    /// that are being linked
    fn compute_quotient_challenge(
        a1_comm: &MultiproverKzgCommitment<E>,
        a2_comm: &MultiproverKzgCommitment<E>,
        quotient_comm: &MultiproverKzgCommitment<E>,
        fabric: &MpcFabric<E::G1>,
    ) -> ScalarResult<E::G1> {
        let mut transcript = MpcTranscript::new(b"MpcPlonkLinkingProof", fabric.clone());

        // We encode the proof linking gates in the first wire polynomial
        let a1 = a1_comm.open_authenticated();
        let a2 = a2_comm.open_authenticated();
        transcript.append_commitments(b"linking_wire_comms", &[a1, a2]);
        transcript.append_commitment(b"quotient_comm", &quotient_comm.open_authenticated());

        transcript.get_and_append_challenge(b"eta")
    }

    /// Compute an opening to the polynomial that represents the identity
    /// checked by the protocol
    ///
    /// Concretely this polynomial is:
    ///     a_1(x) - a_2(x) - q(x) * Z_D(\eta)
    fn compute_identity_opening(
        a1: &AuthenticatedDensePoly<E::G1>,
        a2: &AuthenticatedDensePoly<E::G1>,
        quotient_poly: &AuthenticatedDensePoly<E::G1>,
        challenge: ScalarResult<E::G1>,
        layout: &GroupLayout,
        commit_key: &CommitKey<E>,
    ) -> Result<MultiproverKzgProof<E>, PlonkError> {
        // Compute the identity polynomial
        let a1_minus_a2 = a1 - a2;
        let vanishing_eval = Self::compute_vanishing_poly_eval(&challenge, layout);
        let identity_poly = &a1_minus_a2 - &(quotient_poly * vanishing_eval);

        // Compute the opening
        let (opening_proof, _) = MultiproverKZG::open(commit_key, &identity_poly, &challenge)
            .map_err(PlonkError::PCSError)?;
        Ok(opening_proof)
    }
}
