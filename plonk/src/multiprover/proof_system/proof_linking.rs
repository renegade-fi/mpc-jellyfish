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
    pub fn link_proofs(
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

#[cfg(test)]
mod test {
    use ark_bn254::{Bn254, G1Projective as Curve};
    use ark_ec::pairing::Pairing;
    use ark_ff::Zero;
    use ark_mpc::{
        algebra::{AuthenticatedScalarResult, Scalar},
        test_helpers::execute_mock_mpc,
        MpcFabric, PARTY0,
    };
    use itertools::Itertools;
    use mpc_relation::{
        proof_linking::{GroupLayout, LinkableCircuit},
        PlonkCircuit,
    };
    use rand::{thread_rng, Rng};

    use crate::{
        errors::PlonkError,
        multiprover::proof_system::{
            CollaborativeProof, MpcLinkingHint, MpcPlonkCircuit, MultiproverPlonkKzgSnark,
        },
        proof_system::{
            proof_linking::test_helpers::{
                gen_commit_keys, gen_proving_keys, gen_test_circuit1, gen_test_circuit2,
                CircuitSelector, GROUP_NAME,
            },
            structs::ProvingKey,
            PlonkKzgSnark,
        },
        transcript::SolidityTranscript,
    };

    /// The number of linked witness elements to use in the tests
    const WITNESS_ELEMS: usize = 10;
    /// The test field used
    type TestField = <Bn254 as Pairing>::ScalarField;

    // -----------
    // | Helpers |
    // -----------

    /// Generate a test case proof, group layout, and link hint from the given
    /// circuit
    fn gen_circuit_proof_and_hint(
        witness: &[AuthenticatedScalarResult<Curve>],
        circuit: CircuitSelector,
        layout: Option<GroupLayout>,
        fabric: &MpcFabric<Curve>,
    ) -> (CollaborativeProof<Bn254>, MpcLinkingHint<Bn254>, GroupLayout) {
        let mut cs = MpcPlonkCircuit::new(fabric.clone());
        match circuit {
            CircuitSelector::Circuit1 => gen_test_circuit1(&mut cs, witness, layout),
            CircuitSelector::Circuit2 => gen_test_circuit2(&mut cs, witness, layout),
        };
        cs.finalize_for_arithmetization().unwrap();

        // Generate a proving key
        let pk = gen_pk_from_singleprover(circuit, layout);

        // Get the layout
        let group_layout = cs.get_link_group_layout(GROUP_NAME).unwrap();

        // Generate a proof with a linking hint
        let (proof, hint) = gen_test_proof(&cs, &pk, fabric);
        (proof, hint, group_layout)
    }

    /// Get a proving key by constructing a singleprover circuit of the same
    /// topology
    fn gen_pk_from_singleprover(
        circuit_selector: CircuitSelector,
        layout: Option<GroupLayout>,
    ) -> ProvingKey<Bn254> {
        let mut cs = PlonkCircuit::new_turbo_plonk();
        let dummy_witness = (0..WITNESS_ELEMS).map(|_| TestField::zero()).collect_vec();
        match circuit_selector {
            CircuitSelector::Circuit1 => gen_test_circuit1(&mut cs, &dummy_witness, layout),
            CircuitSelector::Circuit2 => gen_test_circuit2(&mut cs, &dummy_witness, layout),
        };
        cs.finalize_for_arithmetization().unwrap();

        let (pk, _) = gen_proving_keys(&cs);
        pk
    }

    /// Generate a proof and link hint for the circuit by proving its r1cs
    /// relation
    fn gen_test_proof(
        circuit: &MpcPlonkCircuit<Curve>,
        pk: &ProvingKey<Bn254>,
        fabric: &MpcFabric<Curve>,
    ) -> (CollaborativeProof<Bn254>, MpcLinkingHint<Bn254>) {
        MultiproverPlonkKzgSnark::<Bn254>::prove_with_link_hint(circuit, pk, fabric.clone())
            .unwrap()
    }

    /// Prove a link between two circuits and verify the link, return the result
    /// as a result
    async fn prove_and_verify_link(
        lhs_hint: &MpcLinkingHint<Bn254>,
        rhs_hint: &MpcLinkingHint<Bn254>,
        lhs_proof: &CollaborativeProof<Bn254>,
        rhs_proof: &CollaborativeProof<Bn254>,
        layout: &GroupLayout,
        fabric: &MpcFabric<Curve>,
    ) -> Result<(), PlonkError> {
        let (commit_key, open_key) = gen_commit_keys();
        let proof = MultiproverPlonkKzgSnark::<Bn254>::link_proofs(
            lhs_hint,
            rhs_hint,
            layout,
            &commit_key,
            fabric,
        )?;

        let opened_link = proof.open_authenticated().await?;
        let lhs_proof = lhs_proof.clone().open_authenticated().await?;
        let rhs_proof = rhs_proof.clone().open_authenticated().await?;

        PlonkKzgSnark::<Bn254>::verify_link_proof::<SolidityTranscript>(
            &lhs_proof,
            &rhs_proof,
            &opened_link,
            layout,
            &open_key,
        )
    }

    // --------------
    // | Test Cases |
    // --------------

    /// Test the basic case of a valid link on two circuits
    #[tokio::test]
    async fn test_valid_link() {
        let mut rng = thread_rng();
        let witness = (0..WITNESS_ELEMS).map(|_| Scalar::random(&mut rng)).collect_vec();

        // Generate a proof and link in an MPC
        let (res, _) = execute_mock_mpc(move |fabric| {
            let witness = witness.clone();
            async move {
                let witness = fabric.batch_share_scalar(witness, PARTY0);

                // Generate r1cs proofs for the two circuits
                let (lhs_proof, lhs_hint, layout) =
                    gen_circuit_proof_and_hint(&witness, CircuitSelector::Circuit1, None, &fabric);
                let (rhs_proof, rhs_hint, _) = gen_circuit_proof_and_hint(
                    &witness,
                    CircuitSelector::Circuit2,
                    Some(layout),
                    &fabric,
                );

                // Prove and verify the link
                prove_and_verify_link(
                    &lhs_hint, &rhs_hint, &lhs_proof, &rhs_proof, &layout, &fabric,
                )
                .await
            }
        })
        .await;

        assert!(res.is_ok());
    }

    /// Tests a valid link with a layout specified up front
    #[tokio::test]
    #[allow(non_snake_case)]
    async fn test_valid_link__specific_layout() {
        let mut rng = thread_rng();
        let witness = (0..WITNESS_ELEMS).map(|_| Scalar::random(&mut rng)).collect_vec();

        // Generate a proof and link in an MPC
        let (res, _) = execute_mock_mpc(move |fabric| {
            let witness = witness.clone();
            async move {
                let witness = fabric.batch_share_scalar(witness, PARTY0);

                // Generate r1cs proofs for the two circuits
                let layout = GroupLayout { offset: 20, size: WITNESS_ELEMS, alignment: 8 };
                let (lhs_proof, lhs_hint, layout) = gen_circuit_proof_and_hint(
                    &witness,
                    CircuitSelector::Circuit1,
                    Some(layout),
                    &fabric,
                );
                let (rhs_proof, rhs_hint, _) = gen_circuit_proof_and_hint(
                    &witness,
                    CircuitSelector::Circuit2,
                    Some(layout),
                    &fabric,
                );

                // Prove and verify the link
                prove_and_verify_link(
                    &lhs_hint, &rhs_hint, &lhs_proof, &rhs_proof, &layout, &fabric,
                )
                .await
            }
        })
        .await;

        assert!(res.is_ok());
    }

    /// Tests an invalid proof link wherein the witnesses used are different
    #[tokio::test]
    #[allow(non_snake_case)]
    async fn test_invalid_proof_link__different_witnesses() {
        let mut rng = thread_rng();

        // Modify the second witness at a random location
        let witness1 = (0..WITNESS_ELEMS).map(|_| Scalar::random(&mut rng)).collect_vec();
        let mut witness2 = witness1.clone();
        let modification_idx = rng.gen_range(0..WITNESS_ELEMS);
        witness2[modification_idx] = Scalar::random(&mut rng);

        let (res, _) = execute_mock_mpc(move |fabric| {
            let witness1 = witness1.clone();
            let witness2 = witness2.clone();

            async move {
                let witness1 = fabric.batch_share_scalar(witness1, PARTY0);
                let witness2 = fabric.batch_share_scalar(witness2, PARTY0);

                // Generate r1cs proofs for the two circuits
                let (lhs_proof, lhs_hint, layout) =
                    gen_circuit_proof_and_hint(&witness1, CircuitSelector::Circuit1, None, &fabric);
                let (rhs_proof, rhs_hint, _) = gen_circuit_proof_and_hint(
                    &witness2,
                    CircuitSelector::Circuit2,
                    Some(layout),
                    &fabric,
                );

                // Prove and verify the link
                prove_and_verify_link(
                    &lhs_hint, &rhs_hint, &lhs_proof, &rhs_proof, &layout, &fabric,
                )
                .await
            }
        })
        .await;

        assert!(res.is_err());
    }

    /// Tests the case in which the correct witness is used to link but over
    /// incorrectly aligned domains
    #[tokio::test]
    #[allow(non_snake_case)]
    async fn test_invalid_proof_link__wrong_alignment() {
        // Use the same witness between two circuits
        let mut rng = thread_rng();
        let witness = (0..WITNESS_ELEMS).map(|_| Scalar::random(&mut rng)).collect_vec();

        let (res, _) = execute_mock_mpc(move |fabric| {
            let witness = witness.clone();
            async move {
                let witness = fabric.batch_share_scalar(witness, PARTY0);

                // Generate r1cs proofs for the two circuits
                let (lhs_proof, lhs_hint, mut layout) = gen_circuit_proof_and_hint(
                    &witness,
                    CircuitSelector::Circuit1,
                    None, // layout
                    &fabric,
                );

                // Modify the layout to be misaligned
                layout.alignment += 1;
                let (rhs_proof, rhs_hint, _) = gen_circuit_proof_and_hint(
                    &witness,
                    CircuitSelector::Circuit2,
                    Some(layout),
                    &fabric,
                );

                // Prove and verify the link
                prove_and_verify_link(
                    &lhs_hint, &rhs_hint, &lhs_proof, &rhs_proof, &layout, &fabric,
                )
                .await
            }
        })
        .await;

        assert!(res.is_err());
    }

    /// Tests the case in which the correct witness is used to link but over
    /// domains at different offsets
    #[tokio::test]
    #[allow(non_snake_case)]
    async fn test_invalid_proof_link__wrong_offset() {
        // Use the same witness between two circuits
        let mut rng = thread_rng();
        let witness = (0..WITNESS_ELEMS).map(|_| Scalar::random(&mut rng)).collect_vec();

        let (res, _) = execute_mock_mpc(move |fabric| {
            let witness = witness.clone();
            async move {
                let witness = fabric.batch_share_scalar(witness, PARTY0);

                // Generate r1cs proofs for the two circuits
                let (lhs_proof, lhs_hint, mut layout) = gen_circuit_proof_and_hint(
                    &witness,
                    CircuitSelector::Circuit1,
                    None, // layout
                    &fabric,
                );

                // Modify the layout to be misaligned
                layout.offset -= 1;
                let (rhs_proof, rhs_hint, _) = gen_circuit_proof_and_hint(
                    &witness,
                    CircuitSelector::Circuit2,
                    Some(layout),
                    &fabric,
                );

                // Prove and verify the link
                prove_and_verify_link(
                    &lhs_hint, &rhs_hint, &lhs_proof, &rhs_proof, &layout, &fabric,
                )
                .await
            }
        })
        .await;

        assert!(res.is_err());
    }
}
