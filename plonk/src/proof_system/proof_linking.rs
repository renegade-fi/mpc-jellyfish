//! Proof linking primitives for Plonk proofs
//!
//! TODO(@joey): detail the proof-linking protocol

use ark_ec::{
    pairing::Pairing,
    short_weierstrass::{Affine, SWCurveConfig},
};
use ark_ff::{Field, One, Zero};
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial};
use jf_primitives::{
    pcs::{
        prelude::{Commitment, UnivariateKzgPCS, UnivariateKzgProof},
        PolynomialCommitmentScheme,
    },
    rescue::RescueParameter,
};
use mpc_relation::{
    gadgets::ecc::SWToTEConParam,
    proof_linking::{GroupLayout, PROOF_LINK_WIRE_IDX},
};

use crate::{errors::PlonkError, transcript::PlonkTranscript};

use super::{
    structs::{LinkingHint, Proof, ProvingKey, VerifyingKey},
    PlonkKzgSnark,
};

/// A proof that two circuits are linked on a given domain
#[derive(Clone, Debug)]
pub struct LinkingProof<E: Pairing> {
    /// The commitment to the linking quotient polynomial
    pub quotient_commitment: Commitment<E>,
    /// The proof of opening of the linking identity polynomial
    pub opening_proof: UnivariateKzgProof<E>,
}

// ----------
// | Prover |
// ----------

impl<E, F, P> PlonkKzgSnark<E>
where
    E: Pairing<BaseField = F, G1Affine = Affine<P>>,
    F: RescueParameter + SWToTEConParam,
    P: SWCurveConfig<BaseField = F, ScalarField = E::ScalarField>,
{
    /// Link two proofs on a given domain
    pub fn link_proofs<T: PlonkTranscript<F>>(
        lhs_link_hint: &LinkingHint<E>,
        rhs_link_hint: &LinkingHint<E>,
        group_layout: &GroupLayout,
        proving_key: &ProvingKey<E>,
    ) -> Result<LinkingProof<E>, PlonkError> {
        // Compute the wiring polynomials that encode the proof-linked values
        let a1 = &lhs_link_hint.linking_wire_poly;
        let a2 = &rhs_link_hint.linking_wire_poly;

        // Compute the quotient then commit to it
        let quotient = Self::compute_linking_quotient(a1, a2, group_layout)?;
        let quotient_commitment = UnivariateKzgPCS::commit(&proving_key.commit_key, &quotient)
            .map_err(PlonkError::PCSError)?;

        // Squeeze a challenge for the opening
        let opening_challenge = Self::compute_quotient_challenge::<T>(
            &lhs_link_hint.linking_wire_comm,
            &rhs_link_hint.linking_wire_comm,
            &quotient_commitment,
        )?;
        let opening_proof = Self::compute_identity_opening(
            a1,
            a2,
            &quotient,
            opening_challenge,
            group_layout,
            proving_key,
        )?;

        Ok(LinkingProof::<E> { quotient_commitment, opening_proof })
    }

    /// Compute the quotient polynomial for the linking proof
    ///
    /// Let the LHS proof's a(x) wiring polynomial be a_1(x) and the RHS proof's
    /// a(x) wiring polynomial be a_2(x). Then the quotient polynomial is:
    ///     q(x) = (a_1(x) - a_2(x)) / Z_D(x)
    /// Where Z_D(x) is the vanishing polynomial for the linking domain D
    fn compute_linking_quotient(
        a1: &DensePolynomial<E::ScalarField>,
        a2: &DensePolynomial<E::ScalarField>,
        group_layout: &GroupLayout,
    ) -> Result<DensePolynomial<E::ScalarField>, PlonkError> {
        // This may occur for two instances of the same circuit
        if a1 == a2 {
            return Ok(DensePolynomial::from_coefficients_vec(vec![]));
        }

        // Divide the difference by the vanishing polynomial
        let vanishing_poly = Self::compute_vanishing_polynomial(group_layout)?;
        let diff = a1 - a2;

        Ok(&diff / &vanishing_poly)
    }

    /// Compute the vanishing polynomial for the layout of a given group
    fn compute_vanishing_polynomial(
        layout: &GroupLayout,
    ) -> Result<DensePolynomial<E::ScalarField>, PlonkError> {
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

        Ok(vanishing_poly)
    }

    /// Compute the evaluation of the domain zero polynomial at the challenge
    /// point
    fn compute_vanishing_poly_eval(
        challenge: E::ScalarField,
        layout: &GroupLayout,
    ) -> E::ScalarField {
        let base = layout.get_domain_generator::<E::ScalarField>();
        let mut curr_root = base.pow([layout.offset as u64]);

        let mut eval = E::ScalarField::one();
        for _ in 0..layout.size {
            eval *= &(challenge - curr_root);
            curr_root *= &base;
        }

        eval
    }

    /// Squeeze a challenge for the quotient opening proof
    ///
    /// This challenge is squeezed from a transcript that absorbs the wire
    /// polynomials that encode proof linking gates
    /// challenge of each proof, thereby branching off the transcripts of
    /// the proofs _after_ they have committed to the wiring polynomials
    /// that are being linked
    fn compute_quotient_challenge<T: PlonkTranscript<E::BaseField>>(
        a1_comm: &Commitment<E>,
        a2_comm: &Commitment<E>,
        quotient_comm: &Commitment<E>,
    ) -> Result<E::ScalarField, PlonkError> {
        let mut transcript = T::new(b"PlonkLinkingProof");

        // We encode the proof linking gates in the first wire polynomial
        transcript.append_commitments(b"linking_wire_comms", &[*a1_comm, *a2_comm])?;
        transcript.append_commitment(b"quotient_comm", quotient_comm)?;

        transcript.get_and_append_challenge::<E>(b"eta")
    }

    /// Compute an opening to the polynomial that represents the identity
    /// checked by the protocol
    ///
    /// Concretely this polynomial is:
    ///     a_1(x) - a_2(x) - q(x) * Z_D(\eta)
    fn compute_identity_opening(
        a1: &DensePolynomial<E::ScalarField>,
        a2: &DensePolynomial<E::ScalarField>,
        quotient_poly: &DensePolynomial<E::ScalarField>,
        challenge: E::ScalarField,
        layout: &GroupLayout,
        proving_key: &ProvingKey<E>,
    ) -> Result<UnivariateKzgProof<E>, PlonkError> {
        // Compute the identity polynomial
        let a1_minus_a2 = a1 - a2;
        let vanishing_eval = Self::compute_vanishing_poly_eval(challenge, layout);
        let identity_poly = &a1_minus_a2 - &(quotient_poly * vanishing_eval);

        // Compute the opening
        let (opening_proof, _) =
            UnivariateKzgPCS::open(&proving_key.commit_key, &identity_poly, &challenge)
                .map_err(PlonkError::PCSError)?;
        Ok(opening_proof)
    }
}

// ------------
// | Verifier |
// ------------

impl<E, F, P> PlonkKzgSnark<E>
where
    E: Pairing<BaseField = F, G1Affine = Affine<P>>,
    F: RescueParameter + SWToTEConParam,
    P: SWCurveConfig<BaseField = F, ScalarField = E::ScalarField>,
{
    /// Verify a linking proof
    ///
    /// The verifier does not have access to the link hint of the proofs -- this
    /// exposes wiring information -- so it is simpler to pass a proof
    /// reference directly (which the verifier will have). This avoids the need
    /// to index into the commitments at the callsite
    pub fn verify_link_proof<T: PlonkTranscript<E::BaseField>>(
        r1cs_proof1: &Proof<E>,
        r1cs_proof2: &Proof<E>,
        link_proof: &LinkingProof<E>,
        layout: &GroupLayout,
        vk: &VerifyingKey<E>,
    ) -> Result<(), PlonkError> {
        // Squeeze a challenge for the opening
        let quotient_comm = &link_proof.quotient_commitment;
        let a1_comm = &r1cs_proof1.wires_poly_comms[PROOF_LINK_WIRE_IDX];
        let a2_comm = &r1cs_proof2.wires_poly_comms[PROOF_LINK_WIRE_IDX];
        let eta = Self::compute_quotient_challenge::<T>(a1_comm, a2_comm, quotient_comm)?;

        // Compute a commitment to the proof-linking identity polynomial
        let identity_comm =
            Self::compute_identity_commitment(a1_comm, a2_comm, quotient_comm, eta, layout);

        let res = UnivariateKzgPCS::verify(
            &vk.open_key,
            &identity_comm,
            &eta,
            &E::ScalarField::zero(),
            &link_proof.opening_proof,
        )
        .map_err(PlonkError::PCSError)?;

        if res {
            Ok(())
        } else {
            Err(PlonkError::ProofLinkVerification)
        }
    }

    /// Compute a commitment to the evaluated polynomial using the homomorphic
    /// properties of the underlying commitment scheme
    fn compute_identity_commitment(
        a1_comm: &Commitment<E>,
        a2_comm: &Commitment<E>,
        quotient_comm: &Commitment<E>,
        challenge: <E as Pairing>::ScalarField,
        layout: &GroupLayout,
    ) -> Commitment<E> {
        let vanishing_eval = Self::compute_vanishing_poly_eval(challenge, layout);
        let new_comm = a1_comm.0 - a2_comm.0 - quotient_comm.0 * vanishing_eval;

        Commitment::from(Into::<E::G1Affine>::into(new_comm))
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::{Bn254, Fr as FrBn254};
    use ark_std::UniformRand;
    use itertools::Itertools;
    use lazy_static::lazy_static;
    use mpc_relation::{
        traits::{Circuit, LinkGroup},
        PlonkCircuit,
    };
    use rand::{thread_rng, Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    use crate::{
        proof_system::{
            structs::{LinkingHint, Proof, ProvingKey, VerifyingKey},
            PlonkKzgSnark, UniversalSNARK,
        },
        transcript::SolidityTranscript,
    };

    /// The name of the group used in testing
    const GROUP_NAME: &str = "test_group";
    /// The maximum circuit degree used in testing
    const MAX_DEGREE_TESTING: usize = 100;
    lazy_static! {
        /// The rng seed used to generate the test circuit's SRS
        ///
        /// We generate it once at setup to ensure that the SRS is the same between all constructions
        static ref SRS_SEED: u64 = rand::thread_rng().gen();
    }

    /// Generate a test circuit with a witness
    fn gen_test_circuit(witness: &[u64]) -> (PlonkCircuit<FrBn254>, LinkGroup) {
        let mut rng = thread_rng();
        let mut circuit = PlonkCircuit::new_turbo_plonk();
        let sum = witness.iter().cloned().map(Into::<u128>::into).sum::<u128>();

        // Add a few public inputs to the circuit
        for _ in 0..10 {
            circuit.create_public_variable(FrBn254::rand(&mut rng)).unwrap();
        }
        let expected = circuit.create_public_variable(sum.into()).unwrap();

        // Create a proof linking group and add the witnesses to it
        let group = circuit.create_link_group(GROUP_NAME.to_string(), None /* layout */);
        let witness_vars = witness
            .iter()
            .map(|&w| circuit.create_variable_with_link_groups(w.into(), &[group.clone()]).unwrap())
            .collect_vec();

        let sum = circuit.sum(&witness_vars).unwrap();
        circuit.enforce_equal(sum, expected).unwrap();
        circuit.finalize_for_arithmetization().unwrap();

        (circuit, group)
    }

    /// Generate a proof and link hint for the circuit by proving its r1cs
    /// relation
    fn gen_test_proof(circuit: &PlonkCircuit<FrBn254>) -> (Proof<Bn254>, LinkingHint<Bn254>) {
        let mut rng = thread_rng();
        let (pk, _) = gen_keys(circuit);

        PlonkKzgSnark::<Bn254>::prove_with_link_hint::<_, _, SolidityTranscript>(
            &mut rng, circuit, &pk,
        )
        .unwrap()
    }

    /// Setup proving and verifying keys for a test circuit
    fn gen_keys(circuit: &PlonkCircuit<FrBn254>) -> (ProvingKey<Bn254>, VerifyingKey<Bn254>) {
        let mut rng = ChaCha20Rng::seed_from_u64(*SRS_SEED);
        let srs = PlonkKzgSnark::<Bn254>::universal_setup_for_testing(MAX_DEGREE_TESTING, &mut rng)
            .unwrap();

        PlonkKzgSnark::<Bn254>::preprocess(&srs, circuit).unwrap()
    }

    /// Tests a linking proof between two circuits that correctly use the same
    /// values in the linking domain
    #[test]
    fn test_proof_link_valid() {
        const N: usize = 10;
        let mut rng = thread_rng();

        // Use the same witness between two circuits
        let witness = (0..N).map(|_| u64::rand(&mut rng)).collect_vec();

        // Generate the two circuits
        let (mut lhs_circuit, group) = gen_test_circuit(&witness);
        let (rhs_circuit, _) = gen_test_circuit(&witness);

        let circuit_layout = lhs_circuit.gen_circuit_layout().unwrap();
        let group_layout = circuit_layout.group_layouts.get(&group.id).unwrap();

        // Prove each circuit
        let (pk, vk) = gen_keys(&lhs_circuit);
        let (lhs_proof, lhs_hint) = gen_test_proof(&lhs_circuit);
        let (rhs_proof, rhs_hint) = gen_test_proof(&rhs_circuit);

        // Generate a link proof
        let proof = PlonkKzgSnark::link_proofs::<SolidityTranscript>(
            &lhs_hint,
            &rhs_hint,
            group_layout,
            &pk,
        )
        .unwrap();

        // Verify the link proof
        let circuit_layout = lhs_circuit.gen_circuit_layout().unwrap();
        let layout = circuit_layout.group_layouts.get(&group.id).unwrap();
        PlonkKzgSnark::verify_link_proof::<SolidityTranscript>(
            &lhs_proof, &rhs_proof, &proof, layout, &vk,
        )
        .unwrap();
    }
}
