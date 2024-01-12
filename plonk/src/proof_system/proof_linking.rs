//! Proof linking primitives for Plonk proofs
//!
//! TODO(@joey): detail the proof-linking protocol

use ark_ec::{
    pairing::Pairing,
    short_weierstrass::{Affine, SWCurveConfig},
};
use ark_ff::{Field, One, Zero};
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
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
use serde::{ser::SerializeSeq, Deserialize, Serialize};

use crate::{errors::PlonkError, transcript::PlonkTranscript};

use super::{
    structs::{CommitKey, LinkingHint, OpenKey, Proof},
    PlonkKzgSnark,
};

/// A proof that two circuits are linked on a given domain
#[derive(Clone, Debug, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct LinkingProof<E: Pairing> {
    /// The commitment to the linking quotient polynomial
    pub quotient_commitment: Commitment<E>,
    /// The proof of opening of the linking identity polynomial
    pub opening_proof: UnivariateKzgProof<E>,
}

impl<E: Pairing> Serialize for LinkingProof<E> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut bytes = Vec::new();
        self.serialize_compressed(&mut bytes).map_err(serde::ser::Error::custom)?;

        // Serialize explicitly as a sequence to avoid issues with certain serde
        // formats, e.g. flexbuffers
        let mut seq = serializer.serialize_seq(Some(bytes.len()))?;
        for byte in bytes.iter() {
            seq.serialize_element(byte)?;
        }
        seq.end()
    }
}

impl<'de, E: Pairing> Deserialize<'de> for LinkingProof<E> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let bytes = <Vec<u8>>::deserialize(deserializer)?;
        Self::deserialize_compressed(bytes.as_slice()).map_err(serde::de::Error::custom)
    }
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
        commit_key: &CommitKey<E>,
    ) -> Result<LinkingProof<E>, PlonkError> {
        // Compute the wiring polynomials that encode the proof-linked values
        let a1 = &lhs_link_hint.linking_wire_poly;
        let a2 = &rhs_link_hint.linking_wire_poly;

        // Compute the quotient then commit to it
        let quotient = Self::compute_linking_quotient(a1, a2, group_layout)?;
        let quotient_commitment =
            UnivariateKzgPCS::commit(commit_key, &quotient).map_err(PlonkError::PCSError)?;

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
            commit_key,
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
        commit_key: &CommitKey<E>,
    ) -> Result<UnivariateKzgProof<E>, PlonkError> {
        // Compute the identity polynomial
        let a1_minus_a2 = a1 - a2;
        let vanishing_eval = Self::compute_vanishing_poly_eval(challenge, layout);
        let identity_poly = &a1_minus_a2 - &(quotient_poly * vanishing_eval);

        // Compute the opening
        let (opening_proof, _) = UnivariateKzgPCS::open(commit_key, &identity_poly, &challenge)
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
        open_key: &OpenKey<E>,
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
            open_key,
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
pub mod test_helpers {
    //! Helpers exported for proof-linking tests in the `plonk` crate

    use ark_bn254::{Bn254, Fr as FrBn254};
    use itertools::Itertools;
    use jf_primitives::pcs::StructuredReferenceString;
    use lazy_static::lazy_static;
    use mpc_relation::{
        proof_linking::{GroupLayout, LinkableCircuit},
        PlonkCircuit,
    };
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    use crate::proof_system::{
        structs::{CommitKey, OpenKey, ProvingKey, VerifyingKey},
        PlonkKzgSnark, UniversalSNARK,
    };

    /// The field used in testing
    pub type TestField = FrBn254;
    /// The maximum circuit degree used in testing
    pub const MAX_DEGREE_TESTING: usize = 1000;
    /// The name of the group used in testing
    pub const GROUP_NAME: &str = "test_group";

    lazy_static! {
        /// The rng seed used to generate the test circuit's SRS
        ///
        /// We generate it once at setup to ensure that the SRS is the same between all constructions
        pub static ref SRS_SEED: u64 = rand::thread_rng().gen();
    }

    // -----------------
    // | Test Circuits |
    // -----------------

    /// A selector enum for the test circuit
    ///
    /// Aids in the ergonomics of templating tests below
    #[derive(Copy, Clone)]
    pub enum CircuitSelector {
        /// The first circuit
        Circuit1,
        /// The second circuit
        Circuit2,
    }

    /// Generate a summation circuit with the given witness
    pub fn gen_test_circuit1<C: LinkableCircuit<TestField>>(
        c: &mut C,
        witness: &[C::Wire],
        layout: Option<GroupLayout>,
    ) {
        let sum = witness.iter().cloned().sum::<C::Wire>();
        let expected = c.create_public_variable(sum).unwrap();

        // Add a few public inputs to the circuit
        // Simplest just to re-use the witness values directly as these are purely for
        // spacing
        for val in witness.iter() {
            c.create_public_variable(val.clone()).unwrap();
        }

        // Link the witnesses to the group
        let group = c.create_link_group(GROUP_NAME.to_string(), layout);
        let witness_vars = witness
            .iter()
            .map(|w| c.create_variable_with_link_groups(w.clone(), &[group.clone()]).unwrap())
            .collect_vec();

        // Create a few more witnesses that are not linked
        // Again, for interface simplicity just reuse the witness data to generate new
        // values
        witness.iter().map(|w| c.create_variable(w.clone() * w.clone()).unwrap()).collect_vec();

        let sum = c.sum(&witness_vars).unwrap();
        c.enforce_equal(sum, expected).unwrap();
    }

    /// Generate a product circuit with the given witness
    pub fn gen_test_circuit2<C: LinkableCircuit<TestField>>(
        c: &mut C,
        witness: &[C::Wire],
        layout: Option<GroupLayout>,
    ) {
        // Compute the expected result
        let mut product = witness[0].clone();
        for w in witness[1..].iter().cloned() {
            product = product * w;
        }
        let expected = c.create_public_variable(product).unwrap();

        // Add a few public inputs to the circuit
        // Simplest just to re-use the witness values directly as these are purely for
        // spacing
        for val in witness.iter() {
            c.create_public_variable(val.clone()).unwrap();
        }

        // Link half the witnesses to the group
        let group = c.create_link_group(GROUP_NAME.to_string(), layout);
        let witness_vars = witness
            .iter()
            .map(|w| c.create_variable_with_link_groups(w.clone(), &[group.clone()]).unwrap())
            .collect_vec();

        // Create a few more witnesses that are not linked
        // Again, for interface simplicity just reuse the witness data to generate new
        // values
        witness.iter().map(|w| c.create_variable(w.clone() * w.clone()).unwrap()).collect_vec();

        // Constrain the product
        let mut product = c.one();
        for var in &witness_vars {
            product = c.mul(product, *var).unwrap();
        }
        c.enforce_equal(product, expected).unwrap();
    }

    /// Setup proving and verifying keys for a test circuit
    pub fn gen_proving_keys(
        circuit: &PlonkCircuit<FrBn254>,
    ) -> (ProvingKey<Bn254>, VerifyingKey<Bn254>) {
        let mut rng = ChaCha20Rng::seed_from_u64(*SRS_SEED);
        let srs = PlonkKzgSnark::<Bn254>::universal_setup_for_testing(MAX_DEGREE_TESTING, &mut rng)
            .unwrap();

        PlonkKzgSnark::<Bn254>::preprocess(&srs, circuit).unwrap()
    }

    /// Generate commitment keys for a KZG commitment
    ///
    /// This is done separately from the proving key to allow helpers to
    /// generate circuit-agnostic keys
    pub fn gen_commit_keys() -> (CommitKey<Bn254>, OpenKey<Bn254>) {
        let mut rng = ChaCha20Rng::seed_from_u64(*SRS_SEED);
        let srs = PlonkKzgSnark::<Bn254>::universal_setup_for_testing(MAX_DEGREE_TESTING, &mut rng)
            .unwrap();

        (
            srs.extract_prover_param(MAX_DEGREE_TESTING),
            srs.extract_verifier_param(MAX_DEGREE_TESTING),
        )
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::{Bn254, Fr as FrBn254};
    use ark_ec::pairing::Pairing;
    use ark_std::UniformRand;
    use itertools::Itertools;
    use jf_primitives::pcs::prelude::UnivariateKzgProof;
    use mpc_relation::{
        proof_linking::{GroupLayout, LinkableCircuit},
        PlonkCircuit,
    };
    use rand::{thread_rng, Rng};

    use crate::{
        errors::PlonkError,
        proof_system::{
            structs::{LinkingHint, Proof},
            PlonkKzgSnark,
        },
        transcript::SolidityTranscript,
    };

    use super::{
        test_helpers::{
            gen_commit_keys, gen_proving_keys, gen_test_circuit1, gen_test_circuit2,
            CircuitSelector, GROUP_NAME,
        },
        LinkingProof,
    };

    // -----------
    // | Helpers |
    // -----------

    /// Generate a test case proof, group layout, and link hint from the given
    /// circuit
    fn gen_circuit_proof_and_hint(
        witness: &[FrBn254],
        circuit: CircuitSelector,
        layout: Option<GroupLayout>,
    ) -> (Proof<Bn254>, LinkingHint<Bn254>, GroupLayout) {
        let mut cs = PlonkCircuit::new_turbo_plonk();
        match circuit {
            CircuitSelector::Circuit1 => gen_test_circuit1(&mut cs, witness, layout),
            CircuitSelector::Circuit2 => gen_test_circuit2(&mut cs, witness, layout),
        };
        cs.finalize_for_arithmetization().unwrap();

        // Get the layout
        let group_layout = cs.get_link_group_layout(GROUP_NAME).unwrap();

        // Generate a proof with a linking hint
        let (proof, hint) = gen_test_proof(&cs);
        (proof, hint, group_layout)
    }

    /// Generate a proof and link hint for the circuit by proving its r1cs
    /// relation
    fn gen_test_proof(circuit: &PlonkCircuit<FrBn254>) -> (Proof<Bn254>, LinkingHint<Bn254>) {
        let mut rng = thread_rng();
        let (pk, _) = gen_proving_keys(circuit);

        PlonkKzgSnark::<Bn254>::prove_with_link_hint::<_, _, SolidityTranscript>(
            &mut rng, circuit, &pk,
        )
        .unwrap()
    }

    /// Prove a link between two circuits and verify the link, return the result
    /// as a result
    fn prove_and_verify_link(
        lhs_hint: &LinkingHint<Bn254>,
        rhs_hint: &LinkingHint<Bn254>,
        lhs_proof: &Proof<Bn254>,
        rhs_proof: &Proof<Bn254>,
        layout: &GroupLayout,
    ) -> Result<(), PlonkError> {
        let (commit_key, open_key) = gen_commit_keys();
        let proof = PlonkKzgSnark::<Bn254>::link_proofs::<SolidityTranscript>(
            lhs_hint,
            rhs_hint,
            layout,
            &commit_key,
        )?;

        PlonkKzgSnark::<Bn254>::verify_link_proof::<SolidityTranscript>(
            lhs_proof, rhs_proof, &proof, layout, &open_key,
        )
    }

    // --------------
    // | Test Cases |
    // --------------

    /// Tests serialization and deserialization of a linking proof
    #[test]
    fn test_serde() {
        let mut rng = thread_rng();
        let commitment = <Bn254 as Pairing>::G1Affine::rand(&mut rng).into();
        let opening = UnivariateKzgProof { proof: <Bn254 as Pairing>::G1Affine::rand(&mut rng) };

        let proof =
            LinkingProof::<Bn254> { quotient_commitment: commitment, opening_proof: opening };

        let bytes = serde_json::to_vec(&proof).unwrap();
        let recovered = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(proof, recovered);
    }

    /// Tests a linking proof between two circuits that correctly use the same
    /// values in the linking domain
    #[test]
    #[allow(non_snake_case)]
    fn test_valid_proof_link__no_layout() {
        const N: usize = 10;
        let mut rng = thread_rng();

        // Use the same witness between two circuits
        let witness = (0..N).map(|_| FrBn254::rand(&mut rng)).collect_vec();
        let (proof1, hint1, layout) =
            gen_circuit_proof_and_hint(&witness, CircuitSelector::Circuit1, None /* layout */);
        let (proof2, hint2, _) =
            gen_circuit_proof_and_hint(&witness, CircuitSelector::Circuit1, None /* layout */);

        prove_and_verify_link(&hint1, &hint2, &proof1, &proof2, &layout).unwrap();
    }

    /// Tests proof linking two _different_ circuits with a valid witness
    #[test]
    #[allow(non_snake_case)]
    fn test_valid_proof_link__different_circuits() {
        const N: usize = 10;
        let mut rng = thread_rng();

        // Use the same witness between two circuits
        let witness = (0..N).map(|_| FrBn254::rand(&mut rng)).collect_vec();
        let (proof1, hint1, layout) =
            gen_circuit_proof_and_hint(&witness, CircuitSelector::Circuit1, None /* layout */);
        let (proof2, hint2, _) =
            gen_circuit_proof_and_hint(&witness, CircuitSelector::Circuit2, Some(layout));

        prove_and_verify_link(&hint1, &hint2, &proof1, &proof2, &layout).unwrap();
    }

    /// Tests proof linking with a specified layout up front
    #[test]
    #[allow(non_snake_case)]
    fn test_valid_proof_link__specific_layout() {
        const N: usize = 10;
        let mut rng = thread_rng();

        // Construct a pre-specified layout
        let witness = (0..N).map(|_| FrBn254::rand(&mut rng)).collect_vec();
        let layout = GroupLayout { offset: 20, size: N, alignment: 8 };

        // Generate the proofs
        let (proof1, hint1, _) =
            gen_circuit_proof_and_hint(&witness, CircuitSelector::Circuit1, Some(layout));
        let (proof2, hint2, _) =
            gen_circuit_proof_and_hint(&witness, CircuitSelector::Circuit2, Some(layout));

        prove_and_verify_link(&hint1, &hint2, &proof1, &proof2, &layout).unwrap();
    }

    /// Tests an invalid proof link wherein the witnesses used are different
    #[test]
    #[allow(non_snake_case)]
    fn test_invalid_proof_link__different_witnesses() {
        const N: usize = 10;
        let mut rng = thread_rng();

        // -- Same circuit -- //

        // Modify the second witness at a random location
        let witness1 = (0..N).map(|_| FrBn254::rand(&mut rng)).collect_vec();
        let mut witness2 = witness1.clone();
        let modification_idx = rng.gen_range(0..N);
        witness2[modification_idx] = FrBn254::rand(&mut rng);

        // Attempt to prove a link
        let (proof1, hint1, layout) = gen_circuit_proof_and_hint(
            &witness1,
            CircuitSelector::Circuit1,
            None, // layout
        );
        let (proof2, hint2, _) = gen_circuit_proof_and_hint(
            &witness2,
            CircuitSelector::Circuit1,
            None, // layout
        );

        let res = prove_and_verify_link(&hint1, &hint2, &proof1, &proof2, &layout);
        assert!(res.is_err());

        // -- Different circuits -- //

        let (proof1, hint1, layout) = gen_circuit_proof_and_hint(
            &witness1,
            CircuitSelector::Circuit1,
            None, // layout
        );
        let (proof2, hint2, _) =
            gen_circuit_proof_and_hint(&witness2, CircuitSelector::Circuit2, Some(layout));

        let res = prove_and_verify_link(&hint1, &hint2, &proof1, &proof2, &layout);
        assert!(res.is_err());
    }

    /// Tests the case in which the correct witness is used to link but over
    /// incorrectly aligned domains
    #[test]
    #[allow(non_snake_case)]
    fn test_invalid_proof_link__wrong_alignment() {
        const N: usize = 10;
        let mut rng = thread_rng();

        // Use the same witness between two circuits
        let witness = (0..N).map(|_| FrBn254::rand(&mut rng)).collect_vec();
        let (proof1, hint1, mut layout) =
            gen_circuit_proof_and_hint(&witness, CircuitSelector::Circuit1, None /* layout */);

        // Modify the layout to be misaligned
        layout.alignment += 1;
        let (proof2, hint2, _) =
            gen_circuit_proof_and_hint(&witness, CircuitSelector::Circuit2, Some(layout));

        let res = prove_and_verify_link(&hint1, &hint2, &proof1, &proof2, &layout);
        assert!(res.is_err());
    }

    /// Tests the case in which the correct witness is used to link but over
    /// domains at different offsets
    #[test]
    #[allow(non_snake_case)]
    fn test_invalid_proof_link__wrong_offset() {
        const N: usize = 10;
        let mut rng = thread_rng();

        // Use the same witness between two circuits
        let witness = (0..N).map(|_| FrBn254::rand(&mut rng)).collect_vec();
        let (proof1, hint1, mut layout) =
            gen_circuit_proof_and_hint(&witness, CircuitSelector::Circuit1, None /* layout */);

        // Modify the layout to be offset from the first circuit
        layout.offset -= 1;
        let (proof2, hint2, _) =
            gen_circuit_proof_and_hint(&witness, CircuitSelector::Circuit2, Some(layout));

        let res = prove_and_verify_link(&hint1, &hint2, &proof1, &proof2, &layout);
        assert!(res.is_err());
    }
}
