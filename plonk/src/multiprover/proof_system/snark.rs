//! Defines the multiprover analog to the `PlonkKzgSnark` defined in the
//! `proof_system` module of this crate
//!
//! The implementation is designed to closely match the singleprover
//! implementation in structure

use core::marker::PhantomData;

use ark_ec::{
    pairing::Pairing,
    short_weierstrass::{Affine, SWCurveConfig},
};
use ark_mpc::{
    algebra::{AuthenticatedScalarResult, Scalar},
    MpcFabric,
};
use itertools::Itertools;
use mpc_relation::traits::Circuit;

use crate::{
    errors::{PlonkError, SnarkError},
    multiprover::{primitives::MpcTranscript, proof_system::structs::CollaborativeProof},
    proof_system::structs::ProvingKey,
};

use super::{MpcArithmetization, MpcChallenges, MpcOracles, MpcProver};

/// A multiprover Plonk instantiated with KZG as the underlying polynomial
/// commitment scheme
#[derive(Default)]
pub struct MultiproverPlonkKzgSnark<E: Pairing>(PhantomData<E>);

impl<P: SWCurveConfig<BaseField = E::BaseField>, E: Pairing<G1Affine = Affine<P>>>
    MultiproverPlonkKzgSnark<E>
{
    /// Constructor
    pub fn new() -> Self {
        Self(PhantomData)
    }

    /// Create a new multiprover proof
    pub fn prove<C>(
        circuit: &C,
        proving_key: &ProvingKey<E>,
        fabric: MpcFabric<E::G1>,
    ) -> Result<CollaborativeProof<E>, PlonkError>
    where
        C: MpcArithmetization<E::G1>,
        C: Circuit<
            E::ScalarField,
            Wire = AuthenticatedScalarResult<E::G1>,
            Constant = Scalar<E::G1>,
        >,
    {
        let domain_size = circuit.eval_domain_size()?;
        let num_wire_types = circuit.num_wire_types();
        Self::validate_circuit(circuit, proving_key)?;

        // Initialize the transcript
        let mut transcript = MpcTranscript::<E>::new(b"PlonkProof", fabric.clone());

        // Append the public input to the transcript
        let input = AuthenticatedScalarResult::open_authenticated_batch(&circuit.public_input()?)
            .into_iter()
            .map(|r| r.value)
            .collect_vec();
        transcript.append_vk_and_pub_input(proving_key.vk.clone(), &input);

        // Initialize verifier challenges and online polynomial oracles
        let mut challenges = MpcChallenges::default(&fabric);
        let mut online_oracles = MpcOracles::default();
        let prover = MpcProver::new(domain_size, num_wire_types, fabric.clone())?;

        // --- Round 1 --- //
        let ((wires_poly_comms, wire_polys), pi_poly) =
            prover.run_1st_round(&proving_key.commit_key, circuit, true /* mask */)?;

        online_oracles.wire_polys = wire_polys;
        online_oracles.pub_input_poly = pi_poly;

        // Open the commitments and append them to the transcript
        let wires_poly_comms_vec = wires_poly_comms
            .iter()
            .map(|comm| comm.open_authenticated())
            .collect_vec();
        transcript.append_commitments(b"witness_poly_comms", &wires_poly_comms_vec);

        // Though we don't use plookup in the multiprover setting, we still squeeze a
        // challenge to ensure that the transcript is consistent
        transcript.get_and_append_challenge(b"tau");

        // --- Round 2 --- //
        challenges.beta = transcript.get_and_append_challenge(b"beta");
        challenges.gamma = transcript.get_and_append_challenge(b"gamma");

        let (prod_perm_poly_comm, prod_perm_poly) = prover.run_2nd_round(
            &proving_key.commit_key,
            circuit,
            &challenges,
            true, // mask
        )?;
        online_oracles.prod_perm_poly = prod_perm_poly;

        let prod_perm_poly_comm_open = prod_perm_poly_comm.open_authenticated();
        transcript.append_commitment(b"perm_poly_comms", &prod_perm_poly_comm_open);

        // --- Round 3 --- //
        challenges.alpha = transcript.get_and_append_challenge(b"alpha");
        let (split_quot_poly_comms, split_quot_polys) = prover.run_3rd_round(
            &proving_key.commit_key,
            proving_key,
            &challenges,
            &online_oracles,
            num_wire_types,
        )?;

        let split_quote_poly_comm_open = split_quot_poly_comms
            .iter()
            .map(|comm| comm.open_authenticated())
            .collect_vec();
        transcript.append_commitments(b"quot_poly_comms", &split_quote_poly_comm_open);

        // --- Round 4 --- //
        challenges.zeta = transcript.get_and_append_challenge(b"zeta");
        let poly_evals =
            prover.compute_evaluations(proving_key, &challenges, &online_oracles, num_wire_types);

        transcript.append_proof_evaluations(&poly_evals);

        // Compute the linearization polynomial
        let mut lin_poly = prover.compute_quotient_component_for_lin_poly(
            domain_size,
            &challenges.zeta,
            &split_quot_polys,
        )?;

        lin_poly = lin_poly
            + prover.compute_non_quotient_component_for_lin_poly(
                &fabric.one(),
                proving_key,
                &challenges,
                &online_oracles,
                &poly_evals,
            );

        // --- Round 5 --- //
        challenges.v = transcript.get_and_append_challenge(b"v");
        let (opening_proof, shifted_opening_proof) = prover.compute_opening_proofs(
            &proving_key.commit_key,
            proving_key,
            &challenges.zeta,
            &challenges.v,
            &online_oracles,
            &lin_poly,
        )?;

        Ok(CollaborativeProof {
            wire_poly_comms: wires_poly_comms_vec,
            prod_perm_poly_comm: prod_perm_poly_comm_open,
            split_quot_poly_comms: split_quote_poly_comm_open,
            opening_proof,
            shifted_opening_proof,
            poly_evals,
        })
    }

    /// Verify that the shape of the proving key and the circuit match
    fn validate_circuit<C: MpcArithmetization<E::G1>>(
        circuit: &C,
        proving_key: &ProvingKey<E>,
    ) -> Result<(), PlonkError> {
        let n = circuit.eval_domain_size()?;
        if proving_key.domain_size() != n {
            return Err(SnarkError::ParameterError(format!(
                "proving key domain size {} != expected domain size {}",
                proving_key.domain_size(),
                n
            ))
            .into());
        }

        if circuit.num_inputs() != proving_key.vk.num_inputs {
            return Err(SnarkError::ParameterError(format!(
                "circuit.num_inputs {} != prove_key.num_inputs {}",
                circuit.num_inputs(),
                proving_key.vk.num_inputs
            ))
            .into());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ark_mpc::{
        algebra::{AuthenticatedScalarResult, Scalar},
        test_helpers::execute_mock_mpc,
        MpcFabric,
    };
    use futures::future::join_all;
    use itertools::Itertools;
    use mpc_relation::{traits::*, PlonkType};
    use rand::{thread_rng, Rng};

    use crate::{
        errors::PlonkError,
        multiprover::proof_system::{
            test::{setup_snark, test_multiprover_circuit, test_singleprover_circuit, TestGroup},
            MpcPlonkCircuit,
        },
        proof_system::{snark::test::gen_circuit_for_test, PlonkKzgSnark},
        transcript::SolidityTranscript,
    };

    use super::MultiproverPlonkKzgSnark;

    // -----------
    // | Helpers |
    // -----------

    /// A multiprover analog of the circuit used for testing the single-prover
    /// implementation in `plonk/proof_system/snark.rs`
    pub(crate) fn gen_multiprover_circuit_for_test(
        m: usize,
        a0: usize,
        fabric: MpcFabric<TestGroup>,
    ) -> Result<MpcPlonkCircuit<TestGroup>, PlonkError> {
        let mut cs = MpcPlonkCircuit::new(fabric.clone());

        // Create variables
        let mut a = vec![];
        let one = fabric.one_authenticated();
        for i in a0..(a0 + 4 * m) {
            let val = Scalar::from(i) * &one;
            a.push(cs.create_variable(val)?);
        }

        let b0 = Scalar::from(m as u64 * 2) * &one;
        let b1 = Scalar::from(a0 as u64 * 2 + m as u64 * 4 - 1) * &one;

        let b = [
            cs.create_public_variable(b0)?,
            cs.create_public_variable(b1)?,
        ];

        let c = cs.create_public_variable(
            (cs.witness(b[1])? + cs.witness(a[0])?) * (cs.witness(b[1])? - cs.witness(a[0])?),
        )?;

        // Create gates:
        // 1. a0 + ... + a_{4*m-1} = b0 * b1
        // 2. (b1 + a0) * (b1 - a0) = c
        // 3. b0 = 2 * m
        let mut acc = cs.zero();
        a.iter().for_each(|&elem| acc = cs.add(acc, elem).unwrap());
        let b_mul = cs.mul(b[0], b[1])?;
        cs.enforce_equal(acc, b_mul)?;

        let b1_plus_a0 = cs.add(b[1], a[0])?;
        let b1_minus_a0 = cs.sub(b[1], a[0])?;
        cs.mul_gate(b1_plus_a0, b1_minus_a0, c)?;
        cs.enforce_constant(b[0], (m as u64 * 2).into())?;

        // Finalize the circuit
        cs.finalize_for_arithmetization()?;

        Ok(cs)
    }

    // ---------
    // | Tests |
    // ---------

    /// Tests that the proof produced by a collaborative snark correctly
    /// verifies
    #[tokio::test]
    async fn test_prove_simple_circuit() {
        let mut rng = thread_rng();
        let witness = Scalar::random(&mut rng);
        let circuit = test_singleprover_circuit(witness);

        let (pk, vk) = setup_snark(&circuit);

        let (proof, _) = execute_mock_mpc(|fabric| {
            let pk = pk.clone();
            async move {
                let circuit = test_multiprover_circuit(witness, &fabric);
                let proof = MultiproverPlonkKzgSnark::prove(&circuit, &pk, fabric).unwrap();

                proof.open_authenticated().await.unwrap()
            }
        })
        .await;

        PlonkKzgSnark::batch_verify::<SolidityTranscript>(&[&vk], &[&[]], &[&proof], &[None])
            .unwrap();
    }

    /// Test collaborative proving against the circuit defined in the
    /// single-prover tests `plonk/proof_system/snark.rs`
    #[tokio::test]
    async fn test_complex_circuit() {
        let mut rng = thread_rng();
        let m = rng.gen_range(0..10);
        let a0 = rng.gen_range(0..10);
        let circuit = gen_circuit_for_test(m, a0, PlonkType::TurboPlonk).unwrap();

        let (pk, vk) = setup_snark(&circuit);

        let ((public_input, proof), _) = execute_mock_mpc(|fabric| {
            let pk = pk.clone();
            async move {
                let circuit = gen_multiprover_circuit_for_test(m, a0, fabric.clone()).unwrap();
                let input = circuit.public_input().unwrap();

                let proof = MultiproverPlonkKzgSnark::prove(&circuit, &pk, fabric).unwrap();

                (
                    join_all(AuthenticatedScalarResult::open_batch(&input)).await,
                    proof.open_authenticated().await.unwrap(),
                )
            }
        })
        .await;

        let public_input = public_input.iter().map(Scalar::inner).collect_vec();
        PlonkKzgSnark::batch_verify::<SolidityTranscript>(
            &[&vk],
            &[&public_input],
            &[&proof],
            &[None],
        )
        .unwrap();
    }
}
