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
use ark_mpc::{algebra::AuthenticatedScalarResult, MpcFabric};
use itertools::Itertools;

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
            challenges.zeta.clone(),
            &split_quot_polys,
        )?;

        lin_poly = lin_poly
            + prover.compute_non_quotient_component_for_lin_poly(
                &fabric.one(),
                proving_key,
                &challenges,
                &online_oracles,
                &poly_evals,
            )?;

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
    #[test]
    fn test_prove_simple_circuit() {}
}
