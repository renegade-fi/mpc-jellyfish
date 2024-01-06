//! Generic container structs used in the proof system
use core::{
    pin::Pin,
    task::{Context, Poll},
};

use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup};
use ark_mpc::{
    algebra::{AuthenticatedDensePoly, CurvePoint, Scalar, ScalarResult},
    network::PartyId,
    MpcFabric,
};
use futures::{ready, Future};
use itertools::Itertools;

use crate::{
    errors::PlonkError,
    multiprover::primitives::{MultiproverKzgCommitment, MultiproverKzgCommitmentOpening},
    proof_system::structs::{LinkingHint, Proof, ProofEvaluations},
};

/// Multiprover Plonk IOP online polynomial oracles
#[derive(Debug, Clone, Default)]
pub(crate) struct MpcOracles<C: CurveGroup> {
    pub(crate) wire_polys: Vec<AuthenticatedDensePoly<C>>,
    pub(crate) pub_input_poly: AuthenticatedDensePoly<C>,
    pub(crate) prod_perm_poly: AuthenticatedDensePoly<C>,
}

/// Plonk IOP verifier challenges that have been allocated in an MPC fabric
///
/// We hold handles to incomplete computations (transcript evaluations) instead
/// of the underlying values, as per the MPC framework's standard
#[derive(Debug)]
pub(crate) struct MpcChallenges<C: CurveGroup> {
    /// The parameterization of the random linear combination of gate, copy, and
    /// grand product polynomials
    pub alpha: ScalarResult<C>,
    /// The first order permutation challenge
    pub beta: ScalarResult<C>,
    /// The zero'th order (affine) permutation challenge
    pub gamma: ScalarResult<C>,
    /// The challenge at which the quotient is linearized
    pub zeta: ScalarResult<C>,
    /// The opening challenge, used to combine terms in the linearization
    pub v: ScalarResult<C>,
}

impl<C: CurveGroup> MpcChallenges<C> {
    /// A custom default implementation that sets all challenges to zero
    /// initially
    pub fn default(fabric: &MpcFabric<C>) -> Self {
        Self {
            alpha: fabric.zero(),
            beta: fabric.zero(),
            gamma: fabric.zero(),
            zeta: fabric.zero(),
            v: fabric.zero(),
        }
    }
}

/// A struct that stores the polynomial evaluations in a Plonk proof
///
/// Note that this struct differs from the analog in the single-prover
/// implementation in that it stores handles to future evaluations of nodes in
/// the MPC computation graph
///
/// These values exist in the plaintext (are not secret shared) because we
/// immediately append them to the transcript. Our transcript implementation
/// requires opening values before appending to avoid transcript evaluation
/// (hashing) inside the MPC circuit. The elements entered into the transcript
/// definitionally preserve zero knowledge (they are communicated to the
/// verifier), so it is safe to open them
#[derive(Debug, Clone)]
pub struct MpcProofEvaluations<C: CurveGroup> {
    /// Wire witness polynomials evaluations at point `zeta`
    pub wires_evals: Vec<ScalarResult<C>>,

    /// Extended permutation (sigma) polynomials evaluations at point `zeta`
    /// We do not include the last sigma polynomial evaluation
    pub wire_sigma_evals: Vec<ScalarResult<C>>,

    /// Permutation product polynomial evaluation at point `zeta * g`
    pub perm_next_eval: ScalarResult<C>,
}

/// A struct that represents a completed proof in a multiprover context
#[derive(Debug, Clone)]
pub struct CollaborativeProof<E: Pairing> {
    /// The commitments to the wiring polynomials
    pub wire_poly_comms: Vec<MultiproverKzgCommitmentOpening<E>>,
    /// The commitment to the wire permutation polynomial
    pub prod_perm_poly_comm: MultiproverKzgCommitmentOpening<E>,
    /// The split quotient polynomial commitments
    pub split_quot_poly_comms: Vec<MultiproverKzgCommitmentOpening<E>>,
    /// The proof of evaluation of the aggregated argument at a challenge point
    pub opening_proof: MultiproverKzgCommitment<E>,
    /// The proof of evaluation of the aggregated argument at the challenge
    /// point shifted by one multiplication with generator in the
    /// root-of-unit group
    pub shifted_opening_proof: MultiproverKzgCommitment<E>,
    /// Polynomial evaluations of each component in the linearization polynomial
    pub poly_evals: MpcProofEvaluations<E::G1>,
}

impl<E: Pairing> CollaborativeProof<E> {
    /// Open the proof to both parties
    pub fn open_authenticated(self) -> CollaborativeProofOpening<E> {
        CollaborativeProofOpening {
            wire_poly_comms: self.wire_poly_comms,
            prod_perm_poly_comm: self.prod_perm_poly_comm,
            split_quot_poly_comms: self.split_quot_poly_comms,
            opening_proof: self.opening_proof.open_authenticated(),
            shifted_opening_proof: self.shifted_opening_proof.open_authenticated(),
            poly_evals: self.poly_evals,
        }
    }
}

/// A struct that represents a handle on a yet-to-be computed proof opening
///
/// This struct allows us to implement `Future` and target the `Proof` object
/// as the resolved type
#[derive(Debug, Clone)]
pub struct CollaborativeProofOpening<E: Pairing> {
    /// The commitments to the wiring polynomials
    pub wire_poly_comms: Vec<MultiproverKzgCommitmentOpening<E>>,
    /// The commitment to the wire permutation polynomial
    pub prod_perm_poly_comm: MultiproverKzgCommitmentOpening<E>,
    /// The split quotient polynomial commitments
    pub split_quot_poly_comms: Vec<MultiproverKzgCommitmentOpening<E>>,
    /// The proof of evaluation of the aggregated argument at a challenge point
    pub opening_proof: MultiproverKzgCommitmentOpening<E>,
    /// The proof of evaluation of the aggregated argument at the challenge
    /// point shifted by one multiplication with generator in the
    /// root-of-unit group
    pub shifted_opening_proof: MultiproverKzgCommitmentOpening<E>,
    /// Polynomial evaluations of each component in the linearization polynomial
    pub poly_evals: MpcProofEvaluations<E::G1>,
}

impl<E: Pairing> Future for CollaborativeProofOpening<E>
where
    E::ScalarField: Unpin,
{
    type Output = Result<Proof<E>, PlonkError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Poll a single future that returns `Result<T, E>`, and early exit if it
        // resolves to an error
        macro_rules! poll_res {
            // Propagates errors after a future resolves
            ($x:expr, ?) => {{
                match poll_res!($x) {
                    Ok(res) => res,
                    Err(err) => return Poll::Ready(Err(err.into())),
                }
            }};

            // Does not propagate errors
            ($x:expr) => {{
                ready!(Pin::new($x).poll(cx))
            }};
        }

        // Poll a vector of futures, and optionally (with `?`) early exit if
        // any of them resolve to an error
        macro_rules! poll_vec {
            ($x:expr $(, $tail:tt)?) => {{
                let mut results = Vec::new();
                for future in $x.iter_mut() {
                    results.push(poll_res!(future $(, $tail)?));
                }

                results
            }};
        }

        // Open each proof element
        let wires_poly_comms = poll_vec!(self.wire_poly_comms, ?);
        let prod_perm_poly_comm = poll_res!(&mut self.prod_perm_poly_comm, ?);
        let split_quot_poly_comms = poll_vec!(self.split_quot_poly_comms, ?);
        let opening_proof = poll_res!(&mut self.opening_proof, ?);
        let shifted_opening_proof = poll_res!(&mut self.shifted_opening_proof, ?);
        let wires_evals =
            poll_vec!(self.poly_evals.wires_evals).iter().map(Scalar::inner).collect_vec();
        let wire_sigma_evals =
            poll_vec!(self.poly_evals.wire_sigma_evals).iter().map(Scalar::inner).collect_vec();
        let perm_next_eval = poll_res!(&mut self.poly_evals.perm_next_eval).inner();

        Poll::Ready(Ok(Proof {
            wires_poly_comms,
            prod_perm_poly_comm,
            split_quot_poly_comms,
            opening_proof,
            shifted_opening_proof,
            poly_evals: ProofEvaluations { wires_evals, wire_sigma_evals, perm_next_eval },
            plookup_proof: None,
        }))
    }
}

/// A proof-linking hint generated by a set of collaborating provers in the
/// course of proving an R1CS relation
///
/// The linking hint contains information about the prover's witness needed to
/// link the proof to another proof of a different circuit
#[derive(Debug, Clone)]
pub struct MpcLinkingHint<E: Pairing> {
    /// The wire polynomial that encodes the proof-linking gates for the circuit
    pub linking_wire_poly: AuthenticatedDensePoly<E::G1>,
    /// The commitment to the linking wire poly generated while proving
    pub linking_wire_comm: MultiproverKzgCommitment<E>,
}

impl<E: Pairing> MpcLinkingHint<E> {
    /// Share a singleprover linking hint in the fabric to produce a multiprover
    /// linking hint
    pub fn from_singleprover_hint(
        singleprover_hint: &LinkingHint<E>,
        sender: PartyId,
        fabric: &MpcFabric<E::G1>,
    ) -> Self {
        // Share the coefficients of the wiring poly
        let coeffs =
            singleprover_hint.linking_wire_poly.coeffs.iter().map(|c| Scalar::new(*c)).collect();
        let shared_coeffs = fabric.batch_share_scalar(coeffs, sender);
        let shared_poly = AuthenticatedDensePoly::from_coeffs(shared_coeffs);

        // Share the commitment to the wiring poly
        let affine_point: E::G1 = singleprover_hint.linking_wire_comm.as_ref().into_group();
        let commitment = fabric.share_point(CurvePoint::from(affine_point), sender);
        let shared_comm = MultiproverKzgCommitment { commitment };

        Self { linking_wire_poly: shared_poly, linking_wire_comm: shared_comm }
    }
}
