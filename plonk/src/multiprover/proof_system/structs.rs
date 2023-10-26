//! Generic container structs used in the proof system
use ark_ec::CurveGroup;
use ark_mpc::algebra::{AuthenticatedDensePoly, ScalarResult};

/// Multiprover Plonk IOP online polynomial oracles
#[derive(Debug, Clone)]
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
pub struct MpcChallenges<C: CurveGroup> {
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
