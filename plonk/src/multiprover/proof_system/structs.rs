//! Generic container structs used in the proof system
use ark_ec::CurveGroup;
use ark_mpc::algebra::AuthenticatedDensePoly;

/// Multiprover Plonk IOP online polynomial oracles
#[derive(Debug, Clone)]
pub(crate) struct MpcOracles<C: CurveGroup> {
    pub(crate) wire_polys: Vec<AuthenticatedDensePoly<C>>,
    pub(crate) pub_input_poly: AuthenticatedDensePoly<C>,
    pub(crate) prod_perm_poly: AuthenticatedDensePoly<C>,
}
