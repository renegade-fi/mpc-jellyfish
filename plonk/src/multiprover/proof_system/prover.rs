//! A multiprover implementation of the PLONK proof system

use ark_ec::{pairing::Pairing, CurveGroup};
use ark_mpc::{algebra::AuthenticatedDensePoly, MpcFabric};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain, Radix2EvaluationDomain};
use ark_std::rand::{CryptoRng, RngCore};

use crate::{
    constants::domain_size_ratio,
    errors::PlonkError,
    multiprover::primitives::{MultiproverKZG, MultiproverKzgCommitment},
    proof_system::structs::{Challenges, CommitKey},
};

use super::MpcArithmetization;

// -------------------------
// | Prover Implementation |
// -------------------------

/// A type alias for a bundle of commitments and polynomials
/// TODO: Remove this lint allowance
#[allow(unused, type_alias_bounds)]
type MpcCommitmentsAndPolys<E: Pairing> = (
    Vec<MultiproverKzgCommitment<E>>,
    Vec<AuthenticatedDensePoly<E::G1>>,
);

/// A Plonk IOP prover over a secret shared algebra
/// TODO: Remove this lint allowance
#[allow(unused)]
pub(crate) struct MpcProver<E: Pairing>
where
    E: Pairing,
{
    /// The evaluation domain of the PIOP checks hold over
    domain: Radix2EvaluationDomain<E::ScalarField>,
    /// The domain of the quotient polynomial
    quot_domain: GeneralEvaluationDomain<E::ScalarField>,
    /// A reference to the underlying MPC fabric
    fabric: MpcFabric<E::G1>,
}

/// The methods below follow the structure defined in the paper:
///     https://eprint.iacr.org/2019/953.pdf
///
/// Note as well that the Fiat-Shamir brokered challenges are not secret shared.
/// As an optimization we open all inputs to the transcript before inserting
/// them into the transcript. This obviates the need to evaluate the transcript
/// on secret shares (hashing over a secret shared algebra is quite expensive).
///
/// By definition this optimization is safe as we open only elements of the
/// proof, which inherit input privacy from the zero-knowledge property of the
/// proof system
/// TODO: Remove this lint allowance
#[allow(unused)]
impl<E: Pairing> MpcProver<E> {
    /// Construct a Plonk prover that uses a domain with size `domain_size` and
    /// quotient polynomial domain with a size that is larger than the degree of
    /// the quotient polynomial
    pub(crate) fn new(
        domain_size: usize,
        num_wire_types: usize,
        fabric: MpcFabric<E::G1>,
    ) -> Result<Self, PlonkError> {
        let domain = Radix2EvaluationDomain::<E::ScalarField>::new(domain_size)
            .ok_or(PlonkError::DomainCreationError)?;
        let quot_domain = GeneralEvaluationDomain::<E::ScalarField>::new(
            domain_size * domain_size_ratio(domain_size, num_wire_types),
        )
        .ok_or(PlonkError::DomainCreationError)?;

        Ok(Self {
            domain,
            quot_domain,
            fabric,
        })
    }

    /// Round 1:
    /// 1. Compute and commit wire witness polynomials.
    /// 2. Compute public input polynomial.
    /// Return the wire witness polynomials and their commitments,
    /// also return the public input polynomial.
    pub(crate) fn run_1st_round<C: MpcArithmetization<E::G1>, R: CryptoRng + RngCore>(
        &self,
        prng: &mut R,
        ck: &CommitKey<E>,
        cs: &C,
    ) -> Result<(MpcCommitmentsAndPolys<E>, AuthenticatedDensePoly<E::G1>), PlonkError> {
        let wire_polys: Vec<AuthenticatedDensePoly<E::G1>> = cs
            .compute_wire_polynomials()?
            .into_iter()
            .map(|poly| self.mask_polynomial(&poly, 1))
            .collect();
        let wires_poly_comms = MultiproverKZG::batch_commit(ck, &wire_polys)?;
        let pub_input_poly = cs.compute_pub_input_polynomial()?;

        Ok(((wires_poly_comms, wire_polys), pub_input_poly))
    }

    /// Round 2: Compute and commit the permutation grand product polynomial
    /// Return the grand product polynomial and its commitment
    pub(crate) fn run_2nd_round<C: MpcArithmetization<E::G1>>(
        &self,
        ck: &CommitKey<E>,
        cs: &C,
        challenges: &Challenges<E::ScalarField>,
    ) -> Result<(MultiproverKzgCommitment<E>, AuthenticatedDensePoly<E::G1>), PlonkError> {
        let prod_perm_poly = self.mask_polynomial(
            &cs.compute_prod_permutation_polynomial(&challenges.beta, &challenges.gamma)?,
            2, // hiding_degree
        );
        let prod_perm_comm = MultiproverKZG::commit(ck, &prod_perm_poly)?;

        Ok((prod_perm_comm, prod_perm_poly))
    }
}

/// Private helper methods
/// TODO: Remove this lint allowance
#[allow(unused)]
impl<E: Pairing> MpcProver<E> {
    /// Mask the polynomial by adding a random polynomial of degree
    /// `hiding_bound` to it
    fn mask_polynomial(
        &self,
        poly: &AuthenticatedDensePoly<E::G1>,
        hiding_bound: usize,
    ) -> AuthenticatedDensePoly<E::G1> {
        let mask = mul_by_vanishing_poly(
            &AuthenticatedDensePoly::random(hiding_bound, &self.fabric),
            &self.domain,
            &self.fabric,
        );

        mask + poly
    }
}

// -----------
// | Helpers |
// -----------

/// Multiply an authenticated polynomial by the polynomial that vanishes over
/// the given domain
///
/// The domain is comprised of `n`th roots of unity where `n` is the size of the
/// domain, so this is a multiplication by the polynomial `x^n - 1`
///
/// This can be realized by shifting the coefficients by `n` "degrees" (for
/// multiplication by x^n) then subtracting back out the original polynomial
/// (for subtraction of 1)
fn mul_by_vanishing_poly<C: CurveGroup, D: EvaluationDomain<C::ScalarField>>(
    poly: &AuthenticatedDensePoly<C>,
    domain: &D,
    fabric: &MpcFabric<C>,
) -> AuthenticatedDensePoly<C> {
    let n = domain.size();
    let mut shifted_coeffs = fabric.zeros_authenticated(n);
    shifted_coeffs.extend_from_slice(&poly.coeffs);

    let shifted_poly = AuthenticatedDensePoly::from_coeffs(shifted_coeffs);
    shifted_poly - poly
}
