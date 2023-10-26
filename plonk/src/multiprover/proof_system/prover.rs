//! A multiprover implementation of the PLONK proof system

use ark_ec::{pairing::Pairing, CurveGroup};
use ark_ff::{FftField, Field, One};
use ark_mpc::{
    algebra::{AuthenticatedDensePoly, AuthenticatedScalarResult, Scalar},
    MpcFabric,
};
use ark_poly::{
    DenseUVPolynomial, EvaluationDomain, GeneralEvaluationDomain, Radix2EvaluationDomain,
};
use ark_std::rand::{CryptoRng, RngCore};
use jf_relation::constants::GATE_WIDTH;
use jf_utils::par_utils::parallelizable_slice_iter;
use rayon::iter::ParallelIterator;

use crate::{
    constants::domain_size_ratio,
    errors::{PlonkError, SnarkError},
    multiprover::primitives::{MultiproverKZG, MultiproverKzgCommitment},
    proof_system::structs::{Challenges, CommitKey, ProvingKey},
};

use super::{MpcArithmetization, MpcOracles};

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

    /// Round 3: Return the split quotient polynomials and their commitments
    /// Note that the first `num_wire_types`-1 split quotient polynomials
    /// have degree `domain_size`+1.
    pub(crate) fn run_3rd_round<R: CryptoRng + RngCore>(
        &self,
        prng: &mut R,
        ck: &CommitKey<E>,
        pks: &[&ProvingKey<E>],
        challenges: &Challenges<E::ScalarField>,
        online_oracles: &[MpcOracles<E::G1>],
        num_wire_types: usize,
    ) -> Result<MpcCommitmentsAndPolys<E>, PlonkError> {
        let quot_poly =
            self.compute_quotient_polynomial(challenges, pks, online_oracles, num_wire_types)?;
        let split_quot_polys = self.split_quotient_polynomial(prng, &quot_poly, num_wire_types)?;
        let split_quot_poly_comms = MultiproverKZG::batch_commit(ck, &split_quot_polys)?;

        Ok((split_quot_poly_comms, split_quot_polys))
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

    /// Compute the quotient polynomial via (i)FFTs
    ///
    /// The quotient polynomial forms the core of the polynomial argument,
    /// constructed as the quotient of a linear combination of the wire
    /// constraints and copy constraints over the evaluation domain
    fn compute_quotient_polynomial(
        &self,
        challenges: &Challenges<E::ScalarField>,
        pks: &[&ProvingKey<E>],
        online_oracles: &[MpcOracles<E::G1>],
        num_wire_types: usize,
    ) -> Result<AuthenticatedDensePoly<E::G1>, PlonkError> {
        if pks.is_empty() || pks.len() != online_oracles.len() {
            return Err(PlonkError::SnarkError(SnarkError::ParameterError(
                "inconsistent pks/online oracles when computing quotient polys".to_string(),
            )));
        }

        let n = self.domain.size();
        let m = self.quot_domain.size();
        let domain_size_ratio = m / n;

        // Compute 1/Z_H(w^i)
        //
        // We can see that these terms exactly represents the evaluations of Z_H(x) on
        // the quotient domain, as every `n`th root of unity is also an `m`th
        // root of unity, separated by exactly `domain_size_ratio` indices.
        // Therefore, every `domain_size_ratio` indices, the polynomial `Z_H(x)`
        // cycles as the component of the eval point that is an `n`th root of unit is
        // zero'd out.
        //
        // Note that the inverse exists because we retrieve `m`th roots of unity from
        // the "extended" quotient domain (m > n), so taken to the `n`th power
        // does not yield 1
        let z_h_inv: Vec<Scalar<E::G1>> = (0..domain_size_ratio)
            .map(|i| {
                ((E::ScalarField::GENERATOR * self.quot_domain.element(i)).pow([n as u64])
                    - E::ScalarField::one())
                .inverse()
                .unwrap()
            })
            .map(Scalar::new)
            .collect();

        // Compute coset evaluations of the quotient polynomial
        let mut quot_poly_coset_evals_sum = self.fabric.zeros_authenticated(m);
        let mut alpha_base = E::ScalarField::one();
        let alpha_3 = challenges.alpha.square() * challenges.alpha;

        // The coset we use to compute the quotient polynomial
        let coset = self
            .quot_domain
            .get_coset(E::ScalarField::GENERATOR)
            .unwrap();

        // Enumerate proving instances
        for (oracles, pk) in online_oracles.iter().zip(pks.iter()) {
            // Compute evaluations of the selectors, permutations, and wiring polynomials
            let selectors_coset_fft: Vec<Vec<E::ScalarField>> =
                parallelizable_slice_iter(&pk.selectors)
                    .map(|poly| coset.fft(poly.coeffs()))
                    .collect();

            let sigmas_coset_fft: Vec<Vec<E::ScalarField>> = parallelizable_slice_iter(&pk.sigmas)
                .map(|poly| coset.fft(poly.coeffs()))
                .collect();

            let wire_polys_coset_fft: Vec<Vec<AuthenticatedScalarResult<E::G1>>> = oracles
                .wire_polys
                .iter()
                .map(|poly| AuthenticatedScalarResult::fft_with_domain(&poly.coeffs, coset))
                .collect();

            // Compute the evaluations of the z(x) polynomials representing partial products
            // of the larger grand product that argues copy constraints
            let prod_perm_poly_coset_fft =
                AuthenticatedScalarResult::fft_with_domain(&oracles.prod_perm_poly.coeffs, coset);
            let pub_input_poly_coset_fft =
                AuthenticatedScalarResult::fft_with_domain(&oracles.pub_input_poly.coeffs, coset);

            // Compute coset evaluations of the quotient polynomial following the identity
            // in the Plonk paper
            let quot_poly_coset_evals: Vec<AuthenticatedScalarResult<E::G1>> =
                parallelizable_slice_iter(&(0..m).collect::<Vec<_>>())
                    .map(|&i| {
                        // The evaluations of the wiring polynomials at this index
                        let w: Vec<AuthenticatedScalarResult<E::G1>> = (0..num_wire_types)
                            .map(|j| wire_polys_coset_fft[j][i].clone())
                            .collect();

                        // The contribution of the gate constraints to the current quotient
                        // evaluation
                        let t_circ = Self::compute_quotient_circuit_contribution(
                            i,
                            &w,
                            &pub_input_poly_coset_fft[i],
                            &selectors_coset_fft,
                        );

                        // The terms that enforce the copy constraint, the first checks that each
                        // individual index in the grand product is
                        // consistent with the permutation. The second term checks
                        // the grand product
                        let (t_perm_1, t_perm_2) =
                            Self::compute_quotient_copy_constraint_contribution(
                                i,
                                self.quot_domain.element(i) * E::ScalarField::GENERATOR,
                                pk,
                                &w,
                                &prod_perm_poly_coset_fft[i],
                                &prod_perm_poly_coset_fft[(i + domain_size_ratio) % m],
                                challenges,
                                &sigmas_coset_fft,
                            );

                        let mut t1 = t_circ + t_perm_1;
                        let mut t2 = t_perm_2;

                        t1 * z_h_inv[i % domain_size_ratio] + t2
                    })
                    .collect();

            for (a, b) in quot_poly_coset_evals_sum
                .iter_mut()
                .zip(quot_poly_coset_evals.iter())
            {
                *a = &*a + Scalar::new(alpha_base) * b;
            }

            // update the random combiner for aggregating multiple proving instances
            alpha_base *= alpha_3;
        }

        // Compute the coefficient form of the quotient polynomial
        let coeffs = AuthenticatedScalarResult::ifft_with_domain(&quot_poly_coset_evals_sum, coset);
        Ok(AuthenticatedDensePoly::from_coeffs(coeffs))
    }

    // Compute the i-th coset evaluation of the circuit part of the quotient
    // polynomial.
    fn compute_quotient_circuit_contribution(
        i: usize,
        w: &[AuthenticatedScalarResult<E::G1>],
        pi: &AuthenticatedScalarResult<E::G1>,
        selectors_coset_fft: &[Vec<E::ScalarField>],
    ) -> AuthenticatedScalarResult<E::G1> {
        // Selectors in order: q_lc, q_mul, q_hash, q_o, q_c, q_ecc
        let q_lc: Vec<Scalar<E::G1>> = (0..GATE_WIDTH)
            .map(|j| selectors_coset_fft[j][i])
            .map(Scalar::new)
            .collect();
        let q_mul: Vec<Scalar<E::G1>> = (GATE_WIDTH..GATE_WIDTH + 2)
            .map(|j| selectors_coset_fft[j][i])
            .map(Scalar::new)
            .collect();
        let q_hash: Vec<Scalar<E::G1>> = (GATE_WIDTH + 2..2 * GATE_WIDTH + 2)
            .map(|j| selectors_coset_fft[j][i])
            .map(Scalar::new)
            .collect();
        let q_o = Scalar::new(selectors_coset_fft[2 * GATE_WIDTH + 2][i]);
        let q_c = Scalar::new(selectors_coset_fft[2 * GATE_WIDTH + 3][i]);
        let q_ecc = Scalar::new(selectors_coset_fft[2 * GATE_WIDTH + 4][i]);

        // Macro that adds a term to the result only if its selector is non-zero
        // Multiplication in an MPC circuit is expensive so we use this macro to avoid
        // multiplication except when necessary
        let mut res = q_c + pi;
        macro_rules! mask_term {
            ($sel:expr, $x:expr) => {
                if $sel != Scalar::zero() {
                    res = res + $sel * $x;
                }
            };
        }

        mask_term!(q_lc[0], &w[0]);
        mask_term!(q_lc[1], &w[1]);
        mask_term!(q_lc[2], &w[2]);
        mask_term!(q_lc[3], &w[3]);
        mask_term!(q_mul[0], &w[0] * &w[1]);
        mask_term!(q_mul[1], &w[2] * &w[3]);
        mask_term!(q_ecc, &w[0] * &w[1] * &w[2] * &w[3] * &w[4]);
        mask_term!(q_hash[0], w[0].pow(5));
        mask_term!(q_hash[1], w[1].pow(5));
        mask_term!(q_hash[2], w[2].pow(5));
        mask_term!(q_hash[3], w[3].pow(5));
        mask_term!(q_o, -&w[4]);

        res
    }

    /// Compute the i-th coset evaluation of the copy constraint part of the
    /// quotient polynomial.
    /// `eval_point` - the evaluation point.
    /// `w` - the wire polynomial coset evaluations at `eval_point`.
    /// `z_x` - the permutation product polynomial evaluation at `eval_point`.
    /// `z_xw`-  the permutation product polynomial evaluation at `eval_point *
    /// g`, where `g` is the root of unity of the original domain.
    #[allow(clippy::too_many_arguments)]
    fn compute_quotient_copy_constraint_contribution(
        i: usize,
        eval_point: E::ScalarField,
        pk: &ProvingKey<E>,
        w: &[AuthenticatedScalarResult<E::G1>],
        z_x: &AuthenticatedScalarResult<E::G1>,
        z_xw: &AuthenticatedScalarResult<E::G1>,
        challenges: &Challenges<E::ScalarField>,
        sigmas_coset_fft: &[Vec<E::ScalarField>],
    ) -> (
        AuthenticatedScalarResult<E::G1>,
        AuthenticatedScalarResult<E::G1>,
    ) {
        let num_wire_types = w.len();
        let n = pk.domain_size();

        // The check that:
        //   \prod_i [w_i(X) + beta * k_i * X + gamma] * z(X)
        // - \prod_i [w_i(X) + beta * sigma_i(X) + gamma] * z(wX) = 0
        // on the vanishing set.
        // Delay the division of Z_H(X)
        //
        // Extended permutation values
        let sigmas: Vec<E::ScalarField> = (0..num_wire_types)
            .map(|j| sigmas_coset_fft[j][i])
            .collect();

        // Compute the 1st term
        let mut result_1 = Scalar::new(challenges.alpha)
            * w.iter().enumerate().fold(z_x.clone(), |acc, (j, w)| {
                let challenge = pk.k()[j] * eval_point * challenges.beta + challenges.gamma;
                acc * (w + Scalar::new(challenge))
            });

        // Minus the 2nd term
        result_1 = result_1
            - Scalar::new(challenges.alpha)
                * w.iter()
                    .zip(sigmas.iter())
                    .fold(z_xw.clone(), |acc, (w, &sigma)| {
                        let challenge = sigma * challenges.beta + challenges.gamma;
                        acc * (w + Scalar::new(challenge))
                    });

        // The check that z(x) = 1 at point 1
        // (z(x)-1) * L1(x) * alpha^2 / Z_H(x) = (z(x)-1) * alpha^2 / (n * (x - 1))
        let denom =
            Scalar::new(E::ScalarField::from(n as u64) * (eval_point - E::ScalarField::one()));
        let result_2 =
            Scalar::new(challenges.alpha.square()) * (z_x - Scalar::one()) * denom.inverse();

        (result_1, result_2)
    }

    /// Split the quotient polynomial into `num_wire_types` polynomials.
    /// The first `num_wire_types`-1 polynomials have degree `domain_size`+1.
    ///
    /// Let t(X) be the input quotient polynomial, t_i(X) be the output
    /// splitting polynomials. t(X) = \sum_{i=0}^{num_wire_types}
    /// X^{i*(n+2)} * t_i(X)
    ///
    /// NOTE: we have a step polynomial of X^(n+2) instead of X^n as in the
    /// GWC19 paper to achieve better balance among degrees of all splitting
    /// polynomials (especially the highest-degree/last one)
    fn split_quotient_polynomial<R: CryptoRng + RngCore>(
        &self,
        prng: &mut R,
        quot_poly: &AuthenticatedDensePoly<E::G1>,
        num_wire_types: usize,
    ) -> Result<Vec<AuthenticatedDensePoly<E::G1>>, PlonkError> {
        let expected_degree = quotient_polynomial_degree(self.domain.size(), num_wire_types);
        if quot_poly.degree() != expected_degree {
            return Err(PlonkError::SnarkError(SnarkError::WrongQuotientPolyDegree(
                quot_poly.degree(),
                expected_degree,
            )));
        }
        let n = self.domain.size();

        // Compute the splitting polynomials t'_i(X) s.t. t(X) =
        // \sum_{i=0}^{num_wire_types} X^{i*(n+2)} * t'_i(X)
        // Here we effectively just divide the input polynomial into
        // chunks of degree n + 1 contiguous coefficients
        let mut split_quot_polys: Vec<AuthenticatedDensePoly<E::G1>> =
            parallelizable_slice_iter(&(0..num_wire_types).collect::<Vec<_>>())
                .map(|&i| {
                    let end = if i < num_wire_types - 1 {
                        (i + 1) * (n + 2)
                    } else {
                        quot_poly.degree() + 1
                    };

                    // Degree-(n+1) polynomial has n + 2 coefficients.
                    AuthenticatedDensePoly::from_coeffs(quot_poly.coeffs[i * (n + 2)..end].to_vec())
                })
                .collect();

        // Mask splitting polynomials t_i(X), for i in {0..num_wire_types} such that
        // their sum telescopes without boundaries
        //
        // t_i(X) = t'_i(X) - b_last_i + b_now_i * X^(n+2)
        // with t_lowest_i(X) = t_lowest_i(X) - 0 + b_now_i * X^(n+2)
        // and t_highest_i(X) = t_highest_i(X) - b_last_i
        let mut last_randomizer = self.fabric.zero_authenticated();
        let mut randomizers = self
            .fabric
            .random_shared_scalars_authenticated(num_wire_types - 1);

        split_quot_polys
            .iter_mut()
            .enumerate()
            .take(num_wire_types - 1)
            .for_each(|(i, poly)| {
                poly.coeffs[0] = &poly.coeffs[0] - &last_randomizer;
                assert_eq!(poly.degree(), n + 1);

                let next_randomizer = randomizers.pop().unwrap();
                poly.coeffs.push(next_randomizer.clone());

                last_randomizer = next_randomizer;
            });

        // Mask the highest splitting poly
        split_quot_polys[num_wire_types - 1].coeffs[0] =
            &split_quot_polys[num_wire_types - 1].coeffs[0] - last_randomizer;

        Ok(split_quot_polys)
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

/// Compute the expected degree of the quotient polynomial
#[inline]
fn quotient_polynomial_degree(domain_size: usize, num_wire_types: usize) -> usize {
    num_wire_types * (domain_size + 1) + 2
}
