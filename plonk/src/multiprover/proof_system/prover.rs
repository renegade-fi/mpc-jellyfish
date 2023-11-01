//! A multiprover implementation of the PLONK proof system
use core::ops::Neg;

use ark_ec::{pairing::Pairing, CurveGroup};
use ark_ff::{FftField, Field, One};
use ark_mpc::{
    algebra::{
        AuthenticatedDensePoly, AuthenticatedScalarResult, DensePolynomialResult, Scalar,
        ScalarResult,
    },
    MpcFabric, ResultValue,
};
use ark_poly::{
    univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, GeneralEvaluationDomain,
    Polynomial, Radix2EvaluationDomain,
};
use ark_std::rand::{CryptoRng, RngCore};
use itertools::Itertools;
use jf_relation::constants::GATE_WIDTH;
use jf_utils::par_utils::parallelizable_slice_iter;
use rayon::prelude::*;

use crate::{
    constants::domain_size_ratio,
    errors::{PlonkError, SnarkError},
    multiprover::{
        primitives::{MultiproverKZG, MultiproverKzgCommitment},
        proof_system::structs::*,
    },
    proof_system::structs::{CommitKey, ProvingKey},
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
        circuit: &C,
    ) -> Result<(MpcCommitmentsAndPolys<E>, AuthenticatedDensePoly<E::G1>), PlonkError> {
        let wire_polys: Vec<AuthenticatedDensePoly<E::G1>> = circuit
            .compute_wire_polynomials()?
            .into_iter()
            .map(|poly| self.mask_polynomial(&poly, 1))
            .collect();
        let wires_poly_comms = MultiproverKZG::batch_commit(ck, &wire_polys)?;
        let pub_input_poly = circuit.compute_pub_input_polynomial()?;

        Ok(((wires_poly_comms, wire_polys), pub_input_poly))
    }

    /// Round 2: Compute and commit the permutation grand product polynomial
    /// Return the grand product polynomial and its commitment
    pub(crate) fn run_2nd_round<C: MpcArithmetization<E::G1>>(
        &self,
        ck: &CommitKey<E>,
        cs: &C,
        challenges: &MpcChallenges<E::G1>,
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
        pks: &ProvingKey<E>,
        challenges: &MpcChallenges<E::G1>,
        online_oracles: &MpcOracles<E::G1>,
        num_wire_types: usize,
    ) -> Result<MpcCommitmentsAndPolys<E>, PlonkError> {
        let quot_poly =
            self.compute_quotient_polynomial(challenges, pks, online_oracles, num_wire_types)?;
        let split_quot_polys = self.split_quotient_polynomial(prng, &quot_poly, num_wire_types)?;
        let split_quot_poly_comms = MultiproverKZG::batch_commit(ck, &split_quot_polys)?;

        Ok((split_quot_poly_comms, split_quot_polys))
    }

    /// Round 4: Compute then openings needed for the linearization polynomial
    /// and evaluate polynomials to be opened.
    ///
    /// The linearization polynomial reduces the number of openings needed
    /// by linearizing the verification equation with existing openings.
    /// This allows the verifier to use the linear properties of the
    /// commitment scheme to compute the expected opening result
    pub(crate) fn compute_evaluations(
        &self,
        pk: &ProvingKey<E>,
        challenges: &MpcChallenges<E::G1>,
        online_oracles: &MpcOracles<E::G1>,
        num_wire_types: usize,
    ) -> MpcProofEvaluations<E::G1> {
        let wires_evals: Vec<AuthenticatedScalarResult<E::G1>> =
            parallelizable_slice_iter(&online_oracles.wire_polys)
                .map(|poly| poly.eval(&challenges.zeta))
                .collect();

        let wire_sigma_evals: Vec<ScalarResult<E::G1>> = parallelizable_slice_iter(&pk.sigmas)
            .take(num_wire_types - 1)
            .map(|poly| eval_poly_on_result(&challenges.zeta, poly.clone(), &self.fabric))
            .collect();

        let perm_next_eval = online_oracles
            .prod_perm_poly
            .eval(&(&challenges.zeta * Scalar::new(self.domain.group_gen)));

        let wire_evals_open = AuthenticatedScalarResult::open_authenticated_batch(&wires_evals)
            .into_iter()
            .map(|o| o.value)
            .collect();
        let perm_next_eval_open = perm_next_eval.open_authenticated().value;

        MpcProofEvaluations {
            wires_evals: wire_evals_open,
            wire_sigma_evals,
            perm_next_eval: perm_next_eval_open,
        }
    }

    /// Compute linearization polynomial (excluding the quotient part)
    /// i.e. the first four lines in round 5 as described in the paper
    pub(crate) fn compute_non_quotient_component_for_lin_poly(
        &self,
        alpha_base: &ScalarResult<E::G1>,
        pk: &ProvingKey<E>,
        challenges: &MpcChallenges<E::G1>,
        online_oracles: &MpcOracles<E::G1>,
        poly_evals: &MpcProofEvaluations<E::G1>,
    ) -> Result<AuthenticatedDensePoly<E::G1>, PlonkError> {
        let r_circ = self.compute_lin_poly_circuit_contribution(pk, &poly_evals.wires_evals);
        let r_perm = self.compute_lin_poly_copy_constraint_contribution(
            pk,
            challenges,
            poly_evals,
            &online_oracles.prod_perm_poly,
        );

        let mut lin_poly = r_circ + r_perm;
        Ok(lin_poly * alpha_base)
    }

    // Compute the Quotient part of the linearization polynomial:
    //
    // -Z_H(x) * [t1(X) + x^{n+2} * t2(X) + ... + x^{(num_wire_types-1)*(n+2)} *
    // t_{num_wire_types}(X)]
    pub(crate) fn compute_quotient_component_for_lin_poly(
        &self,
        domain_size: usize,
        zeta: ScalarResult<E::G1>,
        quot_polys: &[AuthenticatedDensePoly<E::G1>],
    ) -> Result<AuthenticatedDensePoly<E::G1>, PlonkError> {
        // Compute the term -Z_H(\zeta) and \zeta^{n+2}
        let vanish_eval = zeta.pow(domain_size as u64) - Scalar::one();
        let zeta_to_n_plus_2 = (&vanish_eval + Scalar::one()) * &zeta * &zeta;

        // In this term of the linearization polynomial we take a linear combination
        // of the split quotient polynomials, where the coefficients are powers of
        // \zeta^{n+2}
        let mut r_quot = quot_polys.first().ok_or(PlonkError::IndexError)?.clone();
        let mut coeff = self.fabric.one();
        for poly in quot_polys.iter().skip(1) {
            coeff = coeff * &zeta_to_n_plus_2;
            r_quot = r_quot + poly * &coeff;
        }

        Ok(&r_quot * &vanish_eval.neg())
    }

    /// Compute (aggregated) polynomial opening proofs at point `zeta` and
    /// `zeta * domain_generator`
    pub(crate) fn compute_opening_proofs(
        &self,
        ck: &CommitKey<E>,
        pk: &ProvingKey<E>,
        zeta: &ScalarResult<E::G1>,
        v: &ScalarResult<E::G1>,
        online_oracles: &MpcOracles<E::G1>,
        lin_poly: &AuthenticatedDensePoly<E::G1>,
    ) -> Result<(MultiproverKzgCommitment<E>, MultiproverKzgCommitment<E>), PlonkError> {
        // Combine all polynomials in a random linear combination parameterized by `v`
        let mut batched_poly = lin_poly.clone();
        let mut coeff = v.clone();

        // Accumulate all wiring polynomials in the linear combination
        for poly in online_oracles.wire_polys.iter() {
            batched_poly = batched_poly + poly * &coeff;
            coeff = &coeff * v;
        }

        // Accumulate all the permutation polynomials (except the last one)
        // into the linear combination. The last one is
        // implicitly included in the linearization
        for poly in pk.sigmas.iter().take(pk.sigmas.len() - 1) {
            batched_poly = batched_poly + mul_poly_result(poly.clone(), &coeff, &self.fabric);
            coeff = coeff * v;
        }

        // Divide by X - \zeta
        let divisor = DensePolynomialResult::from_coeffs(vec![-zeta.clone(), self.fabric.one()]);
        let witness_poly = batched_poly / &divisor;

        let commitment = MultiproverKZG::commit(ck, &witness_poly).map_err(PlonkError::PCSError)?;

        // Combine the permutation polynomials of the proving instances
        let mut coeff = v.clone();
        let mut batched_poly = &online_oracles.prod_perm_poly;

        // Divide by X - \zeta
        let witness_poly = batched_poly / &divisor;
        let shifted_commitment = MultiproverKZG::commit(ck, &witness_poly)?;

        Ok((commitment, shifted_commitment))
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
        challenges: &MpcChallenges<E::G1>,
        pk: &ProvingKey<E>,
        online_oracles: &MpcOracles<E::G1>,
        num_wire_types: usize,
    ) -> Result<AuthenticatedDensePoly<E::G1>, PlonkError> {
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
        let mut alpha_base = self.fabric.one();
        let alpha_3 = challenges.alpha.pow(3);

        // The coset we use to compute the quotient polynomial
        let coset = self
            .quot_domain
            .get_coset(E::ScalarField::GENERATOR)
            .unwrap();

        // Compute evaluations of the selectors, permutations, and wiring polynomials
        let selectors_coset_fft: Vec<Vec<E::ScalarField>> =
            parallelizable_slice_iter(&pk.selectors)
                .map(|poly| coset.fft(poly.coeffs()))
                .collect();

        let sigmas_coset_fft: Vec<Vec<E::ScalarField>> = parallelizable_slice_iter(&pk.sigmas)
            .map(|poly| coset.fft(poly.coeffs()))
            .collect();

        let wire_polys_coset_fft: Vec<Vec<AuthenticatedScalarResult<E::G1>>> = online_oracles
            .wire_polys
            .iter()
            .map(|poly| AuthenticatedScalarResult::fft_with_domain(&poly.coeffs, coset))
            .collect();

        // Compute the evaluations of the z(x) polynomials representing partial products
        // of the larger grand product that argues copy constraints
        let prod_perm_poly_coset_fft = AuthenticatedScalarResult::fft_with_domain(
            &online_oracles.prod_perm_poly.coeffs,
            coset,
        );
        let pub_input_poly_coset_fft = AuthenticatedScalarResult::fft_with_domain(
            &online_oracles.pub_input_poly.coeffs,
            coset,
        );

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
                    let (t_perm_1, t_perm_2) = Self::compute_quotient_copy_constraint_contribution(
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
            *a = &*a + &alpha_base * b;
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
        challenges: &MpcChallenges<E::G1>,
        sigmas_coset_fft: &[Vec<E::ScalarField>],
    ) -> (
        AuthenticatedScalarResult<E::G1>,
        AuthenticatedScalarResult<E::G1>,
    ) {
        let num_wire_types = w.len();
        let n = pk.domain_size();
        let eval_point = Scalar::new(eval_point);

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
        let mut result_1 = &challenges.alpha
            * w.iter().enumerate().fold(z_x.clone(), |acc, (j, w)| {
                let challenge =
                    Scalar::new(pk.k()[j]) * eval_point * &challenges.beta + &challenges.gamma;
                acc * (w + challenge)
            });

        // Minus the 2nd term
        result_1 = result_1
            - &challenges.alpha
                * w.iter()
                    .zip(sigmas.iter())
                    .fold(z_xw.clone(), |acc, (w, &sigma)| {
                        let challenge = Scalar::new(sigma) * &challenges.beta + &challenges.gamma;
                        acc * (w + challenge)
                    });

        // The check that z(x) = 1 at point 1
        // (z(x)-1) * L1(x) * alpha^2 / Z_H(x) = (z(x)-1) * alpha^2 / (n * (x - 1))
        let denom = Scalar::from(n as u64) * (eval_point - Scalar::one());
        let result_2 = challenges.alpha.pow(2) * (z_x - Scalar::one()) * denom.inverse();

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

    // Compute the circuit part of the linearization polynomial
    fn compute_lin_poly_circuit_contribution(
        &self,
        pk: &ProvingKey<E>,
        w_evals: &[ScalarResult<E::G1>],
    ) -> DensePolynomialResult<E::G1> {
        // The selectors in order: q_lc, q_mul, q_hash, q_o, q_c, q_ecc
        let q_lc = &pk.selectors[..GATE_WIDTH];
        let q_mul = &pk.selectors[GATE_WIDTH..GATE_WIDTH + 2];
        let q_hash = &pk.selectors[GATE_WIDTH + 2..2 * GATE_WIDTH + 2];
        let q_o = &pk.selectors[2 * GATE_WIDTH + 2];
        let q_c = &pk.selectors[2 * GATE_WIDTH + 3];
        let q_ecc = &pk.selectors[2 * GATE_WIDTH + 4];

        // Note we don't need to compute the constant term of the polynomial.
        mul_poly_result(q_lc[0].clone(), &w_evals[0], &self.fabric)
            + mul_poly_result(q_lc[1].clone(), &w_evals[1], &self.fabric)
            + mul_poly_result(q_lc[2].clone(), &w_evals[2], &self.fabric)
            + mul_poly_result(q_lc[3].clone(), &w_evals[3], &self.fabric)
            + mul_poly_result(q_mul[0].clone(), &(&w_evals[0] * &w_evals[1]), &self.fabric)
            + mul_poly_result(q_mul[1].clone(), &(&w_evals[2] * &w_evals[3]), &self.fabric)
            + mul_poly_result(q_hash[0].clone(), &w_evals[0].pow(5), &self.fabric)
            + mul_poly_result(q_hash[1].clone(), &w_evals[1].pow(5), &self.fabric)
            + mul_poly_result(q_hash[2].clone(), &w_evals[2].pow(5), &self.fabric)
            + mul_poly_result(q_hash[3].clone(), &w_evals[3].pow(5), &self.fabric)
            + mul_poly_result(
                q_ecc.clone(),
                &(&w_evals[0] * &w_evals[1] * &w_evals[2] * &w_evals[3] * &w_evals[4]),
                &self.fabric,
            )
            + mul_poly_result(q_o.clone(), &(-&w_evals[4]), &self.fabric)
            + q_c.clone()
    }

    /// Compute the wire permutation part of the linearization polynomial
    ///
    /// Here we linearize with respect to the polynomial z(X) -- representing
    /// the partial evaluations of the grand product -- and S_{num_wires}(X) --
    /// the permutation polynomial of the last wire
    fn compute_lin_poly_copy_constraint_contribution(
        &self,
        pk: &ProvingKey<E>,
        challenges: &MpcChallenges<E::G1>,
        poly_evals: &MpcProofEvaluations<E::G1>,
        prod_perm_poly: &AuthenticatedDensePoly<E::G1>,
    ) -> AuthenticatedDensePoly<E::G1> {
        let dividend = challenges.zeta.pow(pk.domain_size() as u64) - Scalar::one();
        let divisor = Scalar::from(pk.domain_size() as u32) * (&challenges.zeta - Scalar::one());
        let lagrange_1_eval = dividend * divisor.inverse();

        // Compute the coefficient of z(X)
        let coeff = poly_evals.wires_evals.iter().enumerate().fold(
            challenges.alpha.clone(),
            |acc, (j, wire_eval)| {
                acc * (wire_eval
                    + &challenges.beta * Scalar::new(pk.vk.k[j]) * &challenges.zeta
                    + &challenges.gamma)
            },
        ) + challenges.alpha.pow(2) * lagrange_1_eval;
        let mut r_perm = &coeff * prod_perm_poly;

        // Compute the coefficient of the last sigma wire permutation polynomial
        let num_wire_types = poly_evals.wires_evals.len();
        let coeff = -poly_evals
            .wires_evals
            .iter()
            .take(num_wire_types - 1)
            .zip(poly_evals.wire_sigma_evals.iter())
            .fold(
                // We multiply beta here to isolate the S_{num_wires}(X) term as a formal
                // indeterminate
                &challenges.alpha * &challenges.beta * &poly_evals.perm_next_eval,
                |acc, (wire_eval, sigma_eval)| {
                    acc * (wire_eval + &challenges.beta * sigma_eval + &challenges.gamma)
                },
            );

        r_perm + mul_poly_result(pk.sigmas[num_wire_types - 1].clone(), &coeff, &self.fabric)
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

/// Evaluate a public polynomial on a result in the MPC fabric
///
/// We can't implement this method on `DensePolynomial` because `ark-mpc`
/// does not own that type. So instead we broker evaluation through a helper
/// method
fn eval_poly_on_result<C: CurveGroup>(
    point: &ScalarResult<C>,
    poly: DensePolynomial<C::ScalarField>,
    fabric: &MpcFabric<C>,
) -> ScalarResult<C> {
    // For this we create a new gate op that resolves when the eval point is
    // computed
    fabric.new_gate_op(vec![point.id()], move |mut args| {
        let point: Scalar<C> = args.pop().unwrap().into();
        let res = poly.evaluate(&point.inner());

        ResultValue::Scalar(Scalar::new(res))
    })
}

/// Multiplies a polynomial by a `ScalarResult`
pub fn mul_poly_result<C: CurveGroup>(
    poly: DensePolynomial<C::ScalarField>,
    scaling_factor: &ScalarResult<C>,
    fabric: &MpcFabric<C>,
) -> DensePolynomialResult<C> {
    // Allocate a gate to scale each coefficient individually
    let arity = poly.coeffs.len();
    let coeffs = fabric.new_batch_gate_op(vec![scaling_factor.id()], arity, move |mut args| {
        let scaling_factor: Scalar<C> = args.pop().unwrap().into();

        poly.coeffs
            .into_iter()
            .map(Scalar::new)
            .map(|c| c * scaling_factor)
            .map(ResultValue::Scalar)
            .collect_vec()
    });

    DensePolynomialResult::from_coeffs(coeffs)
}
