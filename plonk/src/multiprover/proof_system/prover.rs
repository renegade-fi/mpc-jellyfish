//! A multiprover implementation of the PLONK proof system
use core::{iter, ops::Neg};

use ark_ec::{pairing::Pairing, CurveGroup};
use ark_ff::{FftField, Field, One, Zero};
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
use itertools::Itertools;
use mpc_relation::constants::GATE_WIDTH;

use crate::{
    constants::domain_size_ratio,
    errors::PlonkError,
    multiprover::{
        primitives::{MultiproverKZG, MultiproverKzgCommitment},
        proof_system::structs::*,
    },
    proof_system::structs::{CommitKey, ProvingKey},
};

use super::MpcArithmetization;

// -------------------------
// | Prover Implementation |
// -------------------------

/// A type alias for a bundle of commitments and polynomials
/// TODO: Remove this lint allowance
#[allow(type_alias_bounds)]
type MpcCommitmentsAndPolys<E: Pairing> =
    (Vec<MultiproverKzgCommitment<E>>, Vec<AuthenticatedDensePoly<E::G1>>);

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

        Ok(Self { domain, quot_domain, fabric })
    }

    /// Round 1:
    /// 1. Compute and commit wire witness polynomials.
    /// 2. Compute public input polynomial.
    /// Return the wire witness polynomials and their commitments,
    /// also return the public input polynomial.
    ///
    /// `mask` indicates whether to mask the wire polynomials, we do not mask in
    /// some tests to make proofs deterministic so that they are comparable
    pub(crate) fn run_1st_round<C: MpcArithmetization<E::G1>>(
        &self,
        ck: &CommitKey<E>,
        circuit: &C,
        mask: bool,
    ) -> Result<(MpcCommitmentsAndPolys<E>, AuthenticatedDensePoly<E::G1>), PlonkError> {
        let wire_polys: Vec<AuthenticatedDensePoly<E::G1>> = circuit
            .compute_wire_polynomials()?
            .into_iter()
            .map(|poly| self.mask_polynomial(&poly, 1 /* hiding_bound */, !mask))
            .collect();
        let wires_poly_comms = MultiproverKZG::batch_commit(ck, &wire_polys)?;
        let pub_input_poly = circuit.compute_pub_input_polynomial()?;

        Ok(((wires_poly_comms, wire_polys), pub_input_poly))
    }

    /// Round 2: Compute and commit the permutation grand product polynomial
    /// Return the grand product polynomial and its commitment
    ///
    /// The `mask` parameter indicates whether to mask the polynomial, we do not
    /// mask in some tests to make proofs deterministic
    pub(crate) fn run_2nd_round<C: MpcArithmetization<E::G1>>(
        &self,
        ck: &CommitKey<E>,
        cs: &C,
        challenges: &MpcChallenges<E::G1>,
        mask: bool,
    ) -> Result<(MultiproverKzgCommitment<E>, AuthenticatedDensePoly<E::G1>), PlonkError> {
        let prod_perm_poly = self.mask_polynomial(
            &cs.compute_prod_permutation_polynomial(&challenges.beta, &challenges.gamma)?,
            2, // hiding_degree
            !mask,
        );
        let prod_perm_comm = MultiproverKZG::commit(ck, &prod_perm_poly)?;

        Ok((prod_perm_comm, prod_perm_poly))
    }

    /// Round 3: Return the split quotient polynomials and their commitments
    /// Note that the first `num_wire_types`-1 split quotient polynomials
    /// have degree `domain_size`+1.
    pub(crate) fn run_3rd_round(
        &self,
        ck: &CommitKey<E>,
        pks: &ProvingKey<E>,
        challenges: &MpcChallenges<E::G1>,
        online_oracles: &MpcOracles<E::G1>,
        num_wire_types: usize,
    ) -> Result<MpcCommitmentsAndPolys<E>, PlonkError> {
        let mut quot_poly =
            self.compute_quotient_polynomial(challenges, pks, online_oracles, num_wire_types)?;
        let split_quot_polys = self.split_quotient_polynomial(&mut quot_poly, num_wire_types)?;
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
            online_oracles.wire_polys.iter().map(|poly| poly.eval(&challenges.zeta)).collect();

        let wire_sigma_evals: Vec<ScalarResult<E::G1>> = pk
            .sigmas
            .iter()
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
    ) -> AuthenticatedDensePoly<E::G1> {
        let r_circ = self.compute_lin_poly_circuit_contribution(pk, &poly_evals.wires_evals);
        let r_perm = self.compute_lin_poly_copy_constraint_contribution(
            pk,
            challenges,
            poly_evals,
            &online_oracles.prod_perm_poly,
        );

        let mut lin_poly = r_circ + r_perm;
        lin_poly * alpha_base
    }

    // Compute the Quotient part of the linearization polynomial:
    //
    // -Z_H(x) * [t1(X) + x^{n+2} * t2(X) + ... + x^{(num_wire_types-1)*(n+2)} *
    // t_{num_wire_types}(X)]
    pub(crate) fn compute_quotient_component_for_lin_poly(
        &self,
        domain_size: usize,
        zeta: &ScalarResult<E::G1>,
        quot_polys: &[AuthenticatedDensePoly<E::G1>],
    ) -> Result<AuthenticatedDensePoly<E::G1>, PlonkError> {
        // Compute the term -Z_H(\zeta) and \zeta^{n+2}
        let vanish_eval = zeta.pow(domain_size as u64) - Scalar::one();
        let zeta_to_n_plus_2 = (&vanish_eval + Scalar::one()) * zeta * zeta;

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
        let divisor = DensePolynomialResult::from_coeffs(vec![-zeta, self.fabric.one()]);
        let witness_poly = batched_poly / &divisor;

        let commitment = MultiproverKZG::commit(ck, &witness_poly).map_err(PlonkError::PCSError)?;

        // Divide by X - \omega * \zeta
        let gen = Scalar::new(self.domain.group_gen);
        let divisor = DensePolynomialResult::from_coeffs(vec![-zeta * gen, self.fabric.one()]);
        let witness_poly = &online_oracles.prod_perm_poly / &divisor;

        let shifted_commitment = MultiproverKZG::commit(ck, &witness_poly)?;

        Ok((commitment, shifted_commitment))
    }
}

/// Private helper methods
impl<E: Pairing> MpcProver<E> {
    /// Mask the polynomial by adding a random polynomial of degree
    /// `hiding_bound` to it
    ///
    /// We optionally allow the operation to be disabled for testing purposes
    fn mask_polynomial(
        &self,
        poly: &AuthenticatedDensePoly<E::G1>,
        hiding_bound: usize,
        disabled: bool,
    ) -> AuthenticatedDensePoly<E::G1> {
        #[cfg(not(test))]
        {
            if disabled {
                panic!("cannot disable proof randomization outside of tests")
            }
        }
        #[cfg(test)]
        {
            if disabled {
                return poly.clone();
            }
        }

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

        // The coset we use to compute the quotient polynomial
        let coset = self.quot_domain.get_coset(E::ScalarField::GENERATOR).unwrap();

        // Compute evaluations of the selectors, permutations, and wiring polynomials
        let selectors_coset_fft: Vec<Vec<E::ScalarField>> =
            pk.selectors.iter().map(|poly| coset.fft(poly.coeffs())).collect();

        let sigmas_coset_fft: Vec<Vec<E::ScalarField>> =
            pk.sigmas.iter().map(|poly| coset.fft(poly.coeffs())).collect();

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

        // The contribution of the gate constraints to the quotient polynomial
        // evaluations
        let t_circ = self.compute_quotient_circuit_contribution(
            &wire_polys_coset_fft,
            &pub_input_poly_coset_fft,
            &selectors_coset_fft,
        );

        // The contribution of the permutation argument (which validates the copy
        // constraints) to the quotient polynomial
        let (t_perms1, t_perms2) = self.compute_quotient_copy_constraint_contribution(
            num_wire_types,
            domain_size_ratio,
            pk,
            &wire_polys_coset_fft,
            &prod_perm_poly_coset_fft,
            challenges,
            &sigmas_coset_fft,
        );

        // Combine the gate and copy constraints
        let zh_inv_values = (0..m).map(|i| z_h_inv[i % domain_size_ratio]).collect_vec();
        let t1 = AuthenticatedScalarResult::batch_add(&t_circ, &t_perms1);
        let t1_mul = AuthenticatedScalarResult::batch_mul_constant(&t1, &zh_inv_values);

        let quot_poly_evals = AuthenticatedScalarResult::batch_add(&t1_mul, &t_perms2);

        // Compute the coefficient form of the quotient polynomial
        let coeffs = AuthenticatedScalarResult::ifft_with_domain(&quot_poly_evals, coset);
        Ok(AuthenticatedDensePoly::from_coeffs(coeffs))
    }

    /// Compute the contribution of the gate constraint to the quotient
    /// polynomial
    fn compute_quotient_circuit_contribution(
        &self,
        w: &[Vec<AuthenticatedScalarResult<E::G1>>],
        pi: &[AuthenticatedScalarResult<E::G1>],
        selectors_coset_fft: &[Vec<E::ScalarField>],
    ) -> Vec<AuthenticatedScalarResult<E::G1>> {
        // Selectors in order: q_lc, q_mul, q_hash, q_o, q_c, q_ecc
        let q_lc0 = &selectors_coset_fft[0];
        let q_lc1 = &selectors_coset_fft[1];
        let q_lc2 = &selectors_coset_fft[2];
        let q_lc3 = &selectors_coset_fft[3];
        let q_mul0 = &selectors_coset_fft[4];
        let q_mul1 = &selectors_coset_fft[5];
        let q_hash0 = &selectors_coset_fft[6];
        let q_hash1 = &selectors_coset_fft[7];
        let q_hash2 = &selectors_coset_fft[8];
        let q_hash3 = &selectors_coset_fft[9];
        let q_o = &selectors_coset_fft[10];
        let q_c = &selectors_coset_fft[11];
        let q_ecc = &selectors_coset_fft[12];

        // Compute the batch contribution of each selector
        let lc0 = self.compute_selector_batch(q_lc0, vec![&w[0]]);
        let lc1 = self.compute_selector_batch(q_lc1, vec![&w[1]]);
        let lc2 = self.compute_selector_batch(q_lc2, vec![&w[2]]);
        let lc3 = self.compute_selector_batch(q_lc3, vec![&w[3]]);
        let mul0 = self.compute_selector_batch(q_mul0, vec![&w[0], &w[1]]);
        let mul1 = self.compute_selector_batch(q_mul1, vec![&w[2], &w[3]]);
        let ecc = self.compute_selector_batch(q_ecc, vec![&w[0], &w[1], &w[2], &w[3]]);
        let hash0 = self.compute_hash_selector_batch(q_hash0, &w[0]);
        let hash1 = self.compute_hash_selector_batch(q_hash1, &w[1]);
        let hash2 = self.compute_hash_selector_batch(q_hash2, &w[2]);
        let hash3 = self.compute_hash_selector_batch(q_hash3, &w[3]);

        let neg_out = AuthenticatedScalarResult::batch_neg(&w[4]);
        let output = self.compute_selector_batch(q_o, vec![&neg_out]);

        let constants = q_c.iter().copied().map(Scalar::new).collect_vec();
        let input = AuthenticatedScalarResult::batch_add_constant(pi, &constants);

        // Sum up all the selectors
        element_wise_sum(&[
            lc0, lc1, lc2, lc3, mul0, mul1, ecc, hash0, hash1, hash2, hash3, output, input,
        ])
    }

    /// Compute the batch circuit contribution of a selector times the product
    /// of its wires, that is, for each entry we have     
    ///     out[i] = sel[i] * w[i][0] * w[i][1] * ... * w[i][m-1]
    ///
    /// We use a sparse multiplication here and splice the sparse result with
    /// zeros to construct the full result
    fn compute_selector_batch(
        &self,
        selectors: &[E::ScalarField],
        wires: Vec<&[AuthenticatedScalarResult<E::G1>]>,
    ) -> Vec<AuthenticatedScalarResult<E::G1>> {
        let n = selectors.len();
        let m = wires.len();
        let mut nonzero_indices = Vec::new();

        let mut nonzero_sel: Vec<Scalar<E::G1>> = Vec::new();
        let mut nonzero_wires: Vec<Vec<AuthenticatedScalarResult<E::G1>>> = vec![vec![]; m];
        for (i, selector) in selectors.iter().enumerate() {
            if !selector.is_zero() {
                wires.iter().enumerate().for_each(|(j, w)| nonzero_wires[j].push(w[i].clone()));
                nonzero_sel.push(Scalar::new(*selector));
                nonzero_indices.push(i);
            }
        }

        // Multiply the non-zero wires and selectors
        let wire_product = element_wise_product(&nonzero_wires);
        let mul_res = AuthenticatedScalarResult::batch_mul_constant(&wire_product, &nonzero_sel);

        self.splice_with_zeros(n, mul_res, nonzero_indices)
    }

    /// Compute the contribution of a hash selector to the quotient polynomial
    ///
    /// This differs from the method above in that we do not take the product
    /// but instead raise the given wire values to their fifth power
    fn compute_hash_selector_batch(
        &self,
        selectors: &[E::ScalarField],
        wires: &[AuthenticatedScalarResult<E::G1>],
    ) -> Vec<AuthenticatedScalarResult<E::G1>> {
        let n = selectors.len();
        let mut nonzero_indices = Vec::new();

        let mut nonzero_sel: Vec<Scalar<E::G1>> = Vec::new();
        let mut nonzero_wires: Vec<AuthenticatedScalarResult<E::G1>> = Vec::new();
        for (i, (selector, wire)) in selectors.iter().zip(wires.iter()).enumerate() {
            if !selector.is_zero() {
                nonzero_wires.push(wire.pow(5));
                nonzero_sel.push(Scalar::new(*selector));
                nonzero_indices.push(i);
            }
        }

        // Multiply the non-zero wires and selectors
        let mul_res = AuthenticatedScalarResult::batch_mul_constant(&nonzero_wires, &nonzero_sel);
        self.splice_with_zeros(n, mul_res, nonzero_indices)
    }

    /// Splice the given vector with a vector of zeros placing the values of the
    /// input vector at the indices specified by `nonzero_indices`
    fn splice_with_zeros(
        &self,
        res_len: usize,
        values: Vec<AuthenticatedScalarResult<E::G1>>,
        nonzero_indices: Vec<usize>,
    ) -> Vec<AuthenticatedScalarResult<E::G1>> {
        let zero = self.fabric.zero_authenticated();

        // Splice together the multiplication results with zeros
        let mut res = Vec::with_capacity(res_len);
        let mut cursor = 0;
        for (nonzero_index, value) in nonzero_indices.iter().zip(values.into_iter()) {
            let num_zeros = *nonzero_index - cursor;
            res.extend(&mut iter::repeat(zero.clone()).take(num_zeros));
            res.push(value);
            cursor = *nonzero_index + 1;
        }

        // Add any extra zeros at the end
        let num_zeros = res_len - cursor;
        res.extend(&mut iter::repeat(zero).take(num_zeros));

        res
    }

    /// Computes the contribution of the copy constraint to the quotient
    /// polynomials.
    ///
    /// The first term is: the wire values summed with a random affine
    /// transformation applied to their indices; minus the same wire values with
    /// the same random affine transformation applied to their *permutation*
    ///
    /// The second term here corresponds to the grand product check that
    /// validates the permutation
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn compute_quotient_copy_constraint_contribution(
        &self,
        num_wire_types: usize,
        domain_size_ratio: usize,
        pk: &ProvingKey<E>,
        wire_evals: &[Vec<AuthenticatedScalarResult<E::G1>>],
        prod_perm_poly_coset_evals: &[AuthenticatedScalarResult<E::G1>],
        challenges: &MpcChallenges<E::G1>,
        sigmas_coset_fft: &[Vec<E::ScalarField>],
    ) -> (Vec<AuthenticatedScalarResult<E::G1>>, Vec<AuthenticatedScalarResult<E::G1>>) {
        let n = pk.domain_size();
        let m = self.quot_domain.size();

        // Tile an array with the challenges
        let alphas = vec![challenges.alpha.clone(); m];
        let betas = vec![challenges.beta.clone(); m];
        let gammas = vec![challenges.gamma.clone(); m];

        // Construct the evaluations shifted by the coset generators
        let coset_generators = pk.k().iter().copied().collect_vec();
        let eval_points =
            (0..m).map(|i| self.quot_domain.element(i) * E::ScalarField::GENERATOR).collect_vec();

        let mut all_evals = Vec::with_capacity(num_wire_types);
        for generator in coset_generators.iter() {
            let mut evals = Vec::with_capacity(m);
            for point in eval_points.iter() {
                evals.push(*point * *generator);
            }

            all_evals.push(evals);
        }

        // --- First Term --- //
        let mut product = Self::compute_shifted_product(&betas, &gammas, wire_evals, &all_evals);
        product = AuthenticatedScalarResult::batch_mul(&product, prod_perm_poly_coset_evals);
        let first_term_evals = AuthenticatedScalarResult::batch_mul_public(&product, &alphas);

        // --- Second Term --- //
        let mut product =
            Self::compute_shifted_product(&betas, &gammas, wire_evals, sigmas_coset_fft);

        let initial_shifted_evals = (0..m)
            .map(|i| prod_perm_poly_coset_evals[(i + domain_size_ratio) % m].clone())
            .collect_vec();
        product = AuthenticatedScalarResult::batch_mul(&product, &initial_shifted_evals);
        let second_term_evals = AuthenticatedScalarResult::batch_mul_public(&product, &alphas);

        // Combine the first and second terms
        let res1 = AuthenticatedScalarResult::batch_sub(&first_term_evals, &second_term_evals);

        // -- Third Term -- //
        let n_scalar = Scalar::from(n);
        let alpha_square = challenges.alpha.pow(2);
        let alphas = vec![alpha_square; m];
        let ones = vec![self.fabric.one(); m];

        let denominators = eval_points
            .into_iter()
            .map(Scalar::new)
            .map(|p| n_scalar * (p - Scalar::one()))
            .map(|s| s.inverse())
            .collect_vec();
        let scaling_factors = ScalarResult::batch_mul_constant(&alphas, &denominators);

        let numerators =
            AuthenticatedScalarResult::batch_sub_public(prod_perm_poly_coset_evals, &ones);
        let res2 = AuthenticatedScalarResult::batch_mul_public(&numerators, &scaling_factors);

        (res1, res2)
    }

    /// Computes the product of a set of wire values with random affine
    /// transformations added to their indices
    fn compute_shifted_product(
        beta: &[ScalarResult<E::G1>],
        gamma: &[ScalarResult<E::G1>],
        wire_evals: &[Vec<AuthenticatedScalarResult<E::G1>>],
        eval_domain: &[Vec<E::ScalarField>],
    ) -> Vec<AuthenticatedScalarResult<E::G1>> {
        let num_wire_types = wire_evals.len();

        // Compute the affine transformations of the given domain values
        let mut challenge_lcs = Vec::with_capacity(num_wire_types);
        for eval_coset in eval_domain.iter() {
            let scalar_coset = eval_coset.iter().copied().map(Scalar::new).collect_vec();
            let scaled_eval = ScalarResult::batch_mul_constant(beta, &scalar_coset);
            let affine_eval = ScalarResult::batch_add(&scaled_eval, gamma);

            challenge_lcs.push(affine_eval);
        }

        // Add these transformed domain values to the wire values
        let mut product_terms = Vec::with_capacity(num_wire_types);
        for (wire_eval, challenge_lc) in wire_evals.iter().zip(challenge_lcs.iter()) {
            let term = AuthenticatedScalarResult::batch_add_public(wire_eval, challenge_lc);
            product_terms.push(term);
        }

        // Reduce via product
        element_wise_product(&product_terms)
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
    fn split_quotient_polynomial(
        &self,
        quot_poly: &mut AuthenticatedDensePoly<E::G1>,
        num_wire_types: usize,
    ) -> Result<Vec<AuthenticatedDensePoly<E::G1>>, PlonkError> {
        let expected_degree = quotient_polynomial_degree(self.domain.size(), num_wire_types);
        // We cannot truncate leading zeros like the single-prover implementation
        // because the coeffs of the polynomial are masked. Instead, we truncate the
        // degree to the expected degree, which is the effectual change of
        // truncating leading zeros *in this case*
        quot_poly.coeffs.truncate(expected_degree + 1);
        let n = self.domain.size();

        // Compute the splitting polynomials t'_i(X) s.t. t(X) =
        // \sum_{i=0}^{num_wire_types} X^{i*(n+2)} * t'_i(X)
        // Here we effectively just divide the input polynomial into
        // chunks of degree n + 1 contiguous coefficients
        let mut split_quot_polys: Vec<AuthenticatedDensePoly<E::G1>> = (0..num_wire_types)
            .map(|i| {
                let end =
                    if i < num_wire_types - 1 { (i + 1) * (n + 2) } else { quot_poly.degree() + 1 };

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
        let mut randomizers = self.fabric.random_shared_scalars(num_wire_types - 1);

        split_quot_polys.iter_mut().take(num_wire_types - 1).for_each(|poly| {
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
        let r_perm = &coeff * prod_perm_poly;

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

/// Take the element-wise product of a set of vectors
pub fn element_wise_product<C: CurveGroup>(
    vectors: &[Vec<AuthenticatedScalarResult<C>>],
) -> Vec<AuthenticatedScalarResult<C>> {
    assert!(!vectors.is_empty(), "must have at least one vector");
    if vectors.len() == 1 {
        return vectors[0].clone();
    }

    let n = vectors[0].len();
    for vec in vectors.iter() {
        assert_eq!(vec.len(), n, "all vectors must have the same length");
    }

    let initial = AuthenticatedScalarResult::batch_mul(&vectors[0], &vectors[1]);

    // If we choose to view the vectors as tiling the columns of matrices, each step
    // in this fold replaces the first and second columns with their
    // element-wise product
    vectors[2..].iter().fold(initial, |acc, vec| AuthenticatedScalarResult::batch_mul(&acc, vec))
}

/// Take the element-wise sum of a set of vectors
fn element_wise_sum<C: CurveGroup>(
    vectors: &[Vec<AuthenticatedScalarResult<C>>],
) -> Vec<AuthenticatedScalarResult<C>> {
    assert!(!vectors.is_empty(), "must have at least one vector");
    if vectors.len() == 1 {
        return vectors[0].clone();
    }

    let n = vectors[0].len();
    for vec in vectors.iter() {
        assert_eq!(vec.len(), n, "all vectors must have the same length");
    }

    let initial = AuthenticatedScalarResult::batch_add(&vectors[0], &vectors[1]);

    // If we choose to view the vectors as tiling the columns of matrices, each step
    // in this fold replaces the first and second columns with their
    // element-wise sum
    vectors[2..].iter().fold(initial, |acc, vec| AuthenticatedScalarResult::batch_add(&acc, vec))
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
        let point: Scalar<C> = args.next().unwrap().into();
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
    if poly.coeffs.is_empty() {
        return DensePolynomialResult::from_coeffs(vec![fabric.zero()]);
    }

    // Allocate a gate to scale each coefficient individually
    let arity = poly.coeffs.len();
    let coeffs = fabric.new_batch_gate_op(vec![scaling_factor.id()], arity, move |mut args| {
        let scaling_factor: Scalar<C> = args.next().unwrap().into();

        poly.coeffs
            .into_iter()
            .map(Scalar::new)
            .map(|c| c * scaling_factor)
            .map(ResultValue::Scalar)
            .collect_vec()
    });

    DensePolynomialResult::from_coeffs(coeffs)
}

#[cfg(test)]
pub(crate) mod test {
    use ark_ec::CurveGroup;
    use ark_ff::{One, Zero};
    use ark_mpc::{
        algebra::{AuthenticatedDensePoly, Scalar, ScalarShare},
        network::PartyId,
        offline_prep::PreprocessingPhase,
        test_helpers::{execute_mock_mpc, execute_mock_mpc_with_beaver_source},
        MpcFabric, PARTY0, PARTY1,
    };
    use ark_poly::univariate::DensePolynomial;
    use futures::{future::join_all, prelude::*};
    use itertools::Itertools;
    use jf_primitives::pcs::prelude::Commitment;
    use mpc_relation::{traits::*, PlonkCircuit};
    use rand::thread_rng;

    use crate::{
        multiprover::proof_system::{
            test_helpers::{setup_snark, TestCurve, TestGroup, TestScalar},
            MpcChallenges, MpcOracles, MpcPlonkCircuit,
        },
        proof_system::{
            prover::Prover,
            structs::{Challenges, Oracles, ProofEvaluations, ProvingKey},
        },
    };

    use super::MpcProver;

    /// A beaver source that always returns zero
    #[cfg(any(feature = "test_helpers", test))]
    struct ZeroBeaverSource {
        /// The ID of the local party
        party_id: PartyId,
    }

    impl ZeroBeaverSource {
        /// Create a new beaver source given the local party_id
        pub fn new(party_id: PartyId) -> Self {
            Self { party_id }
        }
    }

    impl<C: CurveGroup> PreprocessingPhase<C> for ZeroBeaverSource {
        fn get_mac_key_share(&self) -> Scalar<C> {
            Scalar::zero()
        }

        fn next_local_input_mask(&mut self) -> (Scalar<C>, ScalarShare<C>) {
            (Scalar::zero(), ScalarShare::new(Scalar::zero(), Scalar::zero()))
        }

        fn next_counterparty_input_mask(&mut self) -> ScalarShare<C> {
            ScalarShare::new(Scalar::zero(), Scalar::zero())
        }

        fn next_shared_bit(&mut self) -> ScalarShare<C> {
            ScalarShare::new(Scalar::zero(), Scalar::zero())
        }

        fn next_triplet(&mut self) -> (ScalarShare<C>, ScalarShare<C>, ScalarShare<C>) {
            let zero = ScalarShare::new(Scalar::zero(), Scalar::zero());
            (zero, zero, zero)
        }

        /// For the shared inverse pair, we return 1 to give a valid member of
        /// the multiplicative subgroup
        ///
        /// This means that each party holds their party ID as a shared value
        fn next_shared_inverse_pair(&mut self) -> (ScalarShare<C>, ScalarShare<C>) {
            let val = Scalar::from(self.party_id);
            let share = ScalarShare::new(val, val);
            (share, share)
        }

        fn next_shared_value(&mut self) -> ScalarShare<C> {
            ScalarShare::new(Scalar::zero(), Scalar::zero())
        }
    }

    /// Get a randomized set of challenges
    fn randomized_challenges() -> Challenges<TestScalar> {
        let mut rng = thread_rng();
        Challenges {
            alpha: Scalar::<TestGroup>::random(&mut rng).inner(),
            beta: Scalar::<TestGroup>::random(&mut rng).inner(),
            gamma: Scalar::<TestGroup>::random(&mut rng).inner(),
            zeta: Scalar::<TestGroup>::random(&mut rng).inner(),
            v: Scalar::<TestGroup>::random(&mut rng).inner(),
            u: Scalar::<TestGroup>::random(&mut rng).inner(),
            tau: Scalar::<TestGroup>::random(&mut rng).inner(),
        }
    }

    /// Allocate challenges in an MPC fabric
    fn allocate_challenges(
        challenges: &Challenges<TestScalar>,
        fabric: &MpcFabric<TestGroup>,
    ) -> MpcChallenges<TestGroup> {
        MpcChallenges {
            alpha: fabric.share_plaintext(Scalar::new(challenges.alpha), PARTY0),
            beta: fabric.share_plaintext(Scalar::new(challenges.beta), PARTY0),
            gamma: fabric.share_plaintext(Scalar::new(challenges.gamma), PARTY0),
            zeta: fabric.share_plaintext(Scalar::new(challenges.zeta), PARTY0),
            v: fabric.share_plaintext(Scalar::new(challenges.v), PARTY0),
        }
    }

    /// Allocate a set of oracles to get an `MpcOracles` instance
    fn allocate_oracles(
        oracles: &Oracles<TestScalar>,
        fabric: &MpcFabric<TestGroup>,
    ) -> MpcOracles<TestGroup> {
        MpcOracles {
            wire_polys: allocate_polys(&oracles.wire_polys, fabric),
            pub_input_poly: allocate_poly(&oracles.pub_inp_poly, fabric),
            prod_perm_poly: allocate_poly(&oracles.prod_perm_poly, fabric),
        }
    }

    /// Allocate a group of `DensePolynomial` in an MPC fabric
    fn allocate_polys(
        polys: &[DensePolynomial<TestScalar>],
        fabric: &MpcFabric<TestGroup>,
    ) -> Vec<AuthenticatedDensePoly<TestGroup>> {
        polys.iter().map(|p| allocate_poly(p, fabric)).collect_vec()
    }

    /// Allocate a `DensePolynomial` in an MPC fabric
    fn allocate_poly(
        poly: &DensePolynomial<TestScalar>,
        fabric: &MpcFabric<TestGroup>,
    ) -> AuthenticatedDensePoly<TestGroup> {
        if poly.coeffs.is_empty() {
            return AuthenticatedDensePoly::from_coeffs(vec![fabric.zero_authenticated()]);
        }

        let coeffs = fabric
            .batch_share_scalar(poly.coeffs.iter().cloned().map(Scalar::new).collect(), PARTY0);
        AuthenticatedDensePoly::from_coeffs(coeffs)
    }

    /// Generate the testing circuit in a singleprover context
    pub(crate) fn test_singleprover_circuit(
        witness: Scalar<TestGroup>,
    ) -> PlonkCircuit<TestScalar> {
        let mut circuit = PlonkCircuit::new_turbo_plonk();

        let mut res = circuit.create_variable(witness.inner()).unwrap();
        for _ in 0..10 {
            res = circuit.add_constant(res, &TestScalar::one()).unwrap();
            res = circuit.mul(res, res).unwrap();
        }

        // Multiply by zero
        let zero = circuit.mul_constant(res, &TestScalar::zero()).unwrap();
        circuit.enforce_equal(zero, circuit.zero()).unwrap();

        circuit.finalize_for_arithmetization().unwrap();
        circuit
    }

    /// Generate the testing circuit in a multiprover context
    pub(crate) fn test_multiprover_circuit(
        witness: Scalar<TestGroup>,
        fabric: &MpcFabric<TestGroup>,
    ) -> MpcPlonkCircuit<TestGroup> {
        let mut circuit = MpcPlonkCircuit::new(fabric.clone());

        let shared_witness = fabric.share_scalar(witness, PARTY0 /* sender */);
        let mut res = circuit.create_variable(shared_witness).unwrap();

        // Each loop is (res + 1) * (res + 1)
        for _ in 0..10 {
            res = circuit.add_constant(res, &TestScalar::one()).unwrap();
            res = circuit.mul(res, res).unwrap();
        }

        // Multiply by zero
        let zero = circuit.mul_constant(res, &TestScalar::zero()).unwrap();
        circuit.enforce_equal(zero, circuit.zero()).unwrap();

        circuit.finalize_for_arithmetization().unwrap();
        circuit
    }

    /// Execute an MPC with a `ZeroBeaverSource`
    ///
    /// We do this to zero out any proof-randomization that is done during the
    /// MPC so that proofs between single and multiprover are directly
    /// comparable
    async fn execute_deterministic_mpc<T, S, F>(f: F) -> T
    where
        T: Send + 'static,
        S: Future<Output = T> + Send + 'static,
        F: FnMut(MpcFabric<TestGroup>) -> S,
    {
        execute_mock_mpc_with_beaver_source(
            f,
            ZeroBeaverSource::new(PARTY0),
            ZeroBeaverSource::new(PARTY1),
        )
        .await
        .0
    }

    // ----------------------
    // | PIOP Round Helpers |
    // ----------------------

    /// Encapsulates a prover, key, oracles, etc so that a single object may be
    /// conveniently passed around
    struct TestParams {
        /// The circuit being tested
        circuit: PlonkCircuit<TestScalar>,
        /// The witness used to parameterize the circuit
        witness: Scalar<TestGroup>,
        /// The underlying prover
        prover: Prover<TestCurve>,
        /// The oracles to polynomials in the PIOP
        oracles: Oracles<TestScalar>,
        /// The split quotient polynomials, used in the PIOP but not
        /// appended to the oracles
        split_quot_polys: Vec<DensePolynomial<TestScalar>>,
        /// The linearization polynomial created in the fourth round
        lin_poly: DensePolynomial<TestScalar>,
        /// The challenges used in the PIOP
        challenges: Challenges<TestScalar>,
        /// The proving key
        pk: ProvingKey<TestCurve>,
    }

    impl TestParams {
        /// Constructor
        pub fn new() -> Self {
            // Construct a circuit and pre-process it
            let mut rng = thread_rng();
            let witness = Scalar::random(&mut rng);

            let circuit = test_singleprover_circuit(witness);
            let (pk, _) = setup_snark(&circuit);

            // Construct a prover for the circuit
            let domain_size = pk.domain_size();
            let num_wire_types = circuit.num_wire_types();

            let prover = Prover::new(domain_size, num_wire_types).unwrap();

            Self {
                circuit,
                witness,
                prover,
                pk,
                oracles: Default::default(),
                split_quot_polys: Default::default(),
                lin_poly: Default::default(),
                challenges: randomized_challenges(),
            }
        }
    }

    /// Run the first round of a single-prover circuit
    ///
    /// Returns a commitment to the wire polynomials
    fn run_first_round(mask: bool, params: &mut TestParams) -> Vec<Commitment<TestCurve>> {
        let mut rng = thread_rng();
        let ((wire_comms, wire_polys), pub_input_poly) = params
            .prover
            .run_1st_round(
                &mut rng,
                &params.pk.commit_key,
                &params.circuit,
                mask, // mask
            )
            .unwrap();

        params.oracles.wire_polys = wire_polys;
        params.oracles.pub_inp_poly = pub_input_poly;

        wire_comms
    }

    /// Run the second round of a single-prover circuit
    ///
    /// Returns a commitment to the permutation polynomial
    fn run_second_round(mask: bool, params: &mut TestParams) -> Commitment<TestCurve> {
        let mut rng = thread_rng();
        let (perm_commit, perm_poly) = params
            .prover
            .run_2nd_round(
                &mut rng,
                &params.pk.commit_key,
                &params.circuit,
                &params.challenges,
                mask, // mask
            )
            .unwrap();

        params.oracles.prod_perm_poly = perm_poly;
        perm_commit
    }

    /// Run the third round of a single-prover circuit
    ///
    /// Returns commitments to the split quotient polynomials
    fn run_third_round(mask: bool, params: &mut TestParams) -> Vec<Commitment<TestCurve>> {
        let mut rng = thread_rng();
        let (split_quot_comms, split_quot_polys) = params
            .prover
            .run_3rd_round(
                &mut rng,
                &params.pk.commit_key,
                &[&params.pk],
                &params.challenges,
                &[params.oracles.clone()],
                params.circuit.num_wire_types(),
                mask,
            )
            .unwrap();

        params.split_quot_polys = split_quot_polys;
        split_quot_comms
    }

    /// Run the fourth round of a single-prover circuit
    ///
    /// Returns polynomial evaluations
    fn run_fourth_round(params: &mut TestParams) -> ProofEvaluations<TestScalar> {
        let evals = params.prover.compute_evaluations(
            &params.pk,
            &params.challenges,
            &params.oracles,
            params.circuit.num_wire_types(),
        );

        let mut lin_poly = Prover::<TestCurve>::compute_quotient_component_for_lin_poly(
            params.circuit.eval_domain_size().unwrap(),
            params.challenges.zeta,
            &params.split_quot_polys,
        )
        .unwrap();
        lin_poly += &params
            .prover
            .compute_non_quotient_component_for_lin_poly(
                TestScalar::one(),
                &params.pk,
                &params.challenges,
                &params.oracles,
                &evals,
                None, // plookup_evals
            )
            .unwrap();
        params.lin_poly = lin_poly;

        evals
    }

    /// Run the fifth round of a single-prover circuit
    ///
    /// Returns the commitments to the opening and shifted opening polynomials
    fn run_fifth_round(params: &TestParams) -> (Commitment<TestCurve>, Commitment<TestCurve>) {
        params
            .prover
            .compute_opening_proofs(
                &params.pk.commit_key,
                &[&params.pk],
                &params.challenges.zeta,
                &params.challenges.v,
                &[params.oracles.clone()],
                &params.lin_poly,
            )
            .unwrap()
    }

    // ---------
    // | Tests |
    // ---------

    /// Tests equivalence with the single-prover on the first round of the PIOP
    #[tokio::test]
    async fn test_first_round() {
        // Generate a new test
        let mut params = TestParams::new();
        let expected_wire_comms = run_first_round(false /* mask */, &mut params);
        let expected_wire_polys = params.oracles.wire_polys.clone();
        let expected_pub_poly = params.oracles.pub_inp_poly.clone();

        let domain_size = params.pk.domain_size();
        let num_wire_types = params.circuit.num_wire_types();

        let ((wire_comms, wire_polys), pub_poly) = execute_deterministic_mpc(|fabric| {
            let commit_key = params.pk.commit_key.clone();
            async move {
                let multiprover_circuit = test_multiprover_circuit(params.witness, &fabric);
                let prover = MpcProver::new(domain_size, num_wire_types, fabric).unwrap();

                // Run the first round
                let ((wire_comms, wire_polys), pub_poly) = prover
                    .run_1st_round(&commit_key, &multiprover_circuit, false /* mask */)
                    .unwrap();

                // Open each of the values
                let wire_comms_open = stream::iter(wire_comms)
                    .then(|comm| async move { comm.open_authenticated().await.unwrap() })
                    .collect::<Vec<_>>();
                let wire_polys_open = stream::iter(wire_polys)
                    .then(|poly| async move { poly.open_authenticated().await.unwrap() })
                    .collect::<Vec<_>>();
                let pub_poly = pub_poly.open_authenticated();

                ((wire_comms_open.await, wire_polys_open.await), pub_poly.await.unwrap())
            }
        })
        .await;

        assert_eq!(wire_comms, expected_wire_comms);
        assert_eq!(wire_polys, expected_wire_polys);
        assert_eq!(pub_poly, expected_pub_poly);
    }

    #[tokio::test]
    async fn test_second_round() {
        let mut params = TestParams::new();
        let domain_size = params.pk.domain_size();
        let num_wire_types = params.circuit.num_wire_types();

        // Compute the result in a single-prover setup
        let expected_perm_commit = run_second_round(false /* mask */, &mut params);
        let expected_perm_poly = params.oracles.prod_perm_poly.clone();

        // Compute the result in an MPC
        let ((perm_commit, perm_poly), _) = execute_mock_mpc(|fabric| {
            let pk = params.pk.clone();
            async move {
                let challenges = allocate_challenges(&params.challenges, &fabric);
                let multiprover_circuit = test_multiprover_circuit(params.witness, &fabric);
                let prover = MpcProver::new(domain_size, num_wire_types, fabric.clone()).unwrap();

                // Run the second round
                let (perm_commit, perm_poly) = prover
                    .run_2nd_round(
                        &pk.commit_key,
                        &multiprover_circuit,
                        &challenges,
                        false, // mask
                    )
                    .unwrap();

                // Open the results
                (perm_commit.open_authenticated().await.unwrap(), perm_poly.open().await)
            }
        })
        .await;

        assert_eq!(perm_commit, expected_perm_commit);
        assert_eq!(perm_poly, expected_perm_poly);
    }

    #[tokio::test]
    async fn test_third_round() {
        let mut params = TestParams::new();
        let domain_size = params.pk.domain_size();
        let num_wire_types = params.circuit.num_wire_types();

        // Compute the result in a single-prover setup
        run_first_round(true /* mask */, &mut params);
        run_second_round(true /* mask */, &mut params);
        let expected_quot_comms = run_third_round(false /* mask */, &mut params);

        // Compute the result in an MPC
        let (quot_comms, quot_polys) = execute_deterministic_mpc(|fabric| {
            let pk = params.pk.clone();
            let oracles = params.oracles.clone();

            async move {
                let challenges = allocate_challenges(&params.challenges, &fabric);
                let oracles = allocate_oracles(&oracles, &fabric);
                let prover = MpcProver::new(domain_size, num_wire_types, fabric.clone()).unwrap();

                // Run the third round
                let (quot_comms, quot_polys) = prover
                    .run_3rd_round(&pk.commit_key, &pk, &challenges, &oracles, num_wire_types)
                    .unwrap();

                // Open the values
                let comms_open = stream::iter(quot_comms)
                    .then(|c| async move { c.open_authenticated().await.unwrap() })
                    .collect::<Vec<_>>()
                    .await;
                let polys_open = stream::iter(quot_polys)
                    .then(|p| async move { p.open_authenticated().await.unwrap() })
                    .collect::<Vec<_>>()
                    .await;

                (comms_open, polys_open)
            }
        })
        .await;

        assert_eq!(quot_polys, params.split_quot_polys);
        assert_eq!(quot_comms, expected_quot_comms);
    }

    /// Test the fourth round of the PIOP
    #[tokio::test]
    async fn test_fourth_round() {
        let mut params = TestParams::new();
        let domain_size = params.pk.domain_size();
        let num_wire_types = params.circuit.num_wire_types();

        // Compute the result in a single-prover setup
        run_first_round(true /* mask */, &mut params);
        run_second_round(true /* mask */, &mut params);
        run_third_round(false /* mask */, &mut params);
        let expected_evals = run_fourth_round(&mut params);

        // Compute the result in an MPC
        let ((evals, lin_poly), _) = execute_mock_mpc(|fabric| {
            let pk = params.pk.clone();
            let oracles = params.oracles.clone();
            let quot_polys = params.split_quot_polys.clone();

            async move {
                let challenges = allocate_challenges(&params.challenges, &fabric);
                let oracles = allocate_oracles(&oracles, &fabric);
                let prover = MpcProver::new(domain_size, num_wire_types, fabric.clone()).unwrap();

                // Run the fourth round
                let evals = prover.compute_evaluations(&pk, &challenges, &oracles, num_wire_types);
                let mut lin_poly = prover
                    .compute_quotient_component_for_lin_poly(
                        domain_size,
                        &challenges.zeta,
                        &allocate_polys(&quot_polys, &fabric),
                    )
                    .unwrap();
                lin_poly = lin_poly
                    + prover.compute_non_quotient_component_for_lin_poly(
                        &fabric.one(),
                        &pk,
                        &challenges,
                        &oracles,
                        &evals,
                    );

                // Open the values
                (
                    ProofEvaluations {
                        wires_evals: join_all(evals.wires_evals)
                            .await
                            .iter()
                            .map(Scalar::inner)
                            .collect_vec(),
                        wire_sigma_evals: join_all(evals.wire_sigma_evals)
                            .await
                            .iter()
                            .map(Scalar::inner)
                            .collect_vec(),
                        perm_next_eval: evals.perm_next_eval.await.inner(),
                    },
                    lin_poly.open_authenticated().await.unwrap(),
                )
            }
        })
        .await;

        assert_eq!(evals, expected_evals);
        assert_eq!(lin_poly, params.lin_poly);
    }

    /// Test the fifth round of the PIOP
    #[tokio::test]
    async fn test_fifth_round() {
        let mut params = TestParams::new();
        let domain_size = params.pk.domain_size();
        let num_wire_types = params.circuit.num_wire_types();

        // Compute the result in a single-prover setup
        run_first_round(true /* mask */, &mut params);
        run_second_round(true /* mask */, &mut params);
        run_third_round(false /* mask */, &mut params);
        run_fourth_round(&mut params);
        let (expected_open, expected_shift) = run_fifth_round(&params);

        // Compute the result in an MPC
        let ((open, shifted_open), _) = execute_mock_mpc(|fabric| {
            let pk = params.pk.clone();
            let oracles = params.oracles.clone();
            let lin_poly = params.lin_poly.clone();

            async move {
                let challenges = allocate_challenges(&params.challenges, &fabric);
                let oracles = allocate_oracles(&oracles, &fabric);
                let prover = MpcProver::new(domain_size, num_wire_types, fabric.clone()).unwrap();

                // Run the fifth round
                let (open, shifted_open) = prover
                    .compute_opening_proofs(
                        &pk.commit_key,
                        &pk,
                        &challenges.zeta,
                        &challenges.v,
                        &oracles,
                        &allocate_poly(&lin_poly, &fabric),
                    )
                    .unwrap();

                // Open the values
                (
                    open.open_authenticated().await.unwrap(),
                    shifted_open.open_authenticated().await.unwrap(),
                )
            }
        })
        .await;

        assert_eq!(open, expected_open);
        assert_eq!(shifted_open, expected_shift);
    }
}
