//! Defines the prover side of a collaborative KZG proof
//! See the original KZG10 paper for formal details:
//!     https://www.iacr.org/archive/asiacrypt2010/6477178/6477178.pdf
//!
//! Modeled after the univariate, single-prover KZG implementation in this repo

use core::{
    marker::PhantomData,
    pin::Pin,
    task::{Context, Poll},
};

use ark_ec::{pairing::Pairing, AffineRepr};
use ark_mpc::algebra::{
    AuthenticatedDensePoly, AuthenticatedPointOpenResult, AuthenticatedPointResult,
    AuthenticatedScalarResult, CurvePoint, DensePolynomialResult, ScalarResult,
};
use futures::{ready, Future, FutureExt};
use jf_primitives::pcs::prelude::{
    Commitment, PCSError, UnivariateKzgProof, UnivariateProverParam,
};

// ----------------------
// | Struct Definitions |
// ----------------------

/// A multiprover KZG proof
pub struct MultiproverKZG<E: Pairing> {
    _phantom: PhantomData<E>,
}

/// A commitment to the shared polynomial
#[derive(Clone)]
pub struct MultiproverKzgCommitment<E: Pairing> {
    /// The underlying commitment, an element of the curve group
    pub commitment: AuthenticatedPointResult<E::G1>,
}

/// The result of opening a commitment in the MPC
///
/// Wrapping the type in this way allows us to implement `Future` and resolve
/// this opening to a standard, single-prover KZG commitment
#[derive(Clone)]
pub struct MultiproverKzgCommitmentOpening<E: Pairing> {
    /// The result of opening the underlying commitment
    pub opening: AuthenticatedPointOpenResult<E::G1>,
}

impl<E: Pairing> MultiproverKzgCommitment<E> {
    /// Open the commitment
    pub fn open_authenticated(&self) -> MultiproverKzgCommitmentOpening<E> {
        MultiproverKzgCommitmentOpening {
            opening: self.commitment.open_authenticated(),
        }
    }
}

impl<E: Pairing> Future for MultiproverKzgCommitmentOpening<E>
where
    E::ScalarField: Unpin,
{
    type Output = Result<Commitment<E>, PCSError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let res = ready!(self.opening.poll_unpin(cx))
            .map(|opening| Commitment(opening.to_affine()))
            .map_err(|e| PCSError::Multiprover(e.to_string()));

        Poll::Ready(res)
    }
}

/// A proof of evaluation of a polynomial at a point
#[derive(Clone)]
pub struct MultiproverKzgProof<E: Pairing> {
    /// The underlying proof, an element of the curve group
    pub proof: AuthenticatedPointResult<E::G1>,
}

/// The result of opening a proof in the MPC
///
/// Similar to above, we wrap the result in a `Future` to resolve it to a
/// standard, single-prover KZG proof
#[derive(Clone)]
pub struct MultiproverKzgProofOpening<E: Pairing> {
    /// The result of opening the underlying proof
    pub opening: AuthenticatedPointOpenResult<E::G1>,
}

impl<E: Pairing> MultiproverKzgProof<E> {
    /// Open the proof
    pub fn open_authenticated(&self) -> MultiproverKzgProofOpening<E> {
        MultiproverKzgProofOpening {
            opening: self.proof.open_authenticated(),
        }
    }
}

impl<E: Pairing> Future for MultiproverKzgProofOpening<E>
where
    E::ScalarField: Unpin,
{
    type Output = Result<UnivariateKzgProof<E>, PCSError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let res = ready!(self.opening.poll_unpin(cx))
            .map(|opening| UnivariateKzgProof {
                proof: opening.to_affine(),
            })
            .map_err(|e| PCSError::Multiprover(e.to_string()));

        Poll::Ready(res)
    }
}

// ------------------------------------
// | Commitment Scheme Implementation |
// ------------------------------------

impl<E: Pairing> MultiproverKZG<E> {
    /// Commit to a polynomial that is shared between the provers
    pub fn commit(
        prover_params: &UnivariateProverParam<E>,
        poly: &AuthenticatedDensePoly<E::G1>,
    ) -> Result<MultiproverKzgCommitment<E>, PCSError> {
        if poly.degree() > prover_params.powers_of_g.len() {
            return Err(PCSError::InvalidParameters(
                "Polynomial degree exceeds supported degree".to_string(),
            ));
        }

        // Map the prover params to `CurvePoint`s in the MPC framework
        let eval_terms = Self::convert_prover_params(prover_params);
        let commitment = CurvePoint::msm_authenticated(&poly.coeffs, &eval_terms);
        Ok(MultiproverKzgCommitment { commitment })
    }

    /// Prove opening of a polynomial at a point
    ///
    /// Returns both the proof and the opening evaluation
    ///
    /// Note that the KZG implementation of this library is slightly
    /// restructured from the original paper and from the Plonk batching
    /// scheme. In particular, the proof is structured to avoid as
    /// much arithmetic in G_2 as possible
    pub fn open(
        prover_params: &UnivariateProverParam<E>,
        poly: &AuthenticatedDensePoly<E::G1>,
        point: &ScalarResult<E::G1>,
    ) -> Result<(MultiproverKzgProof<E>, AuthenticatedScalarResult<E::G1>), PCSError> {
        let quotient_degree = poly.degree() - 1;
        if quotient_degree > prover_params.powers_of_g.len() {
            return Err(PCSError::InvalidParameters(
                "Polynomial degree exceeds supported degree".to_string(),
            ));
        }

        // Compute the quotient polynomial, i.e. f(x) / (x - z) for opening point z
        let fabric = point.fabric();
        let divisor = DensePolynomialResult::from_coeffs(vec![-point, fabric.one()]);
        let quotient = poly / &divisor;

        // Evaluate the quotient polynomial "in the exponent" of the prover params at
        // the SRS challenge, \beta
        let eval_terms = Self::convert_prover_params(prover_params);
        let proof =
            CurvePoint::msm_authenticated(&quotient.coeffs, &eval_terms[..=quotient_degree]);

        let eval = poly.eval(point);

        Ok((MultiproverKzgProof { proof }, eval))
    }

    /// Convert native prover params to prover params in the MPC framework
    fn convert_prover_params(pp: &UnivariateProverParam<E>) -> Vec<CurvePoint<E::G1>> {
        pp.powers_of_g
            .iter()
            .map(|g| CurvePoint::from(g.into_group()))
            .collect::<Vec<_>>()
    }
}

// ---------
// | Tests |
// ---------

#[cfg(all(test, feature = "all-tests"))]
mod test {
    use ark_bn254::Bn254;
    use ark_ec::pairing::Pairing;
    use ark_mpc::{
        algebra::{AuthenticatedDensePoly, Scalar},
        network::PartyId,
        test_helpers::execute_mock_mpc,
        MpcFabric, PARTY0,
    };
    use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, Polynomial};
    use ark_std::UniformRand;
    use itertools::Itertools;
    use jf_primitives::pcs::{
        prelude::UnivariateKzgPCS, PolynomialCommitmentScheme, StructuredReferenceString,
    };
    use jf_utils::test_rng;
    use rand::Rng;

    use crate::multiprover::multiprover_kzg::MultiproverKZG;

    /// The curve used for testing
    type TestCurve = Bn254;
    /// The curve group to run the MPC fabric over
    type TestGroup = <TestCurve as Pairing>::G1;
    /// The scalar field underlying the MPC
    type ScalarField = <TestCurve as Pairing>::ScalarField;

    /// The polynomial degree bound used in the test suite
    const DEGREE_BOUND: usize = 100;

    // -----------
    // | Helpers |
    // -----------

    /// Generate a random polynomial with a given degree bound (inclusive)
    fn random_poly(degree_bound: usize) -> DensePolynomial<ScalarField> {
        let mut rng = test_rng();
        let degree = rng.gen_range(1..=degree_bound);
        let coeffs = (0..=degree)
            .map(|_| ScalarField::rand(&mut rng))
            .collect_vec();

        DensePolynomial::from_coefficients_vec(coeffs)
    }

    /// Share a polynomial with the counterparty in an MPC
    fn share_poly(
        poly: &DensePolynomial<ScalarField>,
        sender: PartyId,
        fabric: &MpcFabric<TestGroup>,
    ) -> AuthenticatedDensePoly<TestGroup> {
        let coeffs = poly.coeffs.iter().map(|c| Scalar::new(*c)).collect_vec();
        let shared_coeffs = fabric.batch_share_scalar(coeffs, sender);

        AuthenticatedDensePoly::from_coeffs(shared_coeffs)
    }

    // ---------
    // | Tests |
    // ---------

    /// Tests committing to a random polynomial in an MPC circuit, opening the
    /// commitment, then verifying it outside of the circuit
    #[tokio::test]
    async fn test_commit_open() {
        let mut rng = test_rng();
        let poly = random_poly(DEGREE_BOUND);
        let point = ScalarField::rand(&mut rng);

        let pp = <UnivariateKzgPCS<TestCurve> as PolynomialCommitmentScheme>::gen_srs_for_testing(
            &mut rng,
            DEGREE_BOUND,
        )
        .unwrap();
        let (ck, vk) = pp.trim(poly.degree()).unwrap();

        let ((comm, proof, eval), _) = execute_mock_mpc(|fabric| {
            let ck = ck.clone();
            let poly = poly.clone();

            async move {
                let shared_poly = share_poly(&poly, PARTY0, &fabric);
                let allocated_point = fabric.allocate_scalar(Scalar::new(point));

                let comm = MultiproverKZG::<TestCurve>::commit(&ck, &shared_poly).unwrap();
                let (proof, eval) =
                    MultiproverKZG::<TestCurve>::open(&ck, &shared_poly, &allocated_point).unwrap();

                (
                    comm.open_authenticated().await,
                    proof.open_authenticated().await,
                    eval.open_authenticated().await,
                )
            }
        })
        .await;

        assert!(comm.is_ok());
        assert!(proof.is_ok());
        assert!(eval.is_ok());

        let comm = comm.unwrap();
        let proof = proof.unwrap();
        let eval = eval.unwrap().inner();

        assert!(UnivariateKzgPCS::<TestCurve>::verify(&vk, &comm, &point, &eval, &proof).unwrap())
    }
}
