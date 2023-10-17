//! Defines the arithmetization and circuit abstractions over a base
//! MPC-enabled arithmetic

use ark_ec::CurveGroup;
use ark_ff::FftField;
use ark_mpc::algebra::{AuthenticatedDensePoly, AuthenticatedScalarResult};
use ark_poly::univariate::DensePolynomial;

use super::MpcCircuitError;

/// The variable type in an MPC constraint system
///
/// This represents an index into the wire assignments as flattened out
/// canonically
pub type MpcVariable = usize;
/// A variable of boolean type
pub struct MpcBoolVar(usize);

impl From<MpcBoolVar> for MpcVariable {
    fn from(value: MpcBoolVar) -> Self {
        value.0
    }
}

impl MpcBoolVar {
    /// Create a new boolean variable from a variable index
    ///
    /// Do not constrain the underlying value to be boolean
    pub(crate) fn new_unchecked(inner: usize) -> Self {
        Self(inner)
    }
}

/// The circuit abstraction; contains information about circuit structure and
/// methods for constructing the circuit gate-by-gate
///
/// This is largely a re-implementation of the existing `Circuit` trait, made to
/// work over a secret shared field
pub trait MpcCircuit<C: CurveGroup> {
    /// The number of constraints
    fn num_gates(&self) -> usize;

    /// The number of variables
    fn num_variables(&self) -> usize;

    /// The number of public inputs
    fn num_inputs(&self) -> usize;

    /// The number of wire types
    ///
    /// We do not support UltraPlonk so this will likely be static
    fn num_wire_types(&self) -> usize;

    /// The public input to the circuit
    ///
    /// Note that while the input is public, it may not have yet been *made*
    /// public so the result type is a secret shared field element
    fn public_input(&self) -> Result<Vec<AuthenticatedScalarResult<C>>, MpcCircuitError>;

    /// Check whether the circuit constraints are satisfied
    fn check_circuit_satisfiability(
        &self,
        public_input: &[AuthenticatedScalarResult<C>],
    ) -> Result<(), MpcCircuitError>;

    /// Create a constant variable in the circuit, returning the index of the
    /// variable
    fn create_constant_variable(
        &mut self,
        val: AuthenticatedScalarResult<C>,
    ) -> Result<MpcVariable, MpcCircuitError>;

    /// Add a variable to the circuit; returns the index of the variable
    fn create_variable(
        &mut self,
        val: AuthenticatedScalarResult<C>,
    ) -> Result<MpcVariable, MpcCircuitError>;

    /// Add a bool variable to the circuit; return the index of the variable.
    ///
    /// In the single-prover version of this method; the input is a `bool`.
    /// However, inputs to the multi-prover constraint system are secret
    /// shared, so we take in a generic field element.
    ///
    /// We do, however, constrain the underlying shared value to be boolean in
    /// the same way the single-prover constraint system does
    fn create_boolean_variable(
        &mut self,
        val: AuthenticatedScalarResult<C>,
    ) -> Result<MpcBoolVar, MpcCircuitError> {
        let var = self.create_variable(val)?;
        self.enforce_bool(var)?;
        Ok(MpcBoolVar(var))
    }

    /// Add a public input variable; return the index of the variable.
    fn create_public_variable(
        &mut self,
        val: AuthenticatedScalarResult<C>,
    ) -> Result<MpcVariable, MpcCircuitError>;

    /// Set a variable to a public variable
    fn set_variable_public(&mut self, var: MpcVariable) -> Result<(), MpcCircuitError>;

    /// Return a default variable with value zero.
    fn zero(&self) -> MpcVariable;

    /// Return a default variable with value one.
    fn one(&self) -> MpcVariable;

    /// Return a default variable with value `false` (namely zero).
    fn false_var(&self) -> MpcBoolVar {
        MpcBoolVar::new_unchecked(self.zero())
    }

    /// Return a default variable with value `true` (namely one).
    fn true_var(&self) -> MpcBoolVar {
        MpcBoolVar::new_unchecked(self.one())
    }

    /// Return the witness value of variable `idx`.
    /// Return error if the input variable is invalid.
    fn witness(&self, idx: MpcVariable) -> Result<AuthenticatedScalarResult<C>, MpcCircuitError>;

    /// Common gates that should be implemented in any constraint systems.
    ///
    /// Constrain a variable to a constant.
    /// Return error if `var` is an invalid variable.
    fn enforce_constant(
        &mut self,
        var: MpcVariable,
        constant: AuthenticatedScalarResult<C>,
    ) -> Result<(), MpcCircuitError>;

    /// Constrain variable `c` to the addition of `a` and `b`.
    /// Return error if the input variables are invalid.
    fn add_gate(
        &mut self,
        a: MpcVariable,
        b: MpcVariable,
        c: MpcVariable,
    ) -> Result<(), MpcCircuitError>;

    /// Obtain a variable representing an addition.
    /// Return the index of the variable.
    /// Return error if the input variables are invalid.
    fn add(&mut self, a: MpcVariable, b: MpcVariable) -> Result<MpcVariable, MpcCircuitError>;

    /// Constrain variable `c` to the subtraction of `a` and `b`.
    /// Return error if the input variables are invalid.
    fn sub_gate(
        &mut self,
        a: MpcVariable,
        b: MpcVariable,
        c: MpcVariable,
    ) -> Result<(), MpcCircuitError>;

    /// Obtain a variable representing a subtraction.
    /// Return the index of the variable.
    /// Return error if the input variables are invalid.
    fn sub(&mut self, a: MpcVariable, b: MpcVariable) -> Result<MpcVariable, MpcCircuitError>;

    /// Constrain variable `c` to the multiplication of `a` and `b`.
    /// Return error if the input variables are invalid.
    fn mul_gate(
        &mut self,
        a: MpcVariable,
        b: MpcVariable,
        c: MpcVariable,
    ) -> Result<(), MpcCircuitError>;

    /// Obtain a variable representing a multiplication.
    /// Return the index of the variable.
    /// Return error if the input variables are invalid.
    fn mul(&mut self, a: MpcVariable, b: MpcVariable) -> Result<MpcVariable, MpcCircuitError>;

    /// Constrain a variable to a bool.
    /// Return error if the input is invalid.
    fn enforce_bool(&mut self, a: MpcVariable) -> Result<(), MpcCircuitError>;

    /// Constrain two variables to have the same value.
    /// Return error if the input variables are invalid.
    fn enforce_equal(&mut self, a: MpcVariable, b: MpcVariable) -> Result<(), MpcCircuitError>;

    /// Pad the circuit with n dummy gates
    fn pad_gates(&mut self, n: usize);
}

/// An abstraction shimming the `Circuit` abstraction and the PIOP based
/// arguments in the MPC prover. The `MpcArithmetization` takes circuit wire
/// assignments and constructs polynomial representations of the assignment
pub trait MpcArithmetization<C: CurveGroup>: MpcCircuit<C>
where
    C::ScalarField: FftField,
{
    /// The required SRS size for the circuit.
    fn srs_size(&self) -> Result<usize, MpcCircuitError>;

    /// Get the size of the evaluation domain for arithmetization (after circuit
    /// has been finalized).
    fn eval_domain_size(&self) -> Result<usize, MpcCircuitError>;

    /// Compute and return selector polynomials.
    /// Return an error if the circuit has not been finalized yet.
    fn compute_selector_polynomials(
        &self,
    ) -> Result<Vec<DensePolynomial<C::ScalarField>>, MpcCircuitError>;

    /// Compute and return extended permutation polynomials.
    /// Return an error if the circuit has not been finalized yet.
    fn compute_extended_permutation_polynomials(
        &self,
    ) -> Result<Vec<AuthenticatedDensePoly<C>>, MpcCircuitError>;

    /// Compute and return the product polynomial for permutation arguments.
    /// Return an error if the circuit has not been finalized yet.
    fn compute_prod_permutation_polynomial(
        &self,
        beta: &C::ScalarField,
        gamma: &C::ScalarField,
    ) -> Result<AuthenticatedDensePoly<C>, MpcCircuitError>;

    /// Compute and return the list of wiring witness polynomials.
    /// Return an error if the circuit has not been finalized yet.
    fn compute_wire_polynomials(&self) -> Result<Vec<AuthenticatedDensePoly<C>>, MpcCircuitError>;

    /// Compute and return the public input polynomial.
    /// Return an error if the circuit has not been finalized yet.
    /// The IO gates of the circuit are guaranteed to be in the front.
    fn compute_pub_input_polynomial(&self) -> Result<AuthenticatedDensePoly<C>, MpcCircuitError>;
}
