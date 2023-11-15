//! Defines traits for expressing a Plonk relation

use core::ops::{Add, Mul, Neg, Sub};

use ark_ff::{FftField, Field};
use ark_poly::univariate::DensePolynomial;

use crate::{
    constants::{GATE_WIDTH, N_MUL_SELECTORS},
    errors::CircuitError,
    gates::Gate,
    next_multiple, BoolVar, SortedLookupVecAndPolys, Variable,
};

/// An interface for Plonk constraint systems.
pub trait Circuit<F: Field>: ConstraintSystem<F> {
    /// The number of constraints.
    fn num_gates(&self) -> usize;

    /// The number of variables.
    fn num_vars(&self) -> usize;

    /// The number of public input variables.
    fn num_inputs(&self) -> usize;

    /// The number of wire types of the circuit.
    /// E.g., UltraPlonk has 4 different types of input wires, 1 type of output
    /// wires, and 1 type of lookup wires.
    fn num_wire_types(&self) -> usize;

    /// The list of public input values.
    fn public_input(&self) -> Result<Vec<F>, CircuitError>;

    /// Check circuit satisfiability against a public input.
    fn check_circuit_satisfiability(&self, pub_input: &[F]) -> Result<(), CircuitError>;

    /// Add a constant variable to the circuit; return the index of the
    /// variable.
    fn create_constant_variable(&mut self, val: F) -> Result<Variable, CircuitError>;

    /// Add a variable to the circuit; return the index of the variable.
    fn create_variable(&mut self, val: F) -> Result<Variable, CircuitError>;

    /// Add a bool variable to the circuit; return the index of the variable.
    fn create_boolean_variable(&mut self, val: bool) -> Result<BoolVar, CircuitError> {
        let val_scalar = if val { F::one() } else { F::zero() };
        let var = self.create_variable(val_scalar)?;
        self.enforce_bool(var)?;
        Ok(BoolVar(var))
    }

    /// Add a public input variable; return the index of the variable.
    fn create_public_variable(&mut self, val: F) -> Result<Variable, CircuitError>;

    /// Add a public bool variable to the circuit; return the index of the
    /// variable.
    fn create_public_boolean_variable(&mut self, val: bool) -> Result<BoolVar, CircuitError> {
        let val_scalar = if val { F::one() } else { F::zero() };
        let var = self.create_public_variable(val_scalar)?;
        Ok(BoolVar(var))
    }

    /// Set a variable to a public variable
    fn set_variable_public(&mut self, var: Variable) -> Result<(), CircuitError>;

    /// Return the witness value of variable `idx`.
    /// Return error if the input variable is invalid.
    fn witness(&self, idx: Variable) -> Result<F, CircuitError>;
}

/// An interface to add gates to a circuit that generalizes across wiring
/// implementations
///
/// Effectively abstracts the variable-centric gate interface
pub trait ConstraintSystem<F: Field> {
    /// Return a default variable with value zero.
    fn zero(&self) -> Variable;

    /// Return a default variable with value one.
    fn one(&self) -> Variable;

    /// Insert a gate into the constraint system
    fn insert_gate(
        &mut self,
        wire_vars: &[Variable; GATE_WIDTH + 1],
        gate: Box<dyn Gate<F>>,
    ) -> Result<(), CircuitError>;

    // -----------
    // | Boolean |
    // -----------

    /// Return a default variable with value `false` (namely zero).
    fn false_var(&self) -> BoolVar {
        BoolVar::new_unchecked(self.zero())
    }

    /// Return a default variable with value `true` (namely one).
    fn true_var(&self) -> BoolVar {
        BoolVar::new_unchecked(self.one())
    }

    // --------------
    // | Arithmetic |
    // --------------

    /// Constrain variable `c` to the addition of `a` and `b`.
    /// Return error if the input variables are invalid.
    fn add_gate(&mut self, a: Variable, b: Variable, c: Variable) -> Result<(), CircuitError>;

    /// Obtain a variable representing an addition.
    /// Return the index of the variable.
    /// Return error if the input variables are invalid.
    fn add(&mut self, a: Variable, b: Variable) -> Result<Variable, CircuitError>;

    /// Constrain variable `c` to the subtraction of `a` and `b`.
    /// Return error if the input variables are invalid.
    fn sub_gate(&mut self, a: Variable, b: Variable, c: Variable) -> Result<(), CircuitError>;

    /// Obtain a variable representing a subtraction.
    /// Return the index of the variable.
    /// Return error if the input variables are invalid.
    fn sub(&mut self, a: Variable, b: Variable) -> Result<Variable, CircuitError>;

    /// Constrain variable `c` to the multiplication of `a` and `b`.
    /// Return error if the input variables are invalid.
    fn mul_gate(&mut self, a: Variable, b: Variable, c: Variable) -> Result<(), CircuitError>;

    /// Obtain a variable representing a multiplication.
    /// Return the index of the variable.
    /// Return error if the input variables are invalid.
    fn mul(&mut self, a: Variable, b: Variable) -> Result<Variable, CircuitError>;

    /// Constrain a linear combination gate:
    /// q1 * a + q2 * b + q3 * c + q4 * d  = y
    fn lc_gate(
        &mut self,
        wires: &[Variable; GATE_WIDTH + 1],
        coeffs: &[F; GATE_WIDTH],
    ) -> Result<(), CircuitError>;

    /// Obtain a variable representing a linear combination.
    /// Return error if variables are invalid.
    fn lc(
        &mut self,
        wires_in: &[Variable; GATE_WIDTH],
        coeffs: &[F; GATE_WIDTH],
    ) -> Result<Variable, CircuitError>;

    /// Constrain a mul-addition gate:
    /// q_muls\[0\] * wires\[0\] *  wires\[1\] +  q_muls\[1\] * wires\[2\] *
    /// wires\[3\] = wires\[4\]
    fn mul_add_gate(
        &mut self,
        wires: &[Variable; GATE_WIDTH + 1],
        q_muls: &[F; N_MUL_SELECTORS],
    ) -> Result<(), CircuitError>;

    /// Obtain a variable representing `q12 * a * b + q34 * c * d`,
    /// where `a, b, c, d` are input wires, and `q12`, `q34` are selectors.
    /// Return error if variables are invalid.
    fn mul_add(
        &mut self,
        wires_in: &[Variable; GATE_WIDTH],
        q_muls: &[F; N_MUL_SELECTORS],
    ) -> Result<Variable, CircuitError>;

    /// Add two values with coefficients `a` and `b`, i.e. `a * x + b * y`
    fn add_with_coeffs(
        &mut self,
        x: Variable,
        y: Variable,
        a: &F,
        b: &F,
    ) -> Result<Variable, CircuitError> {
        let one = self.one();
        self.mul_add(&[x, one, y, one], &[*a, *b])
    }

    /// Multiply two values with an added coefficient `c`
    ///
    /// I.e. for variables x, y and constant c we compute c * x * y
    fn mul_with_coeff(
        &mut self,
        a: Variable,
        b: Variable,
        c: &F,
    ) -> Result<Variable, CircuitError> {
        let zero = self.zero();
        self.mul_add(&[a, b, zero, zero], &[*c, F::zero()])
    }

    /// Obtain a variable representing the sum of a list of variables.
    /// Return error if variables are invalid.
    fn sum(&mut self, elems: &[Variable]) -> Result<Variable, CircuitError>;

    /// Sum a vector of variables with given coefficients
    fn lc_sum(&mut self, elems: &[Variable], coeffs: &[F]) -> Result<Variable, CircuitError> {
        assert_eq!(elems.len(), coeffs.len());

        // Create partial linear combinations then sum them
        let mut partials = Vec::new();

        let n_lcs = next_multiple(elems.len(), GATE_WIDTH)?;
        let mut padded_wires = elems.to_vec();
        let mut padded_coeffs = coeffs.to_vec();

        padded_wires.resize(n_lcs, self.zero());
        padded_coeffs.resize(n_lcs, F::zero());

        for (wires, coeffs) in padded_wires
            .chunks(GATE_WIDTH)
            .zip(padded_coeffs.chunks(GATE_WIDTH))
        {
            partials.push(self.lc(
                &[wires[0], wires[1], wires[2], wires[3]],
                &[coeffs[0], coeffs[1], coeffs[2], coeffs[3]],
            )?);
        }

        self.sum(&partials)
    }

    /// Constrain variable `y` to the addition of `a` and `c`, where `c` is a
    /// constant value Return error if the input variables are invalid.
    fn add_constant_gate(&mut self, x: Variable, c: F, y: Variable) -> Result<(), CircuitError>;

    /// Obtains a variable representing an addition with a constant value
    /// Return error if the input variable is invalid
    fn add_constant(&mut self, input_var: Variable, elem: &F) -> Result<Variable, CircuitError>;

    /// Constrain variable `y` to the product of `a` and `c`, where `c` is a
    /// constant value Return error if the input variables are invalid.
    fn mul_constant_gate(&mut self, x: Variable, c: F, y: Variable) -> Result<(), CircuitError>;

    /// Obtains a variable representing a multiplication with a constant value
    /// Return error if the input variable is invalid
    fn mul_constant(&mut self, input_var: Variable, elem: &F) -> Result<Variable, CircuitError>;

    /// Takes the input to the fifth power
    fn pow5(&mut self, x: Variable) -> Result<Variable, CircuitError>;

    // ---------------
    // | Logic Gates |
    // ---------------

    /// Constrains variable `c` to be `if sel { a } else b`
    fn mux_gate(
        &mut self,
        sel: BoolVar,
        a: Variable,
        b: Variable,
        c: Variable,
    ) -> Result<(), CircuitError>;

    /// Create a variable representation of `if sel { a } else { b }`
    fn mux(&mut self, sel: BoolVar, a: Variable, b: Variable) -> Result<Variable, CircuitError>;

    // ---------------
    // | Constraints |
    // ---------------

    /// Common gates that should be implemented in any constraint systems.
    ///
    /// Constrain a variable to a constant.
    /// Return error if `var` is an invalid variable.
    fn enforce_constant(&mut self, var: Variable, constant: F) -> Result<(), CircuitError>;

    /// Constrain a variable to a bool.
    /// Return error if the input is invalid.
    fn enforce_bool(&mut self, a: Variable) -> Result<(), CircuitError>;

    /// Constrain two variables to have the same value.
    /// Return error if the input variables are invalid.
    fn enforce_equal(&mut self, a: Variable, b: Variable) -> Result<(), CircuitError>;

    /// Pad the circuit with n dummy gates
    fn pad_gates(&mut self, n: usize);

    /// Plookup-related methods.
    /// Return true if the circuit support lookup gates.
    fn support_lookup(&self) -> bool;
}

/// An interface that transforms Plonk circuits to polynomial used by
/// Plonk-based SNARKs.
pub trait Arithmetization<F: FftField>: Circuit<F> {
    /// The required SRS size for the circuit.
    fn srs_size(&self) -> Result<usize, CircuitError>;

    /// Get the size of the evaluation domain for arithmetization (after circuit
    /// has been finalized).
    fn eval_domain_size(&self) -> Result<usize, CircuitError>;

    /// Compute and return selector polynomials.
    /// Return an error if the circuit has not been finalized yet.
    fn compute_selector_polynomials(&self) -> Result<Vec<DensePolynomial<F>>, CircuitError>;

    /// Compute and return extended permutation polynomials.
    /// Return an error if the circuit has not been finalized yet.
    fn compute_extended_permutation_polynomials(
        &self,
    ) -> Result<Vec<DensePolynomial<F>>, CircuitError>;

    /// Compute and return the product polynomial for permutation arguments.
    /// Return an error if the circuit has not been finalized yet.
    fn compute_prod_permutation_polynomial(
        &self,
        beta: &F,
        gamma: &F,
    ) -> Result<DensePolynomial<F>, CircuitError>;

    /// Compute and return the list of wiring witness polynomials.
    /// Return an error if the circuit has not been finalized yet.
    fn compute_wire_polynomials(&self) -> Result<Vec<DensePolynomial<F>>, CircuitError>;

    /// Compute and return the public input polynomial.
    /// Return an error if the circuit has not been finalized yet.
    /// The IO gates of the circuit are guaranteed to be in the front.
    fn compute_pub_input_polynomial(&self) -> Result<DensePolynomial<F>, CircuitError>;

    /// Plookup-related methods
    /// Return default errors if the constraint system does not support lookup
    /// gates.
    ///
    /// Compute and return the polynomial that interpolates the range table
    /// elements. Return an error if the circuit does not support lookup or
    /// has not been finalized yet.
    fn compute_range_table_polynomial(&self) -> Result<DensePolynomial<F>, CircuitError> {
        Err(CircuitError::LookupUnsupported)
    }

    /// Compute and return the polynomial that interpolates the key table
    /// elements. Return an error if the circuit does not support lookup or
    /// has not been finalized yet.
    fn compute_key_table_polynomial(&self) -> Result<DensePolynomial<F>, CircuitError> {
        Err(CircuitError::LookupUnsupported)
    }

    /// Compute and return the polynomial that interpolates the table domain
    /// sepration ids. Return an error if the circuit does not support
    /// lookup or has not been finalized.
    fn compute_table_dom_sep_polynomial(&self) -> Result<DensePolynomial<F>, CircuitError> {
        Err(CircuitError::LookupUnsupported)
    }

    /// Compute and return the polynomial that interpolates the lookup domain
    /// sepration selectors for the lookup gates. Return an error if the
    /// circuit does not support lookup or has not been finalized.
    fn compute_q_dom_sep_polynomial(&self) -> Result<DensePolynomial<F>, CircuitError> {
        Err(CircuitError::LookupUnsupported)
    }

    /// Compute and return the combined lookup table vector given random
    /// challenge `tau`.
    fn compute_merged_lookup_table(&self, _tau: F) -> Result<Vec<F>, CircuitError> {
        Err(CircuitError::LookupUnsupported)
    }

    /// Compute the sorted concatenation of the (merged) lookup table and the
    /// witness values to be checked in lookup gates. Return the sorted
    /// vector and 2 polynomials that interpolate the vector. Return an
    /// error if the circuit does not support lookup or has not been
    /// finalized yet.
    fn compute_lookup_sorted_vec_polynomials(
        &self,
        _tau: F,
        _lookup_table: &[F],
    ) -> Result<SortedLookupVecAndPolys<F>, CircuitError> {
        Err(CircuitError::LookupUnsupported)
    }

    /// Compute and return the product polynomial for Plookup arguments.
    /// `beta` and `gamma` are random challenges, `sorted_vec` is the sorted
    /// concatenation of the lookup table and the lookup witnesses.
    /// Return an error if the circuit does not support lookup or
    /// has not been finalized yet.
    fn compute_lookup_prod_polynomial(
        &self,
        _tau: &F,
        _beta: &F,
        _gamma: &F,
        _lookup_table: &[F],
        _sorted_vec: &[F],
    ) -> Result<DensePolynomial<F>, CircuitError> {
        Err(CircuitError::LookupUnsupported)
    }
}
