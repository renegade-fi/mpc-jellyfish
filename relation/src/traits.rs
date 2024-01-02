//! Defines traits for expressing a Plonk relation

use ark_ff::{FftField, Field, One, Zero};
use ark_mpc::algebra::FieldWrapper;
use ark_poly::univariate::DensePolynomial;
use core::{
    fmt::Debug,
    iter::Sum,
    ops::{Add, Mul, Neg, Sub},
};

use crate::{
    constants::{GATE_WIDTH, N_MUL_SELECTORS},
    errors::CircuitError,
    gates::{
        AdditionGate, BoolGate, ConstantAdditionGate, ConstantGate, ConstantMultiplicationGate,
        EqualityGate, Gate, LinCombGate, LogicOrGate, LogicOrOutputGate, MulAddGate,
        MultiplicationGate, MuxGate, SubtractionGate,
    },
    next_multiple,
    proof_linking::GroupLayout,
    BoolVar, SortedLookupVecAndPolys, Variable,
};

/// A shorthand for converting a value to a constant in the circuit
macro_rules! felt {
    ($x:expr) => {
        Self::Constant::from_field(&$x)
    };
}

/// A shorthand for converting a slice of values to a slice of constants in the
/// circuit
macro_rules! felts {
    ($x:expr) => {
        $x.iter().map(|x| Self::Constant::from_field(x)).collect::<Vec<_>>()
    };
}

/// Represents the parameterization of a proof-linking group in the circuit
///
/// See `Circuit::create_link_group` for more details on proof-linking
#[derive(Clone, Debug)]
pub struct LinkGroup {
    /// The id of the group
    pub id: String,
}

/// An interface to add gates to a circuit that generalizes across wiring
/// implementations
///
/// Effectively abstracts the variable-centric gate interface
///
/// The generic `W` corresponds to the witness type assigned to wires, this
/// allows both single and multiprover implementations to make use of the
/// underlying default impls
///
/// The generic `C` corresponds to the constant type used to represent public
/// values in the circuit
///
/// The generic `F` corresponds to the field that the circuit is over, this
/// type is stored in gates and used to construct selectors
pub trait Circuit<F: Field> {
    /// The type that wires take in the circuit
    type Wire: 'static
        + Debug
        + Sized
        + Clone
        + Add<Self::Wire, Output = Self::Wire>
        + Sub<Self::Wire, Output = Self::Wire>
        + Mul<Self::Wire, Output = Self::Wire>
        + Neg<Output = Self::Wire>
        + Sum;

    /// The type that constants take in the circuit
    type Constant: FieldWrapper<F>
        + Copy
        + One
        + Zero
        + Mul<Self::Wire, Output = Self::Wire>
        + Add<Self::Wire, Output = Self::Wire>
        + Sub<Self::Wire, Output = Self::Wire>
        + Neg<Output = Self::Constant>;

    // ---------------------
    // | Circuitry Methods |
    // ---------------------

    // --- Metadata --- //

    /// Return a default variable with value zero.
    fn zero(&self) -> Variable;

    /// Return a default variable with value one.
    fn one(&self) -> Variable;

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
    fn public_input(&self) -> Result<Vec<Self::Wire>, CircuitError>;

    /// Return the witness value of variable `idx`.
    /// Return error if the input variable is invalid.
    fn witness(&self, idx: Variable) -> Result<Self::Wire, CircuitError>;

    /// Check circuit satisfiability against a public input.
    fn check_circuit_satisfiability(&self, pub_input: &[Self::Wire]) -> Result<(), CircuitError>;

    /// Plookup-related methods.
    /// Return true if the circuit support lookup gates.
    fn support_lookup(&self) -> bool;

    // --- Variable Allocation --- //

    /// Check a variable before entering it into a constraint
    fn check_var(&self, var: Variable) -> Result<(), CircuitError>;

    /// Check a set of variables before entering them into a constraint
    fn check_vars(&self, vars: &[Variable]) -> Result<(), CircuitError> {
        for var in vars.iter().copied() {
            self.check_var(var)?;
        }

        Ok(())
    }

    /// Add a constant variable to the circuit; return the index of the
    /// variable.
    fn create_constant_variable(&mut self, val: F) -> Result<Variable, CircuitError> {
        self.create_constant_variable_with_link_groups(val, &[])
    }

    /// Add a constant variable to the circuit; with an optional set of link
    /// groups to add it to
    fn create_constant_variable_with_link_groups(
        &mut self,
        val: F,
        link_groups: &[LinkGroup],
    ) -> Result<Variable, CircuitError>;

    /// Add a variable to the circuit; return the index of the variable.
    fn create_variable(&mut self, val: Self::Wire) -> Result<Variable, CircuitError> {
        self.create_variable_with_link_groups(val, &[])
    }

    /// Add a variable to the witness; with an optional set of link groups to
    /// add it to
    fn create_variable_with_link_groups(
        &mut self,
        val: Self::Wire,
        link_groups: &[LinkGroup],
    ) -> Result<Variable, CircuitError>;

    /// Add a bool variable to the circuit; return the index of the variable.
    fn create_boolean_variable<T: Into<Self::Wire>>(
        &mut self,
        val: T,
    ) -> Result<BoolVar, CircuitError> {
        self.create_boolean_variable_with_link_groups(val, &[])
    }

    /// Add a constant bool variable to the circuit; with an optional set of
    /// link groups to add it to
    fn create_boolean_variable_with_link_groups<T: Into<Self::Wire>>(
        &mut self,
        val: T,
        link_groups: &[LinkGroup],
    ) -> Result<BoolVar, CircuitError> {
        let var = self.create_variable_with_link_groups(val.into(), link_groups)?;
        self.enforce_bool(var)?;
        Ok(BoolVar(var))
    }

    /// Add a public input variable; return the index of the variable.
    fn create_public_variable(&mut self, val: Self::Wire) -> Result<Variable, CircuitError> {
        let var = self.create_variable(val)?;
        self.set_variable_public(var)?;
        Ok(var)
    }

    /// Add a public bool variable to the circuit; return the index of the
    /// variable.
    fn create_public_boolean_variable<T: Into<Self::Wire>>(
        &mut self,
        val: T,
    ) -> Result<BoolVar, CircuitError> {
        let var = self.create_public_variable(val.into())?;
        Ok(BoolVar(var))
    }

    /// Set a variable to a public variable
    fn set_variable_public(&mut self, var: Variable) -> Result<(), CircuitError>;

    /// Create a proof linking group in the circuit. Witness elements may be
    /// allocated into this group in order and they will be placed into the
    /// given layout as proof-linking gates (or assigned a default layout if
    /// none is provided).
    ///
    /// These proof linking gates are effictively of the form a(x) * 0 = 0,
    /// where a(x) is the witness element. This allows us to prove
    /// that the a(x) polynomial of one proof equals the a(x) polynomial of
    /// another proof over some proof-linking domain, represented by the group
    fn create_link_group(&mut self, id: String, layout: Option<GroupLayout>) -> LinkGroup;

    // --- Gate Allocation --- //

    /// Insert a gate into the constraint system
    fn insert_gate(
        &mut self,
        wire_vars: &[Variable; GATE_WIDTH + 1],
        gate: Box<dyn Gate<F>>,
    ) -> Result<(), CircuitError>;

    /// Pad the circuit with n dummy gates
    fn pad_gates(&mut self, n: usize) {
        let wire_vars = &[self.zero(), self.zero(), 0, 0, 0];
        for _ in 0..n {
            self.insert_gate(wire_vars, Box::new(EqualityGate)).unwrap();
        }
    }

    // ----------------
    // | Boolean Vars |
    // ----------------

    /// Return a default variable with value `false` (namely zero).
    fn false_var(&self) -> BoolVar {
        BoolVar::new_unchecked(self.zero())
    }

    /// Return a default variable with value `true` (namely one).
    fn true_var(&self) -> BoolVar {
        BoolVar::new_unchecked(self.one())
    }

    // --------------------
    // | Arithmetic Gates |
    // --------------------

    /// Constrain variable `c` to the addition of `a` and `b`.
    /// Return error if the input variables are invalid.
    fn add_gate(&mut self, a: Variable, b: Variable, c: Variable) -> Result<(), CircuitError> {
        self.check_var(a)?;
        self.check_var(b)?;
        self.check_var(c)?;

        let wire_vars = &[a, b, 0, 0, c];
        self.insert_gate(wire_vars, Box::new(AdditionGate))?;
        Ok(())
    }

    /// Obtain a variable representing an addition.
    /// Return the index of the variable.
    /// Return error if the input variables are invalid.
    fn add(&mut self, a: Variable, b: Variable) -> Result<Variable, CircuitError> {
        self.check_var(a)?;
        self.check_var(b)?;
        let val = self.witness(a)? + self.witness(b)?;
        let c = self.create_variable(val)?;
        self.add_gate(a, b, c)?;
        Ok(c)
    }

    /// Constrain variable `c` to the subtraction of `a` and `b`.
    /// Return error if the input variables are invalid.
    fn sub_gate(&mut self, a: Variable, b: Variable, c: Variable) -> Result<(), CircuitError> {
        self.check_var(a)?;
        self.check_var(b)?;
        self.check_var(c)?;

        let wire_vars = &[a, b, 0, 0, c];
        self.insert_gate(wire_vars, Box::new(SubtractionGate))?;
        Ok(())
    }

    /// Obtain a variable representing a subtraction.
    /// Return the index of the variable.
    /// Return error if the input variables are invalid.
    fn sub(&mut self, a: Variable, b: Variable) -> Result<Variable, CircuitError> {
        self.check_var(a)?;
        self.check_var(b)?;
        let val = self.witness(a)? - self.witness(b)?;
        let c = self.create_variable(val)?;
        self.sub_gate(a, b, c)?;
        Ok(c)
    }

    /// Constrain variable `c` to the multiplication of `a` and `b`.
    /// Return error if the input variables are invalid.
    fn mul_gate(&mut self, a: Variable, b: Variable, c: Variable) -> Result<(), CircuitError> {
        self.check_var(a)?;
        self.check_var(b)?;
        self.check_var(c)?;

        let wire_vars = &[a, b, 0, 0, c];
        self.insert_gate(wire_vars, Box::new(MultiplicationGate))?;
        Ok(())
    }

    /// Create a variable representation of the output of `a * b`
    fn mul(&mut self, a: Variable, b: Variable) -> Result<Variable, CircuitError> {
        self.check_var(a)?;
        self.check_var(b)?;
        let val = self.witness(a)? * self.witness(b)?;
        let c = self.create_variable(val)?;
        self.mul_gate(a, b, c)?;
        Ok(c)
    }

    /// Constrain variable `c` to the linear combination of `a`, `b`, `c`, `d`
    fn lc_gate(
        &mut self,
        wires: &[Variable; GATE_WIDTH + 1],
        coeffs: &[F; GATE_WIDTH],
    ) -> Result<(), CircuitError> {
        self.check_vars(wires)?;

        let wire_vars = [wires[0], wires[1], wires[2], wires[3], wires[4]];
        self.insert_gate(&wire_vars, Box::new(LinCombGate { coeffs: *coeffs }))?;

        Ok(())
    }

    /// Create a variable `c` representing a linear combination of `a`, `b`,
    /// `c`, `d`
    fn lc(
        &mut self,
        wires_in: &[Variable; GATE_WIDTH],
        coeffs: &[F; GATE_WIDTH],
    ) -> Result<Variable, CircuitError> {
        self.check_vars(wires_in)?;

        let vals_in: Vec<Self::Wire> = wires_in
            .iter()
            .map(|&var| self.witness(var))
            .collect::<Result<Vec<_>, CircuitError>>()?;

        // calculate y as the linear combination of coeffs and vals_in
        let y_val =
            vals_in.iter().zip(coeffs.iter()).map(|(val, coeff)| felt!(coeff) * val.clone()).sum();
        let y = self.create_variable(y_val)?;

        let wires = [wires_in[0], wires_in[1], wires_in[2], wires_in[3], y];
        self.lc_gate(&wires, coeffs)?;
        Ok(y)
    }

    /// Constrain variable `wires[4]` to the mul-addition of `q_mul[0] * a * b +
    /// q_mul[1] * c * d`
    fn mul_add_gate(
        &mut self,
        wires: &[Variable; GATE_WIDTH + 1],
        q_muls: &[F; N_MUL_SELECTORS],
    ) -> Result<(), CircuitError> {
        self.check_vars(wires)?;

        let wire_vars = [wires[0], wires[1], wires[2], wires[3], wires[4]];
        self.insert_gate(&wire_vars, Box::new(MulAddGate { coeffs: *q_muls }))?;

        Ok(())
    }

    /// Create a variable `y` representing the mul-addition of `q_mul[0] * a *
    /// b + q_mul[1] * c * d`
    fn mul_add(
        &mut self,
        wires_in: &[Variable; GATE_WIDTH],
        q_muls: &[F; N_MUL_SELECTORS],
    ) -> Result<Variable, CircuitError> {
        // Fetch the witness elements for the input wires
        self.check_vars(wires_in)?;
        let [wire0, wire1, wire2, wire3]: [Self::Wire; 4] = wires_in
            .iter()
            .map(|&var| self.witness(var))
            .collect::<Result<Vec<_>, CircuitError>>()?
            .try_into()
            .expect("incorrect number of wires");

        // Calculate y as the mul-addition of coeffs and vals_in
        let coeffs = felts!(q_muls);
        let y_val = coeffs[0] * wire0 * wire1 + coeffs[1] * wire2 * wire3;
        let y = self.create_variable(y_val)?;

        let wires = [wires_in[0], wires_in[1], wires_in[2], wires_in[3], y];
        self.mul_add_gate(&wires, q_muls)?;
        Ok(y)
    }

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

    /// Create a variable representation of the sum of a slice of variables
    fn sum(&mut self, elems: &[Variable]) -> Result<Variable, CircuitError> {
        if elems.is_empty() {
            return Err(CircuitError::ParameterError(
                "Sum over an empty slice of variables is undefined".to_string(),
            ));
        }
        self.check_vars(elems)?;

        // Construct the output wire's assignment
        let sum_val: Self::Wire = elems
            .iter()
            .map(|&elem| self.witness(elem))
            .collect::<Result<Vec<_>, CircuitError>>()?
            .iter()
            .cloned()
            .sum();
        let sum = self.create_variable(sum_val)?;

        // Pad to ("next multiple of 3" + 1) in length
        let mut padded: Vec<Variable> = elems.to_vec();
        let rate = GATE_WIDTH - 1; // rate at which each lc add
        let padded_len = next_multiple(elems.len() - 1, rate)? + 1;
        padded.resize(padded_len, self.zero());

        // z_0 = = x_0
        // z_i = z_i-1 + x_3i-2 + x_3i-1 + x_3i
        let coeffs = [F::one(); GATE_WIDTH];
        let mut accum = padded[0];
        for i in 1..padded_len / rate {
            accum = self.lc(
                &[accum, padded[rate * i - 2], padded[rate * i - 1], padded[rate * i]],
                &coeffs,
            )?;
        }

        // Final round
        let wires =
            [accum, padded[padded_len - 3], padded[padded_len - 2], padded[padded_len - 1], sum];
        self.lc_gate(&wires, &coeffs)?;

        Ok(sum)
    }

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

        for (wires, coeffs) in padded_wires.chunks(GATE_WIDTH).zip(padded_coeffs.chunks(GATE_WIDTH))
        {
            partials.push(self.lc(
                &[wires[0], wires[1], wires[2], wires[3]],
                &[coeffs[0], coeffs[1], coeffs[2], coeffs[3]],
            )?);
        }

        self.sum(&partials)
    }

    /// Constrain variable `y` to the addition of `x` and `c`
    fn add_constant_gate(&mut self, x: Variable, c: &F, y: Variable) -> Result<(), CircuitError> {
        self.check_var(x)?;
        self.check_var(y)?;

        let wire_vars = &[x, self.one(), 0, 0, y];
        self.insert_gate(wire_vars, Box::new(ConstantAdditionGate(*c)))?;
        Ok(())
    }

    /// Create a variable `y` representing the addition of `x` and `c`
    fn add_constant(&mut self, x: Variable, c: &F) -> Result<Variable, CircuitError> {
        self.check_var(x)?;

        let input_val = self.witness(x).unwrap();
        let output_val = felt!(c) + input_val;
        let output_var = self.create_variable(output_val).unwrap();

        self.add_constant_gate(x, c, output_var)?;

        Ok(output_var)
    }

    /// Constrain variable `y` to the multiplication of `x` and `c`
    fn mul_constant_gate(&mut self, x: Variable, c: &F, y: Variable) -> Result<(), CircuitError> {
        self.check_var(x)?;
        self.check_var(y)?;

        let wire_vars = &[x, 0, 0, 0, y];
        self.insert_gate(wire_vars, Box::new(ConstantMultiplicationGate(*c)))?;
        Ok(())
    }

    /// Create a variable `y` representing the multiplication of `x` and `c`
    fn mul_constant(&mut self, input_var: Variable, elem: &F) -> Result<Variable, CircuitError> {
        self.check_var(input_var)?;

        let input_val = self.witness(input_var).unwrap();
        let output_val = felt!(elem) * input_val;
        let output_var = self.create_variable(output_val).unwrap();

        self.mul_constant_gate(input_var, elem, output_var)?;

        Ok(output_var)
    }

    /// Returns the fifth power of a variable
    fn pow5(&mut self, x: Variable) -> Result<Variable, CircuitError>;

    // ---------------
    // | Logic Gates |
    // ---------------

    /// Constrain that `a` is true or `b` is true.
    /// Return error if variables are invalid.
    fn logic_or_gate(&mut self, a: BoolVar, b: BoolVar) -> Result<(), CircuitError> {
        self.check_var(a.into())?;
        self.check_var(b.into())?;

        let wire_vars = &[a.into(), b.into(), 0, 0, 0];
        self.insert_gate(wire_vars, Box::new(LogicOrGate))?;

        Ok(())
    }

    /// Obtain a variable representing the result of a logic OR gate. Return the
    /// index of the variable. Return error if the input variables are
    /// invalid.
    fn logic_or(&mut self, a: BoolVar, b: BoolVar) -> Result<BoolVar, CircuitError> {
        self.check_var(a.into())?;
        self.check_var(b.into())?;

        let a_val = self.witness(a.into())?;
        let b_val = self.witness(b.into())?;
        let c_val = a_val.clone() + b_val.clone() - a_val * b_val;

        let c = self.create_variable(c_val)?;
        let wire_vars = &[a.into(), b.into(), 0, 0, c];
        self.insert_gate(wire_vars, Box::new(LogicOrOutputGate))?;

        // We do not need to constrain the output to be boolean as the inputs already
        // are, and the range of this gate is {0, 1}
        Ok(BoolVar(c))
    }

    /// Given a list of boolean variables, obtain a variable representing the
    /// result of a logic OR gate. Return the index of the variable. Return
    /// error if the input variables are invalid.
    fn logic_or_all(&mut self, vars: &[BoolVar]) -> Result<BoolVar, CircuitError> {
        if vars.is_empty() {
            return Err(CircuitError::ParameterError(
                "logic_or_all: empty variable list".to_string(),
            ));
        }

        let mut res = vars[0];
        for var in vars.iter().skip(1) {
            res = self.logic_or(res, *var)?;
        }

        Ok(res)
    }

    /// Constrain that `a` is true and `b` is true
    fn logic_and_gate(&mut self, a: BoolVar, b: BoolVar) -> Result<(), CircuitError> {
        self.mul_gate(a.into(), b.into(), self.one())
    }

    /// Obtain a variable representing the result of a logic AND gate. Return
    /// the index of the variable. Return error if the input variables are
    /// invalid.
    fn logic_and(&mut self, a: BoolVar, b: BoolVar) -> Result<BoolVar, CircuitError> {
        let c = self.mul(a.into(), b.into())?;

        // We do not need to constrain the output to be boolean as the inputs already
        // are, and the range of this gate is {0, 1}
        Ok(BoolVar(c))
    }

    /// Given a list of boolean variables, obtain a variable representing the
    /// result of a logic AND gate. Return the index of the variable. Return
    /// error if the input variables are invalid.
    fn logic_and_all(&mut self, vars: &[BoolVar]) -> Result<BoolVar, CircuitError> {
        if vars.is_empty() {
            return Err(CircuitError::ParameterError(
                "logic_and_all: empty variable list".to_string(),
            ));
        }

        let mut res = vars[0];
        for &var in vars.iter().skip(1) {
            res = self.logic_and(res, var)?;
        }

        Ok(res)
    }

    /// Obtain a variable representing the result of a logic negation gate.
    /// Return the index of the variable. Return error if the input variable
    /// is invalid.
    fn logic_neg(&mut self, a: BoolVar) -> Result<BoolVar, CircuitError> {
        self.check_var(a.into())?;

        // 1 - a
        let one = self.one();
        let res = self.add_with_coeffs(one, a.into(), &F::one(), &-F::one())?;

        // We do not need to constrain the output to be boolean as the inputs already
        // are, and the range of this gate is {0, 1}
        Ok(BoolVar(res))
    }

    /// Assuming values represented by `a` is boolean
    /// Constrain `a` is true
    fn enforce_true(&mut self, a: BoolVar) -> Result<(), CircuitError> {
        self.enforce_constant(a.into(), F::one())
    }

    /// Assuming values represented by `a` is boolean
    /// Constrain `a` is false
    fn enforce_false(&mut self, a: BoolVar) -> Result<(), CircuitError> {
        self.enforce_constant(a.into(), F::zero())
    }

    /// Constrains variable `c` to be `if sel { a } else { b }`
    fn mux_gate(
        &mut self,
        sel: BoolVar,
        a: Variable,
        b: Variable,
        c: Variable,
    ) -> Result<(), CircuitError> {
        self.check_var(a)?;
        self.check_var(b)?;
        self.check_var(c)?;
        self.check_var(sel.into())?;

        let wire_vars = &[sel.into(), a, sel.into(), b, c];
        self.insert_gate(wire_vars, Box::new(MuxGate))
    }

    /// Returns a variable representing `if sel { a } else { b }`
    fn mux(&mut self, sel: BoolVar, a: Variable, b: Variable) -> Result<Variable, CircuitError> {
        let sel_eval = self.witness(sel.into())?;
        let a_eval = self.witness(a)?;
        let b_eval = self.witness(b)?;

        let one = Self::Constant::one();
        let res = sel_eval.clone() * a_eval + (one - sel_eval) * b_eval;
        let res_var = self.create_variable(res)?;

        self.mux_gate(sel, a, b, res_var)?;
        Ok(res_var)
    }

    // ---------------
    // | Constraints |
    // ---------------

    /// Common gates that should be implemented in any constraint systems.
    ///
    /// Constrain a variable to a constant.
    /// Return error if `var` is an invalid variable.
    fn enforce_constant(&mut self, var: Variable, constant: F) -> Result<(), CircuitError> {
        self.check_var(var)?;

        let wire_vars = &[0, 0, 0, 0, var];
        self.insert_gate(wire_vars, Box::new(ConstantGate(constant)))?;
        Ok(())
    }

    /// Constrain a variable to a bool.
    /// Return error if the input is invalid.
    fn enforce_bool(&mut self, a: Variable) -> Result<(), CircuitError> {
        self.check_var(a)?;

        let wire_vars = &[a, a, 0, 0, a];
        self.insert_gate(wire_vars, Box::new(BoolGate))?;
        Ok(())
    }

    /// Constrain two variables to have the same value.
    /// Return error if the input variables are invalid.
    fn enforce_equal(&mut self, a: Variable, b: Variable) -> Result<(), CircuitError> {
        self.check_var(a)?;
        self.check_var(b)?;

        let wire_vars = &[a, b, 0, 0, 0];
        self.insert_gate(wire_vars, Box::new(EqualityGate))?;
        Ok(())
    }
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
    /// separation ids. Return an error if the circuit does not support
    /// lookup or has not been finalized.
    fn compute_table_dom_sep_polynomial(&self) -> Result<DensePolynomial<F>, CircuitError> {
        Err(CircuitError::LookupUnsupported)
    }

    /// Compute and return the polynomial that interpolates the lookup domain
    /// separation selectors for the lookup gates. Return an error if the
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
