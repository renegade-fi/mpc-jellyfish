//! Defines arithmetic gadgets for use in circuits

use ark_ec::CurveGroup;
use ark_ff::One;
use ark_mpc::algebra::{AuthenticatedScalarResult, Scalar};
use mpc_relation::{
    constants::{GATE_WIDTH, N_MUL_SELECTORS},
    errors::CircuitError,
    gates::{
        ConstantAdditionGate, ConstantMultiplicationGate, FifthRootGate, LinCombGate, MulAddGate,
        QuadPolyGate,
    },
    Variable,
};

use crate::multiprover::proof_system::{MpcCircuit, MpcPlonkCircuit};

use super::{next_multiple, scalar};

impl<C: CurveGroup> MpcPlonkCircuit<C> {
    /// Arithmetic gates
    ///
    /// Quadratic polynomial gate: q1 * a + q2 * b + q3 * c + q4 * d + q12 * a *
    /// b + q34 * c * d + q_c = q_o * e, where q1, q2, q3, q4, q12, q34,
    /// q_c, q_o are selectors; a, b, c, d are input wires; e is the output
    /// wire. Return error if variables are invalid.
    pub fn quad_poly_gate(
        &mut self,
        wires: &[Variable; GATE_WIDTH + 1],
        q_lc: &[C::ScalarField; GATE_WIDTH],
        q_mul: &[C::ScalarField; N_MUL_SELECTORS],
        q_o: C::ScalarField,
        q_c: C::ScalarField,
    ) -> Result<(), CircuitError> {
        self.check_vars_bound(wires)?;

        self.insert_gate(
            wires,
            Box::new(QuadPolyGate {
                q_lc: *q_lc,
                q_mul: *q_mul,
                q_o,
                q_c,
            }),
        )?;
        Ok(())
    }

    /// Arithmetic gates
    ///
    /// Quadratic polynomial gate:
    /// e = q1 * a + q2 * b + q3 * c + q4 * d + q12 * a *
    /// b + q34 * c * d + q_c, where q1, q2, q3, q4, q12, q34,
    /// q_c are selectors; a, b, c, d are input wires
    ///
    /// Return the variable for
    /// Return error if variables are invalid.
    pub fn gen_quad_poly(
        &mut self,
        wires: &[Variable; GATE_WIDTH],
        q_lc: &[C::ScalarField; GATE_WIDTH],
        q_mul: &[C::ScalarField; N_MUL_SELECTORS],
        q_c: C::ScalarField,
    ) -> Result<Variable, CircuitError> {
        self.check_vars_bound(wires)?;
        let output_val = scalar!(q_lc[0]) * self.witness(wires[0])?
            + scalar!(q_lc[1]) * self.witness(wires[1])?
            + scalar!(q_lc[2]) * self.witness(wires[2])?
            + scalar!(q_lc[3]) * self.witness(wires[3])?
            + scalar!(q_mul[0]) * self.witness(wires[0])? * self.witness(wires[1])?
            + scalar!(q_mul[1]) * self.witness(wires[2])? * self.witness(wires[3])?
            + scalar!(q_c);
        let output_var = self.create_variable(output_val)?;
        let wires = [wires[0], wires[1], wires[2], wires[3], output_var];

        self.insert_gate(
            &wires,
            Box::new(QuadPolyGate {
                q_lc: *q_lc,
                q_mul: *q_mul,
                q_o: C::ScalarField::one(),
                q_c,
            }),
        )?;

        Ok(output_var)
    }

    /// Constrain a linear combination gate:
    /// q1 * a + q2 * b + q3 * c + q4 * d  = y
    pub fn lc_gate(
        &mut self,
        wires: &[Variable; GATE_WIDTH + 1],
        coeffs: &[C::ScalarField; GATE_WIDTH],
    ) -> Result<(), CircuitError> {
        self.check_vars_bound(wires)?;

        let wire_vars = [wires[0], wires[1], wires[2], wires[3], wires[4]];
        self.insert_gate(&wire_vars, Box::new(LinCombGate { coeffs: *coeffs }))?;
        Ok(())
    }

    /// Obtain a variable representing a linear combination.
    /// Return error if variables are invalid.
    pub fn lc(
        &mut self,
        wires_in: &[Variable; GATE_WIDTH],
        coeffs: &[C::ScalarField; GATE_WIDTH],
    ) -> Result<Variable, CircuitError> {
        self.check_vars_bound(wires_in)?;

        let vals_in: Vec<AuthenticatedScalarResult<C>> = wires_in
            .iter()
            .map(|&var| self.witness(var))
            .collect::<Result<Vec<_>, CircuitError>>()?;

        // calculate y as the linear combination of coeffs and vals_in
        let y_val = vals_in
            .iter()
            .zip(coeffs.iter())
            .map(|(val, coeff)| val * scalar!(*coeff))
            .sum();
        let y = self.create_variable(y_val)?;

        let wires = [wires_in[0], wires_in[1], wires_in[2], wires_in[3], y];
        self.lc_gate(&wires, coeffs)?;
        Ok(y)
    }

    /// Constrain a mul-addition gate:
    /// q_muls[0] * wires[0] *  wires[1] +  q_muls[1] * wires[2] *
    /// wires[3] = wires[4]
    pub fn mul_add_gate(
        &mut self,
        wires: &[Variable; GATE_WIDTH + 1],
        q_muls: &[C::ScalarField; N_MUL_SELECTORS],
    ) -> Result<(), CircuitError> {
        self.check_vars_bound(wires)?;

        let wire_vars = [wires[0], wires[1], wires[2], wires[3], wires[4]];
        self.insert_gate(&wire_vars, Box::new(MulAddGate { coeffs: *q_muls }))?;
        Ok(())
    }

    /// Obtain a variable representing `q12 * a * b + q34 * c * d`,
    /// where `a, b, c, d` are input wires, and `q12`, `q34` are selectors
    ///
    /// Return an error if variables are invalid
    pub fn mul_add(
        &mut self,
        wires_in: &[Variable; GATE_WIDTH],
        q_muls: &[C::ScalarField; N_MUL_SELECTORS],
    ) -> Result<Variable, CircuitError> {
        self.check_vars_bound(wires_in)?;

        let vals_in: Vec<AuthenticatedScalarResult<C>> = wires_in
            .iter()
            .map(|&var| self.witness(var))
            .collect::<Result<Vec<_>, CircuitError>>()?;

        // calculate y as the mul-addition of coeffs and vals_in
        let y_val = scalar!(q_muls[0]) * &vals_in[0] * &vals_in[1]
            + scalar!(q_muls[1]) * &vals_in[2] * &vals_in[3];
        let y = self.create_variable(y_val)?;

        let wires = [wires_in[0], wires_in[1], wires_in[2], wires_in[3], y];
        self.mul_add_gate(&wires, q_muls)?;
        Ok(y)
    }

    /// Create a variable representing the sum of a list of variables
    ///
    /// Return an error if variables are invalid
    pub fn sum(&mut self, elems: &[Variable]) -> Result<Variable, CircuitError> {
        if elems.is_empty() {
            return Err(CircuitError::ParameterError(
                "Sum over an empty slice of variables is undefined".to_string(),
            ));
        }
        self.check_vars_bound(elems)?;

        let mut elem_iter = elems
            .iter()
            .map(|elem| self.witness(*elem))
            .collect::<Result<Vec<_>, CircuitError>>()?
            .into_iter();
        let sum_val = elem_iter.next().expect("cannot sum empty slice");
        let sum_val = elem_iter.fold(sum_val, |acc, val| acc + val);
        let sum = self.create_variable(sum_val)?;

        // Pad with zeros and pack into as few gates as possibly by adding on all input
        // wires
        let mut padded: Vec<Variable> = elems.to_owned();
        let gate_capacity = GATE_WIDTH - 1;
        let padded_len = next_multiple(elems.len() - 1, gate_capacity)? + 1;
        padded.resize(padded_len, self.zero());

        // Construct a series of gates in which the output wire of the `n`th gate
        // is the accumulation of 3 * n input wires in the sequence
        let coeffs = [C::ScalarField::one(); GATE_WIDTH];
        let mut accum = padded[0];
        for i in 1..padded_len / gate_capacity {
            accum = self.lc(
                &[
                    accum,
                    padded[gate_capacity * i - 2],
                    padded[gate_capacity * i - 1],
                    padded[gate_capacity * i],
                ],
                &coeffs,
            )?;
        }

        // Final round
        let wires = [
            accum,
            padded[padded_len - 3],
            padded[padded_len - 2],
            padded[padded_len - 1],
            sum,
        ];
        self.lc_gate(&wires, &coeffs)?;

        Ok(sum)
    }

    /// Constrain variable `y` to the addition of `a` and `c`, where `c` is a
    /// constant value
    ///
    /// Return an error if the input variables are invalid
    pub fn add_constant_gate(
        &mut self,
        x: Variable,
        c: C::ScalarField,
        y: Variable,
    ) -> Result<(), CircuitError> {
        self.check_var_bound(x)?;
        self.check_var_bound(y)?;

        let wire_vars = &[x, self.one(), 0, 0, y];
        self.insert_gate(wire_vars, Box::new(ConstantAdditionGate(c)))?;
        Ok(())
    }

    /// Create a variable representing an addition with a constant value
    ///
    /// Return an error if the input variable is invalid
    pub fn add_constant(
        &mut self,
        input_var: Variable,
        elem: &C::ScalarField,
    ) -> Result<Variable, CircuitError> {
        self.check_var_bound(input_var)?;

        let input_val = self.witness(input_var).unwrap();
        let output_val = scalar!(*elem) + input_val;
        let output_var = self.create_variable(output_val).unwrap();

        self.add_constant_gate(input_var, *elem, output_var)?;

        Ok(output_var)
    }

    /// Constrain variable `y` to the product of `a` and `c`, where `c` is a
    /// constant value
    ///
    /// Return an error if the input variables are invalid.
    pub fn mul_constant_gate(
        &mut self,
        x: Variable,
        c: C::ScalarField,
        y: Variable,
    ) -> Result<(), CircuitError> {
        self.check_var_bound(x)?;
        self.check_var_bound(y)?;

        let wire_vars = &[x, 0, 0, 0, y];
        self.insert_gate(wire_vars, Box::new(ConstantMultiplicationGate(c)))?;
        Ok(())
    }

    /// Create a variable representing a multiplication with a constant value
    ///
    /// Return an error if the input variable is invalid
    pub fn mul_constant(
        &mut self,
        input_var: Variable,
        elem: &C::ScalarField,
    ) -> Result<Variable, CircuitError> {
        self.check_var_bound(input_var)?;

        let input_val = self.witness(input_var).unwrap();
        let output_val = scalar!(*elem) * input_val;
        let output_var = self.create_variable(output_val).unwrap();

        self.mul_constant_gate(input_var, *elem, output_var)?;

        Ok(output_var)
    }

    /// Create a variable representing the 11th power of the input variable
    ///
    /// Cost: 3 constraints
    pub fn power_11_gen(&mut self, x: Variable) -> Result<Variable, CircuitError> {
        self.check_var_bound(x)?;

        // now we prove that x^11 = x_to_11
        let x_val = self.witness(x)?;
        let x_to_5_val = x_val.pow(5);
        let x_to_5 = self.create_variable(x_to_5_val)?;

        let wire_vars = &[x, 0, 0, 0, x_to_5];
        self.insert_gate(wire_vars, Box::new(FifthRootGate))?;

        let x_to_10 = self.mul(x_to_5, x_to_5)?;
        self.mul(x_to_10, x)
    }

    /// Constraint a variable to be the 11th power of another variable
    ///
    /// Cost: 3 constraints
    pub fn power_11_gate(&mut self, x: Variable, x_to_11: Variable) -> Result<(), CircuitError> {
        self.check_var_bound(x)?;
        self.check_var_bound(x_to_11)?;

        // now we prove that x^11 = x_to_11
        let x_val = self.witness(x)?;
        let x_to_5_val = x_val.pow(5);
        let x_to_5 = self.create_variable(x_to_5_val)?;

        let wire_vars = &[x, 0, 0, 0, x_to_5];
        self.insert_gate(wire_vars, Box::new(FifthRootGate))?;

        let x_to_10 = self.mul(x_to_5, x_to_5)?;
        self.mul_gate(x_to_10, x, x_to_11)
    }
}
