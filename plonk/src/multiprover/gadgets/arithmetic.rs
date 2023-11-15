//! Defines arithmetic gadgets for use in circuits

use ark_ec::CurveGroup;
use ark_ff::One;
use ark_mpc::algebra::Scalar;
use mpc_relation::{
    constants::{GATE_WIDTH, N_MUL_SELECTORS},
    errors::CircuitError,
    gates::{FifthRootGate, QuadPolyGate},
    traits::*,
    Variable,
};

use crate::multiprover::proof_system::{MpcCircuit, MpcPlonkCircuit};

use super::scalar;

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
