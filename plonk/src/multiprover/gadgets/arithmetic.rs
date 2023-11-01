//! Defines arithmetic gadgets for use in circuits

use ark_ec::CurveGroup;
use ark_mpc::algebra::Scalar;
use jf_relation::{
    errors::CircuitError,
    gates::{ConstantAdditionGate, ConstantMultiplicationGate},
};

use crate::multiprover::proof_system::{MpcCircuit, MpcPlonkCircuit, MpcVariable};

impl<C: CurveGroup> MpcPlonkCircuit<C> {
    /// Constrain variable `y` to the addition of `a` and `c`, where `c` is a
    /// constant value Return error if the input variables are invalid.
    pub fn add_constant_gate(
        &mut self,
        x: MpcVariable,
        c: Scalar<C>,
        y: MpcVariable,
    ) -> Result<(), CircuitError> {
        self.check_var_bound(x)?;
        self.check_var_bound(y)?;

        let wire_vars = &[x, self.one(), 0, 0, y];
        self.insert_gate(wire_vars, Box::new(ConstantAdditionGate(c.inner())))?;
        Ok(())
    }

    /// Obtains a variable representing an addition with a constant value
    /// Return error if the input variable is invalid
    pub fn add_constant(
        &mut self,
        input_var: MpcVariable,
        elem: &Scalar<C>,
    ) -> Result<MpcVariable, CircuitError> {
        self.check_var_bound(input_var)?;

        let input_val = self.witness(input_var).unwrap();
        let output_val = *elem + input_val;
        let output_var = self.create_variable(output_val).unwrap();

        self.add_constant_gate(input_var, *elem, output_var)?;

        Ok(output_var)
    }

    /// Constrain variable `y` to the product of `a` and `c`, where `c` is a
    /// constant value Return error if the input variables are invalid.
    pub fn mul_constant_gate(
        &mut self,
        x: MpcVariable,
        c: Scalar<C>,
        y: MpcVariable,
    ) -> Result<(), CircuitError> {
        self.check_var_bound(x)?;
        self.check_var_bound(y)?;

        let wire_vars = &[x, 0, 0, 0, y];
        self.insert_gate(wire_vars, Box::new(ConstantMultiplicationGate(c.inner())))?;
        Ok(())
    }

    /// Obtains a variable representing a multiplication with a constant value
    /// Return error if the input variable is invalid
    pub fn mul_constant(
        &mut self,
        input_var: MpcVariable,
        elem: &Scalar<C>,
    ) -> Result<MpcVariable, CircuitError> {
        self.check_var_bound(input_var)?;

        let input_val = self.witness(input_var).unwrap();
        let output_val = *elem * input_val;
        let output_var = self.create_variable(output_val).unwrap();

        self.mul_constant_gate(input_var, *elem, output_var)?;

        Ok(output_var)
    }
}
