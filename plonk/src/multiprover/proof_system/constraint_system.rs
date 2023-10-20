//! Defines the arithmetization and circuit abstractions over a base
//! MPC-enabled arithmetic

use ark_ec::CurveGroup;
use ark_ff::FftField;
use ark_mpc::{
    algebra::{AuthenticatedDensePoly, AuthenticatedScalarResult, Scalar, ScalarResult},
    MpcFabric,
};
use ark_poly::{univariate::DensePolynomial, EvaluationDomain, Radix2EvaluationDomain};
use itertools::Itertools;
use jf_relation::{
    constants::{GATE_WIDTH, N_MUL_SELECTORS},
    errors::CircuitError,
    gates::{Gate, IoGate, PaddingGate},
    GateId, Variable, WireId,
};

use super::MpcCircuitError;

// --------------------
// | Traits and Types |
// --------------------

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
    fn num_vars(&self) -> usize;

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
    fn create_constant_variable(&mut self, val: Scalar<C>) -> Result<MpcVariable, MpcCircuitError>;

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
        constant: Scalar<C>,
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

// --------------------------------
// | Plonk Circuit Implementation |
// --------------------------------

/// A Plonk circuit instantiated over an MPC fabric
///
/// Largely borrowed from the single-prover implementation with modification of
/// the field to be a secret shared analog
#[derive(Clone)]
/// TODO: Remove this lint allowance
#[allow(unused)]
pub struct MpcPlonkCircuit<C: CurveGroup>
where
    C::ScalarField: FftField,
{
    /// The number of variables
    num_vars: usize,

    /// The gate of each constraint
    gates: Vec<Box<dyn Gate<C::ScalarField>>>,
    /// The map from arithmetic gate wires to variables
    wire_variables: [Vec<MpcVariable>; GATE_WIDTH + 2],
    /// The IO gates for the list of public input variables
    pub_input_gate_ids: Vec<GateId>,
    /// The assignment of the witness variables
    witness: Vec<AuthenticatedScalarResult<C>>,

    /// The permutation over wires.
    /// Each algebraic gate has 5 wires, i.e., 4 input wires and an output
    /// wire; each lookup gate has a single wire that maps to a witness to
    /// be checked over the lookup table. In total there are 6 * n wires
    /// where n is the (padded) number of arithmetic/lookup gates.  
    /// We build a permutation over the set of wires so that each set of wires
    /// that map to the same witness forms a cycle.
    ///
    /// Each wire is represented by a pair (`WireId, GateId`) so that the wire
    /// is in the `GateId`-th arithmetic/lookup gate and `WireId` represents
    /// the wire type (e.g., 0 represents 1st input wires, 4 represents
    /// output wires, and 5 represents lookup wires).
    wire_permutation: Vec<(WireId, GateId)>,
    /// The extended identity permutation.
    extended_id_permutation: Vec<C::ScalarField>,
    /// The number of wire types. 5 for TurboPlonk and 6 for UltraPlonk.
    num_wire_types: usize,

    /// The evaluation domain for arithmetization of the circuit into various
    /// polynomials. This is only relevant after the circuit is finalized for
    /// arithmetization, by default it is a domain with size 1 (only with
    /// element 0).
    eval_domain: Radix2EvaluationDomain<C::ScalarField>,

    /// The underlying MPC fabric that this circuit is allocated within
    fabric: MpcFabric<C>,
}

impl<C: CurveGroup> MpcPlonkCircuit<C>
where
    C::ScalarField: FftField,
{
    /// Constructor
    pub fn new(fabric: MpcFabric<C>) -> Self {
        let zero = fabric.zero_authenticated();
        let one = fabric.one_authenticated();

        // First build the circuit
        let mut circuit = Self {
            num_vars: 2,
            witness: vec![zero, one],
            gates: vec![],
            wire_variables: [vec![], vec![], vec![], vec![], vec![], vec![]],
            pub_input_gate_ids: vec![],

            wire_permutation: vec![],
            extended_id_permutation: vec![],
            num_wire_types: GATE_WIDTH + 1,

            // This is later updated
            eval_domain: Radix2EvaluationDomain::new(1 /* num_coeffs */).unwrap(),
            fabric,
        };

        circuit.enforce_constant(0, Scalar::zero()).unwrap();
        circuit.enforce_constant(1, Scalar::one()).unwrap();

        circuit
    }
}

/// Private helper methods
impl<C: CurveGroup> MpcPlonkCircuit<C> {
    /// Whether the circuit is finalized
    fn is_finalized(&self) -> bool {
        self.eval_domain.size() != 1
    }

    /// Re-arrange the order of the gates so that IO (public input) gates are at
    /// the front
    fn rearrange_gates(&mut self) -> Result<(), MpcCircuitError> {
        self.check_finalize_flag(true)?;
        for (gate_id, io_gate_id) in self.pub_input_gate_ids.iter_mut().enumerate() {
            if *io_gate_id > gate_id {
                // Swap gate types
                self.gates.swap(gate_id, *io_gate_id);

                // Swap wire variables
                for i in 0..GATE_WIDTH + 1 {
                    self.wire_variables[i].swap(gate_id, *io_gate_id);
                }

                // Update io gate index
                *io_gate_id = gate_id;
            }
        }

        Ok(())
    }

    /// Use downcast to check whether a gate is of IoGate type
    fn is_io_gate(&self, gate_id: GateId) -> bool {
        self.gates[gate_id].as_any().is::<IoGate>()
    }

    /// Pad a finalized circuit to match the evaluation domain, prepared for
    /// arithmetization
    ///
    /// This is a pad to a power of two
    fn pad(&mut self) -> Result<(), MpcCircuitError> {
        self.check_finalize_flag(true)?;
        let n = self.eval_domain.size();
        for _ in self.num_gates()..n {
            self.gates.push(Box::new(PaddingGate));
        }
        for wire_id in 0..self.num_wire_types() {
            self.wire_variables[wire_id].resize(n, self.zero());
        }
        Ok(())
    }

    /// Check that the `gate_id`-th gate is satisfied by the circuit's witness
    /// and the public input value `pub_input`. `gate_id` is guaranteed to
    /// be in the range. The gate equation:
    /// qo * wo = pub_input + q_c +
    ///           q_mul0 * w0 * w1 + q_mul1 * w2 * w3 +
    ///           q_lc0 * w0 + q_lc1 * w1 + q_lc2 * w2 + q_lc3 * w3 +
    ///           q_hash0 * w0 + q_hash1 * w1 + q_hash2 * w2 + q_hash3 * w3 +
    ///           q_ecc * w0 * w1 * w2 * w3 * wo
    ///
    /// Returns the result of the gate minus the expected result. I.e. the gate
    /// is correctly computed if the result is zero.
    ///
    /// Note: This method opens the values of the witness to check
    /// satisfiability, it should only be used for testing
    #[cfg(feature = "test_apis")]
    fn check_gate(
        &self,
        gate_id: Variable,
        pub_input: &AuthenticatedScalarResult<C>,
    ) -> ScalarResult<C> {
        // Compute wire values
        let w_vals = (0..=GATE_WIDTH)
            .map(|i| &self.witness[self.wire_variables[i][gate_id]])
            .collect_vec();

        // Compute selector values
        macro_rules! as_scalars {
            ($x:expr) => {
                $x.iter().map(|x| Scalar::new(*x)).collect_vec()
            };
        }
        let q_lc = as_scalars!(self.gates[gate_id].q_lc());
        let q_mul = as_scalars!(self.gates[gate_id].q_mul());
        let q_hash = as_scalars!(self.gates[gate_id].q_hash());
        let q_c = Scalar::new(self.gates[gate_id].q_c());
        let q_o = Scalar::new(self.gates[gate_id].q_o());
        let q_ecc = Scalar::new(self.gates[gate_id].q_ecc());

        // Compute the gate output
        let expected_gate_output =
            pub_input + self.compute_gate_output(q_lc, q_mul, q_hash, q_ecc, q_c, q_o, &w_vals);
        let gate_output = q_o * w_vals[4];

        (gate_output - expected_gate_output).open()
    }

    /// Compute the output of a gate given its selectors and wire values
    ///
    /// This method differs from the single prover case because multiplication
    /// induces a significantly higher overhead. So we only compute the
    /// output of a gate if its selector is non-zero
    fn compute_gate_output(
        &self,
        q_lc: Vec<Scalar<C>>,
        q_mul: Vec<Scalar<C>>,
        q_hash: Vec<Scalar<C>>,
        q_ecc: Scalar<C>,
        q_c: Scalar<C>,
        q_o: Scalar<C>,
        wire_values: &[&AuthenticatedScalarResult<C>],
    ) -> AuthenticatedScalarResult<C> {
        let mut res = self.fabric.zero_authenticated();

        // Macro that adds a term to the result only if its selector is non-zero
        macro_rules! mask_selector {
            ($sel:expr, $x:expr) => {
                if $sel != Scalar::zero() {
                    res = res + $x;
                }
            };
        }

        mask_selector!(q_lc[0], wire_values[0]);
        mask_selector!(q_lc[1], wire_values[1]);
        mask_selector!(q_lc[2], wire_values[2]);
        mask_selector!(q_lc[3], wire_values[3]);
        mask_selector!(q_mul[0], wire_values[0] * wire_values[1]);
        mask_selector!(q_mul[1], wire_values[2] * wire_values[3]);
        mask_selector!(
            q_ecc,
            wire_values[0] * wire_values[1] * wire_values[2] * wire_values[3] * wire_values[4]
        );
        mask_selector!(q_hash[0], wire_values[0].pow(5));
        mask_selector!(q_hash[1], wire_values[1].pow(5));
        mask_selector!(q_hash[2], wire_values[2].pow(5));
        mask_selector!(q_hash[3], wire_values[3].pow(5));
        res = res + q_c;

        res
    }

    // Compute the permutation over wires.
    // The circuit is guaranteed to be padded before calling the method.
    #[inline]
    fn compute_wire_permutation(&mut self) {
        assert!(self.is_finalized());
        let n = self.eval_domain.size();
        let m = self.num_vars();

        // Compute the mapping from variables to wires.
        let mut variable_wires_map = vec![vec![]; m];
        for (gate_wire_id, variables) in self
            .wire_variables
            .iter()
            .take(self.num_wire_types())
            .enumerate()
        {
            for (gate_id, &var) in variables.iter().enumerate() {
                variable_wires_map[var].push((gate_wire_id, gate_id));
            }
        }

        // Compute the wire permutation
        self.wire_permutation = vec![(0usize, 0usize); self.num_wire_types * n];
        for wires_vec in variable_wires_map.iter_mut() {
            // The list of wires that map to the same variable forms a cycle.
            if !wires_vec.is_empty() {
                // Push the first item so that window iterator will visit the last item
                // paired with the first item, forming a cycle
                wires_vec.push(wires_vec[0]);
                for window in wires_vec.windows(2) {
                    self.wire_permutation[window[0].0 * n + window[0].1] = window[1];
                }

                // Remove the extra first item pushed at the beginning of the iterator
                wires_vec.pop();
            }
        }
    }

    // Check whether the circuit is finalized. Return an error if the finalizing
    // status is different from the expected status.
    #[inline]
    fn check_finalize_flag(&self, expect_finalized: bool) -> Result<(), MpcCircuitError> {
        if !self.is_finalized() && expect_finalized {
            return Err(MpcCircuitError::ConstraintSystem(
                CircuitError::UnfinalizedCircuit,
            ));
        }
        if self.is_finalized() && !expect_finalized {
            return Err(MpcCircuitError::ConstraintSystem(
                CircuitError::ModifyFinalizedCircuit,
            ));
        }
        Ok(())
    }

    // Return the variable that maps to a wire `(i, j)` where i is the wire type and
    // j is the gate index. If gate `j` is a padded dummy gate, return zero
    // variable.
    #[inline]
    fn wire_variable(&self, i: WireId, j: GateId) -> Variable {
        match j < self.wire_variables[i].len() {
            true => self.wire_variables[i][j],
            false => self.zero(),
        }
    }

    // Getter for all linear combination selector
    #[inline]
    fn q_lc(&self) -> [Vec<C::ScalarField>; GATE_WIDTH] {
        let mut result = [vec![], vec![], vec![], vec![]];
        for gate in &self.gates {
            let q_lc_vec = gate.q_lc();
            result[0].push(q_lc_vec[0]);
            result[1].push(q_lc_vec[1]);
            result[2].push(q_lc_vec[2]);
            result[3].push(q_lc_vec[3]);
        }
        result
    }

    // Getter for all multiplication selector
    #[inline]
    fn q_mul(&self) -> [Vec<C::ScalarField>; N_MUL_SELECTORS] {
        let mut result = [vec![], vec![]];
        for gate in &self.gates {
            let q_mul_vec = gate.q_mul();
            result[0].push(q_mul_vec[0]);
            result[1].push(q_mul_vec[1]);
        }
        result
    }

    // Getter for all hash selector
    #[inline]
    fn q_hash(&self) -> [Vec<C::ScalarField>; GATE_WIDTH] {
        let mut result = [vec![], vec![], vec![], vec![]];
        for gate in &self.gates {
            let q_hash_vec = gate.q_hash();
            result[0].push(q_hash_vec[0]);
            result[1].push(q_hash_vec[1]);
            result[2].push(q_hash_vec[2]);
            result[3].push(q_hash_vec[3]);
        }
        result
    }

    // Getter for all output selector
    #[inline]
    fn q_o(&self) -> Vec<C::ScalarField> {
        self.gates.iter().map(|g| g.q_o()).collect()
    }

    // Getter for all constant selector
    #[inline]
    fn q_c(&self) -> Vec<C::ScalarField> {
        self.gates.iter().map(|g| g.q_c()).collect()
    }

    // Getter for all ecc selector
    #[inline]
    fn q_ecc(&self) -> Vec<C::ScalarField> {
        self.gates.iter().map(|g| g.q_ecc()).collect()
    }

    // Getter for all selectors in the following order:
    // q_lc, q_mul, q_hash, q_o, q_c, q_ecc, [q_lookup (if support lookup)]
    #[inline]
    fn all_selectors(&self) -> Vec<Vec<C::ScalarField>> {
        let mut selectors = vec![];
        self.q_lc()
            .as_ref()
            .iter()
            .chain(self.q_mul().as_ref().iter())
            .chain(self.q_hash().as_ref().iter())
            .for_each(|s| selectors.push(s.clone()));
        selectors.push(self.q_o());
        selectors.push(self.q_c());
        selectors.push(self.q_ecc());

        selectors
    }
}

impl<C: CurveGroup> MpcCircuit<C> for MpcPlonkCircuit<C> {
    fn num_gates(&self) -> usize {
        self.gates.len()
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }
    fn num_inputs(&self) -> usize {
        self.pub_input_gate_ids.len()
    }

    fn num_wire_types(&self) -> usize {
        self.num_wire_types
    }

    fn public_input(&self) -> Result<Vec<AuthenticatedScalarResult<C>>, MpcCircuitError> {
        self.pub_input_gate_ids
            .iter()
            .map(|&gate_id| {
                let var = self.wire_variables[GATE_WIDTH][gate_id];
                self.witness(var)
            })
            .collect::<Result<Vec<_>, MpcCircuitError>>()
    }

    // Note: This method involves opening the witness values, it should only be
    // used in testing contexts
    #[cfg(feature = "test_apis")]
    fn check_circuit_satisfiability(
        &self,
        public_input: &[AuthenticatedScalarResult<C>],
    ) -> Result<(), MpcCircuitError> {
        unimplemented!()
    }

    #[cfg(not(feature = "test_apis"))]
    fn check_circuit_satisfiability(
        &self,
        public_input: &[AuthenticatedScalarResult<C>],
    ) -> Result<(), MpcCircuitError> {
        panic("`check_circuit_satisfiability` should not be called outside of tests, this method leaks privacy")
    }

    fn create_constant_variable(&mut self, val: Scalar<C>) -> Result<MpcVariable, MpcCircuitError> {
        let authenticated_val = self.fabric.one_authenticated() * &val;
        let var = self.create_variable(authenticated_val)?;
        self.enforce_constant(var, val)?;

        Ok(var)
    }

    fn create_variable(
        &mut self,
        val: AuthenticatedScalarResult<C>,
    ) -> Result<MpcVariable, MpcCircuitError> {
        unimplemented!()
    }

    fn create_public_variable(
        &mut self,
        val: AuthenticatedScalarResult<C>,
    ) -> Result<MpcVariable, MpcCircuitError> {
        unimplemented!()
    }

    fn set_variable_public(&mut self, var: MpcVariable) -> Result<(), MpcCircuitError> {
        unimplemented!()
    }

    fn zero(&self) -> MpcVariable {
        0
    }

    fn one(&self) -> MpcVariable {
        1
    }

    fn witness(&self, idx: MpcVariable) -> Result<AuthenticatedScalarResult<C>, MpcCircuitError> {
        unimplemented!()
    }

    fn enforce_constant(
        &mut self,
        var: MpcVariable,
        constant: Scalar<C>,
    ) -> Result<(), MpcCircuitError> {
        unimplemented!()
    }

    fn add_gate(
        &mut self,
        a: MpcVariable,
        b: MpcVariable,
        c: MpcVariable,
    ) -> Result<(), MpcCircuitError> {
        unimplemented!()
    }

    fn add(&mut self, a: MpcVariable, b: MpcVariable) -> Result<MpcVariable, MpcCircuitError> {
        unimplemented!()
    }

    fn sub_gate(
        &mut self,
        a: MpcVariable,
        b: MpcVariable,
        c: MpcVariable,
    ) -> Result<(), MpcCircuitError> {
        unimplemented!()
    }

    fn sub(&mut self, a: MpcVariable, b: MpcVariable) -> Result<MpcVariable, MpcCircuitError> {
        unimplemented!()
    }

    fn mul_gate(
        &mut self,
        a: MpcVariable,
        b: MpcVariable,
        c: MpcVariable,
    ) -> Result<(), MpcCircuitError> {
        unimplemented!()
    }

    fn mul(&mut self, a: MpcVariable, b: MpcVariable) -> Result<MpcVariable, MpcCircuitError> {
        unimplemented!()
    }

    fn enforce_bool(&mut self, a: MpcVariable) -> Result<(), MpcCircuitError> {
        unimplemented!()
    }

    fn enforce_equal(&mut self, a: MpcVariable, b: MpcVariable) -> Result<(), MpcCircuitError> {
        unimplemented!()
    }

    fn pad_gates(&mut self, n: usize) {
        unimplemented!()
    }
}
