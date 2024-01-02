//! Defines the arithmetization and circuit abstractions over a base
//! MPC-enabled arithmetic

use std::collections::HashMap;

use ark_ec::CurveGroup;
use ark_ff::{FftField, One, Zero};
use ark_mpc::{
    algebra::{AuthenticatedDensePoly, AuthenticatedScalarResult, Scalar, ScalarResult},
    gadgets::prefix_product,
    MpcFabric,
};
use ark_poly::{
    univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, Radix2EvaluationDomain,
};
use async_trait::async_trait;
use futures::{executor::block_on, future};
use itertools::Itertools;
use mpc_relation::{
    constants::{compute_coset_representatives, GATE_WIDTH, N_MUL_SELECTORS},
    errors::CircuitError,
    gates::{FifthRootGate, Gate, IoGate, PaddingGate},
    proof_linking::GroupLayout,
    traits::*,
    BoolVar, GateId, Variable, WireId,
};

use super::element_wise_product;

// --------------------
// | Traits and Types |
// --------------------

/// An abstraction shimming the `Circuit` abstraction and the PIOP based
/// arguments in the MPC prover. The `MpcArithmetization` takes circuit wire
/// assignments and constructs polynomial representations of the assignment
pub trait MpcArithmetization<C: CurveGroup>: Circuit<C::ScalarField>
where
    C::ScalarField: FftField,
{
    /// The required SRS size for the circuit.
    fn srs_size(&self) -> Result<usize, CircuitError>;

    /// Get the size of the evaluation domain for arithmetization (after circuit
    /// has been finalized).
    fn eval_domain_size(&self) -> Result<usize, CircuitError>;

    /// Compute and return selector polynomials.
    /// Return an error if the circuit has not been finalized yet.
    fn compute_selector_polynomials(
        &self,
    ) -> Result<Vec<DensePolynomial<C::ScalarField>>, CircuitError>;

    /// Compute and return extended permutation polynomials.
    /// Return an error if the circuit has not been finalized yet.
    fn compute_extended_permutation_polynomials(
        &self,
    ) -> Result<Vec<DensePolynomial<C::ScalarField>>, CircuitError>;

    /// Compute and return the product polynomial for permutation arguments.
    /// Return an error if the circuit has not been finalized yet.
    fn compute_prod_permutation_polynomial(
        &self,
        beta: &ScalarResult<C>,
        gamma: &ScalarResult<C>,
    ) -> Result<AuthenticatedDensePoly<C>, CircuitError>;

    /// Compute and return the list of wiring witness polynomials.
    /// Return an error if the circuit has not been finalized yet.
    fn compute_wire_polynomials(&self) -> Result<Vec<AuthenticatedDensePoly<C>>, CircuitError>;

    /// Compute and return the public input polynomial.
    /// Return an error if the circuit has not been finalized yet.
    /// The IO gates of the circuit are guaranteed to be in the front.
    fn compute_pub_input_polynomial(&self) -> Result<AuthenticatedDensePoly<C>, CircuitError>;
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
    C::ScalarField: FftField + Unpin,
{
    /// The number of variables
    num_vars: usize,

    /// The gate of each constraint
    gates: Vec<Box<dyn Gate<C::ScalarField>>>,
    /// The map from arithmetic gate wires to variables
    wire_variables: [Vec<Variable>; GATE_WIDTH + 2],
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

    // -- Proof linking --
    // Proof linking allows us to prove that a subset of the witness of two proofs is the same. We
    // do so by adding proof-linking gates at specific indices in the circuit's arithmetization
    // of the form a(x) * 0 = 0, where a(x) encodes the witness to be linked. We can then
    // invoke a polynomial subprotocol to prove that the a(x) polynomial is the same between
    // proofs. The following fields are used to track membership in groups and placement of
    // groups in the arithmetization
    /// The proof-linking group layouts for the circuit. Maps a group ID to the
    /// indices of the witness contained in the group
    link_groups: HashMap<String, Vec<Variable>>,
    /// The offsets at which to place the link groups in the arithmetization
    link_group_layouts: HashMap<String, GroupLayout>,

    /// The underlying MPC fabric that this circuit is allocated within
    fabric: MpcFabric<C>,
}

impl<C: CurveGroup> MpcPlonkCircuit<C>
where
    C::ScalarField: FftField + Unpin,
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
            link_groups: HashMap::new(),
            link_group_layouts: HashMap::new(),
            fabric,
        };

        circuit.enforce_constant(0, C::ScalarField::zero()).unwrap();
        circuit.enforce_constant(1, C::ScalarField::one()).unwrap();

        circuit
    }

    /// Checks if a variable is strictly less than the number of variables.
    /// This function must be invoked for each gate as this check is not applied
    /// in the function `insert_gate`
    #[inline]
    pub fn check_var_bound(&self, var: Variable) -> Result<(), CircuitError> {
        if var >= self.num_vars {
            return Err(CircuitError::VarIndexOutOfBound(var, self.num_vars));
        }
        Ok(())
    }

    /// Check if a list of variables are strictly less than the number of
    /// variables
    pub fn check_vars_bound(&self, vars: &[Variable]) -> Result<(), CircuitError> {
        for &var in vars {
            self.check_var_bound(var)?
        }
        Ok(())
    }

    /// Change the value of a variable. Only used for testing.
    #[cfg(feature = "test_apis")]
    pub fn witness_mut(&mut self, idx: Variable) -> &mut AuthenticatedScalarResult<C> {
        &mut self.witness[idx]
    }

    /// Add a variable to a set of link groups
    pub fn add_to_link_groups(
        &mut self,
        var: Variable,
        link_groups: &[LinkGroup],
    ) -> Result<(), CircuitError> {
        self.check_finalize_flag(false)?;
        self.check_var_bound(var)?;

        for link_group in link_groups {
            self.link_groups
                .get_mut(&link_group.id)
                .ok_or(CircuitError::LinkGroupNotFound(link_group.id.to_string()))?
                .push(var);
        }

        Ok(())
    }

    /// Creating a `BoolVar` without checking if `v` is a boolean value!
    ///
    /// You should absolutely sure about what you are doing
    /// You should normally only use this API if you already enforce `v` to be a
    /// boolean value using other constraints.
    #[allow(unused)]
    pub(crate) fn create_boolean_variable_unchecked(
        &mut self,
        a: AuthenticatedScalarResult<C>,
    ) -> Result<BoolVar, CircuitError> {
        let var = self.create_variable(a)?;
        Ok(BoolVar::new_unchecked(var))
    }
}

/// Private helper methods
impl<C: CurveGroup> MpcPlonkCircuit<C>
where
    C::ScalarField: FftField + Unpin,
{
    /// Whether the circuit is finalized
    fn is_finalized(&self) -> bool {
        self.eval_domain.size() != 1
    }

    /// Re-arrange the order of the gates so that IO (public input) gates are at
    /// the front
    fn rearrange_gates(&mut self) -> Result<(), CircuitError> {
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
    fn pad(&mut self) -> Result<(), CircuitError> {
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
    pub fn check_gate(
        &self,
        gate_id: Variable,
        pub_input: &AuthenticatedScalarResult<C>,
    ) -> ScalarResult<C> {
        // Compute wire values
        let w_vals =
            (0..=GATE_WIDTH).map(|i| &self.witness[self.wire_variables[i][gate_id]]).collect_vec();

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
            pub_input + self.compute_gate_output(q_lc, q_mul, q_hash, q_ecc, q_c, &w_vals);
        let gate_output = q_o * w_vals[4];

        (gate_output - expected_gate_output).open()
    }

    /// Compute the output of a gate given its selectors and wire values
    ///
    /// This method differs from the single prover case because multiplication
    /// induces a significantly higher overhead. So we only compute the
    /// output of a gate if its selector is non-zero
    #[allow(clippy::too_many_arguments)]
    fn compute_gate_output(
        &self,
        q_lc: Vec<Scalar<C>>,
        q_mul: Vec<Scalar<C>>,
        q_hash: Vec<Scalar<C>>,
        q_ecc: Scalar<C>,
        q_c: Scalar<C>,
        wire_values: &[&AuthenticatedScalarResult<C>],
    ) -> AuthenticatedScalarResult<C> {
        let mut res = self.fabric.zero_authenticated();

        // Macro that adds a term to the result only if its selector is non-zero
        macro_rules! mask_selector {
            ($sel:expr, $x:expr) => {
                if $sel != Scalar::zero() {
                    res = res + $sel * $x;
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
        for (gate_wire_id, variables) in
            self.wire_variables.iter().take(self.num_wire_types()).enumerate()
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
    fn check_finalize_flag(&self, expect_finalized: bool) -> Result<(), CircuitError> {
        if !self.is_finalized() && expect_finalized {
            return Err(CircuitError::UnfinalizedCircuit);
        }
        if self.is_finalized() && !expect_finalized {
            return Err(CircuitError::ModifyFinalizedCircuit);
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

#[async_trait]
impl<C: CurveGroup> Circuit<C::ScalarField> for MpcPlonkCircuit<C>
where
    C::ScalarField: Unpin,
{
    type Wire = AuthenticatedScalarResult<C>;
    type Constant = Scalar<C>;

    fn zero(&self) -> Variable {
        0
    }

    fn one(&self) -> Variable {
        1
    }

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

    fn public_input(&self) -> Result<Vec<AuthenticatedScalarResult<C>>, CircuitError> {
        self.pub_input_gate_ids
            .iter()
            .map(|&gate_id| {
                let var = self.wire_variables[GATE_WIDTH][gate_id];
                self.witness(var)
            })
            .collect::<Result<Vec<_>, CircuitError>>()
    }

    fn witness(&self, idx: Variable) -> Result<AuthenticatedScalarResult<C>, CircuitError> {
        self.check_var_bound(idx)?;

        Ok(self.witness[idx].clone())
    }

    // Note: This method involves opening the witness values, it should only be
    // used in testing contexts
    #[cfg(feature = "test_apis")]
    fn check_circuit_satisfiability(
        &self,
        public_input: &[AuthenticatedScalarResult<C>],
    ) -> Result<(), CircuitError> {
        let n = public_input.len();
        if n != self.num_inputs() {
            return Err(CircuitError::PubInputLenMismatch(n, self.pub_input_gate_ids.len()));
        }

        let mut gate_results = Vec::new();

        // Check public I/O gates
        for (i, gate_id) in self.pub_input_gate_ids.iter().enumerate() {
            let pi = &public_input[i];
            gate_results.push(self.check_gate(*gate_id, pi));
        }

        // Check rest of the gates
        let zero = self.fabric.zero_authenticated();
        for gate_id in 0..self.num_gates() {
            if !self.is_io_gate(gate_id) {
                let res = self.check_gate(gate_id, &zero /* public_input */);
                gate_results.push(res);
            }
        }

        // Await all the gate results
        block_on(future::join_all(gate_results))
            .into_iter()
            .enumerate()
            .map(|(idx, res)| {
                if res == Scalar::zero() {
                    Ok(())
                } else {
                    Err(CircuitError::GateCheckFailure(idx, "gate check failed".to_string()))
                }
            })
            .collect::<Result<Vec<_>, CircuitError>>()
            .map(|_| ())
    }

    #[cfg(not(feature = "test_apis"))]
    fn check_circuit_satisfiability(
        &self,
        _public_input: &[AuthenticatedScalarResult<C>],
    ) -> Result<(), CircuitError> {
        panic!("`check_circuit_satisfiability` should not be called outside of tests, this method leaks privacy")
    }

    fn support_lookup(&self) -> bool {
        false
    }

    #[inline]
    fn check_var(&self, var: Variable) -> Result<(), CircuitError> {
        if var >= self.num_vars {
            return Err(CircuitError::VarIndexOutOfBound(var, self.num_vars));
        }
        Ok(())
    }

    fn create_constant_variable_with_link_groups(
        &mut self,
        val: C::ScalarField,
        link_groups: &[LinkGroup],
    ) -> Result<Variable, CircuitError> {
        let authenticated_val = self.fabric.one_authenticated() * Scalar::new(val);
        let var = self.create_variable_with_link_groups(authenticated_val, link_groups)?;
        self.enforce_constant(var, val)?;

        Ok(var)
    }

    fn create_variable_with_link_groups(
        &mut self,
        val: AuthenticatedScalarResult<C>,
        link_groups: &[LinkGroup],
    ) -> Result<Variable, CircuitError> {
        self.check_finalize_flag(false)?;
        self.witness.push(val);
        self.num_vars += 1;

        let var = self.num_vars - 1;
        self.add_to_link_groups(var, link_groups)?;

        Ok(var)
    }

    fn set_variable_public(&mut self, var: Variable) -> Result<(), CircuitError> {
        self.check_finalize_flag(false)?;
        self.pub_input_gate_ids.push(self.num_gates());

        // Create an io gate that forces `witness[var] = public_input`.
        let wire_vars = &[0, 0, 0, 0, var];
        self.insert_gate(wire_vars, Box::new(IoGate))?;
        Ok(())
    }

    fn create_link_group(&mut self, id: String, layout: Option<GroupLayout>) -> LinkGroup {
        self.link_groups.insert(id.clone(), Vec::new());
        if let Some(layout) = layout {
            self.link_group_layouts.insert(id.clone(), layout);
        }

        LinkGroup { id }
    }

    fn insert_gate(
        &mut self,
        wire_vars: &[Variable; GATE_WIDTH + 1],
        gate: Box<dyn Gate<C::ScalarField>>,
    ) -> Result<(), CircuitError> {
        self.check_finalize_flag(false)?;

        for (wire_var, wire_variable) in
            wire_vars.iter().zip(self.wire_variables.iter_mut().take(GATE_WIDTH + 1))
        {
            wire_variable.push(*wire_var)
        }

        self.gates.push(gate);
        Ok(())
    }

    fn pow5(&mut self, x: Variable) -> Result<Variable, CircuitError> {
        let val = self.witness(x)?;
        let res = val.pow(5);
        let res_var = self.create_variable(res)?;

        let wire_vars = &[x, 0, 0, 0, res_var];
        self.insert_gate(wire_vars, Box::new(FifthRootGate))?;
        Ok(res_var)
    }
}

/// Private permutation related methods
impl<C: CurveGroup> MpcPlonkCircuit<C>
where
    C::ScalarField: FftField + Unpin,
{
    /// Copy constraints: precompute the extended permutation over circuit
    /// wires. Refer to Sec 5.2 and Sec 8.1 of https://eprint.iacr.org/2019/953.pdf for more details.
    #[inline]
    fn compute_extended_id_permutation(&mut self) {
        assert!(self.is_finalized());
        let n = self.eval_domain.size();

        // Compute the extended identity permutation
        // id[i*n+j] = k[i] * g^j
        let k: Vec<C::ScalarField> = compute_coset_representatives(self.num_wire_types, Some(n));

        // Precompute domain elements
        let group_elems: Vec<C::ScalarField> = self.eval_domain.elements().collect();

        // Compute extended identity permutation
        self.extended_id_permutation = vec![C::ScalarField::zero(); self.num_wire_types * n];
        for (i, &coset_repr) in k.iter().enumerate() {
            for (j, &group_elem) in group_elems.iter().enumerate() {
                self.extended_id_permutation[i * n + j] = coset_repr * group_elem;
            }
        }
    }

    #[inline]
    fn compute_extended_permutation(&self) -> Result<Vec<C::ScalarField>, CircuitError> {
        assert!(self.is_finalized());
        let n = self.eval_domain.size();

        // The extended wire permutation can be computed as
        // extended_perm[i] = id[wire_perm[i].into() * n + wire_perm[i].1]
        let extended_perm: Vec<C::ScalarField> = self
            .wire_permutation
            .iter()
            .map(|&(wire_id, gate_id)| {
                // if permutation value undefined, return 0
                if wire_id >= self.num_wire_types {
                    C::ScalarField::zero()
                } else {
                    self.extended_id_permutation[wire_id * n + gate_id]
                }
            })
            .collect();

        if extended_perm.len() != self.num_wire_types * n {
            return Err(CircuitError::ParameterError(
                "Length of the extended permutation vector should be number of gate \
                         (including padded dummy gates) * number of wire types"
                    .to_string(),
            ));
        }
        Ok(extended_perm)
    }
}

/// Finalization
impl<C: CurveGroup> MpcPlonkCircuit<C>
where
    C::ScalarField: FftField + Unpin,
{
    /// Finalize the setup of the circuit before arithmetization.
    pub fn finalize_for_arithmetization(&mut self) -> Result<(), CircuitError> {
        if self.is_finalized() {
            return Ok(());
        }

        self.eval_domain = Radix2EvaluationDomain::new(self.num_gates())
            .ok_or(CircuitError::DomainCreationError)?;
        self.pad()?;
        self.rearrange_gates()?;
        self.compute_wire_permutation();
        self.compute_extended_id_permutation();
        Ok(())
    }
}

impl<C: CurveGroup> MpcArithmetization<C> for MpcPlonkCircuit<C>
where
    C::ScalarField: Unpin,
{
    fn srs_size(&self) -> Result<usize, CircuitError> {
        Ok(self.eval_domain_size()? + 2)
    }

    fn eval_domain_size(&self) -> Result<usize, CircuitError> {
        self.check_finalize_flag(true)?;
        Ok(self.eval_domain.size())
    }

    fn compute_selector_polynomials(
        &self,
    ) -> Result<Vec<DensePolynomial<<C>::ScalarField>>, CircuitError> {
        self.check_finalize_flag(true)?;
        let domain = &self.eval_domain;
        if domain.size() < self.num_gates() {
            return Err(CircuitError::ParameterError(
                "Domain size should be bigger than number of constraint".to_string(),
            ));
        }

        // Order: (lc, mul, hash, o, c, ecc) as specified in spec
        let selector_polys = self
            .all_selectors()
            .iter()
            .map(|selector| DensePolynomial::from_coefficients_vec(domain.ifft(selector)))
            .collect();
        Ok(selector_polys)
    }

    fn compute_extended_permutation_polynomials(
        &self,
    ) -> Result<Vec<DensePolynomial<C::ScalarField>>, CircuitError> {
        self.check_finalize_flag(true)?;
        let domain = &self.eval_domain;
        let n = domain.size();
        let extended_perm = self.compute_extended_permutation()?;

        let extended_perm_polys: Vec<DensePolynomial<C::ScalarField>> = (0..self.num_wire_types)
            .map(|i| {
                DensePolynomial::from_coefficients_vec(
                    domain.ifft(&extended_perm[i * n..(i + 1) * n]),
                )
            })
            .collect();

        Ok(extended_perm_polys)
    }

    fn compute_prod_permutation_polynomial(
        &self,
        beta: &ScalarResult<C>,
        gamma: &ScalarResult<C>,
    ) -> Result<AuthenticatedDensePoly<C>, CircuitError> {
        self.check_finalize_flag(true)?;
        let n = self.eval_domain.size();
        let one = self.fabric.one_authenticated();

        let gammas = vec![gamma.clone(); n - 1];
        let betas = vec![beta.clone(); n - 1];

        let mut numerator_terms = Vec::with_capacity(self.num_wire_types());
        let mut denominator_terms = Vec::with_capacity(self.num_wire_types());
        for i in 0..self.num_wire_types() {
            let wire_values =
                (0..(n - 1)).map(|j| self.witness[self.wire_variable(i, j)].clone()).collect_vec();
            let id_perm_values = (0..(n - 1))
                .map(|j| Scalar::new(self.extended_id_permutation[i * n + j]))
                .collect_vec();
            let perm_values = (0..(n - 1))
                .map(|j| self.wire_permutation[i * n + j])
                .map(|(perm_i, perm_j)| self.extended_id_permutation[perm_i * n + perm_j])
                .map(Scalar::new)
                .collect_vec();

            let tmp = AuthenticatedScalarResult::batch_add_public(&wire_values, &gammas);
            let beta_times_id_perm = ScalarResult::batch_mul_constant(&betas, &id_perm_values);
            let beta_times_sigma_perm = ScalarResult::batch_mul_constant(&betas, &perm_values);

            let num_term = AuthenticatedScalarResult::batch_add_public(&tmp, &beta_times_id_perm);
            let denom_term =
                AuthenticatedScalarResult::batch_add_public(&tmp, &beta_times_sigma_perm);

            numerator_terms.push(num_term);
            denominator_terms.push(denom_term);
        }

        let numerators = element_wise_product(&numerator_terms);
        let denominators = element_wise_product(&denominator_terms);

        // Divide the numerators and denominators, create a prefix product of this
        // division, and then convert into a polynomial from evaluation form
        let div_res = AuthenticatedScalarResult::batch_div(&numerators, &denominators);
        let products = prefix_product(&div_res, &self.fabric);

        // The last element of this product is one for a valid proof, we match the
        // single-prover implementation and put this first in the resulting
        // prefix product vec
        let product_vec = [vec![one], products].concat();
        let coeffs =
            AuthenticatedScalarResult::ifft::<Radix2EvaluationDomain<C::ScalarField>>(&product_vec);

        Ok(AuthenticatedDensePoly::from_coeffs(coeffs))
    }

    fn compute_wire_polynomials(&self) -> Result<Vec<AuthenticatedDensePoly<C>>, CircuitError> {
        self.check_finalize_flag(true)?;
        let domain = &self.eval_domain;
        if domain.size() < self.num_gates() {
            return Err(CircuitError::ParameterError(format!(
                "Domain size {} should be bigger than number of constraint {}",
                domain.size(),
                self.num_gates()
            )));
        }

        let witness = &self.witness;
        let wire_polys: Vec<AuthenticatedDensePoly<C>> = self
            .wire_variables
            .iter()
            .take(self.num_wire_types())
            .map(|wire_vars| {
                let wire_vec: Vec<AuthenticatedScalarResult<C>> =
                    wire_vars.iter().map(|&var| witness[var].clone()).collect_vec();

                let coeffs = AuthenticatedScalarResult::ifft::<
                    Radix2EvaluationDomain<C::ScalarField>,
                >(&wire_vec);

                AuthenticatedDensePoly::from_coeffs(coeffs)
            })
            .collect();

        assert_eq!(wire_polys.len(), self.num_wire_types());
        Ok(wire_polys)
    }

    fn compute_pub_input_polynomial(&self) -> Result<AuthenticatedDensePoly<C>, CircuitError> {
        self.check_finalize_flag(true)?;

        let domain = &self.eval_domain;
        let mut pub_input_vec = self.fabric.zeros_authenticated(domain.size());

        self.pub_input_gate_ids.iter().for_each(|&io_gate_id| {
            let var = self.wire_variables[GATE_WIDTH][io_gate_id];
            pub_input_vec[io_gate_id] = self.witness[var].clone();
        });

        let coeffs = AuthenticatedScalarResult::ifft::<Radix2EvaluationDomain<C::ScalarField>>(
            &pub_input_vec,
        );
        Ok(AuthenticatedDensePoly::from_coeffs(coeffs))
    }
}
