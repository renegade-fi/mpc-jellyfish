//! Defines the `LinkableCircuit` trait, which allows us to provide default
//! implementations of the proof-linking arithmetization methods for circuits.
//! These default implementations require a slim interface and allow us to
//! centralize the logical complexity of generating a proof-linkable
//! arithmetization

use ark_ff::Field;
use ark_std::iterable::Iterable;
use itertools::Itertools;
use std::iter;

use crate::{
    constants::GATE_WIDTH,
    errors::CircuitError,
    gates::{Gate, PaddingGate, ProofLinkingGate},
    traits::Circuit,
    BoolVar, Variable,
};

use super::{CircuitLayout, GroupLayout, LinkGroup};

/// The index in the wire polynomials that encodes the proof linking gates
pub const PROOF_LINK_WIRE_IDX: usize = 0;

/// A circuit that can be arithmetized for proof linking
pub trait LinkableCircuit<F: Field>: Circuit<F> {
    // --- Group Creation & Allocation --- //

    /// The number of proof-linked inputs to the circuit
    fn num_links(&self) -> usize;

    /// Add a variable to the witness; with an optional set of link groups to
    /// allocate it within
    fn create_variable_with_link_groups(
        &mut self,
        val: Self::Wire,
        link_groups: &[LinkGroup],
    ) -> Result<Variable, CircuitError>;

    /// Add a constant variable to the circuit; with an optional set of link
    /// groups to allocate it within
    fn create_constant_variable_with_link_groups(
        &mut self,
        val: Self::Constant,
        link_groups: &[LinkGroup],
    ) -> Result<Variable, CircuitError>;

    /// Add a constant bool variable to the circuit; with an optional set of
    /// link groups to allocate it within
    fn create_boolean_variable_with_link_groups<T: Into<Self::Wire>>(
        &mut self,
        val: T,
        link_groups: &[LinkGroup],
    ) -> Result<BoolVar, CircuitError> {
        let var = self.create_variable_with_link_groups(val.into(), link_groups)?;
        self.enforce_bool(var)?;
        Ok(BoolVar(var))
    }

    /// Set the layout of a link group
    fn set_group_layout(&mut self, id: String, layout: GroupLayout);

    /// Create a proof linking group in the circuit. Witness elements may be
    /// allocated into this group in order and they will be placed into the
    /// given layout as proof-linking gates (or assigned a default layout if
    /// none is provided).
    ///
    /// These proof linking gates are effectively of the form a(x) * 0 = 0,
    /// where a(x) is the witness element. This allows us to prove
    /// that the a(x) polynomial of one proof equals the a(x) polynomial of
    /// another proof over some proof-linking domain, represented by the group
    fn create_link_group(&mut self, id: String, layout: Option<GroupLayout>) -> LinkGroup;

    // --- Gate + Wire Access --- //

    /// Get a reference to the gates of the circuit
    fn gates(&self) -> &[Box<dyn Gate<F>>];

    /// Get a reference to the wires of the circuit
    fn wires(&self) -> &[Vec<Variable>; GATE_WIDTH + 2];

    /// Get a mutable reference to the gates of the circuit
    ///
    /// Warning: this is an internal method and should not be used outside of
    /// the crate
    #[doc(hidden)]
    unsafe fn gates_mut(&mut self) -> &mut Vec<Box<dyn Gate<F>>>;

    /// Get a mutable reference to the wires of the circuit
    ///
    /// Warning: this is an internal method and should not be used outside of
    /// the crate
    #[doc(hidden)]
    unsafe fn wires_mut(&mut self) -> &mut [Vec<Variable>; GATE_WIDTH + 2];

    // --- Layout and Arithmetization Methods --- //

    /// Setup the order of the gates before the proof linking gates are added
    ///
    /// E.g. to move public input gates to the front or plookup gates to the
    /// back
    fn order_gates(&mut self) -> Result<(), CircuitError>;

    /// Get the current alignment of the circuit
    fn current_alignment(&self) -> usize {
        // The maximum alignment needed by any link group
        let mut max_alignment = 0;
        for id in self.link_group_ids().iter() {
            if let Some(layout) = self.get_link_group_layout(id) {
                max_alignment = usize::max(max_alignment, layout.alignment);
            }
        }

        // The total number of gates after proof linking gates are added
        let n_gates = self.num_gates() + self.num_links();
        let gates_order = n_gates.next_power_of_two().ilog2() as usize;
        usize::max(max_alignment, gates_order)
    }

    /// Get the IDs of all the link groups in the circuit
    fn link_group_ids(&self) -> Vec<String>;

    /// Get a link group's membership list by its ID
    ///
    /// Returns `None` if the group does not exist
    fn get_link_group_members(&self, id: &str) -> Option<&[Variable]>;

    /// Get the group layout of a link group if available
    ///
    /// Returns `None` if the group does not have a layout specified
    fn get_link_group_layout(&self, id: &str) -> Option<GroupLayout>;

    /// Generate a layout for the circuit including the placement of link groups
    ///
    /// Mutates the circuit by placing the link groups with unspecified layouts
    fn generate_layout(&mut self) -> Result<CircuitLayout, CircuitError> {
        // 1. Separate out the already placed and unplaced link groups
        let mut sorted_placements = Vec::new();
        let mut unplaced_groups = Vec::new();
        for group in self.link_group_ids().into_iter() {
            if let Some(layout) = self.get_link_group_layout(&group) {
                sorted_placements.push((group, layout));
            } else {
                unplaced_groups.push(group);
            }
        }

        // Sort the placed groups
        let mut alignment = self.current_alignment();
        sorted_placements.sort_by_key(|(_, layout)| layout.range_in_nth_roots(alignment));

        // 2. Place the unplaced groups
        let inputs = self.num_inputs();
        for id in unplaced_groups.into_iter() {
            let size = self.get_link_group_members(&id).unwrap().len();
            while !place_group_with_alignment(size, inputs, alignment, &id, &mut sorted_placements)
            {
                alignment += 1;
            }
        }

        // 3. Create a layout and validate it
        let group_layouts = sorted_placements.into_iter().collect();
        let circuit_layout =
            CircuitLayout { n_inputs: self.num_inputs(), n_gates: self.num_gates(), group_layouts };

        validate_layout(self.num_inputs(), &circuit_layout)?;

        // 4. Set the group layouts in the circuit so that subsequent calls to this
        //    method do not need to place the groups again
        for (id, layout) in circuit_layout.group_layouts.iter() {
            self.set_group_layout(id.clone(), *layout);
        }

        Ok(circuit_layout)
    }

    /// Applies a layout to the circuit, this includes:
    /// - Padding the circuit to the next power of two
    /// - Placing proof linking gates
    /// - Rearranging circuit as per the `order_gates` method
    fn apply_layout(&mut self, layout: &CircuitLayout) -> Result<(), CircuitError> {
        // 1. Order the existing gates before adding proof linking gates
        self.order_gates()?;

        // 2. Collect the existing gates and wire variables for merge
        let mut new_gates: Vec<Box<dyn Gate<F>>> = Vec::with_capacity(layout.circuit_size());
        let mut new_wire_variables: [Vec<Variable>; GATE_WIDTH + 2] = (0..GATE_WIDTH + 2)
            .map(|_| Vec::with_capacity(layout.circuit_size()))
            .collect_vec()
            .try_into()
            .unwrap();

        let mut old_gates = self.gates().iter().cloned();
        let mut old_vars = self.wires().iter().map(|v| v.iter().copied()).collect_vec();

        let gates = &mut new_gates;
        let wire_variables = &mut new_wire_variables;
        let gates_iter = &mut old_gates;
        let vars_iter = &mut old_vars;

        // 3. Place the input gates from the gates iter
        place_gates(layout.n_inputs, gates, wire_variables, gates_iter, vars_iter);

        // 4. Place the proof linking gates
        let circuit_alignment = layout.circuit_alignment();
        for (id, placement) in layout.sorted_groups_iter() {
            // Insert existing gates until the start of the next group
            let (group_start, _) = placement.range_in_nth_roots(circuit_alignment);
            let n = group_start - gates.len();
            place_gates(n, gates, wire_variables, gates_iter, vars_iter);

            // Insert the group
            let group = self.get_link_group_members(id).unwrap();
            append_group(group, placement, layout, gates, wire_variables, gates_iter, vars_iter);
        }

        // 5. Pad to the end of the circuit
        let n = layout.circuit_size() - gates.len();
        place_gates(n, gates, wire_variables, gates_iter, vars_iter);

        // 6. Replace the gates and wire variables
        // SAFETY: The unsafe markers on these methods are to warn against external use
        unsafe {
            *self.gates_mut() = new_gates;
            *self.wires_mut() = new_wire_variables;
        }

        Ok(())
    }
}

// -----------
// | Helpers |
// -----------

/// A helper to place a group with a given alignment
///
/// Assumes the `placed_groups` are sorted by their range
///
/// Returns false if this alignment is not possible
fn place_group_with_alignment(
    size: usize,
    n_inputs: usize,
    alignment: usize,
    group_id: &str,
    placed_groups: &mut Vec<(String, GroupLayout)>,
) -> bool {
    // Map all placed groups to their ranges in the `2^alignment`-th roots
    let placed_ranges = placed_groups
        .iter()
        .map(|(_, layout)| layout.range_in_nth_roots(alignment))
        .sorted()
        .collect_vec();

    // Find the first gap in the roots of unity large enough to accommodate the
    // group
    let mut offset = n_inputs;
    let mut curr_idx = 0;

    while curr_idx < placed_ranges.len() {
        let (start, end) = placed_ranges[curr_idx];
        if offset + size <= start {
            // We've found a gap large enough to accommodate the group
            let layout = GroupLayout { alignment, offset, size };
            placed_groups.insert(curr_idx, (group_id.to_string(), layout));

            return true;
        } else {
            // We've found a gap that's too small, so we skip over it
            offset = end + 1;
            curr_idx += 1;
        }
    }

    // Attempt to place the group after all others
    let roots_bound = 1 << alignment;
    if offset + size < roots_bound {
        let layout = GroupLayout { alignment, offset, size };
        placed_groups.push((group_id.to_string(), layout));

        return true;
    }

    // Otherwise, we've exhausted all possible placements
    false
}

/// Copy the next `n` gates from the original gates iterator into the new
/// gates vector
///
/// Assumes that the wire variables and gate iters are the same size, i.e.
/// that we can `unwrap` on the wire variable iters
fn place_gates<F: Field>(
    n: usize,
    gates: &mut Vec<Box<dyn Gate<F>>>,
    wire_variables: &mut [Vec<Variable>; GATE_WIDTH + 2],
    gates_iter: &mut impl Iterator<Item = Box<dyn Gate<F>>>,
    vars_iter: &mut [impl Iterator<Item = Variable>],
) {
    // May be hit in `append_group` if spacing is one, so best to short circuit
    if n == 0 {
        return;
    }

    let padding_iter = iter::repeat(Box::new(PaddingGate)).map(|g| g as Box<dyn Gate<F>>);
    let mut new_gates = gates_iter.chain(padding_iter).take(n);
    let new_vars = vars_iter.iter_mut().map(|v| v.chain(iter::repeat(0)).take(n)).collect_vec();

    gates.extend(&mut new_gates);
    for (wire, var) in wire_variables.iter_mut().zip(new_vars) {
        wire.extend(var);
    }
}

/// Place a group in the circuit interspersed with gates from the original
/// circuit
fn append_group<F: Field>(
    group: &[Variable],
    layout: &GroupLayout,
    circuit_layout: &CircuitLayout,
    gates: &mut Vec<Box<dyn Gate<F>>>,
    wire_variables: &mut [Vec<Variable>; GATE_WIDTH + 2],
    gates_iter: &mut impl Iterator<Item = Box<dyn Gate<F>>>,
    vars_iter: &mut [impl Iterator<Item = Variable>],
) {
    // Translate the layout alignment to the circuit's alignment
    let spacing = 1 << (circuit_layout.circuit_alignment() - layout.alignment);

    // Place the group at the appropriate spacing
    for link_var in group.iter().copied() {
        // Insert a proof linking gate and wires
        gates.push(Box::new(ProofLinkingGate));
        wire_variables[PROOF_LINK_WIRE_IDX].push(link_var);
        for (i, wire_var) in wire_variables.iter_mut().enumerate() {
            if i != PROOF_LINK_WIRE_IDX {
                wire_var.push(0);
            }
        }

        // Insert gates in between the proof linking gates
        place_gates(spacing - 1, gates, wire_variables, gates_iter, vars_iter);
    }
}

/// Validate a circuit layout against a circuit's arithmetization
///
/// This currently comprises three checks:
///  - Each link group's offset occurs after the public inputs
///  - Each link group's offset is within its alignment
///  - No two link groups overlap
fn validate_layout(num_inputs: usize, layout: &CircuitLayout) -> Result<(), CircuitError> {
    // Check that each offset occurs after the public inputs
    let n = layout.circuit_alignment();
    for (id, layout) in layout.group_layouts.iter() {
        // Check that the layout does not exceed its alignment
        let alignment_bound = 1 << layout.alignment;
        if layout.offset + layout.size >= alignment_bound {
            return Err(CircuitError::Layout(format!(
                "Link group {id} (layout = {layout:?}) exceeds its alignment",
            )));
        }

        // Check that each offset occurs after the public inputs
        let (start, _) = layout.range_in_nth_roots(n);
        if start < num_inputs {
            return Err(CircuitError::Layout(format!(
                "Link group {id} (layout = {layout:?}) would mangle public inputs"
            )));
        }
    }

    // Check that adjacent link groups are non-overlapping
    let sorted = layout.sorted_groups_iter().collect_vec();
    for window in sorted.windows(2 /* size */) {
        let (id1, layout1) = window[0];
        let (id2, layout2) = window[1];

        let range1 = layout1.range_in_nth_roots(n);
        let range2 = layout2.range_in_nth_roots(n);

        if ranges_intersect(range1, range2) {
            return Err(CircuitError::Layout(format!(
                "Link group {} (layout = {:?}) overlaps with group {} (layout = {:?})",
                id1, layout1, id2, layout2
            )));
        }
    }

    Ok(())
}

/// Returns whether the specified ranges intersect
fn ranges_intersect(range1: (usize, usize), range2: (usize, usize)) -> bool {
    let (start1, end1) = range1;
    let (start2, end2) = range2;

    let range_start = usize::max(start1, start2);
    let range_end = usize::min(end1, end2);

    range_start <= range_end
}

#[cfg(test)]
mod test {
    use ark_ed_on_bn254::Fq as FqEd254;
    use ark_ff::{One, PrimeField};
    use ark_std::{
        rand::{distributions::uniform::SampleRange, thread_rng},
        UniformRand,
    };
    use itertools::Itertools;

    use crate::{
        constants::GATE_WIDTH,
        gates::{Gate, IoGate, PaddingGate, ProofLinkingGate},
        proof_linking::{linkable_circuit::LinkableCircuit, LinkGroup},
        traits::Circuit,
        PlonkCircuit, Variable,
    };

    use super::GroupLayout;

    // -----------
    // | Helpers |
    // -----------

    /// A dummy gate that we use to test circuit layouts
    #[derive(Clone, Debug)]
    pub struct DummyGate;
    impl<F: PrimeField> Gate<F> for DummyGate {
        fn name(&self) -> &'static str {
            "DummyGate"
        }
    }

    /// Add a random link group to the circuit
    fn add_random_link_group(
        alignment: usize,
        size: usize,
        cs: &mut PlonkCircuit<FqEd254>,
    ) -> LinkGroup {
        assert!(size <= 2usize.pow(alignment as u32));
        let mut rng = thread_rng();

        let n = cs.current_alignment();
        let n_inputs = cs.num_inputs();

        // Scale the offset so it is after the public inputs
        let offset_min = if n <= alignment {
            n_inputs
        } else {
            let scale = 1 << (n - alignment);
            n_inputs.next_multiple_of(scale) / scale
        };
        let offset_max = 2usize.pow(alignment as u32) - size - 1;
        let offset = (offset_min..offset_max).sample_single(&mut rng);

        // Create the group and add variables to it
        let placement = GroupLayout::new(alignment, offset);
        let group = cs.create_link_group("group1".to_string(), Some(placement));
        for _ in 0..size {
            cs.create_variable_with_link_groups(FqEd254::rand(&mut rng), &[group.clone()]).unwrap();
        }

        group
    }

    /// Check that the given link group has been placed correctly
    fn assert_link_group_placement(
        cs: &PlonkCircuit<FqEd254>,
        group_id: &str,
        layout: &GroupLayout,
    ) {
        // This method assumes that the circuit's gates have been padded to a power of
        // two, which happens when the arithmetization is finalized
        let n = cs.num_gates().ilog2() as usize;
        let spacing = 1 << (n - layout.alignment);
        let offset = layout.offset * spacing;

        for (i, elem) in cs.link_groups[group_id].iter().enumerate() {
            let idx = offset + i * spacing;
            // Verify that a proof linking gate has been added at the correct offset
            assert!(cs.gates[idx].as_any().is::<ProofLinkingGate>());

            // Verify the wire variables are correct
            assert_eq!(cs.wire_variables[0][idx], *elem);
            for i in 1..GATE_WIDTH + 1 {
                assert_eq!(cs.wire_variables[i][idx], 0);
            }

            // Check for padding gates in between the proof linking elements
            for i in idx + 1..idx + spacing {
                assert!(cs.gates[i].as_any().is::<PaddingGate>());
            }
        }
    }

    // ---------
    // | Tests |
    // ---------

    /// Tests the layout method of the circuit
    #[test]
    fn test_circuit_layout() {
        // The power of two we round up to when padding
        const MIN_ALIGNMENT: usize = 6;
        const MAX_ALIGNMENT: usize = 10;

        let mut rng = thread_rng();
        let mut cs = PlonkCircuit::<FqEd254>::new_turbo_plonk();

        // Create three groups at varying specified and unspecified offsets,
        // leaving room before the first group so that the second group (unspecified
        // layout) is slotted in before it
        let alignment1 = (MIN_ALIGNMENT..MAX_ALIGNMENT).sample_single(&mut rng);
        let offset1 = (10..20).sample_single(&mut rng);
        let placement1 = GroupLayout::new(alignment1, offset1);

        // Create the third group on a larger or equal alignment, allocate it after the
        // first group
        let alignment3 = (alignment1..=MAX_ALIGNMENT).sample_single(&mut rng);
        let scaling = 1 << (alignment3 - alignment1);

        let scaled_offset1 = offset1 * scaling;
        let offset3 = (scaled_offset1 + 1..2usize.pow(alignment3 as u32)).sample_single(&mut rng);
        let placement3 = GroupLayout::new(alignment3, offset3);

        // Create the groups
        let group1 = cs.create_link_group(String::from("group1"), Some(placement1));
        let group2 = cs.create_link_group(String::from("group2"), None);
        let group3 = cs.create_link_group(String::from("group3"), Some(placement3));

        // Allocate two vars into the groups
        let val1 = FqEd254::rand(&mut rng);
        let val2 = FqEd254::rand(&mut rng);

        let _public1 = cs.create_public_variable(FqEd254::one()).unwrap();
        let _public2 = cs.create_public_variable(FqEd254::one()).unwrap();
        cs.create_variable_with_link_groups(val1, &[group1, group2.clone()]).unwrap();
        cs.create_variable_with_link_groups(val2, &[group2, group3.clone()]).unwrap();

        // Generate a layout for the circuit
        let layout = cs.generate_layout().unwrap();

        assert_eq!(layout.n_inputs, 2);
        assert_eq!(layout.n_gates, 4); // two i/o gates, two constant gates

        // Group 1
        let placement1 = layout.group_layouts.get("group1").unwrap();
        assert_eq!(placement1.alignment, alignment1);
        assert_eq!(placement1.offset, offset1);
        assert_eq!(placement1.size, 1);

        // Group 2, should be placed immediately after public inputs
        let placement2 = layout.group_layouts.get("group2").unwrap();
        assert_eq!(placement2.alignment, alignment3);
        assert_eq!(placement2.offset, 2);
        assert_eq!(placement2.size, 2);

        // Group 3
        let placement3 = layout.group_layouts.get("group3").unwrap();
        assert_eq!(placement3.alignment, alignment3);
        assert_eq!(placement3.offset, offset3);
        assert_eq!(placement3.size, 1);

        // Check that the link groups have been placed correctly
        cs.finalize_for_arithmetization().unwrap();
        assert_link_group_placement(&cs, "group1", placement1);
        assert_link_group_placement(&cs, "group2", placement2);
        assert_link_group_placement(&cs, "group3", placement3);
    }

    /// Tests the circuit layout when an invalid offset is given
    #[test]
    fn test_invalid_circuit_layout() {
        let mut rng = thread_rng();

        // Add a link group in the public inputs
        let mut cs = PlonkCircuit::<FqEd254>::new_turbo_plonk();
        let val = FqEd254::rand(&mut rng);

        let placement = GroupLayout::new(4 /* alignment */, 0 /* offset */);
        let group = cs.create_link_group(String::from("test"), Some(placement));

        cs.create_public_variable(FqEd254::one()).unwrap();
        cs.create_variable_with_link_groups(val, &[group]).unwrap();

        assert!(cs.generate_layout().is_err());

        // Attempt to add two link groups at conflicting offsets
        let mut cs = PlonkCircuit::<FqEd254>::new_turbo_plonk();

        let alignment1 = (0..10).sample_single(&mut rng);
        let offset1 = (0..2usize.pow(alignment1)).sample_single(&mut rng);
        let placement1 = GroupLayout::new(alignment1 as usize, offset1);

        let alignment2 = (alignment1..=10).sample_single(&mut rng);
        let spacing = 1 << (alignment2 - alignment1);

        let offset2 = offset1 * spacing + spacing; // Start group2 on the last element of group1
        let placement2 = GroupLayout::new(alignment2 as usize, offset2);

        let group1 = cs.create_link_group(String::from("group1"), Some(placement1));
        let group2 = cs.create_link_group(String::from("group2"), Some(placement2));

        let val1 = FqEd254::rand(&mut rng);
        let val2 = FqEd254::rand(&mut rng);
        cs.create_variable_with_link_groups(val1, &[group1.clone(), group2.clone()]).unwrap();
        cs.create_variable_with_link_groups(val2, &[group1, group2]).unwrap();

        assert!(cs.generate_layout().is_err());
    }

    /// Tests that gates added to the circuit remain in the circuit after proof
    /// linking
    ///
    /// Does so by effectively creating a random circuit, and ensuring the gates
    /// and wires are the same after proof-linking is done
    #[test]
    fn test_proof_linking_gates() {
        const N: usize = 100;
        const MIN_ALIGNMENT: usize = (N.ilog2() + 1) as usize;
        const MAX_ALIGNMENT: usize = MIN_ALIGNMENT + 3;
        let mut rng = thread_rng();

        // Sample circuit dimensions
        let n_inputs = (1..N).sample_single(&mut rng);
        let n_gates = (n_inputs..2 * n_inputs).sample_single(&mut rng);
        let n_witness = (1..N).sample_single(&mut rng);
        let n_links = (1..N).sample_single(&mut rng);

        // Construct the circuit
        let mut cs = PlonkCircuit::<FqEd254>::new_turbo_plonk();

        // Create the inputs
        for _ in 0..n_inputs {
            cs.create_public_variable(FqEd254::rand(&mut rng)).unwrap();
        }

        // Create witness variables for the gates
        for _ in 0..n_witness {
            cs.create_variable(FqEd254::rand(&mut rng)).unwrap();
        }

        // Create a random set of gates
        let mut wiring: Vec<[Variable; GATE_WIDTH + 1]> = Vec::with_capacity(n_gates);
        for _ in 0..n_gates {
            let wires = (0..GATE_WIDTH + 1)
                .map(|_| (0..n_witness).sample_single(&mut rng))
                .collect_vec()
                .try_into()
                .unwrap();
            cs.insert_gate(&wires, Box::new(DummyGate)).unwrap();

            wiring.push(wires);
        }

        // Add a random link group and allocate variables in it
        let alignment = (MIN_ALIGNMENT..MAX_ALIGNMENT).sample_single(&mut rng);
        add_random_link_group(alignment, n_links, &mut cs);

        // Arithmetize the circuit
        cs.finalize_for_arithmetization().unwrap();

        // Now check the gates and wires are the same
        let mut inputs = 0;
        let mut gates = 0;
        let mut arithmetized_wiring = Vec::with_capacity(wiring.len());

        for (i, gate) in cs.gates.iter().enumerate() {
            if gate.as_any().is::<DummyGate>() {
                gates += 1;
                let mut wires = [0; GATE_WIDTH + 1];
                #[allow(clippy::needless_range_loop)]
                for wire_idx in 0..GATE_WIDTH + 1 {
                    wires[wire_idx] = cs.wire_variables[wire_idx][i];
                }

                arithmetized_wiring.push(wires);
            } else if gate.as_any().is::<IoGate>() {
                inputs += 1;
            }
        }

        assert_eq!(inputs, n_inputs);
        assert_eq!(gates, n_gates);

        wiring.sort();
        arithmetized_wiring.sort();
        assert_eq!(wiring, arithmetized_wiring);
    }
}
