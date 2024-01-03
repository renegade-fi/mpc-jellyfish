use core::iter;

use ark_ff::PrimeField;
use itertools::Itertools;

use crate::{
    constants::GATE_WIDTH,
    errors::CircuitError,
    gates::{Gate, PaddingGate, ProofLinkingGate},
    traits::Circuit,
    CircuitLayout, PlonkCircuit, Variable,
};

/// The placement parameterization of a group
#[derive(Clone, Copy, Debug)]
pub struct GroupLayout {
    /// The roots-of-unity alignment of the group, i.e. the group is allocated
    /// on the 2^{alignment}-th roots of unity
    pub alignment: usize,
    /// The offset within the alignment, i.e. the group is allocated on the
    /// 2^{alignment}-th roots of unity starting at offset
    pub offset: usize,
    /// The size of the group in the trace
    pub size: usize,
}

impl GroupLayout {
    /// Constructor
    pub fn new(alignment: usize, offset: usize) -> Self {
        Self { alignment, offset, size: 0 }
    }

    /// Get the inclusive range this group takes up in the trace when embedded
    /// in the 2^n-th roots of unity
    pub fn range_in_nth_roots(&self, n: usize) -> (usize, usize) {
        assert!(n >= self.alignment, "Group alignment must be <= n");

        // Adjust the spacing for the larger roots of unity
        let spacing = 1 << (n - self.alignment);

        let start = self.offset * spacing;
        let end = start + (self.size - 1) * spacing;

        (start, end)
    }
}

/// Methods for proof linking and group placement
impl<F: PrimeField> PlonkCircuit<F> {
    /// Generate a layout of the circuit, including where proof-linking gates
    /// will be placed
    pub fn gen_circuit_layout(&self) -> Result<CircuitLayout, CircuitError> {
        // 1. Place the proof linking groups with specific layouts into the circuit
        let alignment = self.current_circuit_alignment();
        let mut sorted_placements = self
            .link_group_layouts
            .clone()
            .into_iter()
            .sorted_by_key(|(_, layout)| layout.range_in_nth_roots(alignment))
            .collect_vec();

        // 2. Place the rest of the link groups into the circuit
        for group_id in
            self.link_groups.keys().sorted().filter(|id| !self.link_group_layouts.contains_key(*id))
        {
            self.place_group(group_id.clone(), &mut sorted_placements);
        }

        // 3. Create a layout and validate it
        let group_layouts = sorted_placements.into_iter().collect();
        let circuit_layout =
            CircuitLayout { n_inputs: self.num_inputs(), n_gates: self.num_gates(), group_layouts };

        self.validate_layout(&circuit_layout)?;
        Ok(circuit_layout)
    }

    /// The alignment of the circuit before placing the unconstrained groups,
    /// i.e. those groups that do not have a specified layout
    ///
    /// That is, for return value `n`, the circuit will be aligned to the 2^n-th
    /// roots of unity
    fn current_circuit_alignment(&self) -> usize {
        let max_alignment =
            self.link_group_layouts.values().map(|l| l.alignment).max().unwrap_or(0);

        let n_proof_links = self.link_groups.values().map(|v| v.len()).sum::<usize>();
        let n_gates = (self.num_gates() + n_proof_links).next_power_of_two().ilog2() as usize;

        usize::max(n_gates, max_alignment)
    }

    /// Place a group into a list of already allocated groups
    ///
    /// Assumes the `placed_groups` are sorted by their range
    ///
    /// Mutates `placed_groups` to insert the new group
    ///
    /// We first attempt to place the group in the largest set of roots of unity
    /// that already has a group allocated. If successful, we do not need to
    /// increase the proof size. If unsuccessful, we try the next largest set
    /// of roots of unity, and so on
    fn place_group(&self, group_id: String, placed_groups: &mut Vec<(String, GroupLayout)>) {
        // Iterate over alignments until we find one that fits the group
        let mut alignment = self.current_circuit_alignment();
        while !self.place_group_with_alignment(&group_id, alignment, placed_groups) {
            alignment += 1;
        }
    }

    /// A helper to place a group with a given alignment
    ///
    /// Assumes the `placed_groups` are sorted by their range
    ///
    /// Returns false if this alignment is not possible
    fn place_group_with_alignment(
        &self,
        group_id: &str,
        alignment: usize,
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
        let size = self.link_groups.get(group_id).unwrap().len();
        let mut offset = self.num_inputs();
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

    /// Validate a circuit layout against the circuit's arithmetization
    fn validate_layout(&self, layout: &CircuitLayout) -> Result<(), CircuitError> {
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
            if start < self.num_inputs() {
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

    /// Applies a layout to the circuit, this includes:
    /// - Padding the circuit to the next power of two
    /// - Placing proof linking gates
    /// - Rearranging the input gates to the front of the circuit
    pub(crate) fn apply_layout(&mut self, layout: &CircuitLayout) -> Result<(), CircuitError> {
        // 1. Rearrange the input gates to the front of the circuit
        self.rearrange_gates()?;

        // 2. Collect the existing gates and wire variables for merge
        let mut new_gates: Vec<Box<dyn Gate<F>>> = Vec::with_capacity(layout.circuit_size());
        let mut new_wire_variables: [Vec<Variable>; GATE_WIDTH + 2] = (0..GATE_WIDTH + 2)
            .map(|_| Vec::with_capacity(layout.circuit_size()))
            .collect_vec()
            .try_into()
            .unwrap();

        let mut old_gates = self.gates.iter().cloned();
        let mut old_vars = self.wire_variables.iter().map(|v| v.iter().copied()).collect_vec();

        let gates = &mut new_gates;
        let wire_variables = &mut new_wire_variables;
        let gates_iter = &mut old_gates;
        let vars_iter = &mut old_vars;

        // 3. Place the input gates from the gates iter
        Self::place_gates(layout.n_inputs, gates, wire_variables, gates_iter, vars_iter);

        // 4. Place the proof linking gates
        let circuit_alignment = layout.circuit_alignment();
        for (id, placement) in layout.sorted_groups_iter() {
            // Insert existing gates until the start of the next group
            let (group_start, _) = placement.range_in_nth_roots(circuit_alignment);
            let n = group_start - gates.len();
            Self::place_gates(n, gates, wire_variables, gates_iter, vars_iter);

            // Insert the group
            self.append_group(id, placement, layout, gates, wire_variables, gates_iter, vars_iter);
        }

        // 5. Pad to the end of the circuit
        let n = layout.circuit_size() - gates.len();
        Self::place_gates(n, gates, wire_variables, gates_iter, vars_iter);

        // 6. Replace the gates and wire variables
        self.gates = new_gates;
        self.wire_variables = new_wire_variables;

        Ok(())
    }

    /// Copy the next `n` gates from the original gates iterator into the new
    /// gates vector
    ///
    /// Assumes that the wire variables and gate iters are the same size, i.e.
    /// that we can `unwrap` on the wire variable iters
    fn place_gates(
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
    fn append_group(
        &self,
        id: &str,
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
        for link_var in self.link_groups[id].iter().copied() {
            // Insert a proof linking gate and wires
            gates.push(Box::new(ProofLinkingGate));
            wire_variables[0].push(link_var);
            for wire_var in wire_variables.iter_mut().skip(1) {
                wire_var.push(0);
            }

            // Insert gates in between the proof linking gates
            Self::place_gates(spacing - 1, gates, wire_variables, gates_iter, vars_iter);
        }
    }
}

// -----------
// | Helpers |
// -----------

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
        traits::{Circuit, LinkGroup},
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

        let n = cs.current_circuit_alignment();
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
        let layout = cs.gen_circuit_layout().unwrap();

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
        assert_link_group_placement(&cs, "group1", &placement1);
        assert_link_group_placement(&cs, "group2", &placement2);
        assert_link_group_placement(&cs, "group3", &placement3);
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

        assert!(cs.gen_circuit_layout().is_err());

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

        assert!(cs.gen_circuit_layout().is_err());
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
