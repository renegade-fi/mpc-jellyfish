//! Proof linking sub-protocol methods and definitions

mod linkable_circuit;
pub use linkable_circuit::*;

use std::collections::HashMap;

use ark_ff::FftField;
use itertools::Itertools;

/// Represents the parameterization of a proof-linking group in the circuit
///
/// See `Circuit::create_link_group` for more details on proof-linking
pub type LinkGroup = String;

/// The placement parameterization of a group
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    pub fn new(alignment: usize, offset: usize, size: usize) -> Self {
        Self { alignment, offset, size }
    }

    /// Get the inclusive range this group takes up in the trace when embedded
    /// in the 2^n-th roots of unity
    pub fn range_in_nth_roots(&self, n: usize) -> (usize, usize) {
        assert!(n >= self.alignment, "Group alignment must be <= n");

        // Adjust the spacing for the larger roots of unity
        let spacing = 1 << (n - self.alignment);

        let start = self.offset * spacing;
        let end = start + self.size.saturating_sub(1) * spacing;

        (start, end)
    }

    /// Get the domain generator of a group when embedded in the given field
    pub fn get_domain_generator<F: FftField>(&self) -> F {
        let roots_order = 1 << self.alignment;
        F::get_root_of_unity(roots_order).expect("field 2-adicity too small for layout {self:?}")
    }
}

/// The layout of a circuit
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CircuitLayout {
    /// The number of public inputs to the circuit
    pub n_inputs: usize,
    /// The number of gates in the circuit
    pub n_gates: usize,
    /// The offsets and sizes of the proof linking groups in the circuit
    pub group_layouts: HashMap<String, GroupLayout>,
}

impl CircuitLayout {
    /// Get the layout for a given group
    ///
    /// # Panics
    /// Panics if the group does not exist
    pub fn get_group_layout(&self, id: &str) -> GroupLayout {
        self.group_layouts[id]
    }

    /// Get the domain size used to represent the circuit after proof linking
    /// gates are accounted for
    pub fn circuit_size(&self) -> usize {
        // Check for link group alignments that need larger roots
        let max_alignment =
            self.group_layouts.values().map(|layout| layout.alignment).max().unwrap_or(1);
        let link_gates = self.group_layouts.values().map(|layout| layout.size).sum::<usize>();

        let gates = (self.n_gates + link_gates).next_power_of_two();
        let alignment_domain_size = 1 << max_alignment;
        usize::max(gates, alignment_domain_size)
    }

    /// Get the alignment of the circuit, i.e. for return value `n` we use the
    /// roots of unity
    pub fn circuit_alignment(&self) -> usize {
        // `circuit_size` will pad to a power of two, so ilog2 is equivalent to the
        // actual base 2 logarithm
        self.circuit_size().ilog2() as usize
    }

    /// Get an iterator over the groups in the circuit, sorted by the ranges
    /// they are allocated in
    ///
    /// We assume that the group placements are disjoint, so it is safe to
    /// sort them in any common alignment
    pub(crate) fn sorted_groups_iter(&self) -> impl Iterator<Item = (&String, &GroupLayout)> {
        // Compare the ranges in the maximum alignment specified
        let align = self.group_layouts.values().map(|layout| layout.alignment).max().unwrap_or(1);
        self.group_layouts
            .iter()
            .sorted_by_key(|(_, placement)| placement.range_in_nth_roots(align))
    }
}
