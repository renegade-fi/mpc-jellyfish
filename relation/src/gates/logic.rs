// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Jellyfish library.

// You should have received a copy of the MIT License
// along with the Jellyfish library. If not, see <https://mit-license.org/>.

//! Implementation of logic gates

use super::Gate;
use crate::constants::{GATE_WIDTH, N_MUL_SELECTORS};
use ark_ff::Field;

/// A gate for logic OR
#[derive(Clone)]
pub struct LogicOrGate;

impl<F> Gate<F> for LogicOrGate
where
    F: Field,
{
    fn name(&self) -> &'static str {
        "Logic OR Gate"
    }
    fn q_lc(&self) -> [F; GATE_WIDTH] {
        [F::one(), F::one(), F::zero(), F::zero()]
    }
    fn q_mul(&self) -> [F; N_MUL_SELECTORS] {
        [-F::one(), F::zero()]
    }
    fn q_c(&self) -> F {
        -F::one()
    }
}

/// A gate for computing the logic OR value of 2 variables
#[derive(Clone)]
pub struct LogicOrOutputGate;

impl<F> Gate<F> for LogicOrOutputGate
where
    F: Field,
{
    fn name(&self) -> &'static str {
        "Logic OR Value Gate"
    }
    fn q_lc(&self) -> [F; GATE_WIDTH] {
        [F::one(), F::one(), F::zero(), F::zero()]
    }
    fn q_mul(&self) -> [F; N_MUL_SELECTORS] {
        [-F::one(), F::zero()]
    }
    fn q_o(&self) -> F {
        F::one()
    }
}

/// A gate for selecting one of two inputs based on a selector
///
/// I.e. implements if self { a } else { b }
///
/// Assuming that the selector is either 0 or 1 (this should be constrained
/// elsewhere) we can do this via the following:
///     sel * a + (1 - sel) * b
///     = sel * a + b - sel * b
/// We can achieve this by setting q_lc = [0, 0, 0, 1]
/// and q_mul to [1, -1]
///
/// Then the input wires are [sel, a, sel, b] we end up with
///    q_mul[0] * sel * a + q_lc[3] * b + q_mul[1] * sel * b
#[derive(Clone)]
pub struct MuxGate;
impl<F: Field> Gate<F> for MuxGate {
    fn name(&self) -> &'static str {
        "Mux Gate"
    }

    fn q_lc(&self) -> [F; GATE_WIDTH] {
        [F::zero(), F::zero(), F::zero(), F::one()]
    }

    fn q_mul(&self) -> [F; N_MUL_SELECTORS] {
        [F::one(), -F::one()]
    }

    fn q_o(&self) -> F {
        F::one()
    }
}
