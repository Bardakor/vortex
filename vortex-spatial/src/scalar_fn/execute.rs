// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared unary execution for native geometry scalar functions.

mod unary;

pub(crate) use unary::dispatch_unary;
use vortex_array::ArrayRef;
use vortex_array::scalar::Scalar;

/// A non-null operand presented to a geometry kernel.
pub(crate) enum Operand {
    /// One scalar value repeated for every row.
    Constant(Scalar),
    /// A column with one value per row.
    Column(ArrayRef),
}

/// Shared batch state presented to a null-propagating geometry kernel with `N` operands.
pub(crate) struct Execution<const N: usize, V> {
    /// Constant/column shape of each operand.
    pub(crate) operands: [Operand; N],
    /// Validity state required by the kernel.
    pub(crate) valid: V,
    /// Number of output rows.
    pub(crate) len: usize,
}
