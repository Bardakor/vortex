// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A borrowed execution view passed to one row-kernel invocation.

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::dtype::DType;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::unstable::row::visitor::RowPolicy;

/// A borrowed [`ExecutionArgs`] view with the planning metadata selected for its row kernel.
///
/// `arrays` may be filtered or sliced, while `dtypes` and `output_dtype` always describe the
/// original planned batch. Keeping them together prevents an execution path from pairing an input
/// view with unrelated planning metadata.
#[derive(Clone, Copy)]
pub(in crate::scalar_fn::unstable::row) struct BorrowedExecutionArgs<'a> {
    /// The input arrays for this kernel invocation.
    arrays: &'a [ArrayRef],

    /// The number of rows in this kernel invocation.
    row_count: usize,

    /// The original input dtypes used to select the row implementation.
    dtypes: &'a [DType],

    /// The non-nullable dtype built by the selected output capability.
    output_dtype: &'a DType,

    /// The nullable execution policy selected during planning.
    policy: RowPolicy,
}

impl<'a> BorrowedExecutionArgs<'a> {
    /// Pair one input view with the planning metadata selected for its batch.
    pub(in crate::scalar_fn::unstable::row) fn new(
        arrays: &'a [ArrayRef],
        row_count: usize,
        dtypes: &'a [DType],
        output_dtype: &'a DType,
        policy: RowPolicy,
    ) -> Self {
        Self {
            arrays,
            row_count,
            dtypes,
            output_dtype,
            policy,
        }
    }

    /// Return the concrete arrays used by this row-kernel invocation.
    pub(in crate::scalar_fn::unstable::row) fn arrays(&self) -> &'a [ArrayRef] {
        self.arrays
    }

    /// Return the original input dtypes used to select the row implementation.
    pub(in crate::scalar_fn::unstable::row) fn dtypes(&self) -> &'a [DType] {
        self.dtypes
    }

    /// Return the non-nullable dtype built by the selected output capability.
    pub(in crate::scalar_fn::unstable::row) fn output_dtype(&self) -> &'a DType {
        self.output_dtype
    }

    /// Return the nullable execution policy selected during planning.
    pub(in crate::scalar_fn::unstable::row) fn policy(&self) -> RowPolicy {
        self.policy
    }
}

impl ExecutionArgs for BorrowedExecutionArgs<'_> {
    fn get(&self, index: usize) -> VortexResult<ArrayRef> {
        self.arrays.get(index).cloned().ok_or_else(|| {
            vortex_err!(
                "row-function input index must be less than {}, got {index}",
                self.arrays.len(),
            )
        })
    }

    fn num_inputs(&self) -> usize {
        self.arrays.len()
    }

    fn row_count(&self) -> usize {
        self.row_count
    }
}
