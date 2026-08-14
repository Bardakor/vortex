// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Batch execution around a strict row kernel.
//!
//! A row kernel handles typed values for one row. This module adds the columnar concerns around it:
//! planning the output and null strategy, preserving batch constants, propagating strict validity,
//! selecting an execution strategy, and validating the finished output.
//!
//! [`BatchPlan`] carries the nullable execution strategy selected by a concrete dispatch. [`Batch`]
//! applies that strategy, and [`BorrowedExecutionArgs`] pairs each kernel invocation with its
//! planning metadata.

use smallvec::SmallVec;

use crate::ArrayRef;
use crate::dtype::DType;
use crate::scalar_fn::ScalarFnId;
use crate::validity::Validity;

mod args;
pub(super) use args::BorrowedExecutionArgs;

mod execute;
pub(super) use execute::finalize_kernel_output;

mod planning;

pub(super) use super::visitor::BatchPlan;
pub(super) use super::visitor::RowPolicy;

/// One batch of inputs and the metadata needed before its row kernel runs.
pub(crate) struct Batch {
    /// The function being executed, named in the errors this raises.
    id: ScalarFnId,

    /// The number of rows in the original execution scope.
    row_count: usize,

    /// The input columns, collected once: constant folding inspects them and the filter strategy
    /// filters them.
    inputs: SmallVec<[ArrayRef; 4]>,

    /// The input dtypes, collected with the columns and reused by both planning and execution.
    arg_dtypes: SmallVec<[DType; 4]>,

    /// The conjoined input validity, so a row of the output is valid iff it is valid in every
    /// input. Conjoining is lazy, and nothing materializes it unless the null handling asks.
    validity: Validity,

    /// The dtype the function declares for these inputs, which the kernel's output is reconciled
    /// against. Already widened to nullable if any input is nullable.
    result_dtype: DType,

    /// The non-nullable dtype the dispatched output capability builds, computed while planning.
    output_dtype: DType,

    /// How the concrete dispatch executes nullable rows.
    policy: RowPolicy,
}

#[cfg(test)]
mod tests;
