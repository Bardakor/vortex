// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Selects a batch execution strategy.
//!
//! [`Batch::execute`] handles universal fast paths and encoded reductions, then delegates to dense
//! or valid-only execution.

use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::Batch;
use super::RowPolicy;
use super::args::BorrowedExecutionArgs;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::Constant;
use crate::scalar_fn::unstable::row::execute::RowExecution;
use crate::scalar_fn::unstable::row::types::batch_constant;
use crate::validity::Validity;

mod constant;
mod dense;
mod filter_scatter;
mod valid_only;

mod output;
pub(crate) use output::finalize_kernel_output;

impl Batch {
    /// Apply encoded reductions, constant folding, and null handling around `kernel`.
    ///
    /// `reduce` receives the original inputs before constant broadcasting. When the mask contains
    /// valid and invalid rows, `try_unfiltered` may avoid filtering. `Ok(None)` filters the valid
    /// rows and scatters the output back. Every result is checked against the planned shape and
    /// dtype.
    pub(crate) fn execute(
        &self,
        reduce: impl FnOnce(
            BorrowedExecutionArgs<'_>,
            &mut ExecutionCtx,
        ) -> VortexResult<Option<RowExecution>>,
        kernel: impl Fn(BorrowedExecutionArgs<'_>, &mut ExecutionCtx) -> VortexResult<RowExecution>,
        try_unfiltered: impl FnOnce(
            BorrowedExecutionArgs<'_>,
            &Mask,
            &mut ExecutionCtx,
        ) -> VortexResult<Option<RowExecution>>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        // Strictness: an all-null batch has no observable row work. Keep the literal-constant
        // check explicit alongside the conjoined validity invariant.
        if matches!(self.validity, Validity::AllInvalid)
            || self.inputs.iter().any(|input| {
                input
                    .as_opt::<Constant>()
                    .is_some_and(|constant| constant.scalar().is_null())
            })
        {
            return Ok(self.all_null());
        }

        // An empty mask is both all-true and all-false, so deferred encoded evidence cannot be
        // attributed to an observable row. Let the ordinary policy construct the typed empty
        // output instead.
        if self.row_count > 0
            && let Some(execution) = reduce(self.execution_args(&self.inputs, self.row_count), ctx)?
        {
            match execution {
                RowExecution::Output(values) => return self.finalize_reduced(values, ctx),
                RowExecution::DeferredError(error) => {
                    return self.resolve_reduced_error(error, kernel, try_unfiltered, ctx);
                }
            }
        }

        // All inputs constant, and their conjoined validity proves every row non-null. This sees
        // through extension and masked wrappers just like argument decoding does.
        if self.row_count > 0
            && self.validity.definitely_no_nulls()
            && self
                .inputs
                .iter()
                .all(|input| batch_constant(input).is_some())
        {
            return self.broadcast_one_row(kernel, ctx);
        }

        match self.policy {
            RowPolicy::Dense => self.execute_dense(kernel, false, ctx),
            RowPolicy::DenseWithRetry => self.execute_dense(kernel, true, ctx),
            RowPolicy::ValidOnly => self.execute_valid_only(kernel, try_unfiltered, ctx),
        }
    }
}
