// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::super::Batch;
use super::super::args::BorrowedExecutionArgs;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::builtins::ArrayBuiltins;
use crate::scalar_fn::unstable::row::execute::RowExecution;
use crate::validity::Validity;

/// The result of resolving batch validity.
enum ResolvedValidity {
    /// The output for an all-valid or all-null batch.
    Output(ArrayRef),

    /// A mask with both valid and invalid rows.
    PartiallyValid(Mask),
}

impl Batch {
    /// Resolve deferred evidence from the encoded path by executing only observable rows.
    pub(super) fn resolve_reduced_error(
        &self,
        error: VortexError,
        kernel: impl Fn(BorrowedExecutionArgs<'_>, &mut ExecutionCtx) -> VortexResult<RowExecution>,
        try_unfiltered: impl FnOnce(
            BorrowedExecutionArgs<'_>,
            &Mask,
            &mut ExecutionCtx,
        ) -> VortexResult<Option<RowExecution>>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let valid = self.validity.clone().execute_mask(self.row_count, ctx)?;

        if valid.all_true() {
            return Err(error);
        }
        if valid.all_false() {
            return Ok(self.all_null());
        }

        if let Some(result) = self.try_execute_unfiltered(try_unfiltered, &valid, ctx)? {
            return Ok(result);
        }

        self.filter_and_scatter(kernel, &valid, ctx)
    }

    /// Resolve validity, try unfiltered execution, then fall back to filtering.
    pub(super) fn execute_valid_only(
        &self,
        kernel: impl Fn(BorrowedExecutionArgs<'_>, &mut ExecutionCtx) -> VortexResult<RowExecution>,
        try_unfiltered: impl FnOnce(
            BorrowedExecutionArgs<'_>,
            &Mask,
            &mut ExecutionCtx,
        ) -> VortexResult<Option<RowExecution>>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let valid = match self.resolve_validity(&kernel, ctx)? {
            ResolvedValidity::Output(output) => return Ok(output),
            ResolvedValidity::PartiallyValid(valid) => valid,
        };

        if let Some(result) = self.try_execute_unfiltered(try_unfiltered, &valid, ctx)? {
            return Ok(result);
        }

        self.filter_and_scatter(kernel, &valid, ctx)
    }

    /// Materialize validity and handle all-valid or all-null batches.
    fn resolve_validity(
        &self,
        kernel: &impl Fn(BorrowedExecutionArgs<'_>, &mut ExecutionCtx) -> VortexResult<RowExecution>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ResolvedValidity> {
        let valid = self.validity.clone().execute_mask(self.row_count, ctx)?;

        // Check all-true before all-false: an empty mask is both, and must not be treated as
        // all-null (a zero-length non-nullable execution keeps its non-nullable dtype).
        if valid.all_true() {
            let values = VortexResult::from(kernel(
                self.execution_args(&self.inputs, self.row_count),
                ctx,
            )?)?;
            let values = self.validate_kernel_output(values, self.row_count, ctx)?;
            let values = self.finalize_output(values, self.row_count)?;

            return Ok(ResolvedValidity::Output(values));
        }

        if valid.all_false() {
            return Ok(ResolvedValidity::Output(self.all_null()));
        }

        Ok(ResolvedValidity::PartiallyValid(valid))
    }

    /// Try execution against the original inputs, then mask a returned full-length result.
    fn try_execute_unfiltered(
        &self,
        try_unfiltered: impl FnOnce(
            BorrowedExecutionArgs<'_>,
            &Mask,
            &mut ExecutionCtx,
        ) -> VortexResult<Option<RowExecution>>,
        valid: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(execution) = try_unfiltered(
            self.execution_args(&self.inputs, self.row_count),
            valid,
            ctx,
        )?
        else {
            return Ok(None);
        };
        let values = VortexResult::from(execution)?;
        let values = self.validate_kernel_output(values, valid.len(), ctx)?;

        let mask = BoolArray::new(valid.to_bit_buffer(), Validity::NonNullable).into_array();
        self.finalize_output(values.mask(mask)?, valid.len())
            .map(Some)
    }
}
