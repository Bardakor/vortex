// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::super::Batch;
use super::super::args::BorrowedExecutionArgs;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::builtins::ArrayBuiltins;
use crate::scalar_fn::unstable::row::execute::RowExecution;
use crate::validity::Validity;

impl Batch {
    /// Run every stored payload, then attach the input validity without materializing its mask.
    pub(super) fn execute_dense(
        &self,
        kernel: impl Fn(BorrowedExecutionArgs<'_>, &mut ExecutionCtx) -> VortexResult<RowExecution>,
        retry_deferred_error: bool,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let values = match kernel(self.execution_args(&self.inputs, self.row_count), ctx)? {
            RowExecution::Output(values) => values,
            RowExecution::DeferredError(error) if retry_deferred_error => {
                let valid = self.validity.clone().execute_mask(self.row_count, ctx)?;

                // Unlike `resolve_validity`, all-true preserves the deferred error and all-false
                // suppresses evidence that came entirely from null rows. An empty loop cannot
                // produce deferred evidence, so the ambiguous empty mask cannot reach this arm.
                if valid.all_true() {
                    return Err(error);
                }
                if valid.all_false() {
                    return Ok(self.all_null());
                }

                // Deferred retry receives only the dense kernel. Filter first so the retry cannot
                // evaluate null rows again.
                return self.filter_and_scatter(kernel, &valid, ctx);
            }
            RowExecution::DeferredError(error) => return Err(error),
        };
        let values = self.validate_kernel_output(values, self.row_count, ctx)?;

        match self.validity.clone() {
            Validity::NonNullable | Validity::AllValid => {
                self.finalize_output(values, self.row_count)
            }
            Validity::Array(valid) => self.finalize_output(values.mask(valid)?, self.row_count),
            // Handled by the guard in `Batch::execute`, before the kernel ran.
            Validity::AllInvalid => Ok(self.all_null()),
        }
    }
}
