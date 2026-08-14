// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use smallvec::SmallVec;
use vortex_error::VortexResult;

use super::super::Batch;
use super::super::args::BorrowedExecutionArgs;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::scalar_fn::unstable::row::execute::RowExecution;

impl Batch {
    /// Evaluate one row of constant inputs and broadcast the validated result.
    pub(super) fn broadcast_one_row(
        &self,
        kernel: impl Fn(BorrowedExecutionArgs<'_>, &mut ExecutionCtx) -> VortexResult<RowExecution>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let one_row: SmallVec<[ArrayRef; 4]> = self
            .inputs
            .iter()
            .map(|input| input.slice(0..1))
            .collect::<VortexResult<_>>()?;

        let result = VortexResult::from(kernel(self.execution_args(&one_row, 1), ctx)?)?;
        let result = self.validate_kernel_output(result, 1, ctx)?;
        let result = self.finalize_output(result, 1)?;
        let scalar = result.execute_scalar(0, ctx)?;

        Ok(ConstantArray::new(scalar, self.row_count).into_array())
    }
}
