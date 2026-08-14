// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use smallvec::SmallVec;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use super::super::Batch;
use super::super::args::BorrowedExecutionArgs;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::MaskedArray;
use crate::arrays::PrimitiveArray;
use crate::builtins::ArrayBuiltins;
use crate::dtype::Nullability;
use crate::scalar_fn::unstable::row::execute::RowExecution;
use crate::validity::Validity;

impl Batch {
    /// Filter to valid rows, run the kernel, then scatter into a null-padded output.
    pub(super) fn filter_and_scatter(
        &self,
        kernel: impl Fn(BorrowedExecutionArgs<'_>, &mut ExecutionCtx) -> VortexResult<RowExecution>,
        valid: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let filtered: SmallVec<[ArrayRef; 4]> = self
            .inputs
            .iter()
            .map(|input| input.filter(valid.clone()))
            .collect::<VortexResult<_>>()?;

        let values = VortexResult::from(kernel(
            self.execution_args(&filtered, valid.true_count()),
            ctx,
        )?)?;
        let values = self.validate_kernel_output(values, valid.true_count(), ctx)?;

        self.finalize_output(self.scatter_valid(values, valid)?, valid.len())
    }

    /// Scatter `values` (one per set bit of `valid`, in order) back to the positions of the set
    /// bits, producing an array of length `valid.len()` that is null at every unset position.
    fn scatter_valid(&self, values: ArrayRef, valid: &Mask) -> VortexResult<ArrayRef> {
        vortex_ensure_eq!(
            values.len(),
            valid.true_count(),
            "the {} kernel output must contain {} filtered rows, got {}",
            self.id,
            valid.true_count(),
            values.len(),
        );

        let AllOr::Some(slices) = valid.slices() else {
            // The caller handles the all-true and all-false masks.
            vortex_bail!(
                "scatter_valid requires valid and invalid rows, got an all-valid or all-invalid mask"
            );
        };

        // Gather indices: row i of the output reads values[rank(i)]. Rows behind nulls read index
        // 0, and any in-bounds index would do since they are masked out below (values is non-empty
        // here).
        let mut indices = vec![0u64; valid.len()];
        let mut rank = 0u64;
        for &(start, end) in slices {
            for index in &mut indices[start..end] {
                *index = rank;
                rank += 1;
            }
        }
        let indices = PrimitiveArray::new(indices, Validity::NonNullable).into_array();

        let scattered = values.take(indices)?;

        // A nullable gathered array cannot be wrapped because a `Masked` child must be all valid.
        // The general masking pass unions its nulls with the batch validity instead.
        if scattered.dtype().is_nullable() {
            let mask = BoolArray::new(valid.to_bit_buffer(), Validity::NonNullable).into_array();
            return scattered.mask(mask);
        }

        // The gathered values are all valid, so attaching validity is sufficient.
        Ok(MaskedArray::try_new(
            scattered,
            Validity::from_mask(valid.clone(), Nullability::Nullable),
        )?
        .into_array())
    }
}
