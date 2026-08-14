// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;

use super::super::Batch;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::scalar::Scalar;
use crate::scalar_fn::ScalarFnId;
use crate::validity::Validity;

impl Batch {
    pub(super) fn all_null(&self) -> ArrayRef {
        ConstantArray::new(Scalar::null(self.result_dtype.clone()), self.row_count).into_array()
    }

    pub(super) fn finalize_output(
        &self,
        values: ArrayRef,
        expected_len: usize,
    ) -> VortexResult<ArrayRef> {
        reconcile_output(self.id, &self.result_dtype, expected_len, values)
    }

    /// Reconcile an encoding-aware result and apply the batch's strict input validity.
    pub(super) fn finalize_reduced(
        &self,
        values: ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        validate_output(self.id, &self.result_dtype, self.row_count, &values)?;

        let input_valid = self.validity.execute_mask(self.row_count, ctx)?;
        let output_valid = values.validity()?.execute_mask(self.row_count, ctx)?;
        vortex_ensure!(
            input_valid.bitand_not(&output_valid).all_false(),
            "the {} encoded reduction produced nulls for valid rows",
            self.id,
        );

        let values = match self.validity.clone() {
            Validity::NonNullable | Validity::AllValid => values,
            Validity::Array(valid) => values.mask(valid)?,
            // Handled before the encoding-aware hook runs.
            Validity::AllInvalid => return Ok(self.all_null()),
        };

        cast_output_nullability(&self.result_dtype, values)
    }

    /// Validate the output from a row kernel before batch validity is attached.
    pub(super) fn validate_kernel_output(
        &self,
        values: ArrayRef,
        expected_len: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        finalize_kernel_output(self.id, &self.output_dtype, expected_len, values, ctx)
    }
}

/// Validate the output produced directly by a row kernel.
///
/// `values` **must** contain `expected_len` rows. Its dtype must match `result_dtype` when ignoring
/// nullability, and every produced row **must** be valid. Batch execution owns strict null
/// propagation and attaches input-derived validity only after this boundary.
pub(crate) fn finalize_kernel_output(
    id: ScalarFnId,
    result_dtype: &DType,
    expected_len: usize,
    values: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    validate_output(id, result_dtype, expected_len, &values)?;
    vortex_ensure!(
        values.all_valid(ctx)?,
        "the {id} row kernel must produce only valid rows, got at least one null row",
    );

    cast_output_nullability(result_dtype, values)
}

/// Reconcile an output with the function's declared shape and nullability.
fn reconcile_output(
    id: ScalarFnId,
    result_dtype: &DType,
    expected_len: usize,
    values: ArrayRef,
) -> VortexResult<ArrayRef> {
    validate_output(id, result_dtype, expected_len, &values)?;

    cast_output_nullability(result_dtype, values)
}

/// Validate an output's shape and logical dtype without executing a nullability cast.
fn validate_output(
    id: ScalarFnId,
    result_dtype: &DType,
    expected_len: usize,
    values: &ArrayRef,
) -> VortexResult<()> {
    vortex_ensure_eq!(
        values.len(),
        expected_len,
        "the {id} kernel output must contain {expected_len} rows, got {}",
        values.len(),
    );
    vortex_ensure!(
        values.dtype().eq_ignore_nullability(result_dtype),
        "the {id} kernel output dtype must match {result_dtype} ignoring nullability, got {}",
        values.dtype(),
    );

    Ok(())
}

/// Cast only the output nullability after its shape, dtype, and validity are accepted.
fn cast_output_nullability(result_dtype: &DType, values: ArrayRef) -> VortexResult<ArrayRef> {
    if values.dtype() == result_dtype {
        Ok(values)
    } else {
        values.cast(result_dtype.clone())
    }
}
