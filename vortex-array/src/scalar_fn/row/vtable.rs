// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The explicit [`ScalarFnVTable`](crate::scalar_fn::ScalarFnVTable) adapter for [`RowFn`].
//!
//! The [`visitor`](super::visitor) module validates and executes the concrete row signature
//! selected by dispatch. This module connects those visits to batch execution and exposes the
//! resulting scalar function behavior to the rest of the compute stack.

use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::row_fn::RowFn;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::scalar_fn::BorrowedExecutionArgs;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::row::batch::Batch;
use crate::scalar_fn::row::batch::KernelArgs;
use crate::scalar_fn::row::batch::finalize_kernel_output;
use crate::scalar_fn::row::execute::RowExecution;
use crate::scalar_fn::row::visitor::ExecuteRows;
use crate::scalar_fn::row::visitor::ExecuteValidRows;
use crate::scalar_fn::row::visitor::PlanRows;

/// Compute the return dtype for a [`RowFn`] without adopting its complete scalar-function vtable.
pub fn row_fn_return_dtype<F: RowFn>(
    function: &F,
    options: &F::Options,
    args: &[DType],
) -> VortexResult<DType> {
    let plan = function.dispatch(options, args, PlanRows::<F>::new(args, options))?;

    Ok(plan.result_dtype(args))
}

/// Execute a [`RowFn`] while preserving a caller-owned scalar-function vtable.
///
/// Existing vtables delegate here when they need row execution but retain custom hooks for other
/// capabilities. A RowFn-only function implements the remaining vtable methods mechanically and
/// delegates its return-dtype and execution methods here.
pub fn execute_rows<F: RowFn>(
    function: &F,
    options: &F::Options,
    args: &dyn ExecutionArgs,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    // Nullary functions have no input validity to propagate, so they skip batch execution.
    if args.num_inputs() == 0 {
        let result_dtype = row_fn_return_dtype(function, options, &[])?;
        let nullary_args = KernelArgs {
            arrays: &[],
            row_count: args.row_count(),
            dtypes: &[],
            output_dtype: &result_dtype,
        };

        let execution = execute_row_kernel(function, options, nullary_args, ctx)?;
        let values = VortexResult::from(execution)?;

        return finalize_kernel_output(
            RowFn::id(function),
            &result_dtype,
            args.row_count(),
            values,
            ctx,
        );
    }

    let batch = prepare_batch(function, options, args)?;
    batch.execute(
        |args, ctx| function.reduce_encoded(options, args.arrays, ctx),
        |args, ctx| execute_row_kernel(function, options, args, ctx),
        |args, valid, ctx| try_execute_rows_unfiltered(function, options, args, valid, ctx),
        ctx,
    )
}

/// Run the encoding-aware rewrite when available, or execute the selected row loop.
fn execute_row_kernel<F: RowFn>(
    function: &F,
    options: &F::Options,
    args: KernelArgs<'_>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<RowExecution> {
    let execution = BorrowedExecutionArgs::new(args.arrays, args.row_count);

    function.dispatch(
        options,
        args.dtypes,
        ExecuteRows::<F>::new(&execution, args.output_dtype, ctx),
    )
}

/// Try execution against the original inputs, returning `None` when batch execution must filter.
fn try_execute_rows_unfiltered<F: RowFn>(
    function: &F,
    options: &F::Options,
    args: KernelArgs<'_>,
    valid: &Mask,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<RowExecution>> {
    let execution = BorrowedExecutionArgs::new(args.arrays, args.row_count);

    function.dispatch(
        options,
        args.dtypes,
        ExecuteValidRows::<F>::new(&execution, args.output_dtype, valid, ctx),
    )
}

/// Prepare the batch inputs and execution plan for `function`.
fn prepare_batch<F: RowFn>(
    function: &F,
    options: &F::Options,
    args: &dyn ExecutionArgs,
) -> VortexResult<Batch> {
    Batch::new(RowFn::id(function), args, |arg_dtypes| {
        function.dispatch(options, arg_dtypes, PlanRows::<F>::new(arg_dtypes, options))
    })
}
