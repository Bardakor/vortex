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

/// Implement the standard [`ScalarFnVTable`](crate::scalar_fn::ScalarFnVTable) behavior for one
/// concrete [`RowFn`].
///
/// This opt-in is explicit so a function that needs custom coercion, simplification, reduction,
/// formatting, or validity hooks can implement the vtable itself and delegate only execution to
/// [`execute_rows`](crate::scalar_fn::execute_rows). The standard adapter delegates serialization
/// to [`RowFn`], derives arity and child names from [`RowFn::ARG_NAMES`], propagates child validity,
/// and reports the function as strict.
#[macro_export]
macro_rules! impl_row_fn_vtable {
    ($function:ty) => {
        impl $crate::scalar_fn::ScalarFnVTable for $function {
            type Options = <$function as $crate::scalar_fn::RowFn>::Options;

            fn id(&self) -> $crate::scalar_fn::ScalarFnId {
                $crate::scalar_fn::RowFn::id(self)
            }

            fn serialize(
                &self,
                options: &Self::Options,
            ) -> $crate::scalar_fn::row_fn_macro_support::VortexResult<Option<Vec<u8>>> {
                $crate::scalar_fn::RowFn::serialize(self, options)
            }

            fn deserialize(
                &self,
                metadata: &[u8],
                session: &$crate::scalar_fn::row_fn_macro_support::VortexSession,
            ) -> $crate::scalar_fn::row_fn_macro_support::VortexResult<Self::Options> {
                $crate::scalar_fn::RowFn::deserialize(self, metadata, session)
            }

            fn arity(&self, _options: &Self::Options) -> $crate::scalar_fn::Arity {
                $crate::scalar_fn::Arity::Exact(
                    <$function as $crate::scalar_fn::RowFn>::ARG_NAMES.len(),
                )
            }

            fn child_name(
                &self,
                _options: &Self::Options,
                child_index: usize,
            ) -> $crate::scalar_fn::ChildName {
                $crate::scalar_fn::ChildName::from(
                    <$function as $crate::scalar_fn::RowFn>::ARG_NAMES[child_index],
                )
            }

            fn return_dtype(
                &self,
                options: &Self::Options,
                args: &[$crate::dtype::DType],
            ) -> $crate::scalar_fn::row_fn_macro_support::VortexResult<$crate::dtype::DType> {
                $crate::scalar_fn::row_fn_return_dtype(self, options, args)
            }

            fn execute(
                &self,
                options: &Self::Options,
                args: &dyn $crate::scalar_fn::ExecutionArgs,
                ctx: &mut $crate::ExecutionCtx,
            ) -> $crate::scalar_fn::row_fn_macro_support::VortexResult<$crate::ArrayRef> {
                $crate::scalar_fn::execute_rows(self, options, args, ctx)
            }

            fn validity(
                &self,
                _options: &Self::Options,
                expression: &$crate::expr::Expression,
            ) -> $crate::scalar_fn::row_fn_macro_support::VortexResult<
                Option<$crate::expr::Expression>,
            > {
                $crate::expr::union_child_validities(expression)
            }

            fn is_strict(&self, _options: &Self::Options) -> bool {
                true
            }

            fn is_fallible(&self, _options: &Self::Options) -> bool {
                <$function as $crate::scalar_fn::RowFn>::FALLIBLE
            }
        }
    };
}

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
/// capabilities. A function that needs only the standard hooks can use [`impl_row_fn_vtable`] to
/// generate its complete vtable.
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
