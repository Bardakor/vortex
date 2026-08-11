// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`ScalarFnVTable`](crate::scalar_fn::ScalarFnVTable) adapter for [`RowFn`].
//!
//! The [`visitor`](super::visitor) module validates and executes the concrete row signature
//! selected by dispatch. This module connects those visits to batch execution and exposes the
//! resulting scalar function behavior to the rest of the compute stack.

use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use super::row_fn::RowFn;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::expr::Expression;
use crate::expr::union_child_validities;
use crate::scalar_fn::Arity;
use crate::scalar_fn::BorrowedExecutionArgs;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::row::batch::Batch;
use crate::scalar_fn::row::batch::KernelArgs;
use crate::scalar_fn::row::batch::finalize_kernel_output;
use crate::scalar_fn::row::execute::RowExecution;
use crate::scalar_fn::row::visitor::ExecuteRows;
use crate::scalar_fn::row::visitor::ExecuteValidRows;
use crate::scalar_fn::row::visitor::PlanRows;

impl<F: RowFn> ScalarFnVTable for F {
    type Options = F::Options;

    fn id(&self) -> ScalarFnId {
        RowFn::id(self)
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        RowFn::serialize(self, options)
    }

    fn deserialize(&self, metadata: &[u8], session: &VortexSession) -> VortexResult<Self::Options> {
        RowFn::deserialize(self, metadata, session)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(F::ARG_NAMES.len())
    }

    fn child_name(&self, _options: &Self::Options, child_index: usize) -> ChildName {
        ChildName::from(F::ARG_NAMES[child_index])
    }

    fn return_dtype(&self, options: &Self::Options, args: &[DType]) -> VortexResult<DType> {
        row_fn_return_dtype(self, options, args)
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        execute_rows(self, options, args, ctx)
    }

    fn validity(
        &self,
        _options: &Self::Options,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        union_child_validities(expression)
    }

    fn is_strict(&self, _options: &Self::Options) -> bool {
        true
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        F::FALLIBLE
    }
}

/// Compute the return dtype of a [`RowFn`] kernel without invoking its blanket vtable.
pub fn row_fn_return_dtype<F: RowFn>(
    function: &F,
    options: &F::Options,
    args: &[DType],
) -> VortexResult<DType> {
    let plan = function.dispatch(options, args, PlanRows::<F>::new(args, options))?;

    Ok(plan.result_dtype(args))
}

/// Execute a [`RowFn`] without using its blanket [`ScalarFnVTable`] implementation.
///
/// A type cannot implement both [`RowFn`] and [`ScalarFnVTable`] because every `RowFn` receives the
/// standard vtable automatically. Existing vtables can keep their custom hooks on one type and
/// delegate row execution to a private `RowFn` kernel through this function.
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

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;
    use vortex_session::registry::CachedId;

    use crate::dtype::DType;
    use crate::scalar_fn::Arity;
    use crate::scalar_fn::EmptyOptions;
    use crate::scalar_fn::RowFn;
    use crate::scalar_fn::RowVisitor;
    use crate::scalar_fn::ScalarFnId;
    use crate::scalar_fn::ScalarFnVTable;

    #[derive(Clone)]
    struct TestRowFn;

    impl RowFn for TestRowFn {
        type Options = EmptyOptions;

        const ARG_NAMES: &'static [&'static str] = &[];

        fn id(&self) -> ScalarFnId {
            static ID: CachedId = CachedId::new("test.row_fn_vtable");
            *ID
        }

        fn dispatch<V: RowVisitor<Self::Options>>(
            &self,
            _options: &Self::Options,
            _args: &[DType],
            _visitor: V,
        ) -> VortexResult<V::VisitResult> {
            vortex_bail!("compile-only RowFn must not execute")
        }
    }

    #[test]
    fn test_row_fn_implements_standard_vtable() {
        let function = TestRowFn;
        let options = EmptyOptions;

        assert_eq!(ScalarFnVTable::arity(&function, &options), Arity::Exact(0));
        assert!(ScalarFnVTable::is_strict(&function, &options));
        assert!(!ScalarFnVTable::is_fallible(&function, &options));
    }
}
