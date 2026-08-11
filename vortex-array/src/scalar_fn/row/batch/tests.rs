// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use rstest::rstest;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;

use super::super::execute::RowExecution;
use super::Batch;
use super::BatchPlan;
use super::RowPolicy;
use super::finalize_kernel_output;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::PrimitiveArray;
use crate::assert_arrays_eq;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::OutputElement;
use crate::scalar_fn::OutputSink;
use crate::scalar_fn::RowFn;
use crate::scalar_fn::RowVisitor;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::VecExecutionArgs;
use crate::scalar_fn::execute_rows;
use crate::scalar_fn::row_fn_return_dtype;
use crate::validity::Validity;

#[derive(Clone)]
struct RetryConstantAdd;

#[derive(Clone)]
struct NullarySeven;

#[derive(Clone)]
struct OriginalInputReducer;

#[derive(Clone)]
struct InvalidEncodedReduction;

#[derive(Clone)]
struct DeferredOriginalReducer;

#[derive(Clone)]
struct SinkOptions;

struct OptionsCheckingSink;

#[derive(Clone)]
struct InvalidKernelOutput;

/// Deliberately violates [`OutputElement::build`] to test validation at the public boundary.
struct NullProducingI64(i64);

#[derive(Clone)]
struct PreparedAdd {
    visit: PreparedVisit,
    prepares: Arc<AtomicUsize>,
}

#[derive(Clone, Copy)]
enum PreparedVisit {
    Owned,
    Sink,
    Deferred,
}

// SAFETY: `with_capacity` always returns an error, so no sink value can reach `rows`, `row`, or
// `finish` through the executor. The row-initialization requirements are therefore vacuous.
unsafe impl OutputSink<bool> for OptionsCheckingSink {
    type Rows<'a> = ();
    type Row<'a> = ();
    type WriteToken = ();

    fn sink_dtype(enabled: &bool, _args: &[DType]) -> VortexResult<DType> {
        if !enabled {
            vortex_bail!(InvalidArgument: "the test sink is disabled");
        }

        Ok(DType::from(i64::PTYPE))
    }

    fn with_capacity(_rows: usize, _dtype: &DType) -> VortexResult<Self> {
        vortex_bail!("the planning-only test sink must not be allocated")
    }

    fn rows(&mut self) -> Self::Rows<'_> {}

    fn row_count_matches(_rows: &Self::Rows<'_>, _row_count: usize) -> bool {
        true
    }

    fn row<'a>(_rows: &'a mut Self::Rows<'_>, _index: usize) -> Self::Row<'a> {}

    unsafe fn finish(self) -> VortexResult<ArrayRef> {
        vortex_bail!("the planning-only test sink must not finish")
    }
}

impl OutputElement for NullProducingI64 {
    fn element_dtype() -> DType {
        DType::from(i64::PTYPE)
    }

    fn build(values: Vec<Self>) -> ArrayRef {
        let values: Vec<_> = values.into_iter().map(|value| value.0).collect();
        let validity = Validity::from_iter((0..values.len()).map(|index| index != 0));

        PrimitiveArray::new(values, validity).into_array()
    }
}

struct I64Sink(BufferMut<i64>);

// SAFETY: every row is initialized by `BufferMut::zeroed`, and the sink exposes exactly that
// initialized slice. The `()` write token therefore proves no additional invariant.
unsafe impl<Options> OutputSink<Options> for I64Sink {
    type Rows<'a> = &'a mut [i64];
    type Row<'a> = &'a mut i64;
    type WriteToken = ();

    fn sink_dtype(_options: &Options, _args: &[DType]) -> VortexResult<DType> {
        Ok(DType::from(i64::PTYPE))
    }

    fn with_capacity(rows: usize, _dtype: &DType) -> VortexResult<Self> {
        Ok(Self(BufferMut::zeroed(rows)))
    }

    fn rows(&mut self) -> Self::Rows<'_> {
        self.0.as_mut_slice()
    }

    fn row_count_matches(rows: &Self::Rows<'_>, row_count: usize) -> bool {
        rows.len() == row_count
    }

    fn row<'a>(rows: &'a mut Self::Rows<'_>, index: usize) -> Self::Row<'a> {
        &mut rows[index]
    }

    unsafe fn finish(self) -> VortexResult<ArrayRef> {
        Ok(PrimitiveArray::new(self.0.freeze(), Validity::NonNullable).into_array())
    }
}

impl RowFn for NullarySeven {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &[];

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("test.nullary_seven");
        *ID
    }

    fn dispatch<V: RowVisitor<Self::Options>>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit_into::<(), I64Sink, _>(|(), output| {
            *output = 7;
        })
    }
}

impl RowFn for RetryConstantAdd {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];
    const FALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("test.retry_constant_add");
        *ID
    }

    fn dispatch<V: RowVisitor<Self::Options>>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit_deferred::<(u8, u8), u8, bool>(
            |(lhs, rhs)| lhs.overflowing_add(rhs),
            |failed| {
                if failed {
                    return Err(vortex_err!(InvalidArgument: "checked add overflowed"));
                }

                Ok(())
            },
        )
    }

    fn reduce_encoded(
        &self,
        _options: &Self::Options,
        args: &[ArrayRef],
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<RowExecution>> {
        if args[0].len() == 1 {
            return Ok(Some(RowExecution::Output(
                ConstantArray::new(0u8, args[0].len()).into_array(),
            )));
        }

        Ok(None)
    }
}

impl RowFn for OriginalInputReducer {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["value"];

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("test.original_input_reducer");
        *ID
    }

    fn dispatch<V: RowVisitor<Self::Options>>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit::<(i64,), i64>(|(value,)| value)
    }

    fn reduce_encoded(
        &self,
        _options: &Self::Options,
        args: &[ArrayRef],
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<RowExecution>> {
        if args[0].len() == 3 {
            return Ok(Some(RowExecution::Output(
                ConstantArray::new(42_i64, 3).into_array(),
            )));
        }

        Ok(None)
    }
}

impl RowFn for InvalidEncodedReduction {
    type Options = usize;

    const ARG_NAMES: &'static [&'static str] = &["value"];

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("test.invalid_encoded_reduction");
        *ID
    }

    fn dispatch<V: RowVisitor<Self::Options>>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit::<(i64,), i64>(|(value,)| value)
    }

    fn reduce_encoded(
        &self,
        null_index: &Self::Options,
        _args: &[ArrayRef],
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<RowExecution>> {
        Ok(Some(RowExecution::Output(
            PrimitiveArray::new(
                vec![10_i64, 20],
                Validity::from_iter((0..2).map(|index| index != *null_index)),
            )
            .into_array(),
        )))
    }
}

impl RowFn for DeferredOriginalReducer {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["value"];
    const FALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("test.deferred_original_reducer");
        *ID
    }

    fn dispatch<V: RowVisitor<Self::Options>>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit::<(i64,), i64>(|(value,)| value)
    }

    fn reduce_encoded(
        &self,
        _options: &Self::Options,
        _args: &[ArrayRef],
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<RowExecution>> {
        Ok(Some(RowExecution::DeferredError(vortex_err!(
            InvalidArgument: "encoded payload failed"
        ))))
    }
}

impl RowFn for SinkOptions {
    type Options = bool;

    const ARG_NAMES: &'static [&'static str] = &[];

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("test.sink_options");
        *ID
    }

    fn dispatch<V: RowVisitor<Self::Options>>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit_into::<(), OptionsCheckingSink, _>(|(), ()| ())
    }
}

impl RowFn for InvalidKernelOutput {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["value"];

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("test.invalid_kernel_output");
        *ID
    }

    fn dispatch<V: RowVisitor<Self::Options>>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit::<(i64,), NullProducingI64>(|(value,)| NullProducingI64(value))
    }
}

impl RowFn for PreparedAdd {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];
    const FALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("test.prepared_add");
        *ID
    }

    fn dispatch<V: RowVisitor<Self::Options>>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        let prepares = Arc::clone(&self.prepares);
        let prepare = move |(_lhs, rhs): (Option<i64>, Option<i64>)| {
            prepares.fetch_add(1, Ordering::Relaxed);
            rhs
        };

        match self.visit {
            PreparedVisit::Owned => visitor
                .visit_prepared::<(i64, i64), i64, _>(prepare, |constant_rhs, (lhs, rhs)| {
                    lhs.wrapping_add(constant_rhs.unwrap_or(rhs))
                }),
            PreparedVisit::Sink => visitor.visit_prepared_into::<(i64, i64), I64Sink, _, ()>(
                prepare,
                |constant_rhs, (lhs, rhs), output| {
                    *output = lhs.wrapping_add(constant_rhs.unwrap_or(rhs));
                },
            ),
            PreparedVisit::Deferred => visitor.visit_prepared_deferred::<(i64, i64), i64, _, bool>(
                prepare,
                |constant_rhs, (lhs, rhs)| lhs.overflowing_add(constant_rhs.unwrap_or(rhs)),
                |failed| {
                    if failed {
                        return Err(vortex_err!(InvalidArgument: "prepared add overflowed"));
                    }

                    Ok(())
                },
            ),
        }
    }
}

#[test]
fn test_batch_rejects_input_length_mismatch() -> VortexResult<()> {
    static ID: CachedId = CachedId::new("test.row_batch");

    let input = PrimitiveArray::new(vec![1i64, 2], Validity::NonNullable).into_array();
    let args = VecExecutionArgs::new(vec![input], 3);
    let result = Batch::new(*ID, &args, |_| {
        Ok(BatchPlan {
            output_dtype: DType::from(i64::PTYPE),
            policy: RowPolicy::Dense,
        })
    });

    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_dense_retry_does_not_reduce_filtered_inputs() -> VortexResult<()> {
    let lhs =
        PrimitiveArray::new(vec![u8::MAX, 1], Validity::from_iter([true, false])).into_array();
    let rhs = ConstantArray::new(1u8, 2).into_array();
    let args = VecExecutionArgs::new(vec![lhs, rhs], 2);
    let mut ctx = array_session().create_execution_ctx();

    let result = execute_rows(&RetryConstantAdd, &EmptyOptions, &args, &mut ctx);

    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_dense_retry_suppresses_null_row_failure() -> VortexResult<()> {
    let lhs =
        PrimitiveArray::new(vec![1, u8::MAX], Validity::from_iter([true, false])).into_array();
    let rhs = ConstantArray::new(1_u8, 2).into_array();
    let args = VecExecutionArgs::new(vec![lhs, rhs], 2);
    let mut ctx = array_session().create_execution_ctx();

    let actual = execute_rows(&RetryConstantAdd, &EmptyOptions, &args, &mut ctx)?;
    let expected = PrimitiveArray::new(vec![2_u8, 0], Validity::from_iter([true, false]));

    assert_arrays_eq!(&actual, expected.as_ref(), &mut ctx);
    Ok(())
}

#[test]
fn test_reduce_encoded_defers_errors_behind_nulls() -> VortexResult<()> {
    let input =
        PrimitiveArray::new(vec![10_i64, 20], Validity::from_iter([true, false])).into_array();
    let args = VecExecutionArgs::new(vec![input.clone()], 2);
    let mut ctx = array_session().create_execution_ctx();

    let actual = execute_rows(&DeferredOriginalReducer, &EmptyOptions, &args, &mut ctx)?;

    assert_arrays_eq!(&actual, &input, &mut ctx);
    Ok(())
}

#[rstest]
#[case::all_valid(Validity::AllValid)]
#[case::mixed(Validity::from_iter([true, false]))]
fn test_reduce_encoded_rejects_nulls_on_valid_rows(#[case] validity: Validity) -> VortexResult<()> {
    let input = PrimitiveArray::new(vec![10_i64, 20], validity).into_array();
    let args = VecExecutionArgs::new(vec![input], 2);
    let mut ctx = array_session().create_execution_ctx();

    let error = match execute_rows(&InvalidEncodedReduction, &0, &args, &mut ctx) {
        Err(error) => error,
        Ok(_) => vortex_bail!("an encoded reduction introduced a null on a valid row"),
    };
    let error = error.to_string();

    assert!(
        error.contains("test.invalid_encoded_reduction"),
        "the boundary error must name the function, got {error}",
    );
    assert!(
        error.contains("encoded reduction produced nulls for valid rows"),
        "the boundary error must identify invalid reduced output, got {error}",
    );
    Ok(())
}

#[test]
fn test_reduce_encoded_preserves_input_nulls() -> VortexResult<()> {
    let input =
        PrimitiveArray::new(vec![10_i64, 20], Validity::from_iter([true, false])).into_array();
    let args = VecExecutionArgs::new(vec![input.clone()], 2);
    let mut ctx = array_session().create_execution_ctx();

    let actual = execute_rows(&InvalidEncodedReduction, &1, &args, &mut ctx)?;

    assert_arrays_eq!(&actual, &input, &mut ctx);
    Ok(())
}

#[test]
fn test_reduce_encoded_precedes_constant_broadcast() -> VortexResult<()> {
    let input = ConstantArray::new(7_i64, 3).into_array();
    let args = VecExecutionArgs::new(vec![input], 3);
    let mut ctx = array_session().create_execution_ctx();

    let actual = execute_rows(&OriginalInputReducer, &EmptyOptions, &args, &mut ctx)?;
    let expected = ConstantArray::new(42_i64, 3).into_array();

    assert_arrays_eq!(&actual, &expected, &mut ctx);
    Ok(())
}

#[test]
fn test_constant_input_broadcasts_one_row() -> VortexResult<()> {
    let input = ConstantArray::new(7_i64, 2).into_array();
    let args = VecExecutionArgs::new(vec![input.clone()], 2);
    let mut ctx = array_session().create_execution_ctx();

    let actual = execute_rows(&OriginalInputReducer, &EmptyOptions, &args, &mut ctx)?;

    assert_arrays_eq!(&actual, &input, &mut ctx);
    Ok(())
}

#[rstest]
#[case::all_valid([true, true])]
#[case::all_invalid([false, false])]
fn test_resolve_validity_array_masks(#[case] validity: [bool; 2]) -> VortexResult<()> {
    static ID: CachedId = CachedId::new("test.resolve_validity");

    let validity = Validity::Array(BoolArray::from_iter(validity).into_array());
    let input = PrimitiveArray::new(vec![4_i64, 5], validity).into_array();
    let args = VecExecutionArgs::new(vec![input.clone()], 2);
    let batch = Batch::new(*ID, &args, |_| {
        Ok(BatchPlan {
            output_dtype: DType::from(i64::PTYPE),
            policy: RowPolicy::ValidOnly,
        })
    })?;
    let mut ctx = array_session().create_execution_ctx();

    let actual = batch.execute(
        |_args, _ctx| Ok(None),
        |args, _ctx| Ok(RowExecution::Output(args.arrays[0].clone())),
        |_args, _valid, _ctx| Ok(None),
        &mut ctx,
    )?;

    assert_arrays_eq!(&actual, &input, &mut ctx);
    Ok(())
}

#[test]
fn test_valid_only_filters_and_scatters() -> VortexResult<()> {
    static ID: CachedId = CachedId::new("test.filter_and_scatter");

    let input = PrimitiveArray::new(
        vec![10_i64, 20, 30, 40],
        Validity::from_iter([true, false, true, false]),
    )
    .into_array();
    let args = VecExecutionArgs::new(vec![input.clone()], 4);
    let batch = Batch::new(*ID, &args, |_| {
        Ok(BatchPlan {
            output_dtype: DType::from(i64::PTYPE),
            policy: RowPolicy::ValidOnly,
        })
    })?;
    let mut ctx = array_session().create_execution_ctx();

    let actual = batch.execute(
        |_args, _ctx| Ok(None),
        |args, _ctx| Ok(RowExecution::Output(args.arrays[0].clone())),
        |_args, _valid, _ctx| Ok(None),
        &mut ctx,
    )?;

    assert_arrays_eq!(&actual, &input, &mut ctx);
    Ok(())
}

#[test]
fn test_finalize_kernel_output_validates_shape_and_dtype() -> VortexResult<()> {
    static ID: CachedId = CachedId::new("test.finalize_kernel_output");

    let values = PrimitiveArray::from_iter([1_i64, 2]).into_array();
    let result_dtype = DType::Primitive(i64::PTYPE, Nullability::Nullable);
    let mut ctx = array_session().create_execution_ctx();

    let actual = finalize_kernel_output(*ID, &result_dtype, 2, values.clone(), &mut ctx)?;
    let expected = PrimitiveArray::new(vec![1_i64, 2], Validity::AllValid).into_array();
    assert_eq!(actual.dtype(), &result_dtype);
    assert_arrays_eq!(&actual, &expected, &mut ctx);

    assert!(finalize_kernel_output(*ID, &result_dtype, 3, values, &mut ctx).is_err());

    let bools = BoolArray::from_iter([true, false]).into_array();
    assert!(finalize_kernel_output(*ID, &result_dtype, 2, bools, &mut ctx).is_err());
    Ok(())
}

#[test]
fn test_sink_dtype_receives_function_options() -> VortexResult<()> {
    assert_eq!(
        row_fn_return_dtype(&SinkOptions, &true, &[])?,
        DType::from(i64::PTYPE)
    );
    assert!(row_fn_return_dtype(&SinkOptions, &false, &[]).is_err());
    Ok(())
}

#[test]
fn test_nonnullable_kernel_output_rejects_nulls_at_function_boundary() -> VortexResult<()> {
    let input = PrimitiveArray::from_iter([1_i64, 2]).into_array();

    assert_invalid_kernel_output(input)
}

#[test]
fn test_all_valid_kernel_output_rejects_nulls_at_function_boundary() -> VortexResult<()> {
    let input = PrimitiveArray::new(vec![1_i64, 2], Validity::AllValid).into_array();

    assert_invalid_kernel_output(input)
}

#[track_caller]
fn assert_invalid_kernel_output(input: ArrayRef) -> VortexResult<()> {
    let args = VecExecutionArgs::new(vec![input], 2);
    let mut ctx = array_session().create_execution_ctx();
    let execution = execute_rows(&InvalidKernelOutput, &EmptyOptions, &args, &mut ctx);
    let error = match execution {
        Err(error) => error,
        Ok(output) => match output.execute::<PrimitiveArray>(&mut ctx) {
            Err(error) => error,
            Ok(_) => vortex_bail!("an invalid row kernel output passed boundary validation"),
        },
    };
    let error = error.to_string();

    assert!(
        error.contains("test.invalid_kernel_output"),
        "the boundary error must name the function, got {error}",
    );
    assert!(
        error.contains("row kernel produced nulls for valid rows"),
        "the boundary error must identify invalid row output, got {error}",
    );
    Ok(())
}

#[rstest]
#[case::owned_constant(PreparedVisit::Owned, true)]
#[case::owned_varying(PreparedVisit::Owned, false)]
#[case::sink_constant(PreparedVisit::Sink, true)]
#[case::sink_varying(PreparedVisit::Sink, false)]
#[case::deferred_constant(PreparedVisit::Deferred, true)]
#[case::deferred_varying(PreparedVisit::Deferred, false)]
fn test_prepared_visits(
    #[case] visit: PreparedVisit,
    #[case] constant_rhs: bool,
) -> VortexResult<()> {
    let lhs = PrimitiveArray::from_iter([1_i64, 2]).into_array();
    let rhs = if constant_rhs {
        ConstantArray::new(3_i64, 2).into_array()
    } else {
        PrimitiveArray::from_iter([3_i64, 4]).into_array()
    };
    let args = VecExecutionArgs::new(vec![lhs, rhs], 2);
    let prepares = Arc::new(AtomicUsize::new(0));
    let function = PreparedAdd {
        visit,
        prepares: Arc::clone(&prepares),
    };
    let mut ctx = array_session().create_execution_ctx();

    let actual = execute_rows(&function, &EmptyOptions, &args, &mut ctx)?;
    let expected = if constant_rhs {
        PrimitiveArray::from_iter([4_i64, 5]).into_array()
    } else {
        PrimitiveArray::from_iter([4_i64, 6]).into_array()
    };

    assert_arrays_eq!(&actual, &expected, &mut ctx);
    assert_eq!(prepares.load(Ordering::Relaxed), 1);
    Ok(())
}

#[rstest]
#[case::dense(RowPolicy::Dense)]
#[case::dense_with_retry(RowPolicy::DenseWithRetry)]
#[case::valid_only(RowPolicy::ValidOnly)]
fn test_strategy_matrix(#[case] policy: RowPolicy) -> VortexResult<()> {
    static ID: CachedId = CachedId::new("test.row_strategy");

    let input = PrimitiveArray::new(vec![1i64, 2, 3], Validity::from_iter([true, false, true]))
        .into_array();
    let args = VecExecutionArgs::new(vec![input.clone()], 3);
    let batch = Batch::new(*ID, &args, |_| {
        Ok(BatchPlan {
            output_dtype: DType::from(i64::PTYPE),
            policy,
        })
    })?;
    let mut ctx = array_session().create_execution_ctx();

    let actual = batch.execute(
        |_args, _ctx| Ok(None),
        |args, _ctx| Ok(RowExecution::Output(args.arrays[0].fill_null(0_i64)?)),
        |args, _valid, _ctx| Ok(Some(RowExecution::Output(args.arrays[0].fill_null(0_i64)?))),
        &mut ctx,
    )?;

    assert_arrays_eq!(&actual, &input, &mut ctx);
    Ok(())
}

#[test]
fn test_nullary_row_function_broadcasts() -> VortexResult<()> {
    let args = VecExecutionArgs::new(vec![], 3);
    let mut ctx = array_session().create_execution_ctx();

    let actual = execute_rows(&NullarySeven, &EmptyOptions, &args, &mut ctx)?;
    let expected = PrimitiveArray::from_iter([7i64, 7, 7]).into_array();

    assert_arrays_eq!(&actual, &expected, &mut ctx);
    Ok(())
}
