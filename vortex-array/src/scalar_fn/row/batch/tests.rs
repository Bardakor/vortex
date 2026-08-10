// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;

use super::Batch;
use super::BatchPlan;
use super::RowPolicy;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::ConstantArray;
use crate::arrays::PrimitiveArray;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::scalar_fn::DeferredError;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::OutputSink;
use crate::scalar_fn::RowFn;
use crate::scalar_fn::RowVisitor;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::VecExecutionArgs;
use crate::validity::Validity;

#[derive(Clone)]
struct RetryConstantAdd;

#[derive(Clone)]
struct NullarySeven;

struct I64Sink(BufferMut<i64>);

impl OutputSink for I64Sink {
    type Rows<'a> = &'a mut [i64];
    type Row<'a> = &'a mut i64;
    type WriteToken = ();

    fn sink_dtype(_args: &[DType]) -> VortexResult<DType> {
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

    unsafe fn finish(self, _error: DeferredError) -> VortexResult<ArrayRef> {
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

    fn dispatch<V: RowVisitor>(
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

    fn dispatch<V: RowVisitor>(
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
    ) -> VortexResult<Option<ArrayRef>> {
        if args[0].len() == 1 {
            return Ok(Some(ConstantArray::new(0u8, args[0].len()).into_array()));
        }

        Ok(None)
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

    let result = ScalarFnVTable::execute(&RetryConstantAdd, &EmptyOptions, &args, &mut ctx);

    assert!(result.is_err());
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
        |args, _ctx| {
            Ok(super::super::execute::RowExecution::Output(
                args.arrays[0].clone(),
            ))
        },
        |args, _valid, _ctx| {
            Ok(Some(super::super::execute::RowExecution::Output(
                args.arrays[0].clone(),
            )))
        },
        &mut ctx,
    )?;

    assert_arrays_eq!(&actual, &input, &mut ctx);
    Ok(())
}

#[test]
fn test_nullary_row_function_broadcasts() -> VortexResult<()> {
    let args = VecExecutionArgs::new(vec![], 3);
    let mut ctx = array_session().create_execution_ctx();

    let actual = ScalarFnVTable::execute(&NullarySeven, &EmptyOptions, &args, &mut ctx)?;
    let expected = PrimitiveArray::from_iter([7i64, 7, 7]).into_array();

    assert_arrays_eq!(&actual, &expected, &mut ctx);
    Ok(())
}
