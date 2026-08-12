// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution that writes through an output sink.

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use super::RowExecution;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::unstable::row::ElementTuple;
use crate::scalar_fn::unstable::row::OutputSink;
use crate::scalar_fn::unstable::row::SinkResult;

/// Ensure that every decoded input addresses the complete row loop.
fn ensure_decoded_lengths<Args: ElementTuple>(
    columns: &Args::Columns,
    views: Option<&Args::Views<'_>>,
    row_count: usize,
) -> VortexResult<()> {
    let lengths_match = match views {
        Some(views) => Args::view_lens_match(views, row_count),
        None => Args::decoded_lens_match(columns, row_count),
    };
    vortex_ensure!(
        lengths_match,
        "a decoded row input does not address exactly {row_count} rows",
    );

    Ok(())
}

/// Decode every input column once, allocate the sink once, then write one row at a time.
///
/// The sink lives here rather than in the closure, so `apply` stays [`Fn`] and mutable output state
/// does not need to be captured by the closure.
pub fn execute_sink<Args, Prepared, Sink, ApplyResult, Options>(
    args: &dyn ExecutionArgs,
    sink_dtype: &DType,
    ctx: &mut ExecutionCtx,
    prepare: impl FnOnce(Args::ConstElems<'_>) -> Prepared,
    apply: impl Fn(&Prepared, Args::Elems<'_>, <Sink as OutputSink<Options>>::Row<'_>) -> ApplyResult,
) -> VortexResult<RowExecution>
where
    Args: ElementTuple,
    Sink: OutputSink<Options>,
    ApplyResult: SinkResult<WriteToken = <Sink as OutputSink<Options>>::WriteToken>,
{
    let row_count = args.row_count();
    let mut sink = <Sink as OutputSink<Options>>::with_capacity(row_count, sink_dtype)?;
    let columns = Args::decode(args, ctx)?;
    let prepared = prepare(Args::constants(&columns));
    let views = Args::per_row_views(&columns);
    ensure_decoded_lengths::<Args>(&columns, views.as_ref(), row_count)?;

    {
        // Borrow the sink once so its shape and buffer descriptor remain loop invariants. This
        // scope releases the borrow before `finish_sink` consumes the sink.
        let mut rows = <Sink as OutputSink<Options>>::rows(&mut sink);
        vortex_ensure!(
            <Sink as OutputSink<Options>>::row_count_matches(&rows, row_count),
            "the output sink does not address exactly {row_count} rows",
        );

        // The all-per-row representation removes argument-shape dispatch from the hot loop. The
        // mixed path instead reads collapsed batch constants at row zero.
        if let Some(views) = views {
            for index in 0..row_count {
                // SAFETY: `ensure_decoded_lengths` proved every view has `row_count` rows before
                // the loop.
                let elements = unsafe { Args::get_from_views_unchecked(&views, index) };
                // SAFETY: `row_count_matches` proved the sink addresses every loop index.
                let output =
                    unsafe { <Sink as OutputSink<Options>>::row_unchecked(&mut rows, index) };
                apply(&prepared, elements, output).into_result()?;
            }
        } else {
            for index in 0..row_count {
                // SAFETY: `row_count_matches` proved the sink addresses every loop index.
                let output =
                    unsafe { <Sink as OutputSink<Options>>::row_unchecked(&mut rows, index) };
                apply(&prepared, Args::get(&columns, index), output).into_result()?;
            }
        }
    }

    finish_sink::<Sink, Options>(sink)
}

/// Run a prepared sink over only the rows set in `valid`, or decline when the sink cannot skip.
pub fn execute_sink_valid_rows<Args, Prepared, Sink, ApplyResult, Options>(
    args: &dyn ExecutionArgs,
    sink_dtype: &DType,
    valid: &Mask,
    ctx: &mut ExecutionCtx,
    prepare: impl FnOnce(Args::ConstElems<'_>) -> Prepared,
    apply: impl Fn(&Prepared, Args::Elems<'_>, <Sink as OutputSink<Options>>::Row<'_>) -> ApplyResult,
) -> VortexResult<Option<RowExecution>>
where
    Args: ElementTuple,
    Sink: OutputSink<Options>,
    ApplyResult: SinkResult<WriteToken = <Sink as OutputSink<Options>>::WriteToken>,
{
    // Decline before input decoding or sink allocation when this sink cannot initialize rows that
    // the mask skips. The capability and the operation are the same function pointer.
    let Some(initialize_skipped_rows) = <Sink as OutputSink<Options>>::skipped_rows_initializer()
    else {
        return Ok(None);
    };

    // Null-tolerant decoding exposes values behind nulls without filtering the inputs first. An
    // element representation may decline when it cannot provide those values safely.
    let Some(columns) = Args::decode_null_tolerant(args, ctx)? else {
        return Ok(None);
    };
    let prepared = prepare(Args::constants(&columns));
    let row_count = args.row_count();
    let mut sink = <Sink as OutputSink<Options>>::with_capacity(row_count, sink_dtype)?;

    // Batch execution resolves all-valid and all-null inputs before selecting this path.
    let AllOr::Some(valid) = valid.bit_buffer() else {
        vortex_bail!("execute_sink_valid_rows requires a mixed mask");
    };
    vortex_ensure!(
        valid.len() == row_count,
        "the validity mask does not address exactly {row_count} rows",
    );

    {
        let mut rows = <Sink as OutputSink<Options>>::rows(&mut sink);
        vortex_ensure!(
            <Sink as OutputSink<Options>>::row_count_matches(&rows, row_count),
            "the output sink does not address exactly {row_count} rows",
        );

        let views = Args::per_row_views(&columns);
        ensure_decoded_lengths::<Args>(&columns, views.as_ref(), row_count)?;

        // The loop writes only valid indices, but the sink still finishes a full-length output.
        // Initialize placeholders now; batch execution masks them before the result escapes.
        initialize_skipped_rows(&mut rows);

        // Mask traversal is callback-based and cannot return a `VortexResult`. Record the first
        // immediate error, turn later callbacks into no-ops, and return before finishing the sink.
        let mut error = None;
        valid.for_each_set_index(|index| {
            if error.is_some() {
                return;
            }

            // SAFETY: `row_count_matches` proved that the sink addresses every mask index, which
            // is below the mask's validated `row_count`.
            let output = unsafe { <Sink as OutputSink<Options>>::row_unchecked(&mut rows, index) };
            let result = match &views {
                Some(views) => {
                    // SAFETY: `ensure_decoded_lengths` proved every view has `row_count` rows, and
                    // mask indices are below `row_count`.
                    let elements = unsafe { Args::get_from_views_unchecked(views, index) };
                    apply(&prepared, elements, output)
                }
                None => apply(&prepared, Args::get(&columns, index), output),
            };
            if let Err(err) = result.into_result() {
                error = Some(err);
            }
        });

        if let Some(error) = error {
            return Err(error);
        }
    }

    finish_sink::<Sink, Options>(sink).map(Some)
}

fn finish_sink<S, Options>(sink: S) -> VortexResult<RowExecution>
where
    S: OutputSink<Options>,
{
    // SAFETY: callers reach this helper only after every completed callback returned the sink's
    // write token. Skipped-row traversal also ran the sink's initializer before visiting its mask.
    // The sink contract defines how that evidence establishes initialization of its row storage.
    unsafe { <S as OutputSink<Options>>::finish(sink) }.map(RowExecution::Output)
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_mask::Mask;

    use super::execute_sink_valid_rows;
    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::PrimitiveArray;
    use crate::dtype::DType;
    use crate::dtype::NativePType;
    use crate::scalar_fn::EmptyOptions;
    use crate::scalar_fn::VecExecutionArgs;
    use crate::scalar_fn::unstable::row::OutputSink;
    use crate::validity::Validity;

    struct NonSkippingSink;

    // SAFETY: `with_capacity` always returns an error, so no sink value can reach `rows`, `row`, or
    // `finish` through the executor. The row-initialization requirements are therefore vacuous.
    unsafe impl<Options> OutputSink<Options> for NonSkippingSink {
        type Rows<'a> = ();
        type Row<'a> = ();
        type WriteToken = ();

        fn sink_dtype(_options: &Options, _args: &[DType]) -> VortexResult<DType> {
            Ok(DType::from(i64::PTYPE))
        }

        fn with_capacity(_rows: usize, _dtype: &DType) -> VortexResult<Self> {
            Err(vortex_err!(
                "a non-skipping sink must decline before allocation"
            ))
        }

        fn rows(&mut self) -> Self::Rows<'_> {}

        fn row_count_matches(_rows: &Self::Rows<'_>, _row_count: usize) -> bool {
            true
        }

        unsafe fn row_unchecked<'a>(_rows: &'a mut Self::Rows<'_>, _index: usize) -> Self::Row<'a> {
        }

        unsafe fn finish(self) -> VortexResult<ArrayRef> {
            Err(vortex_err!("a non-skipping sink must not finish"))
        }
    }

    #[test]
    fn test_non_skipping_sink_declines_before_allocation() -> VortexResult<()> {
        let input = PrimitiveArray::new(vec![1_i64, 2], Validity::NonNullable).into_array();
        let args = VecExecutionArgs::new(vec![input], 2);
        let valid = Mask::from_iter([true, false]);
        let mut ctx = array_session().create_execution_ctx();

        let execution = execute_sink_valid_rows::<(i64,), (), NonSkippingSink, (), EmptyOptions>(
            &args,
            &DType::from(i64::PTYPE),
            &valid,
            &mut ctx,
            |_| (),
            |_, _, _| (),
        )?;

        assert!(execution.is_none());
        Ok(())
    }
}
