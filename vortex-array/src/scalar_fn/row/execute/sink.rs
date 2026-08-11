// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution that writes through an output sink.

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use super::RowExecution;
use super::ensure_decoded_lengths;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::scalar_fn::ElementTuple;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::OutputSink;
use crate::scalar_fn::SinkResult;

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
    let mut accumulated = ApplyResult::Accumulated::default();

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
                apply(
                    &prepared,
                    elements,
                    <Sink as OutputSink<Options>>::row(&mut rows, index),
                )
                .accumulate(&mut accumulated)?;
            }
        } else {
            for index in 0..row_count {
                apply(
                    &prepared,
                    Args::get(&columns, index),
                    <Sink as OutputSink<Options>>::row(&mut rows, index),
                )
                .accumulate(&mut accumulated)?;
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
    let mut accumulated = ApplyResult::Accumulated::default();

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

            let result = match &views {
                Some(views) => apply(
                    &prepared,
                    // SAFETY: `ensure_decoded_lengths` proved every view has `row_count` rows, and
                    // mask indices are below `row_count`.
                    unsafe { Args::get_from_views_unchecked(views, index) },
                    <Sink as OutputSink<Options>>::row(&mut rows, index),
                ),
                None => apply(
                    &prepared,
                    Args::get(&columns, index),
                    <Sink as OutputSink<Options>>::row(&mut rows, index),
                ),
            };
            if let Err(err) = result.accumulate(&mut accumulated) {
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
mod tests;
