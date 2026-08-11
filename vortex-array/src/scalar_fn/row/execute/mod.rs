// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Row-loop execution for owned outputs and output sinks.
//!
//! [`owned`] stores one independent value per row and can reduce failure evidence. [`sink`]
//! drives output builders whose row handles may refer to shared batch state.

use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::scalar_fn::ElementTuple;

mod owned;
pub(super) use owned::execute_owned;
pub(super) use owned::execute_owned_infallible;

mod sink;
pub(super) use sink::execute_sink;
pub(super) use sink::execute_sink_valid_rows;

/// The outcome of a row loop before batch execution decides whether an error is observable.
///
/// Together with the surrounding [`VortexResult`], this represents three outcomes:
///
/// - `Err(error)` is a non-retryable execution or immediate row error.
/// - [`Output`](Self::Output) is a successful row loop.
/// - [`DeferredError`](Self::DeferredError) is failure evidence from a completed row loop.
///
/// A dense loop may evaluate values behind nulls. Its deferred error is therefore not necessarily
/// observable: batch execution can retry over only valid rows, suppressing an error that came from
/// a null row while preserving one from a valid row. A plain `VortexResult<ArrayRef>` would lose
/// the distinction between that retryable error and an error for which retrying cannot help.
///
/// Once execution is known to contain only valid rows, converting this outcome into a
/// `VortexResult<ArrayRef>` turns [`DeferredError`](Self::DeferredError) into an ordinary error.
pub enum RowExecution {
    /// The successfully built, full-length output column.
    Output(ArrayRef),

    /// An error constructed from failure evidence reduced across a completed row loop.
    DeferredError(VortexError),
}

impl From<RowExecution> for VortexResult<ArrayRef> {
    fn from(execution: RowExecution) -> Self {
        match execution {
            RowExecution::Output(output) => Ok(output),
            RowExecution::DeferredError(error) => Err(error),
        }
    }
}

/// Ensure that every decoded input addresses the complete row loop.
pub(super) fn ensure_decoded_lengths<Args: ElementTuple>(
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
