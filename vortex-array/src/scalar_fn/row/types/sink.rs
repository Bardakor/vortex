// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The column builders a row function can write its output into.

use std::mem::MaybeUninit;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::dtype::DType;
use crate::scalar_fn::DeferredError;
use crate::scalar_fn::OutputElement;

/// A column allocated once per batch that a row closure writes into, one row at a time.
///
/// A sink may use the input dtypes to build a runtime-shaped output or own shared batch state. The
/// executor passes each row slot into an [`Fn`] closure, keeping mutable state out of its capture.
///
/// Rows arrive in increasing index order. Ordinary execution visits `0..row_count` exactly once;
/// skip-invalid execution can omit invalid rows when
/// [`initialize_skipped_rows`](Self::initialize_skipped_rows) returns `true`.
pub trait OutputSink: 'static + Sized {
    /// Whether this sink accepts [`DeferredError`] from its row closure instead of requiring a
    /// per-row [`VortexResult`].
    ///
    /// A supporting sink must return an error from [`finish`](Self::finish) when its deferred error
    /// argument occurred.
    const ERRORS_ARE_DEFERRED: bool = false;

    /// A loop-local view of all output rows.
    ///
    /// Borrowed once before execution so the sink's buffer descriptor and shape become loop
    /// invariants rather than being re-read through `&mut Self` for every row.
    type Rows<'a>
    where
        Self: 'a;

    /// The place a row closure writes one row through, borrowed from the sink.
    type Row<'a>
    where
        Self: 'a;

    /// Proof that a successful row closure left its row handle initialized.
    ///
    /// Use `()` for initialized row handles. A sink exposing uninitialized storage uses a distinct
    /// token returned after initialization. A sink that uses this token to justify unsafe code
    /// **must** prevent safe construction that does not establish the invariant. Make construction
    /// unsafe when Rust cannot tie the token to the supplied row handle.
    type WriteToken: 'static;

    /// The dtype of the column this sink builds, given the function's input dtypes.
    ///
    /// **Must** be non-nullable: batch execution derives nullability from the inputs, widens the
    /// result, and masks the null rows.
    fn sink_dtype(args: &[DType]) -> VortexResult<DType>;

    /// Allocate a sink for `rows` rows of `dtype`, which is this sink's own
    /// [`sink_dtype`](Self::sink_dtype). Called once per batch.
    fn with_capacity(rows: usize, dtype: &DType) -> VortexResult<Self>;

    /// Borrow all output rows for the hot loop.
    fn rows(&mut self) -> Self::Rows<'_>;

    /// Whether every index in `0..row_count` is addressable through [`row`](Self::row).
    ///
    /// Called once before the hot loop. Besides validating the sink contract, this gives the
    /// optimizer the output bounds it needs to remove the bounds check hidden in each row accessor.
    fn row_count_matches(rows: &Self::Rows<'_>, row_count: usize) -> bool;

    /// Initialize every output position so skip-invalid execution may omit rows.
    ///
    /// Return `true` after leaving a legal arbitrary value in every row. Return `false` without
    /// changing `rows` when the sink cannot support skipped rows. Combining the capability probe
    /// and initialization prevents a separate support flag from disagreeing with a no-op method.
    fn initialize_skipped_rows(_rows: &mut Self::Rows<'_>) -> bool {
        false
    }

    /// Hand out the place to write row `index`. Must be `O(1)`: it is called in the row loop.
    fn row<'a>(rows: &'a mut Self::Rows<'_>, index: usize) -> Self::Row<'a>;

    /// Finish into the built column, whose dtype **must** be this sink's
    /// [`sink_dtype`](Self::sink_dtype). Called once per batch with whether any deferred row error
    /// occurred.
    ///
    /// # Safety
    ///
    /// The executor must have completed every row callback successfully, and each callback must
    /// have returned this sink's [`WriteToken`](Self::WriteToken). When skipped rows are allowed,
    /// [`initialize_skipped_rows`](Self::initialize_skipped_rows) must have returned `true` before
    /// traversal.
    unsafe fn finish(self, error: DeferredError) -> VortexResult<ArrayRef>;
}

/// Proof that one uninitialized element row was initialized.
#[must_use = "return this token from the row closure to prove that it initialized the output"]
pub struct InitializedElement(
    /// Private so constructing initialization evidence requires an unsafe operation.
    (),
);

impl InitializedElement {
    /// Write `value` into an uninitialized row and return its proof token.
    ///
    /// # Safety
    ///
    /// `row` must be the [`UninitElementSink`] row supplied to the current callback. The caller must
    /// return the token from that callback. Using another row or returning the token from another
    /// callback can cause undefined behavior.
    #[inline]
    pub unsafe fn write<T>(row: &mut MaybeUninit<T>, value: T) -> Self {
        row.write(value);

        Self(())
    }
}

/// An element sink that leaves dense output uninitialized before the row loop.
///
/// The row closure must return the [`InitializedElement`] from [`InitializedElement::write`] on
/// success. The token is zero-sized, so the proof adds no runtime row state.
///
/// Skip-invalid execution initializes placeholders before omitting rows. Immediate failures are
/// safe because [`OutputSink::finish`] is not called after one.
pub struct UninitElementSink<T> {
    /// Spare storage written in increasing row order.
    values: Vec<T>,

    /// The number of slots exposed to the row loop and initialized before finishing.
    row_count: usize,
}

impl<T: OutputElement + Copy + Default> OutputSink for UninitElementSink<T> {
    type Rows<'a> = &'a mut [MaybeUninit<T>];
    type Row<'a> = &'a mut MaybeUninit<T>;
    type WriteToken = InitializedElement;

    fn sink_dtype(_args: &[DType]) -> VortexResult<DType> {
        Ok(T::element_dtype())
    }

    fn with_capacity(rows: usize, _dtype: &DType) -> VortexResult<Self> {
        Ok(Self {
            values: Vec::with_capacity(rows),
            row_count: rows,
        })
    }

    fn rows(&mut self) -> Self::Rows<'_> {
        &mut self.values.spare_capacity_mut()[..self.row_count]
    }

    fn row_count_matches(rows: &Self::Rows<'_>, row_count: usize) -> bool {
        rows.len() == row_count
    }

    fn initialize_skipped_rows(rows: &mut Self::Rows<'_>) -> bool {
        for row in rows.iter_mut() {
            row.write(T::default());
        }

        true
    }

    fn row<'a>(rows: &'a mut Self::Rows<'_>, index: usize) -> Self::Row<'a> {
        &mut rows[index]
    }

    unsafe fn finish(mut self, _error: DeferredError) -> VortexResult<ArrayRef> {
        // SAFETY: the caller guarantees every slot in `0..row_count` was initialized, and
        // `with_capacity` reserved every slot in that range.
        unsafe { self.values.set_len(self.row_count) };

        Ok(T::build(self.values))
    }
}
