// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Output builders for row kernels that cannot return independent owned values.
//!
//! [`OutputSink`] allocates batch-wide state and lends one row handle to each callback.
//! [`UninitElementSink`] is the fixed-width implementation used when avoiding output
//! initialization matters.

use std::mem::MaybeUninit;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::dtype::DType;
use crate::scalar_fn::unstable::row::OutputElement;

/// A column allocated once per batch that a row closure writes into, one row at a time.
///
/// A sink may use function options and input dtypes to build a runtime-shaped output or own shared
/// batch state. The executor passes each row slot into an [`Fn`] closure.
///
/// Rows arrive in increasing index order. Ordinary execution visits `0..row_count` exactly once;
/// skip-invalid execution can omit invalid rows when [`skipped_rows_initializer`] returns an
/// initializer.
///
/// # Errors
///
/// Lifecycle methods report only incidental failures such as allocation. A semantic error that
/// depends on input values **must** come from the row callback through a fallible [`SinkResult`], or
/// [`RowFn::FALLIBLE`] cannot protect optimizations such as dictionary push-down.
///
/// # Safety
///
/// An implementation must uphold all of these requirements:
///
/// - When [`row_count_matches`] returns `true`, every index in `0..row_count` **must** identify one
///   distinct row owned by this sink.
/// - A row must either be initialized before the callback or require a
///   [`WriteToken`] that safe code cannot produce without initializing that exact row. Evidence for
///   an uninitialized row **must not** be safely forgeable, reusable, or substitutable.
/// - An initializer returned by [`skipped_rows_initializer`] **must** initialize every row.
/// - `Self` and every borrowed [`Rows`] view **must** remain safe to drop if decoding,
///   preparation, skipped-row initialization, or a row callback returns an error or unwinds. The
///   executor can abandon a sink after any prefix of rows.
/// - [`finish`] **must** be sound once every visited callback returned its required token and the
///   skipped-row initializer, when present, ran successfully.
///
/// The executor relies on these guarantees when it calls `finish`.
///
/// [`Rows`]: Self::Rows
/// [`WriteToken`]: Self::WriteToken
/// [`finish`]: Self::finish
/// [`row_count_matches`]: Self::row_count_matches
/// [`RowFn::FALLIBLE`]: crate::scalar_fn::unstable::row::RowFn::FALLIBLE
/// [`SinkResult`]: crate::scalar_fn::unstable::row::SinkResult
/// [`skipped_rows_initializer`]: Self::skipped_rows_initializer
pub unsafe trait OutputSink<Options>: 'static + Sized {
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

    /// The operation that initializes every output position before skip-invalid execution.
    ///
    /// `Some` enables skip-invalid execution. The initializer **must** make every row safe to
    /// finish; callbacks overwrite valid rows and batch execution masks skipped rows.
    ///
    /// `None` makes the executor fall back to filtering the inputs.
    fn skipped_rows_initializer() -> Option<for<'a> fn(&mut Self::Rows<'a>)> {
        None
    }

    /// The dtype of the column this sink builds, given the function options and input dtypes.
    ///
    /// **Must** be non-nullable: batch execution derives nullability from the inputs, widens the
    /// result, and masks the null rows.
    fn sink_dtype(options: &Options, args: &[DType]) -> VortexResult<DType>;

    /// Allocate a sink for `rows` rows of `dtype`, which is this sink's own
    /// [`sink_dtype`](Self::sink_dtype). Called once per batch.
    fn with_capacity(rows: usize, dtype: &DType) -> VortexResult<Self>;

    /// Borrow all output rows for the hot loop.
    fn rows(&mut self) -> Self::Rows<'_>;

    /// Whether every index in `0..row_count` is addressable through
    /// [`row_unchecked`](Self::row_unchecked).
    ///
    /// Called once before the hot loop. Besides validating the sink contract, this gives the
    /// optimizer the output bounds it needs to remove the bounds check hidden in each row accessor.
    fn row_count_matches(rows: &Self::Rows<'_>, row_count: usize) -> bool;

    /// Hand out the place to write row `index`. Must be `O(1)`: it is called in the row loop.
    ///
    /// # Safety
    ///
    /// [`row_count_matches`](Self::row_count_matches) must have returned `true` for `rows` and the
    /// same `row_count`, and `index` must be less than that `row_count`.
    unsafe fn row_unchecked<'a>(rows: &'a mut Self::Rows<'_>, index: usize) -> Self::Row<'a>;

    /// Finish into the built column, whose dtype **must** be this sink's
    /// [`sink_dtype`](Self::sink_dtype). Called once per batch.
    ///
    /// # Safety
    ///
    /// The executor must have completed every row callback successfully, and each callback must
    /// have returned this sink's [`WriteToken`](Self::WriteToken). When skipped rows are allowed,
    /// the initializer returned by
    /// [`skipped_rows_initializer`](Self::skipped_rows_initializer) must have run before traversal.
    unsafe fn finish(self) -> VortexResult<ArrayRef>;
}

/// Proof that one uninitialized element row was initialized.
///
/// The private field prevents safe construction without calling [`write`](Self::write):
///
/// ```compile_fail,E0423
/// use vortex_array::scalar_fn::unstable::row::InitializedElement;
///
/// let _evidence = InitializedElement(());
/// ```
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
    /// `row` must be the [`UninitElementSink`] row supplied to the current callback. The caller
    /// must return the token from that callback. Using another row or returning the token from
    /// another callback can cause undefined behavior.
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
/// Skip-invalid execution initializes placeholders before omitting rows. Errors and unwinds are
/// safe because `values` keeps length zero until `finish`; `T: Copy` means initialized
/// spare-capacity elements require no destruction.
pub struct UninitElementSink<T> {
    /// Spare storage written in increasing row order.
    values: Vec<T>,

    /// The number of slots exposed to the row loop and initialized before finishing.
    row_count: usize,
}

// SAFETY: the row slice covers exactly the reserved spare-capacity range, so each accepted index
// names one distinct slot. `InitializedElement` cannot be constructed by safe code; its unsafe
// constructor writes the supplied slot and requires the caller to return that exact evidence. The
// skipped-row initializer writes `T::default()` into every slot before masked traversal.
unsafe impl<T: OutputElement + Copy + Default, Options> OutputSink<Options>
    for UninitElementSink<T>
{
    type Rows<'a> = &'a mut [MaybeUninit<T>];
    type Row<'a> = &'a mut MaybeUninit<T>;
    type WriteToken = InitializedElement;

    fn skipped_rows_initializer() -> Option<for<'a> fn(&mut Self::Rows<'a>)> {
        Some(|rows| {
            for row in rows.iter_mut() {
                row.write(T::default());
            }
        })
    }

    fn sink_dtype(_options: &Options, _args: &[DType]) -> VortexResult<DType> {
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

    unsafe fn row_unchecked<'a>(rows: &'a mut Self::Rows<'_>, index: usize) -> Self::Row<'a> {
        // SAFETY: required by this method's contract.
        unsafe { rows.get_unchecked_mut(index) }
    }

    unsafe fn finish(mut self) -> VortexResult<ArrayRef> {
        // SAFETY: the caller guarantees every slot in `0..row_count` was initialized, and
        // `with_capacity` reserved every slot in that range.
        unsafe { self.values.set_len(self.row_count) };

        Ok(T::build(self.values))
    }
}
