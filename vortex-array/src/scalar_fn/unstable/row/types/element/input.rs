// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;

/// An element type that can be read row-wise out of an input column.
///
/// # Safety
///
/// For every view returned by [`view`](Self::view), every index below
/// [`view_len`](Self::view_len) **must** satisfy the safety contract of
/// [`get_from_view_unchecked`](Self::get_from_view_unchecked). Shared execution relies on this
/// proof to perform unchecked reads after one pre-loop length check.
pub unsafe trait InputElement: 'static {
    /// The decoded column representation supporting `O(1)` row access.
    type Column;

    /// The view of a per-row decoded column read by the hot row loop.
    ///
    /// This may borrow a cheaper representation than [`Column`](Self::Column). Primitive elements,
    /// for example, expose a slice so its pointer and length are loop invariants rather than
    /// re-reading a [`Buffer`](vortex_buffer::Buffer) descriptor for every row.
    type View<'a>;

    /// The borrowed element value handed to a row closure.
    type Elem<'a>;

    /// Whether every dense decode and access path tolerates rows that are null in the input.
    ///
    /// Arrays only guarantee payloads for valid rows. This is `false` when a null row's stored
    /// offset or pointer may not address anything, and `true` only when [`decode`](Self::decode),
    /// [`get`](Self::get), [`view`](Self::view), [`view_len`](Self::view_len), and
    /// [`get_from_view`](Self::get_from_view) remain safe and correct for null rows.
    ///
    /// Dense execution requires this of every argument; otherwise the row layer executes only
    /// valid rows.
    ///
    /// Dense execution can pass unspecified values from null rows. The closure must be total over
    /// every stored value: it cannot panic or cause side effects beyond its declared output.
    const DENSE_SAFE: bool = false;

    /// Whether [`decode`](Self::decode) can fail on _legal_ input data.
    ///
    /// This excludes infrastructural failures such as IO or allocation. Set it when legal input may
    /// contain a value that the decoder rejects.
    const DECODE_FALLIBLE: bool = true;

    /// Validate that `dtype` is an acceptable input column dtype for this element type.
    fn validate(dtype: &DType) -> VortexResult<()>;

    /// Decode `array` into its column representation.
    ///
    /// The executor calls this once per row-kernel invocation. A dense deferred-error retry starts
    /// another invocation over filtered valid rows. Hoist dtype checks, downcasts, and other
    /// invocation-invariant work into this method.
    fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column>;

    /// Decode `array` _without_ assuming every row is valid, or `Ok(None)` when this element
    /// cannot for this particular array.
    ///
    /// An element with [`DENSE_SAFE`](Self::DENSE_SAFE) set **should not** override this: its
    /// ordinary decode already tolerates null payloads, so the default is already correct and an
    /// override just restates it. Overriding is for an element that is _not_ dense-safe but can
    /// still write an arbitrary placeholder into null slots; the caller guarantees
    /// [`get`](Self::get) is never called for such a row. The skip-invalid strategy uses this
    /// representation to avoid filtering the input.
    ///
    /// Return `Ok(None)` rather than an error when an array has no null-tolerant decode; the
    /// batch execution falls back to the filter strategy.
    fn decode_null_tolerant(
        array: ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Self::Column>> {
        if Self::DENSE_SAFE {
            Self::decode(array, ctx).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Read the element at `index`, the one function called once per row.
    ///
    /// This must not repeat work that is constant across the batch; do that work in
    /// [`decode`](Self::decode).
    fn get(column: &Self::Column, index: usize) -> Self::Elem<'_>;

    /// Borrow the representation used when this argument varies within the batch.
    ///
    /// Called once before the hot loop. Constants do not use this view because the tuple adapter
    /// keeps their one-row decoded representation separate.
    fn view(column: &Self::Column) -> Self::View<'_>;

    /// Number of rows addressable through a [`View`](Self::View).
    ///
    /// Every index below this length must be valid for
    /// [`get_from_view_unchecked`](Self::get_from_view_unchecked).
    fn view_len(view: &Self::View<'_>) -> usize;

    /// Read one row from a [`View`](Self::View).
    fn get_from_view<'a>(view: &Self::View<'a>, index: usize) -> Self::Elem<'a>
    where
        Self: 'a;

    /// Read one row without checking that `index` is in bounds.
    ///
    /// # Safety
    ///
    /// `index` must be less than [`view_len`](Self::view_len) for `view`.
    unsafe fn get_from_view_unchecked<'a>(view: &Self::View<'a>, index: usize) -> Self::Elem<'a>
    where
        Self: 'a,
    {
        Self::get_from_view(view, index)
    }
}
