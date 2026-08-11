// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Read-only lane source — the [`IndexedSource`] trait and the [`LaneZip`] adapter.

/// A length-known source supporting unchecked indexed reads.
///
/// Implemented for `&[T]` (with `T: Copy`) and for [`LaneZip`] over two `IndexedSource`s.
/// The kernels in this crate require this trait instead of `Iterator` so that lane
/// reads carry no inter-iteration data dependency — the autovectorizer treats each
/// lane independently.
pub trait IndexedSource {
    /// The per-lane item type passed through the kernel by value.
    type Item;
    /// Logical lane count.
    fn len(&self) -> usize;
    /// Returns true when there are no lanes.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Read the lane at `i` without bounds checking.
    ///
    /// # Safety
    ///
    /// `i` must be strictly less than `self.len()`.
    unsafe fn get_unchecked(&self, i: usize) -> Self::Item;
}

impl<T: Copy> IndexedSource for &[T] {
    type Item = T;
    #[inline]
    fn len(&self) -> usize {
        <[T]>::len(self)
    }
    #[inline]
    unsafe fn get_unchecked(&self, i: usize) -> T {
        // SAFETY: caller guarantees i < self.len().
        unsafe { *<[T]>::get_unchecked(self, i) }
    }
}

impl<T: Copy> IndexedSource for &mut [T] {
    type Item = T;
    #[inline]
    fn len(&self) -> usize {
        <[T]>::len(self)
    }
    #[inline]
    unsafe fn get_unchecked(&self, i: usize) -> T {
        // SAFETY: caller guarantees i < self.len().
        unsafe { *<[T]>::get_unchecked(self, i) }
    }
}

/// Pair of two [`IndexedSource`]s of equal length. Yields `(A::Item, B::Item)` per lane.
///
/// Use this to drive a binary kernel from two columns. [`LaneZip::new`] enforces length equality
/// at construction, and [`IndexedSource::len`] checks it for callers of the public tuple
/// constructor.
#[derive(Clone, Copy)]
pub struct LaneZip<A, B>(pub A, pub B);

impl<A: IndexedSource, B: IndexedSource> LaneZip<A, B> {
    /// Build a `LaneZip` from two equal-length sources.
    ///
    /// # Panics
    ///
    /// Panics if the two operands have different lengths.
    pub fn new(a: A, b: B) -> Self {
        assert_eq!(
            a.len(),
            b.len(),
            "LaneZip operands must have the same length"
        );
        Self(a, b)
    }
}

impl<A: IndexedSource, B: IndexedSource> IndexedSource for LaneZip<A, B> {
    type Item = (A::Item, B::Item);
    #[inline]
    fn len(&self) -> usize {
        assert_eq!(
            self.0.len(),
            self.1.len(),
            "LaneZip operands must have the same length"
        );
        self.0.len()
    }
    #[inline]
    unsafe fn get_unchecked(&self, i: usize) -> (A::Item, B::Item) {
        // SAFETY: caller guarantees i < self.len(), which also verifies matching lengths.
        unsafe { (self.0.get_unchecked(i), self.1.get_unchecked(i)) }
    }
}

#[cfg(test)]
mod tests {
    use super::IndexedSource;
    use super::LaneZip;

    #[test]
    #[should_panic(expected = "LaneZip operands must have the same length")]
    fn direct_construction_checks_lengths() {
        LaneZip(&[1_u8][..], &[2_u8, 3][..]).len();
    }
}
