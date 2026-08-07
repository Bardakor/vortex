// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The checked arithmetic one row of a primitive column is computed with.
//!
//! Each operator is a type implementing [`CheckedPrimitiveOp`] at every native width, and each
//! width implements [`CheckedArithmetic`] with the value and failure evidence written separately.
//! Keeping them apart is what lets [`row`](super::row) write a value for every row and reduce the
//! evidence without a branch, so the loop vectorizes.

use crate::dtype::NativePType;
use crate::dtype::half::f16;
use crate::scalar_fn::SinkResult;

/// Checked addition, failing on integer overflow.
pub(super) struct CheckedAdd;

/// Checked subtraction, failing on integer overflow.
pub(super) struct CheckedSub;

/// Checked multiplication, failing on integer overflow.
pub(super) struct CheckedMul;

/// Checked division, failing on integer division by zero and on `MIN / -1`.
pub(super) struct CheckedDiv;

/// Evidence that some row failed, in a form that OR-reduces across the batch.
///
/// A plain `bool` is the obvious choice and the right one for most operations. Unsigned
/// multiplication is the exception: deriving a `bool` from the widened product costs a comparison,
/// and LLVM rewrites that comparison plus the product into `llvm.umul.with.overflow`, which has no
/// vector form and scalarizes the whole loop. Carrying the discarded high half instead means the row
/// never compares, so the multiply stays a widening vector multiply and the reduction stays a
/// vector OR. **The width must not exceed the element's**, or the reduction becomes the loop's
/// bottleneck instead of the arithmetic.
pub(super) trait Failure: SinkResult<Accumulated = Self> + Copy + Default {}

impl<T: SinkResult<Accumulated = T> + Copy + Default> Failure for T {}

/// One arithmetic operator at one width, as a value and its failure evidence.
///
/// The pair rather than an `Option<T>` is what a row can write unconditionally: the value is stored
/// whatever the evidence says, and a failing row is either masked away as null or turned into a
/// batch error before anything reads it.
pub(super) trait CheckedPrimitiveOp<T: NativePType>: 'static + Sized {
    /// The error reported for a batch in which some valid row failed.
    const ERROR: &'static str;

    /// How this operation reports a failing row. See [`Failure`].
    type Failure: Failure;

    /// The result of this operation, paired with evidence of whether the row failed.
    fn apply(lhs: T, rhs: T) -> (T, Self::Failure);
}

impl<T: CheckedArithmetic> CheckedPrimitiveOp<T> for CheckedAdd {
    const ERROR: &'static str = "integer overflow in checked add";

    type Failure = bool;

    #[inline(always)]
    fn apply(lhs: T, rhs: T) -> (T, bool) {
        (lhs.add_value(rhs), lhs.add_error(rhs))
    }
}

impl<T: CheckedArithmetic> CheckedPrimitiveOp<T> for CheckedSub {
    const ERROR: &'static str = "integer overflow in checked sub";

    type Failure = bool;

    #[inline(always)]
    fn apply(lhs: T, rhs: T) -> (T, bool) {
        (lhs.sub_value(rhs), lhs.sub_error(rhs))
    }
}

impl<T: CheckedArithmetic> CheckedPrimitiveOp<T> for CheckedMul {
    const ERROR: &'static str = "integer overflow in checked mul";

    type Failure = T::MulFailure;

    #[inline(always)]
    fn apply(lhs: T, rhs: T) -> (T, T::MulFailure) {
        (lhs.mul_value(rhs), lhs.mul_failure(rhs))
    }
}

impl<T: CheckedArithmetic> CheckedPrimitiveOp<T> for CheckedDiv {
    const ERROR: &'static str = "integer division by zero or overflow in checked div";

    type Failure = bool;

    #[inline(always)]
    fn apply(lhs: T, rhs: T) -> (T, bool) {
        let failed = lhs.div_error(rhs);
        let value = if failed {
            T::default()
        } else {
            lhs.div_value(rhs)
        };
        (value, failed)
    }
}

/// The per-width arithmetic behind [`CheckedPrimitiveOp`], with each operation split into the value
/// it produces and whether producing it failed.
///
/// Every `_value` method **must** be total: it is called for rows behind nulls, whose operands are
/// arbitrary, so it may not panic or trap. Integer division is the one that needs care, and
/// [`CheckedDiv`] supplies the default instead of dividing when the divisor is rejected.
pub(super) trait CheckedArithmetic: NativePType {
    /// How multiplication reports a failing row.
    ///
    /// `Self` for the unsigned widths that have a widening multiply, so the row can hand back the
    /// discarded high half rather than comparing. `bool` everywhere else: the narrow signed widths
    /// already vectorize through a two-sided range check, floats never overflow, and the 64-bit
    /// widths use a full-width evidence word.
    type MulFailure: Failure;

    fn add_value(self, rhs: Self) -> Self;
    fn add_error(self, rhs: Self) -> bool;
    fn sub_value(self, rhs: Self) -> Self;
    fn sub_error(self, rhs: Self) -> bool;
    fn mul_value(self, rhs: Self) -> Self;
    fn mul_failure(self, rhs: Self) -> Self::MulFailure;
    fn div_value(self, rhs: Self) -> Self;
    fn div_error(self, rhs: Self) -> bool;
}

macro_rules! impl_checked_unsigned {
    ($ty:ty,widening_mul: $wide:ty) => {
        impl CheckedArithmetic for $ty {
            type MulFailure = $ty;

            #[inline(always)]
            fn add_value(self, rhs: Self) -> Self {
                self.wrapping_add(rhs)
            }

            #[inline(always)]
            fn add_error(self, rhs: Self) -> bool {
                self > <$ty>::MAX - rhs
            }

            #[inline(always)]
            fn sub_value(self, rhs: Self) -> Self {
                self.wrapping_sub(rhs)
            }

            #[inline(always)]
            fn sub_error(self, rhs: Self) -> bool {
                self < rhs
            }

            #[inline(always)]
            fn mul_value(self, rhs: Self) -> Self {
                self.wrapping_mul(rhs)
            }

            /// The bits the narrow product discards, which are non-zero exactly when the multiply
            /// overflowed and cost none of the comparison LLVM folds into `umul.with.overflow`.
            #[inline(always)]
            fn mul_failure(self, rhs: Self) -> $ty {
                (((self as $wide) * (rhs as $wide)) >> <$ty>::BITS) as $ty
            }

            #[inline(always)]
            fn div_value(self, rhs: Self) -> Self {
                self / rhs
            }

            #[inline(always)]
            fn div_error(self, rhs: Self) -> bool {
                rhs == 0
            }
        }
    };
    ($ty:ty,overflowing_mul) => {
        impl CheckedArithmetic for $ty {
            type MulFailure = u64;

            #[inline(always)]
            fn add_value(self, rhs: Self) -> Self {
                self.wrapping_add(rhs)
            }

            #[inline(always)]
            fn add_error(self, rhs: Self) -> bool {
                self > <$ty>::MAX - rhs
            }

            #[inline(always)]
            fn sub_value(self, rhs: Self) -> Self {
                self.wrapping_sub(rhs)
            }

            #[inline(always)]
            fn sub_error(self, rhs: Self) -> bool {
                self < rhs
            }

            #[inline(always)]
            fn mul_value(self, rhs: Self) -> Self {
                self.wrapping_mul(rhs)
            }

            /// As the narrower widths, but through `u128`, because this arm is the 64-bit width and
            /// has no wider native type to widen into.
            #[inline(always)]
            fn mul_failure(self, rhs: Self) -> u64 {
                const { assert!(<$ty>::BITS == 64) };

                (((self as u128) * (rhs as u128)) >> 64) as u64
            }

            #[inline(always)]
            fn div_value(self, rhs: Self) -> Self {
                self / rhs
            }

            #[inline(always)]
            fn div_error(self, rhs: Self) -> bool {
                rhs == 0
            }
        }
    };
}

macro_rules! impl_checked_signed {
    ($ty:ty,widening_mul: $wide:ty) => {
        impl CheckedArithmetic for $ty {
            type MulFailure = bool;

            #[inline(always)]
            fn add_value(self, rhs: Self) -> Self {
                self.wrapping_add(rhs)
            }

            #[inline(always)]
            fn add_error(self, rhs: Self) -> bool {
                let value = self.wrapping_add(rhs);
                ((self ^ value) & (rhs ^ value)) < 0
            }

            #[inline(always)]
            fn sub_value(self, rhs: Self) -> Self {
                self.wrapping_sub(rhs)
            }

            #[inline(always)]
            fn sub_error(self, rhs: Self) -> bool {
                let value = self.wrapping_sub(rhs);
                ((self ^ rhs) & (self ^ value)) < 0
            }

            #[inline(always)]
            fn mul_value(self, rhs: Self) -> Self {
                self.wrapping_mul(rhs)
            }

            /// A plain `bool`, because the two-sided range check is not the shape LLVM folds
            /// into an overflow intrinsic, so these widths vectorize without wider evidence.
            #[inline(always)]
            fn mul_failure(self, rhs: Self) -> bool {
                let product = (self as $wide) * (rhs as $wide);

                product < <$ty>::MIN as $wide || product > <$ty>::MAX as $wide
            }

            #[inline(always)]
            fn div_value(self, rhs: Self) -> Self {
                self / rhs
            }

            #[inline(always)]
            fn div_error(self, rhs: Self) -> bool {
                rhs == 0 || (self == <$ty>::MIN && rhs == -1)
            }
        }
    };
    ($ty:ty,overflowing_mul) => {
        impl CheckedArithmetic for $ty {
            type MulFailure = u64;

            #[inline(always)]
            fn add_value(self, rhs: Self) -> Self {
                self.wrapping_add(rhs)
            }

            #[inline(always)]
            fn add_error(self, rhs: Self) -> bool {
                let value = self.wrapping_add(rhs);
                ((self ^ value) & (rhs ^ value)) < 0
            }

            #[inline(always)]
            fn sub_value(self, rhs: Self) -> Self {
                self.wrapping_sub(rhs)
            }

            #[inline(always)]
            fn sub_error(self, rhs: Self) -> bool {
                let value = self.wrapping_sub(rhs);
                ((self ^ rhs) & (self ^ value)) < 0
            }

            #[inline(always)]
            fn mul_value(self, rhs: Self) -> Self {
                self.wrapping_mul(rhs)
            }

            /// A signed multiply overflows exactly when the discarded half of the true product
            /// differs from the sign extension of the half that was kept, so the two XOR to zero
            /// on the products that fit.
            #[inline(always)]
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the truncated half is the result, and the discarded half is the evidence"
            )]
            fn mul_failure(self, rhs: Self) -> u64 {
                const { assert!(<$ty>::BITS == 64) };

                let wide = (self as i128) * (rhs as i128);
                let kept = wide as i64;
                let discarded = (wide >> 64) as i64;

                (discarded ^ (kept >> 63)) as u64
            }

            #[inline(always)]
            fn div_value(self, rhs: Self) -> Self {
                self / rhs
            }

            #[inline(always)]
            fn div_error(self, rhs: Self) -> bool {
                rhs == 0 || (self == <$ty>::MIN && rhs == -1)
            }
        }
    };
}

macro_rules! impl_checked_float {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl CheckedArithmetic for $ty {
                type MulFailure = bool;

                #[inline(always)]
                fn add_value(self, rhs: Self) -> Self {
                    self + rhs
                }

                #[inline(always)]
                fn add_error(self, _rhs: Self) -> bool {
                    false
                }

                #[inline(always)]
                fn sub_value(self, rhs: Self) -> Self {
                    self - rhs
                }

                #[inline(always)]
                fn sub_error(self, _rhs: Self) -> bool {
                    false
                }

                #[inline(always)]
                fn mul_value(self, rhs: Self) -> Self {
                    self * rhs
                }

                #[inline(always)]
                fn mul_failure(self, _rhs: Self) -> bool {
                    false
                }

                #[inline(always)]
                fn div_value(self, rhs: Self) -> Self {
                    self / rhs
                }

                #[inline(always)]
                fn div_error(self, _rhs: Self) -> bool {
                    false
                }
            }
        )+
    };
}

impl_checked_unsigned!(u8, widening_mul: u16);
impl_checked_unsigned!(u16, widening_mul: u32);
impl_checked_unsigned!(u32, widening_mul: u64);
impl_checked_unsigned!(u64, overflowing_mul);
impl_checked_signed!(i8, widening_mul: i16);
impl_checked_signed!(i16, widening_mul: i32);
impl_checked_signed!(i32, widening_mul: i64);
impl_checked_signed!(i64, overflowing_mul);
impl_checked_float!(f16, f32, f64);

#[cfg(test)]
mod tests {
    use super::CheckedArithmetic;
    use crate::scalar_fn::SinkResult;

    /// Values whose pairwise products are worth probing: the saturating boundaries, the sign-change
    /// pivots, and a spread of magnitudes that straddles the 64-bit split.
    const PROBES: &[i64] = &[
        0,
        1,
        -1,
        2,
        -2,
        3,
        i64::MIN,
        i64::MIN + 1,
        i64::MAX,
        i64::MAX - 1,
        1 << 31,
        1 << 32,
        1 << 62,
        -(1 << 62),
        0x7FFF_FFFF,
        -0x8000_0000,
    ];

    /// Every `mul_failure` implementation is either a bit trick or a two-sided range check, so
    /// hold each against `checked_mul`, whose `None` is the definition of overflow.
    #[track_caller]
    fn assert_agrees_with_checked_mul<T: CheckedArithmetic>(lhs: T, rhs: T, reference: Option<T>) {
        let failed = <T::MulFailure as SinkResult>::occurred(lhs.mul_failure(rhs));

        assert_eq!(failed, reference.is_none(), "{lhs:?} * {rhs:?}");
    }

    #[test]
    fn mul_failure_agrees_with_checked_mul_at_64_bits() {
        for &lhs in PROBES {
            for &rhs in PROBES {
                assert_agrees_with_checked_mul(lhs, rhs, lhs.checked_mul(rhs));

                let (lhs, rhs) = (lhs as u64, rhs as u64);
                assert_agrees_with_checked_mul(lhs, rhs, lhs.checked_mul(rhs));
            }
        }
    }

    /// The 8-bit widths are cheap enough to check exhaustively, pinning the unsigned shift and the
    /// signed range check against every product that exists.
    #[test]
    fn mul_failure_agrees_with_checked_mul_exhaustively_at_8_bits() {
        for lhs in u8::MIN..=u8::MAX {
            for rhs in u8::MIN..=u8::MAX {
                assert_agrees_with_checked_mul(lhs, rhs, lhs.checked_mul(rhs));
            }
        }

        for lhs in i8::MIN..=i8::MAX {
            for rhs in i8::MIN..=i8::MAX {
                assert_agrees_with_checked_mul(lhs, rhs, lhs.checked_mul(rhs));
            }
        }
    }
}
