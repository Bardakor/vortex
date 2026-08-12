// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::Constant;
use crate::arrays::Extension;
use crate::arrays::Masked;
use crate::arrays::extension::ExtensionArrayExt;
use crate::arrays::masked::MaskedArraySlotsExt;
use crate::dtype::DType;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::unstable::row::InputElement;

/// One decoded input, collapsed to a single row when it is constant for the batch.
pub struct ArgColumn<T: InputElement>(
    /// The decoded column, classified by whether it stores one value per row.
    ArgColumnKind<T>,
);

enum ArgColumnKind<T: InputElement> {
    PerRow(T::Column),
    Constant(T::Column),
}

impl<T: InputElement> ArgColumn<T> {
    fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        // An empty input has no row 0 to slice, and its row loop runs zero times either way.
        if let Some(constant) = batch_constant(&array)
            && !array.is_empty()
        {
            return Ok(Self(ArgColumnKind::Constant(T::decode(
                constant.slice(0..1)?,
                ctx,
            )?)));
        }

        Ok(Self(ArgColumnKind::PerRow(T::decode(array, ctx)?)))
    }

    fn decode_null_tolerant(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Option<Self>> {
        // Batch execution short-circuits null constants before selecting a strategy, so a
        // constant reaching this path is non-null and can use the ordinary decode.
        if let Some(constant) = batch_constant(&array)
            && !array.is_empty()
        {
            return Ok(Some(Self(ArgColumnKind::Constant(T::decode(
                constant.slice(0..1)?,
                ctx,
            )?))));
        }

        Ok(T::decode_null_tolerant(array, ctx)?
            .map(ArgColumnKind::PerRow)
            .map(Self))
    }

    fn get(&self, index: usize) -> T::Elem<'_> {
        match &self.0 {
            ArgColumnKind::PerRow(column) => T::get(column, index),
            ArgColumnKind::Constant(column) => T::get(column, 0),
        }
    }

    fn per_row_column(&self) -> Option<&T::Column> {
        match &self.0 {
            ArgColumnKind::PerRow(column) => Some(column),
            ArgColumnKind::Constant(_) => None,
        }
    }

    fn addresses_rows(&self, row_count: usize) -> bool {
        // A constant is always read at index zero, so it addresses any batch length.
        match &self.0 {
            ArgColumnKind::PerRow(column) => T::view_len(&T::view(column)) == row_count,
            ArgColumnKind::Constant(_) => true,
        }
    }

    fn constant(&self) -> Option<T::Elem<'_>> {
        match &self.0 {
            ArgColumnKind::PerRow(_) => None,
            ArgColumnKind::Constant(column) => Some(T::get(column, 0)),
        }
    }
}

/// Return the batch-constant array, looking through masked and extension wrappers.
///
/// Batch execution owns mask validity, so a masked constant may expose its constant child here. An
/// extension over constant storage remains wrapped to preserve its extension dtype.
pub fn batch_constant(array: &ArrayRef) -> Option<ArrayRef> {
    if array.is::<Constant>() {
        return Some(array.clone());
    }

    if let Some(masked) = array.as_opt::<Masked>() {
        return Some(masked.child().clone()).filter(|child| child.is::<Constant>());
    }

    array
        .as_opt::<Extension>()
        .is_some_and(|ext| ext.storage_array().is::<Constant>())
        .then(|| array.clone())
}

/// Typed argument tuples for arities zero through twelve.
///
/// This trait is sealed; add a new row representation by implementing [`InputElement`] and placing
/// it in one of the supplied tuples.
pub trait ElementTuple: 'static + private::Sealed {
    /// The decoded column representations.
    type Columns;

    /// Borrowed views of decoded columns when every argument stores one value per row.
    type Views<'a>;

    /// The borrowed row of element values.
    type Elems<'a>;

    /// The batch-constant element values: [`Elems`](Self::Elems) with every argument wrapped in
    /// `Option`.
    ///
    /// `Some` carries the value of a batch-constant argument; `None` marks a per-row argument. A
    /// [`RowVisitor`] passes these values to its prepare closure so constant work can leave the row
    /// loop.
    ///
    /// [`RowVisitor`]: crate::scalar_fn::unstable::row::RowVisitor
    type ConstElems<'a>;

    /// The number of arguments.
    const ARITY: usize;

    /// Whether every argument is [`InputElement::DENSE_SAFE`].
    const DENSE_SAFE: bool;

    /// Whether _any_ argument is [`InputElement::DECODE_FALLIBLE`].
    const DECODE_FALLIBLE: bool;

    /// Validate the input dtypes and exact arity.
    fn validate(dtypes: &[DType]) -> VortexResult<()>;

    /// Decode every input column once for one row-kernel invocation.
    ///
    /// A dense deferred-error retry starts another invocation over filtered valid rows.
    fn decode(args: &dyn ExecutionArgs, ctx: &mut ExecutionCtx) -> VortexResult<Self::Columns>;

    /// Decode every input column once while tolerating null rows.
    ///
    /// Return `Ok(None)` when an argument has no null-tolerant representation. The skip-invalid
    /// strategy calls this once per batch.
    fn decode_null_tolerant(
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Self::Columns>>;

    /// Read the row of elements at `index`. Must be `O(1)`: it is called in the row loop.
    fn get(columns: &Self::Columns, index: usize) -> Self::Elems<'_>;

    /// Borrow every decoded column directly, or `None` when any argument is batch-constant.
    ///
    /// This is selected once outside the hot loop. Keeping `ArgColumn` out of the resulting tuple
    /// gives the optimizer ordinary contiguous column access without a per-row constant check.
    fn per_row_views(columns: &Self::Columns) -> Option<Self::Views<'_>>;

    /// Whether every view contains exactly `row_count` rows.
    fn view_lens_match(views: &Self::Views<'_>, row_count: usize) -> bool;

    /// Whether every per-row argument contains exactly `row_count` rows.
    ///
    /// This provides the same guarantee as [`view_lens_match`](Self::view_lens_match) when
    /// [`per_row_views`](Self::per_row_views) declines a mixed per-row and batch-constant tuple. A
    /// batch constant is exempt because it was collapsed to one row.
    fn decoded_lens_match(columns: &Self::Columns, row_count: usize) -> bool;

    /// Read one row from borrowed views.
    fn get_from_views<'a>(views: &Self::Views<'a>, index: usize) -> Self::Elems<'a>;

    /// Read one row from borrowed views without checking bounds.
    ///
    /// # Safety
    ///
    /// `index` must be in bounds for every column.
    unsafe fn get_from_views_unchecked<'a>(
        views: &Self::Views<'a>,
        index: usize,
    ) -> Self::Elems<'a>;

    /// Read the batch-constant elements out of the decoded columns once for one row-kernel
    /// invocation.
    fn constants(columns: &Self::Columns) -> Self::ConstElems<'_>;
}

impl private::Sealed for () {}

impl ElementTuple for () {
    type Columns = ();
    type Views<'a> = ();
    type Elems<'a> = ();
    type ConstElems<'a> = ();

    const ARITY: usize = 0;
    const DENSE_SAFE: bool = true;
    const DECODE_FALLIBLE: bool = false;

    fn validate(dtypes: &[DType]) -> VortexResult<()> {
        vortex_ensure_eq!(
            dtypes.len(),
            0,
            "expected 0 argument dtypes, got {}",
            dtypes.len(),
        );
        Ok(())
    }

    fn decode(_args: &dyn ExecutionArgs, _ctx: &mut ExecutionCtx) -> VortexResult<Self::Columns> {
        Ok(())
    }

    fn decode_null_tolerant(
        _args: &dyn ExecutionArgs,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Self::Columns>> {
        Ok(Some(()))
    }

    fn get(_columns: &Self::Columns, _index: usize) -> Self::Elems<'_> {}

    fn per_row_views(_columns: &Self::Columns) -> Option<Self::Views<'_>> {
        Some(())
    }

    fn view_lens_match(_views: &Self::Views<'_>, _row_count: usize) -> bool {
        true
    }

    fn decoded_lens_match(_columns: &Self::Columns, _row_count: usize) -> bool {
        true
    }

    fn get_from_views<'a>(_views: &Self::Views<'a>, _index: usize) -> Self::Elems<'a> {}

    unsafe fn get_from_views_unchecked<'a>(
        _views: &Self::Views<'a>,
        _index: usize,
    ) -> Self::Elems<'a> {
    }

    fn constants(_columns: &Self::Columns) -> Self::ConstElems<'_> {}
}

macro_rules! element_tuple {
    ($arity:literal; $($t:ident : $idx:tt),+) => {
        impl<$($t: InputElement),+> private::Sealed for ($($t,)+) {}

        impl<$($t: InputElement),+> ElementTuple for ($($t,)+) {
            type Columns = ($(ArgColumn<$t>,)+);
            type Views<'a> = ($($t::View<'a>,)+);
            type Elems<'a> = ($($t::Elem<'a>,)+);
            type ConstElems<'a> = ($(Option<$t::Elem<'a>>,)+);

            const ARITY: usize = $arity;
            const DENSE_SAFE: bool = $($t::DENSE_SAFE &&)+ true;
            const DECODE_FALLIBLE: bool = $($t::DECODE_FALLIBLE ||)+ false;

            fn validate(dtypes: &[DType]) -> VortexResult<()> {
                vortex_ensure_eq!(
                    dtypes.len(),
                    $arity,
                    "expected {} argument dtypes, got {}",
                    $arity,
                    dtypes.len(),
                );

                $($t::validate(&dtypes[$idx])?;)+
                Ok(())
            }

            fn decode(
                args: &dyn ExecutionArgs,
                ctx: &mut ExecutionCtx,
            ) -> VortexResult<Self::Columns> {
                Ok(($(ArgColumn::<$t>::decode(args.get($idx)?, ctx)?,)+))
            }

            fn decode_null_tolerant(
                args: &dyn ExecutionArgs,
                ctx: &mut ExecutionCtx,
            ) -> VortexResult<Option<Self::Columns>> {
                Ok(Some((
                    $(match ArgColumn::<$t>::decode_null_tolerant(args.get($idx)?, ctx)? {
                        Some(column) => column,
                        None => return Ok(None),
                    },)+
                )))
            }

            fn get(columns: &Self::Columns, index: usize) -> Self::Elems<'_> {
                ($(columns.$idx.get(index),)+)
            }

            fn per_row_views(columns: &Self::Columns) -> Option<Self::Views<'_>> {
                Some(($($t::view(columns.$idx.per_row_column()?),)+))
            }

            fn view_lens_match(
                views: &Self::Views<'_>,
                row_count: usize,
            ) -> bool {
                $($t::view_len(&views.$idx) == row_count &&)+ true
            }

            fn decoded_lens_match(columns: &Self::Columns, row_count: usize) -> bool {
                $(columns.$idx.addresses_rows(row_count) &&)+ true
            }

            fn get_from_views<'a>(
                views: &Self::Views<'a>,
                index: usize,
            ) -> Self::Elems<'a> {
                ($($t::get_from_view(&views.$idx, index),)+)
            }

            unsafe fn get_from_views_unchecked<'a>(
                views: &Self::Views<'a>,
                index: usize,
            ) -> Self::Elems<'a> {
                // SAFETY: forwarded from this method's contract.
                ($(unsafe { $t::get_from_view_unchecked(&views.$idx, index) },)+)
            }

            fn constants(columns: &Self::Columns) -> Self::ConstElems<'_> {
                ($(columns.$idx.constant(),)+)
            }
        }
    };
}

element_tuple!(1; A:0);
element_tuple!(2; A:0, B:1);
element_tuple!(3; A:0, B:1, C:2);
element_tuple!(4; A:0, B:1, C:2, D:3);
element_tuple!(5; A:0, B:1, C:2, D:3, E:4);
element_tuple!(6; A:0, B:1, C:2, D:3, E:4, F:5);
element_tuple!(7; A:0, B:1, C:2, D:3, E:4, F:5, G:6);
element_tuple!(8; A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7);
element_tuple!(9; A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8);
element_tuple!(10; A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9);
element_tuple!(11; A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10);
element_tuple!(12; A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, L:11);

mod private {
    pub trait Sealed {}
}
