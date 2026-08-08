// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use num_traits::CheckedAdd;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use super::checked_add_i64;
use super::checked_add_u64;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::GroupedState;
use crate::arrays::DecimalArray;
use crate::arrays::PrimitiveArray;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::DecimalType;
use crate::dtype::NativeDecimalType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::match_each_decimal_value_type;
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;
use crate::validity::Validity;

pub(super) enum SumGroupedValues {
    Unsigned(Vec<u64>),
    Signed(Vec<i64>),
    Float(Vec<f64>),
    Decimal8(Vec<i8>),
    Decimal16(Vec<i16>),
    Decimal32(Vec<i32>),
    Decimal64(Vec<i64>),
    Decimal128(Vec<i128>),
    Decimal256(Vec<crate::dtype::i256>),
}

impl SumGroupedValues {
    fn len(&self) -> usize {
        match self {
            Self::Unsigned(values) => values.len(),
            Self::Signed(values) => values.len(),
            Self::Float(values) => values.len(),
            Self::Decimal8(values) => values.len(),
            Self::Decimal16(values) => values.len(),
            Self::Decimal32(values) => values.len(),
            Self::Decimal64(values) => values.len(),
            Self::Decimal128(values) => values.len(),
            Self::Decimal256(values) => values.len(),
        }
    }

    fn resize(&mut self, len: usize) {
        match self {
            Self::Unsigned(values) => values.resize(len, 0),
            Self::Signed(values) => values.resize(len, 0),
            Self::Float(values) => values.resize(len, 0.0),
            Self::Decimal8(values) => values.resize(len, 0),
            Self::Decimal16(values) => values.resize(len, 0),
            Self::Decimal32(values) => values.resize(len, 0),
            Self::Decimal64(values) => values.resize(len, 0),
            Self::Decimal128(values) => values.resize(len, 0),
            Self::Decimal256(values) => values.resize(len, crate::dtype::i256::ZERO),
        }
    }
}

pub(crate) struct SumGroupedState {
    values: SumGroupedValues,
    overflowed: Vec<u8>,
    partial_dtype: DType,
}

impl SumGroupedState {
    pub(crate) fn try_new(partial_dtype: DType) -> VortexResult<Self> {
        let values = match &partial_dtype {
            DType::Primitive(PType::U64, _) => SumGroupedValues::Unsigned(Vec::new()),
            DType::Primitive(PType::I64, _) => SumGroupedValues::Signed(Vec::new()),
            DType::Primitive(PType::F64, _) => SumGroupedValues::Float(Vec::new()),
            DType::Decimal(dtype, _) => match DecimalType::smallest_decimal_value_type(dtype) {
                DecimalType::I8 => SumGroupedValues::Decimal8(Vec::new()),
                DecimalType::I16 => SumGroupedValues::Decimal16(Vec::new()),
                DecimalType::I32 => SumGroupedValues::Decimal32(Vec::new()),
                DecimalType::I64 => SumGroupedValues::Decimal64(Vec::new()),
                DecimalType::I128 => SumGroupedValues::Decimal128(Vec::new()),
                DecimalType::I256 => SumGroupedValues::Decimal256(Vec::new()),
            },
            dtype => vortex_bail!("Unsupported grouped sum partial dtype: {dtype}"),
        };
        Ok(Self {
            values,
            overflowed: Vec::new(),
            partial_dtype,
        })
    }

    pub(super) fn parts_mut(&mut self) -> (&mut SumGroupedValues, &mut [u8]) {
        (&mut self.values, &mut self.overflowed)
    }

    pub(super) fn decimal_dtype(&self) -> Option<DecimalDType> {
        self.partial_dtype.as_decimal_opt().copied()
    }
}

impl GroupedState for SumGroupedState {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    fn ensure_groups(&mut self, num_groups: usize) -> VortexResult<()> {
        let len = num_groups.max(self.len());
        self.values.resize(len);
        self.overflowed.resize(len, 0);
        Ok(())
    }

    fn is_saturated(&self, group_id: usize) -> bool {
        if self.overflowed[group_id] != 0 {
            return true;
        }
        matches!(&self.values, SumGroupedValues::Float(values) if values[group_id].is_nan())
    }

    fn combine_scalar(&mut self, group_id: usize, partial: Scalar) -> VortexResult<()> {
        if partial.is_null() {
            self.overflowed[group_id] = 1;
            return Ok(());
        }
        if self.overflowed[group_id] != 0 {
            return Ok(());
        }

        let decimal_dtype = self.decimal_dtype();
        let (values, overflowed) = self.parts_mut();
        match values {
            SumGroupedValues::Unsigned(values) => {
                let value = partial
                    .as_primitive()
                    .typed_value::<u64>()
                    .vortex_expect("checked non-null");
                overflowed[group_id] = u8::from(checked_add_u64(&mut values[group_id], value));
            }
            SumGroupedValues::Signed(values) => {
                let value = partial
                    .as_primitive()
                    .typed_value::<i64>()
                    .vortex_expect("checked non-null");
                overflowed[group_id] = u8::from(checked_add_i64(&mut values[group_id], value));
            }
            SumGroupedValues::Float(values) => {
                values[group_id] += partial
                    .as_primitive()
                    .typed_value::<f64>()
                    .vortex_expect("checked non-null");
            }
            SumGroupedValues::Decimal8(values) => combine_decimal_scalar(
                values,
                overflowed,
                group_id,
                &partial,
                decimal_dtype.vortex_expect("decimal state dtype"),
            ),
            SumGroupedValues::Decimal16(values) => combine_decimal_scalar(
                values,
                overflowed,
                group_id,
                &partial,
                decimal_dtype.vortex_expect("decimal state dtype"),
            ),
            SumGroupedValues::Decimal32(values) => combine_decimal_scalar(
                values,
                overflowed,
                group_id,
                &partial,
                decimal_dtype.vortex_expect("decimal state dtype"),
            ),
            SumGroupedValues::Decimal64(values) => combine_decimal_scalar(
                values,
                overflowed,
                group_id,
                &partial,
                decimal_dtype.vortex_expect("decimal state dtype"),
            ),
            SumGroupedValues::Decimal128(values) => combine_decimal_scalar(
                values,
                overflowed,
                group_id,
                &partial,
                decimal_dtype.vortex_expect("decimal state dtype"),
            ),
            SumGroupedValues::Decimal256(values) => combine_decimal_scalar(
                values,
                overflowed,
                group_id,
                &partial,
                decimal_dtype.vortex_expect("decimal state dtype"),
            ),
        }
        Ok(())
    }

    fn partial_scalar(&self, group_id: usize) -> VortexResult<Scalar> {
        if self
            .overflowed
            .get(group_id)
            .is_some_and(|&overflowed| overflowed != 0)
        {
            return Ok(Scalar::null(self.partial_dtype.clone()));
        }

        Ok(match &self.values {
            SumGroupedValues::Unsigned(values) => Scalar::primitive(
                values.get(group_id).copied().unwrap_or(0),
                Nullability::Nullable,
            ),
            SumGroupedValues::Signed(values) => Scalar::primitive(
                values.get(group_id).copied().unwrap_or(0),
                Nullability::Nullable,
            ),
            SumGroupedValues::Float(values) => Scalar::primitive(
                values.get(group_id).copied().unwrap_or(0.0),
                Nullability::Nullable,
            ),
            SumGroupedValues::Decimal8(values) => decimal_scalar(
                values.get(group_id).copied().unwrap_or(0),
                self.decimal_dtype().vortex_expect("decimal state dtype"),
            ),
            SumGroupedValues::Decimal16(values) => decimal_scalar(
                values.get(group_id).copied().unwrap_or(0),
                self.decimal_dtype().vortex_expect("decimal state dtype"),
            ),
            SumGroupedValues::Decimal32(values) => decimal_scalar(
                values.get(group_id).copied().unwrap_or(0),
                self.decimal_dtype().vortex_expect("decimal state dtype"),
            ),
            SumGroupedValues::Decimal64(values) => decimal_scalar(
                values.get(group_id).copied().unwrap_or(0),
                self.decimal_dtype().vortex_expect("decimal state dtype"),
            ),
            SumGroupedValues::Decimal128(values) => decimal_scalar(
                values.get(group_id).copied().unwrap_or(0),
                self.decimal_dtype().vortex_expect("decimal state dtype"),
            ),
            SumGroupedValues::Decimal256(values) => decimal_scalar(
                values
                    .get(group_id)
                    .copied()
                    .unwrap_or(crate::dtype::i256::ZERO),
                self.decimal_dtype().vortex_expect("decimal state dtype"),
            ),
        })
    }

    fn accumulate_partials(
        &mut self,
        partials: &ArrayRef,
        group_ids: &[u32],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let validity = partials.validity()?.execute_mask(partials.len(), ctx)?;
        let decimal_dtype = self.decimal_dtype();
        let (values, overflowed) = self.parts_mut();
        match values {
            SumGroupedValues::Unsigned(values) => {
                let partials = partials.clone().execute::<PrimitiveArray>(ctx)?;
                accumulate_primitive_partials(
                    values,
                    overflowed,
                    partials.as_slice::<u64>(),
                    group_ids,
                    &validity,
                    checked_add_u64,
                );
            }
            SumGroupedValues::Signed(values) => {
                let partials = partials.clone().execute::<PrimitiveArray>(ctx)?;
                accumulate_primitive_partials(
                    values,
                    overflowed,
                    partials.as_slice::<i64>(),
                    group_ids,
                    &validity,
                    checked_add_i64,
                );
            }
            SumGroupedValues::Float(values) => {
                let partials = partials.clone().execute::<PrimitiveArray>(ctx)?;
                accumulate_float_partials(
                    values,
                    overflowed,
                    partials.as_slice::<f64>(),
                    group_ids,
                    &validity,
                );
            }
            SumGroupedValues::Decimal8(values) => accumulate_decimal_partials(
                values,
                overflowed,
                &partials.clone().execute::<DecimalArray>(ctx)?,
                group_ids,
                &validity,
                decimal_dtype.vortex_expect("decimal state dtype"),
            ),
            SumGroupedValues::Decimal16(values) => accumulate_decimal_partials(
                values,
                overflowed,
                &partials.clone().execute::<DecimalArray>(ctx)?,
                group_ids,
                &validity,
                decimal_dtype.vortex_expect("decimal state dtype"),
            ),
            SumGroupedValues::Decimal32(values) => accumulate_decimal_partials(
                values,
                overflowed,
                &partials.clone().execute::<DecimalArray>(ctx)?,
                group_ids,
                &validity,
                decimal_dtype.vortex_expect("decimal state dtype"),
            ),
            SumGroupedValues::Decimal64(values) => accumulate_decimal_partials(
                values,
                overflowed,
                &partials.clone().execute::<DecimalArray>(ctx)?,
                group_ids,
                &validity,
                decimal_dtype.vortex_expect("decimal state dtype"),
            ),
            SumGroupedValues::Decimal128(values) => accumulate_decimal_partials(
                values,
                overflowed,
                &partials.clone().execute::<DecimalArray>(ctx)?,
                group_ids,
                &validity,
                decimal_dtype.vortex_expect("decimal state dtype"),
            ),
            SumGroupedValues::Decimal256(values) => accumulate_decimal_partials(
                values,
                overflowed,
                &partials.clone().execute::<DecimalArray>(ctx)?,
                group_ids,
                &validity,
                decimal_dtype.vortex_expect("decimal state dtype"),
            ),
        }
        Ok(())
    }

    fn flush_partials(&mut self, num_groups: usize) -> VortexResult<ArrayRef> {
        vortex_ensure!(
            num_groups >= self.len(),
            "Cannot flush {} groups after accumulating {} groups",
            num_groups,
            self.len()
        );
        self.ensure_groups(num_groups)?;
        let validity = validity_from_overflow(&self.overflowed);
        self.overflowed.clear();
        let decimal_dtype = self.decimal_dtype();
        Ok(match &mut self.values {
            SumGroupedValues::Unsigned(values) => {
                PrimitiveArray::new(Buffer::from(std::mem::take(values)), validity).into_array()
            }
            SumGroupedValues::Signed(values) => {
                PrimitiveArray::new(Buffer::from(std::mem::take(values)), validity).into_array()
            }
            SumGroupedValues::Float(values) => {
                PrimitiveArray::new(Buffer::from(std::mem::take(values)), validity).into_array()
            }
            SumGroupedValues::Decimal8(values) => decimal_array(
                std::mem::take(values),
                decimal_dtype.vortex_expect("decimal state dtype"),
                validity,
            ),
            SumGroupedValues::Decimal16(values) => decimal_array(
                std::mem::take(values),
                decimal_dtype.vortex_expect("decimal state dtype"),
                validity,
            ),
            SumGroupedValues::Decimal32(values) => decimal_array(
                std::mem::take(values),
                decimal_dtype.vortex_expect("decimal state dtype"),
                validity,
            ),
            SumGroupedValues::Decimal64(values) => decimal_array(
                std::mem::take(values),
                decimal_dtype.vortex_expect("decimal state dtype"),
                validity,
            ),
            SumGroupedValues::Decimal128(values) => decimal_array(
                std::mem::take(values),
                decimal_dtype.vortex_expect("decimal state dtype"),
                validity,
            ),
            SumGroupedValues::Decimal256(values) => decimal_array(
                std::mem::take(values),
                decimal_dtype.vortex_expect("decimal state dtype"),
                validity,
            ),
        })
    }
}

fn for_each_partial_row(validity: &Mask, len: usize, mut f: impl FnMut(usize, bool)) {
    match validity.indices() {
        AllOr::All => (0..len).for_each(|idx| f(idx, true)),
        AllOr::None => (0..len).for_each(|idx| f(idx, false)),
        AllOr::Some(valid_indices) => {
            let mut valid_indices = valid_indices.iter().copied().peekable();
            for idx in 0..len {
                let valid = valid_indices.next_if_eq(&idx).is_some();
                f(idx, valid);
            }
        }
    }
}

fn accumulate_primitive_partials<T: Copy>(
    values: &mut [T],
    overflowed: &mut [u8],
    partials: &[T],
    group_ids: &[u32],
    validity: &Mask,
    checked_add: fn(&mut T, T) -> bool,
) {
    for_each_partial_row(validity, partials.len(), |idx, valid| {
        let group = group_ids[idx] as usize;
        if !valid || checked_add(&mut values[group], partials[idx]) {
            overflowed[group] = 1;
        }
    });
}

fn accumulate_float_partials(
    values: &mut [f64],
    overflowed: &mut [u8],
    partials: &[f64],
    group_ids: &[u32],
    validity: &Mask,
) {
    for_each_partial_row(validity, partials.len(), |idx, valid| {
        let group = group_ids[idx] as usize;
        if !valid {
            overflowed[group] = 1;
        } else {
            values[group] += partials[idx];
        }
    });
}

fn accumulate_decimal_partials<I>(
    values: &mut [I],
    overflowed: &mut [u8],
    partials: &DecimalArray,
    group_ids: &[u32],
    validity: &Mask,
    dtype: DecimalDType,
) where
    I: NativeDecimalType + CheckedAdd,
{
    match_each_decimal_value_type!(partials.values_type(), |T| {
        accumulate_decimal_partial_values(
            values,
            overflowed,
            &partials.buffer::<T>(),
            group_ids,
            validity,
            dtype,
        );
    });
}

fn accumulate_decimal_partial_values<T, I>(
    values: &mut [I],
    overflowed: &mut [u8],
    partials: &[T],
    group_ids: &[u32],
    validity: &Mask,
    dtype: DecimalDType,
) where
    T: NativeDecimalType,
    I: NativeDecimalType + CheckedAdd,
{
    for_each_partial_row(validity, partials.len(), |idx, valid| {
        let group = group_ids[idx] as usize;
        if !valid {
            overflowed[group] = 1;
        } else {
            let Some(value) = <I as crate::dtype::BigCast>::from(partials[idx]) else {
                overflowed[group] = 1;
                return;
            };
            add_decimal(values, overflowed, group, value, dtype);
        }
    });
}

fn combine_decimal_scalar<T>(
    values: &mut [T],
    overflowed: &mut [u8],
    group_id: usize,
    partial: &Scalar,
    dtype: DecimalDType,
) where
    T: NativeDecimalType + CheckedAdd,
{
    let value = partial
        .as_decimal()
        .decimal_value()
        .vortex_expect("checked non-null")
        .cast::<T>()
        .vortex_expect("decimal partial must use grouped state width");
    add_decimal(values, overflowed, group_id, value, dtype);
}

pub(super) fn add_decimal<T>(
    values: &mut [T],
    overflowed: &mut [u8],
    group_id: usize,
    value: T,
    dtype: DecimalDType,
) where
    T: NativeDecimalType + CheckedAdd,
{
    if overflowed[group_id] != 0 {
        return;
    }
    let Some(result) = values[group_id].checked_add(&value) else {
        overflowed[group_id] = 1;
        return;
    };
    let precision = usize::from(dtype.precision());
    if T::MIN_BY_PRECISION[precision] <= result && result <= T::MAX_BY_PRECISION[precision] {
        values[group_id] = result;
    } else {
        overflowed[group_id] = 1;
    }
}

fn validity_from_overflow(overflowed: &[u8]) -> Validity {
    if overflowed.iter().all(|&overflowed| overflowed == 0) {
        Validity::AllValid
    } else {
        Validity::from_iter(overflowed.iter().map(|&overflowed| overflowed == 0))
    }
}

fn decimal_scalar<T: NativeDecimalType>(value: T, dtype: DecimalDType) -> Scalar
where
    DecimalValue: From<T>,
{
    Scalar::decimal(DecimalValue::from(value), dtype, Nullability::Nullable)
}

fn decimal_array<T: NativeDecimalType>(
    values: Vec<T>,
    dtype: DecimalDType,
    validity: Validity,
) -> ArrayRef {
    DecimalArray::new(Buffer::from(values), dtype, validity).into_array()
}
