// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use num_traits::CheckedAdd;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;

use super::IS_EMPTY_FIELD;
use super::IS_OVERFLOW_FIELD;
use super::SUM_FIELD;
use super::checked_add_i64;
use super::checked_add_u64;
use super::decode_sum_partial_scalar;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::GroupedState;
use crate::arrays::BoolArray;
use crate::arrays::DecimalArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::struct_::StructArrayExt;
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
    empty: Vec<u8>,
    partial_dtype: DType,
    return_dtype: DType,
}

impl SumGroupedState {
    pub(crate) fn try_new(partial_dtype: DType, return_dtype: DType) -> VortexResult<Self> {
        let values = match &return_dtype {
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
            dtype => vortex_bail!("Unsupported grouped sum return dtype: {dtype}"),
        };
        Ok(Self {
            values,
            overflowed: Vec::new(),
            empty: Vec::new(),
            partial_dtype,
            return_dtype,
        })
    }

    pub(super) fn parts_mut(&mut self) -> (&mut SumGroupedValues, &mut [u8], &mut [u8]) {
        (&mut self.values, &mut self.overflowed, &mut self.empty)
    }

    pub(super) fn decimal_dtype(&self) -> Option<DecimalDType> {
        self.return_dtype.as_decimal_opt().copied()
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
        self.empty.resize(len, 1);
        Ok(())
    }

    fn is_saturated(&self, group_id: usize) -> bool {
        if self.overflowed[group_id] != 0 {
            return true;
        }
        matches!(&self.values, SumGroupedValues::Float(values) if values[group_id].is_nan())
    }

    fn combine_scalar(&mut self, group_id: usize, partial: Scalar) -> VortexResult<()> {
        let (partial, partial_overflowed, partial_empty) = decode_sum_partial_scalar(partial)?;
        if partial_empty {
            return Ok(());
        }
        self.empty[group_id] = 0;
        if partial_overflowed {
            self.overflowed[group_id] = 1;
            return Ok(());
        }
        if self.overflowed[group_id] != 0 {
            return Ok(());
        }

        let decimal_dtype = self.decimal_dtype();
        let (values, overflowed, _) = self.parts_mut();
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
        let sum = match &self.values {
            SumGroupedValues::Unsigned(values) => Scalar::primitive(
                values.get(group_id).copied().unwrap_or(0),
                Nullability::NonNullable,
            ),
            SumGroupedValues::Signed(values) => Scalar::primitive(
                values.get(group_id).copied().unwrap_or(0),
                Nullability::NonNullable,
            ),
            SumGroupedValues::Float(values) => Scalar::primitive(
                values.get(group_id).copied().unwrap_or(0.0),
                Nullability::NonNullable,
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
        };
        Ok(Scalar::struct_(
            self.partial_dtype.clone(),
            vec![
                sum,
                Scalar::bool(
                    self.overflowed
                        .get(group_id)
                        .is_some_and(|&overflowed| overflowed != 0),
                    Nullability::NonNullable,
                ),
                Scalar::bool(
                    self.empty.get(group_id).is_none_or(|&empty| empty != 0),
                    Nullability::NonNullable,
                ),
            ],
        ))
    }

    fn accumulate_partials(
        &mut self,
        partials: &ArrayRef,
        group_ids: &[u32],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let partials = partials.clone().execute::<StructArray>(ctx)?;
        let validity = partials
            .as_ref()
            .validity()?
            .execute_mask(partials.as_ref().len(), ctx)?;
        let sums = partials.unmasked_field_by_name(SUM_FIELD)?.clone();
        let partial_overflowed = partials
            .unmasked_field_by_name(IS_OVERFLOW_FIELD)?
            .clone()
            .execute::<BoolArray>(ctx)?
            .into_bit_buffer();
        let partial_empty = partials
            .unmasked_field_by_name(IS_EMPTY_FIELD)?
            .clone()
            .execute::<BoolArray>(ctx)?
            .into_bit_buffer();
        let rows = PartialRows {
            group_ids,
            validity: &validity,
            overflowed: &partial_overflowed,
            empty: &partial_empty,
        };
        let decimal_dtype = self.decimal_dtype();
        let (values, overflowed, empty) = self.parts_mut();
        match values {
            SumGroupedValues::Unsigned(values) => {
                let sums = sums.execute::<PrimitiveArray>(ctx)?;
                accumulate_primitive_partials(
                    PartialState {
                        values,
                        overflowed,
                        empty,
                    },
                    sums.as_slice::<u64>(),
                    &rows,
                    checked_add_u64,
                );
            }
            SumGroupedValues::Signed(values) => {
                let sums = sums.execute::<PrimitiveArray>(ctx)?;
                accumulate_primitive_partials(
                    PartialState {
                        values,
                        overflowed,
                        empty,
                    },
                    sums.as_slice::<i64>(),
                    &rows,
                    checked_add_i64,
                );
            }
            SumGroupedValues::Float(values) => {
                let sums = sums.execute::<PrimitiveArray>(ctx)?;
                accumulate_float_partials(
                    PartialState {
                        values,
                        overflowed,
                        empty,
                    },
                    sums.as_slice::<f64>(),
                    &rows,
                );
            }
            SumGroupedValues::Decimal8(values) => accumulate_decimal_partials(
                PartialState {
                    values,
                    overflowed,
                    empty,
                },
                &sums.execute::<DecimalArray>(ctx)?,
                &rows,
                decimal_dtype.vortex_expect("decimal state dtype"),
            ),
            SumGroupedValues::Decimal16(values) => accumulate_decimal_partials(
                PartialState {
                    values,
                    overflowed,
                    empty,
                },
                &sums.execute::<DecimalArray>(ctx)?,
                &rows,
                decimal_dtype.vortex_expect("decimal state dtype"),
            ),
            SumGroupedValues::Decimal32(values) => accumulate_decimal_partials(
                PartialState {
                    values,
                    overflowed,
                    empty,
                },
                &sums.execute::<DecimalArray>(ctx)?,
                &rows,
                decimal_dtype.vortex_expect("decimal state dtype"),
            ),
            SumGroupedValues::Decimal64(values) => accumulate_decimal_partials(
                PartialState {
                    values,
                    overflowed,
                    empty,
                },
                &sums.execute::<DecimalArray>(ctx)?,
                &rows,
                decimal_dtype.vortex_expect("decimal state dtype"),
            ),
            SumGroupedValues::Decimal128(values) => accumulate_decimal_partials(
                PartialState {
                    values,
                    overflowed,
                    empty,
                },
                &sums.execute::<DecimalArray>(ctx)?,
                &rows,
                decimal_dtype.vortex_expect("decimal state dtype"),
            ),
            SumGroupedValues::Decimal256(values) => accumulate_decimal_partials(
                PartialState {
                    values,
                    overflowed,
                    empty,
                },
                &sums.execute::<DecimalArray>(ctx)?,
                &rows,
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
        let overflowed = std::mem::take(&mut self.overflowed);
        let empty = std::mem::take(&mut self.empty);
        let decimal_dtype = self.decimal_dtype();
        let sums = match &mut self.values {
            SumGroupedValues::Unsigned(values) => {
                PrimitiveArray::new(Buffer::from(std::mem::take(values)), Validity::NonNullable)
                    .into_array()
            }
            SumGroupedValues::Signed(values) => {
                PrimitiveArray::new(Buffer::from(std::mem::take(values)), Validity::NonNullable)
                    .into_array()
            }
            SumGroupedValues::Float(values) => {
                PrimitiveArray::new(Buffer::from(std::mem::take(values)), Validity::NonNullable)
                    .into_array()
            }
            SumGroupedValues::Decimal8(values) => decimal_array(
                std::mem::take(values),
                decimal_dtype.vortex_expect("decimal state dtype"),
                Validity::NonNullable,
            ),
            SumGroupedValues::Decimal16(values) => decimal_array(
                std::mem::take(values),
                decimal_dtype.vortex_expect("decimal state dtype"),
                Validity::NonNullable,
            ),
            SumGroupedValues::Decimal32(values) => decimal_array(
                std::mem::take(values),
                decimal_dtype.vortex_expect("decimal state dtype"),
                Validity::NonNullable,
            ),
            SumGroupedValues::Decimal64(values) => decimal_array(
                std::mem::take(values),
                decimal_dtype.vortex_expect("decimal state dtype"),
                Validity::NonNullable,
            ),
            SumGroupedValues::Decimal128(values) => decimal_array(
                std::mem::take(values),
                decimal_dtype.vortex_expect("decimal state dtype"),
                Validity::NonNullable,
            ),
            SumGroupedValues::Decimal256(values) => decimal_array(
                std::mem::take(values),
                decimal_dtype.vortex_expect("decimal state dtype"),
                Validity::NonNullable,
            ),
        };
        let names = self.partial_dtype.as_struct_fields().names().clone();
        Ok(StructArray::try_new(
            names,
            vec![
                sums,
                BoolArray::from_iter(overflowed.into_iter().map(|value| value != 0)).into_array(),
                BoolArray::from_iter(empty.into_iter().map(|value| value != 0)).into_array(),
            ],
            num_groups,
            Validity::AllValid,
        )?
        .into_array())
    }
}

struct PartialRows<'a> {
    group_ids: &'a [u32],
    validity: &'a Mask,
    overflowed: &'a BitBuffer,
    empty: &'a BitBuffer,
}

struct PartialState<'a, T> {
    values: &'a mut [T],
    overflowed: &'a mut [u8],
    empty: &'a mut [u8],
}

fn for_each_nonempty_partial(rows: &PartialRows<'_>, mut f: impl FnMut(usize, bool)) {
    for (idx, ((valid, overflowed), empty)) in rows
        .validity
        .iter()
        .zip(rows.overflowed.iter())
        .zip(rows.empty.iter())
        .enumerate()
    {
        if valid && !empty {
            f(idx, overflowed);
        }
    }
}

fn accumulate_primitive_partials<T: Copy>(
    state: PartialState<'_, T>,
    partials: &[T],
    rows: &PartialRows<'_>,
    checked_add: fn(&mut T, T) -> bool,
) {
    for_each_nonempty_partial(rows, |idx, is_overflowed| {
        let group = rows.group_ids[idx] as usize;
        state.empty[group] = 0;
        if is_overflowed || checked_add(&mut state.values[group], partials[idx]) {
            state.overflowed[group] = 1;
        }
    });
}

fn accumulate_float_partials(
    state: PartialState<'_, f64>,
    partials: &[f64],
    rows: &PartialRows<'_>,
) {
    for_each_nonempty_partial(rows, |idx, is_overflowed| {
        let group = rows.group_ids[idx] as usize;
        state.empty[group] = 0;
        if is_overflowed {
            state.overflowed[group] = 1;
        } else {
            state.values[group] += partials[idx];
        }
    });
}

fn accumulate_decimal_partials<I>(
    state: PartialState<'_, I>,
    partials: &DecimalArray,
    rows: &PartialRows<'_>,
    dtype: DecimalDType,
) where
    I: NativeDecimalType + CheckedAdd,
{
    match_each_decimal_value_type!(partials.values_type(), |T| {
        accumulate_decimal_partial_values(state, &partials.buffer::<T>(), rows, dtype);
    });
}

fn accumulate_decimal_partial_values<T, I>(
    state: PartialState<'_, I>,
    partials: &[T],
    rows: &PartialRows<'_>,
    dtype: DecimalDType,
) where
    T: NativeDecimalType,
    I: NativeDecimalType + CheckedAdd,
{
    for_each_nonempty_partial(rows, |idx, is_overflowed| {
        let group = rows.group_ids[idx] as usize;
        state.empty[group] = 0;
        if is_overflowed {
            state.overflowed[group] = 1;
        } else {
            let Some(value) = <I as crate::dtype::BigCast>::from(partials[idx]) else {
                state.overflowed[group] = 1;
                return;
            };
            add_decimal(state.values, state.overflowed, group, value, dtype);
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

fn decimal_scalar<T: NativeDecimalType>(value: T, dtype: DecimalDType) -> Scalar
where
    DecimalValue: From<T>,
{
    Scalar::decimal(DecimalValue::from(value), dtype, Nullability::NonNullable)
}

fn decimal_array<T: NativeDecimalType>(
    values: Vec<T>,
    dtype: DecimalDType,
    validity: Validity,
) -> ArrayRef {
    DecimalArray::new(Buffer::from(values), dtype, validity).into_array()
}
