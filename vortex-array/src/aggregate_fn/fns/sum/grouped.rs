// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use num_traits::CheckedAdd;
use num_traits::ToPrimitive;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use super::Sum;
use super::SumAggregateOpts;
use super::checked_add_i64;
use super::checked_add_u64;
use super::grouped_state::SumGroupedState;
use super::grouped_state::SumGroupedValues;
use super::grouped_state::add_decimal;
use super::primitive::sum_float_all;
use super::primitive::sum_signed_all;
use super::primitive::sum_unsigned_all;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::GroupIds;
use crate::aggregate_fn::kernels::GroupedAggregateKernel;
use crate::aggregate_fn::kernels::GroupedAggregateKernelAdapter;
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::bool::BoolArrayExt;
use crate::dtype::NativeDecimalType;
use crate::dtype::NativePType;
use crate::match_each_decimal_value_type;
use crate::match_each_native_ptype;

const MIN_GROUP_RUN_LENGTH: usize = 4;

pub(crate) static SUM_GROUPED_KERNEL: GroupedAggregateKernelAdapter<Sum, SumGroupedKernel> =
    GroupedAggregateKernelAdapter::new(SumGroupedKernel);

#[derive(Debug)]
pub(crate) struct SumGroupedKernel;

impl GroupedAggregateKernel<Sum> for SumGroupedKernel {
    type State = SumGroupedState;

    fn grouped_accumulate(
        &self,
        options: &SumAggregateOpts,
        state: &mut Self::State,
        batch: &ArrayRef,
        group_ids: &GroupIds,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        if let Some(primitive) = batch.as_opt::<Primitive>() {
            let group_ids = group_ids.validated_ids(ctx)?;
            accumulate_grouped_primitive(
                state,
                &primitive.into_owned(),
                group_ids.as_ref(),
                options.skip_nans,
                ctx,
            )?;
            return Ok(true);
        }

        if let Some(bools) = batch.as_opt::<Bool>() {
            let group_ids = group_ids.validated_ids(ctx)?;
            accumulate_grouped_bool(state, &bools.into_owned(), group_ids.as_ref(), ctx)?;
            return Ok(true);
        }

        if let Some(decimals) = batch.as_opt::<Decimal>() {
            let group_ids = group_ids.validated_ids(ctx)?;
            accumulate_grouped_decimal(state, &decimals.into_owned(), group_ids.as_ref(), ctx)?;
            return Ok(true);
        }

        Ok(false)
    }
}

fn for_each_valid_idx(validity: &Mask, len: usize, mut f: impl FnMut(usize)) {
    match validity.indices() {
        AllOr::All => (0..len).for_each(f),
        AllOr::None => {}
        AllOr::Some(indices) => indices.iter().copied().for_each(&mut f),
    }
}

fn for_each_group_run(group_ids: &[u32], mut f: impl FnMut(u32, usize, usize)) {
    let Some((&first, rest)) = group_ids.split_first() else {
        return;
    };
    let mut group_id = first;
    let mut start = 0usize;
    for (idx, &next_group_id) in rest.iter().enumerate() {
        let idx = idx + 1;
        if next_group_id != group_id {
            f(group_id, start, idx);
            group_id = next_group_id;
            start = idx;
        }
    }
    f(group_id, start, group_ids.len());
}

fn has_long_group_runs(group_ids: &[u32]) -> bool {
    let mut run_length = 1;
    for ids in group_ids[..group_ids.len().min(256)].windows(2) {
        if ids[0] == ids[1] {
            run_length += 1;
            if run_length >= MIN_GROUP_RUN_LENGTH {
                return true;
            }
        } else {
            run_length = 1;
        }
    }
    false
}

fn accumulate_grouped_unsigned(
    values: &mut [u64],
    overflowed: &mut [u8],
    empty: &mut [u8],
    group_id: u32,
    value: u64,
) {
    let group = group_id as usize;
    empty[group] = 0;
    if checked_add_u64(&mut values[group], value) {
        overflowed[group] = 1;
    }
}

fn accumulate_grouped_unsigned_run<T>(
    sums: &mut [u64],
    overflowed: &mut [u8],
    empty: &mut [u8],
    group_id: u32,
    values: &[T],
) where
    T: NativePType + AsPrimitive<u64>,
{
    let group = group_id as usize;
    empty[group] = 0;
    if sum_unsigned_all(&mut sums[group], values) {
        overflowed[group] = 1;
    }
}

fn accumulate_grouped_unsigned_all<T>(
    sums: &mut [u64],
    overflowed: &mut [u8],
    empty: &mut [u8],
    values: &[T],
    group_ids: &[u32],
) where
    T: NativePType + AsPrimitive<u64>,
{
    if !has_long_group_runs(group_ids) {
        for (&value, &group_id) in values.iter().zip(group_ids) {
            accumulate_grouped_unsigned(sums, overflowed, empty, group_id, value.as_());
        }
        return;
    }

    for_each_group_run(group_ids, |group_id, start, end| {
        empty[group_id as usize] = 0;
        if end - start >= MIN_GROUP_RUN_LENGTH {
            accumulate_grouped_unsigned_run(sums, overflowed, empty, group_id, &values[start..end]);
        } else {
            for &value in &values[start..end] {
                accumulate_grouped_unsigned(sums, overflowed, empty, group_id, value.as_());
            }
        }
    });
}

fn accumulate_grouped_signed(
    values: &mut [i64],
    overflowed: &mut [u8],
    empty: &mut [u8],
    group_id: u32,
    value: i64,
) {
    let group = group_id as usize;
    empty[group] = 0;
    if checked_add_i64(&mut values[group], value) {
        overflowed[group] = 1;
    }
}

fn accumulate_grouped_signed_run<T>(
    sums: &mut [i64],
    overflowed: &mut [u8],
    empty: &mut [u8],
    group_id: u32,
    values: &[T],
) where
    T: NativePType + AsPrimitive<i64>,
{
    let group = group_id as usize;
    empty[group] = 0;
    if sum_signed_all(&mut sums[group], values) {
        overflowed[group] = 1;
    }
}

fn accumulate_grouped_signed_all<T>(
    sums: &mut [i64],
    overflowed: &mut [u8],
    empty: &mut [u8],
    values: &[T],
    group_ids: &[u32],
) where
    T: NativePType + AsPrimitive<i64>,
{
    if !has_long_group_runs(group_ids) {
        for (&value, &group_id) in values.iter().zip(group_ids) {
            accumulate_grouped_signed(sums, overflowed, empty, group_id, value.as_());
        }
        return;
    }

    for_each_group_run(group_ids, |group_id, start, end| {
        if end - start >= MIN_GROUP_RUN_LENGTH {
            accumulate_grouped_signed_run(sums, overflowed, empty, group_id, &values[start..end]);
        } else {
            for &value in &values[start..end] {
                accumulate_grouped_signed(sums, overflowed, empty, group_id, value.as_());
            }
        }
    });
}

fn accumulate_grouped_float(
    sums: &mut [f64],
    empty: &mut [u8],
    group_id: u32,
    value: f64,
    skip_nans: bool,
) {
    empty[group_id as usize] = 0;
    if !skip_nans || !value.is_nan() {
        sums[group_id as usize] += value;
    }
}

fn accumulate_grouped_float_all<T: NativePType>(
    sums: &mut [f64],
    empty: &mut [u8],
    values: &[T],
    group_ids: &[u32],
    skip_nans: bool,
) {
    if !has_long_group_runs(group_ids) {
        for (value, &group_id) in values.iter().zip(group_ids) {
            let value = ToPrimitive::to_f64(value).vortex_expect("float to f64");
            accumulate_grouped_float(sums, empty, group_id, value, skip_nans);
        }
        return;
    }

    for_each_group_run(group_ids, |group_id, start, end| {
        empty[group_id as usize] = 0;
        if end - start >= MIN_GROUP_RUN_LENGTH {
            sum_float_all(&mut sums[group_id as usize], &values[start..end], skip_nans);
        } else {
            for value in &values[start..end] {
                let value = ToPrimitive::to_f64(value).vortex_expect("float to f64");
                accumulate_grouped_float(sums, empty, group_id, value, skip_nans);
            }
        }
    });
}

fn accumulate_grouped_primitive(
    state: &mut SumGroupedState,
    primitive: &PrimitiveArray,
    group_ids: &[u32],
    skip_nans: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let validity = primitive
        .as_ref()
        .validity()?
        .execute_mask(primitive.as_ref().len(), ctx)?;
    let all_valid = matches!(validity.slices(), AllOr::All);
    let (state, overflowed, empty) = state.parts_mut();

    match_each_native_ptype!(primitive.ptype(),
        unsigned: |T| {
            let SumGroupedValues::Unsigned(sums) = state else {
                vortex_panic!("unsigned input with non-unsigned grouped sum state")
            };
            let values = primitive.as_slice::<T>();
            if all_valid {
                accumulate_grouped_unsigned_all(sums, overflowed, empty, values, group_ids);
            } else {
                for_each_valid_idx(&validity, values.len(), |idx| {
                    accumulate_grouped_unsigned(
                        sums,
                        overflowed,
                        empty,
                        group_ids[idx],
                        values[idx].as_(),
                    );
                });
            }
        },
        signed: |T| {
            let SumGroupedValues::Signed(sums) = state else {
                vortex_panic!("signed input with non-signed grouped sum state")
            };
            let values = primitive.as_slice::<T>();
            if all_valid {
                accumulate_grouped_signed_all(sums, overflowed, empty, values, group_ids);
            } else {
                for_each_valid_idx(&validity, values.len(), |idx| {
                    accumulate_grouped_signed(
                        sums,
                        overflowed,
                        empty,
                        group_ids[idx],
                        values[idx].as_(),
                    );
                });
            }
        },
        floating: |T| {
            let SumGroupedValues::Float(sums) = state else {
                vortex_panic!("float input with non-float grouped sum state")
            };
            let values = primitive.as_slice::<T>();
            if all_valid {
                accumulate_grouped_float_all(sums, empty, values, group_ids, skip_nans);
            } else {
                for_each_valid_idx(&validity, values.len(), |idx| {
                    let value = ToPrimitive::to_f64(&values[idx]).vortex_expect("float to f64");
                    accumulate_grouped_float(sums, empty, group_ids[idx], value, skip_nans);
                });
            }
        }
    );
    Ok(())
}

fn accumulate_grouped_bool(
    state: &mut SumGroupedState,
    bools: &BoolArray,
    group_ids: &[u32],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let validity = bools
        .as_ref()
        .validity()?
        .execute_mask(bools.as_ref().len(), ctx)?;
    let values = bools.to_bit_buffer();
    let valid_true = match validity.bit_buffer() {
        AllOr::All => values,
        AllOr::None => return Ok(()),
        AllOr::Some(validity) => &values & validity,
    };
    let (state, overflowed, empty) = state.parts_mut();
    let SumGroupedValues::Unsigned(sums) = state else {
        vortex_panic!("boolean input with non-unsigned grouped sum state")
    };
    for_each_valid_idx(&validity, bools.as_ref().len(), |idx| {
        empty[group_ids[idx] as usize] = 0;
    });
    valid_true.for_each_set_index(|idx| {
        accumulate_grouped_unsigned(sums, overflowed, empty, group_ids[idx], 1);
    });
    Ok(())
}

fn accumulate_grouped_decimal(
    state: &mut SumGroupedState,
    decimals: &DecimalArray,
    group_ids: &[u32],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let validity = decimals
        .as_ref()
        .validity()?
        .execute_mask(decimals.as_ref().len(), ctx)?;
    let output_dtype = state
        .decimal_dtype()
        .vortex_expect("decimal sum state dtype");
    let (state, overflowed, empty) = state.parts_mut();
    match_each_decimal_value_type!(decimals.values_type(), |T| {
        let values = decimals.buffer::<T>();
        match state {
            SumGroupedValues::Decimal8(sums) => accumulate_grouped_decimal_values(
                sums,
                overflowed,
                empty,
                values,
                group_ids,
                &validity,
                output_dtype,
            ),
            SumGroupedValues::Decimal16(sums) => accumulate_grouped_decimal_values(
                sums,
                overflowed,
                empty,
                values,
                group_ids,
                &validity,
                output_dtype,
            ),
            SumGroupedValues::Decimal32(sums) => accumulate_grouped_decimal_values(
                sums,
                overflowed,
                empty,
                values,
                group_ids,
                &validity,
                output_dtype,
            ),
            SumGroupedValues::Decimal64(sums) => accumulate_grouped_decimal_values(
                sums,
                overflowed,
                empty,
                values,
                group_ids,
                &validity,
                output_dtype,
            ),
            SumGroupedValues::Decimal128(sums) => accumulate_grouped_decimal_values(
                sums,
                overflowed,
                empty,
                values,
                group_ids,
                &validity,
                output_dtype,
            ),
            SumGroupedValues::Decimal256(sums) => accumulate_grouped_decimal_values(
                sums,
                overflowed,
                empty,
                values,
                group_ids,
                &validity,
                output_dtype,
            ),
            _ => vortex_panic!("decimal input with non-decimal grouped sum state"),
        }
    });
    Ok(())
}

fn accumulate_grouped_decimal_values<T, I>(
    sums: &mut [I],
    overflowed: &mut [u8],
    empty: &mut [u8],
    values: Buffer<T>,
    group_ids: &[u32],
    validity: &Mask,
    dtype: crate::dtype::DecimalDType,
) where
    T: NativeDecimalType + AsPrimitive<I>,
    I: NativeDecimalType + CheckedAdd,
{
    for_each_valid_idx(validity, values.len(), |idx| {
        empty[group_ids[idx] as usize] = 0;
        add_decimal(
            sums,
            overflowed,
            group_ids[idx] as usize,
            values[idx].as_(),
            dtype,
        );
    });
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::DynGroupedAccumulator;
    use crate::aggregate_fn::GroupIds;
    use crate::aggregate_fn::GroupedAccumulator;
    use crate::aggregate_fn::fns::sum::Sum;
    use crate::aggregate_fn::fns::sum::SumAggregateOpts;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::arrays::DecimalArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::FieldName;
    use crate::dtype::FieldNames;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::i256;
    use crate::scalar::DecimalValue;
    use crate::validity::Validity;

    fn sum_partials(
        sums: crate::ArrayRef,
        overflowed: impl IntoIterator<Item = bool>,
        empty: impl IntoIterator<Item = bool>,
    ) -> VortexResult<crate::ArrayRef> {
        let len = sums.len();
        Ok(StructArray::try_new(
            FieldNames::from_iter([
                FieldName::from("sum"),
                FieldName::from("is_overflow"),
                FieldName::from("is_empty"),
            ]),
            vec![
                sums,
                BoolArray::from_iter(overflowed).into_array(),
                BoolArray::from_iter(empty).into_array(),
            ],
            len,
            Validity::AllValid,
        )?
        .into_array())
    }

    fn run_grouped_sum(
        values: &crate::ArrayRef,
        ids: impl IntoIterator<Item = u32>,
        num_groups: usize,
        options: SumAggregateOpts,
    ) -> VortexResult<crate::ArrayRef> {
        let mut acc = GroupedAccumulator::try_new(Sum, options, values.dtype().clone())?;
        let group_ids = GroupIds::from_iter(ids, num_groups)?;
        let mut ctx = array_session().create_execution_ctx();
        acc.accumulate(values, &group_ids, &mut ctx)?;
        acc.finish(num_groups)
    }

    #[test]
    fn dense_ids_repeat_reorder_and_omit_groups() -> VortexResult<()> {
        let values = PrimitiveArray::from_option_iter([
            Some(1i32),
            None,
            Some(3),
            Some(4),
            Some(5),
            Some(6),
        ])
        .into_array();
        let actual = run_grouped_sum(&values, [2, 0, 2, 0, 2, 0], 4, SumAggregateOpts::default())?;
        let expected =
            PrimitiveArray::from_option_iter([Some(10i64), None, Some(9), None]).into_array();
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(&actual, &expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn bool_and_overflow_are_group_local() -> VortexResult<()> {
        let bools: BoolArray = [true, false, true, true].into_iter().collect();
        let actual = run_grouped_sum(
            &bools.into_array(),
            [1, 0, 1, 0],
            2,
            SumAggregateOpts::default(),
        )?;
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(
            &actual,
            &PrimitiveArray::from_option_iter([Some(1u64), Some(2)]).into_array(),
            &mut ctx
        );

        let values =
            PrimitiveArray::new(buffer![i64::MAX, 1, 2, 3], Validity::NonNullable).into_array();
        let actual = run_grouped_sum(&values, [0, 0, 1, 1], 2, SumAggregateOpts::default())?;
        assert_arrays_eq!(
            &actual,
            &PrimitiveArray::from_option_iter([None, Some(5i64)]).into_array(),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn float_nan_options_match_scalar_sum() -> VortexResult<()> {
        let values =
            PrimitiveArray::new(buffer![1.0f64, f64::NAN, 2.0, 4.0], Validity::NonNullable)
                .into_array();
        let skipped = run_grouped_sum(&values, [0, 0, 1, 1], 2, SumAggregateOpts::default())?;
        let included = run_grouped_sum(&values, [0, 0, 1, 1], 2, SumAggregateOpts::include_nans())?;
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(
            &skipped,
            &PrimitiveArray::from_option_iter([Some(1.0f64), Some(6.0)]).into_array(),
            &mut ctx
        );
        let group_zero = included.execute_scalar(0, &mut ctx)?;
        assert!(
            group_zero
                .as_primitive()
                .typed_value::<f64>()
                .vortex_expect("grouped float sum should be non-null")
                .is_nan()
        );
        Ok(())
    }

    #[test]
    fn exact_decimal_sum_with_reordered_ids_and_nulls() -> VortexResult<()> {
        let input_dtype = DecimalDType::new(10, 2);
        let values = DecimalArray::new(
            buffer![100i64, 200, -50, 300, 400],
            input_dtype,
            Validity::from_iter([true, true, true, false, true]),
        )
        .into_array();
        let actual = run_grouped_sum(&values, [2, 0, 2, 0, 2], 4, SumAggregateOpts::default())?;
        let output_dtype = DecimalDType::new(20, 2);
        let expected = DecimalArray::new(
            buffer![200i64, 0, 450, 0],
            output_dtype,
            Validity::from_iter([true, false, true, false]),
        )
        .into_array();
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(&actual, &expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn exact_decimal_overflow_is_group_local() -> VortexResult<()> {
        let one = i256::from_i128(1);
        let large = i256::from_i128(10)
            .checked_pow(76)
            .vortex_expect("10^76 must fit in i256")
            - one;
        let dtype = DecimalDType::new(76, 0);
        let values = DecimalArray::new(
            buffer![large, i256::from_i128(7), large],
            dtype,
            Validity::NonNullable,
        )
        .into_array();
        let actual = run_grouped_sum(&values, [0, 1, 0], 2, SumAggregateOpts::default())?;
        let expected = DecimalArray::new(
            buffer![i256::ZERO, i256::from_i128(7)],
            dtype,
            Validity::from_iter([false, true]),
        )
        .into_array();
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(&actual, &expected, &mut ctx);

        let group_one = actual.execute_scalar(1, &mut ctx)?;
        assert_eq!(
            group_one.as_decimal().decimal_value(),
            Some(DecimalValue::I256(i256::from_i128(7)))
        );
        Ok(())
    }

    #[test]
    fn accumulates_typed_primitive_partials() -> VortexResult<()> {
        let input_dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let partials = sum_partials(
            PrimitiveArray::new(buffer![2i64, 3, 5, 0], Validity::NonNullable).into_array(),
            [false, false, false, true],
            [false; 4],
        )?;
        let mut ctx = array_session().create_execution_ctx();
        let mut acc = GroupedAccumulator::try_new(Sum, SumAggregateOpts::default(), input_dtype)?;
        acc.accumulate_partials(
            &partials,
            &GroupIds::from_iter([0u32, 1, 1, 0], 2)?,
            &mut ctx,
        )?;
        let actual = acc.finish(2)?;
        let expected = PrimitiveArray::from_option_iter([None, Some(8i64)]).into_array();
        assert_arrays_eq!(&actual, &expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn accumulates_typed_decimal_partials() -> VortexResult<()> {
        let input_dtype = DecimalDType::new(10, 2);
        let partial_dtype = DecimalDType::new(20, 2);
        let partials = sum_partials(
            DecimalArray::new(
                buffer![200i64, 300, 500, 0],
                partial_dtype,
                Validity::NonNullable,
            )
            .into_array(),
            [false, false, false, true],
            [false; 4],
        )?;
        let mut ctx = array_session().create_execution_ctx();
        let mut acc = GroupedAccumulator::try_new(
            Sum,
            SumAggregateOpts::default(),
            DType::Decimal(input_dtype, Nullability::Nullable),
        )?;
        acc.accumulate_partials(
            &partials,
            &GroupIds::from_iter([0u32, 1, 1, 0], 2)?,
            &mut ctx,
        )?;
        let actual = acc.finish(2)?;
        let expected = DecimalArray::new(
            buffer![0i128, 800],
            partial_dtype,
            Validity::from_iter([false, true]),
        )
        .into_array();
        assert_arrays_eq!(&actual, &expected, &mut ctx);
        Ok(())
    }
}
