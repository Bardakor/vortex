// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use num_traits::ToPrimitive;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use super::Sum;
use super::SumPartial;
use super::SumState;
use super::checked_add_i64;
use super::checked_add_u64;
use super::primitive::sum_float_all;
use super::primitive::sum_signed_all;
use super::primitive::sum_unsigned_all;
use super::sum_decimal_dtype;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::GroupIds;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::kernels::GroupedAggregateKernel;
use crate::aggregate_fn::kernels::GroupedAggregateKernelAdapter;
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::bool::BoolArrayExt;
use crate::dtype::DecimalType;
use crate::dtype::NativeDecimalType;
use crate::dtype::NativePType;
use crate::match_each_decimal_value_type;
use crate::match_each_native_ptype;
use crate::scalar::DecimalValue;

const MIN_GROUP_RUN_LENGTH: usize = 4;

pub(crate) static SUM_GROUPED_KERNEL: GroupedAggregateKernelAdapter<Sum, SumGroupedKernel> =
    GroupedAggregateKernelAdapter::new(SumGroupedKernel);

#[derive(Debug)]
pub(crate) struct SumGroupedKernel;

impl GroupedAggregateKernel<Sum> for SumGroupedKernel {
    fn grouped_accumulate(
        &self,
        options: &NumericalAggregateOpts,
        partials: &mut [SumPartial],
        batch: &ArrayRef,
        group_ids: &GroupIds,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        if let Some(primitive) = batch.as_opt::<Primitive>() {
            let group_ids = group_ids.validated_ids(ctx)?;
            accumulate_grouped_primitive(
                partials,
                &primitive.into_owned(),
                group_ids.as_ref(),
                options.skip_nans,
                ctx,
            )?;
            return Ok(true);
        }

        if let Some(bools) = batch.as_opt::<Bool>() {
            let group_ids = group_ids.validated_ids(ctx)?;
            accumulate_grouped_bool(partials, &bools.into_owned(), group_ids.as_ref(), ctx)?;
            return Ok(true);
        }

        if let Some(decimals) = batch.as_opt::<Decimal>() {
            let group_ids = group_ids.validated_ids(ctx)?;
            accumulate_grouped_decimal(partials, &decimals.into_owned(), group_ids.as_ref(), ctx)?;
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

fn accumulate_grouped_unsigned(partials: &mut [SumPartial], group_id: u32, value: u64) {
    let partial = &mut partials[group_id as usize];
    let saturated = match partial.current.as_mut() {
        None => return,
        Some(SumState::Unsigned(acc)) => checked_add_u64(acc, value),
        Some(_) => vortex_panic!("unsigned sum state with non-unsigned input"),
    };
    if saturated {
        partial.current = None;
    }
}

fn accumulate_grouped_unsigned_run<T>(partials: &mut [SumPartial], group_id: u32, values: &[T])
where
    T: NativePType + AsPrimitive<u64>,
{
    let partial = &mut partials[group_id as usize];
    let saturated = match partial.current.as_mut() {
        None => return,
        Some(SumState::Unsigned(acc)) => sum_unsigned_all(acc, values),
        Some(_) => vortex_panic!("unsigned sum state with non-unsigned input"),
    };
    if saturated {
        partial.current = None;
    }
}

fn accumulate_grouped_unsigned_all<T>(partials: &mut [SumPartial], values: &[T], group_ids: &[u32])
where
    T: NativePType + AsPrimitive<u64>,
{
    for_each_group_run(group_ids, |group_id, start, end| {
        if end - start >= MIN_GROUP_RUN_LENGTH {
            accumulate_grouped_unsigned_run(partials, group_id, &values[start..end]);
        } else {
            for &value in &values[start..end] {
                accumulate_grouped_unsigned(partials, group_id, value.as_());
            }
        }
    });
}

fn accumulate_grouped_signed(partials: &mut [SumPartial], group_id: u32, value: i64) {
    let partial = &mut partials[group_id as usize];
    let saturated = match partial.current.as_mut() {
        None => return,
        Some(SumState::Signed(acc)) => checked_add_i64(acc, value),
        Some(_) => vortex_panic!("signed sum state with non-signed input"),
    };
    if saturated {
        partial.current = None;
    }
}

fn accumulate_grouped_signed_run<T>(partials: &mut [SumPartial], group_id: u32, values: &[T])
where
    T: NativePType + AsPrimitive<i64>,
{
    let partial = &mut partials[group_id as usize];
    let saturated = match partial.current.as_mut() {
        None => return,
        Some(SumState::Signed(acc)) => sum_signed_all(acc, values),
        Some(_) => vortex_panic!("signed sum state with non-signed input"),
    };
    if saturated {
        partial.current = None;
    }
}

fn accumulate_grouped_signed_all<T>(partials: &mut [SumPartial], values: &[T], group_ids: &[u32])
where
    T: NativePType + AsPrimitive<i64>,
{
    for_each_group_run(group_ids, |group_id, start, end| {
        if end - start >= MIN_GROUP_RUN_LENGTH {
            accumulate_grouped_signed_run(partials, group_id, &values[start..end]);
        } else {
            for &value in &values[start..end] {
                accumulate_grouped_signed(partials, group_id, value.as_());
            }
        }
    });
}

fn accumulate_grouped_float(
    partials: &mut [SumPartial],
    group_id: u32,
    value: f64,
    skip_nans: bool,
) {
    if skip_nans && value.is_nan() {
        return;
    }
    match partials[group_id as usize].current.as_mut() {
        None => {}
        Some(SumState::Float(acc)) => *acc += value,
        Some(_) => vortex_panic!("float sum state with non-float input"),
    }
}

fn accumulate_grouped_float_run<T: NativePType>(
    partials: &mut [SumPartial],
    group_id: u32,
    values: &[T],
    skip_nans: bool,
) {
    match partials[group_id as usize].current.as_mut() {
        None => {}
        Some(SumState::Float(acc)) => sum_float_all(acc, values, skip_nans),
        Some(_) => vortex_panic!("float sum state with non-float input"),
    }
}

fn accumulate_grouped_float_all<T: NativePType>(
    partials: &mut [SumPartial],
    values: &[T],
    group_ids: &[u32],
    skip_nans: bool,
) {
    for_each_group_run(group_ids, |group_id, start, end| {
        if end - start >= MIN_GROUP_RUN_LENGTH {
            accumulate_grouped_float_run(partials, group_id, &values[start..end], skip_nans);
        } else {
            for value in &values[start..end] {
                let value = ToPrimitive::to_f64(value).vortex_expect("float to f64");
                accumulate_grouped_float(partials, group_id, value, skip_nans);
            }
        }
    });
}

fn accumulate_grouped_primitive(
    partials: &mut [SumPartial],
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

    match_each_native_ptype!(primitive.ptype(),
        unsigned: |T| {
            let values = primitive.as_slice::<T>();
            if all_valid {
                accumulate_grouped_unsigned_all(partials, values, group_ids);
            } else {
                for_each_valid_idx(&validity, values.len(), |idx| {
                    accumulate_grouped_unsigned(partials, group_ids[idx], values[idx].as_());
                });
            }
        },
        signed: |T| {
            let values = primitive.as_slice::<T>();
            if all_valid {
                accumulate_grouped_signed_all(partials, values, group_ids);
            } else {
                for_each_valid_idx(&validity, values.len(), |idx| {
                    accumulate_grouped_signed(partials, group_ids[idx], values[idx].as_());
                });
            }
        },
        floating: |T| {
            let values = primitive.as_slice::<T>();
            if all_valid {
                accumulate_grouped_float_all(partials, values, group_ids, skip_nans);
            } else {
                for_each_valid_idx(&validity, values.len(), |idx| {
                    let value = ToPrimitive::to_f64(&values[idx]).vortex_expect("float to f64");
                    accumulate_grouped_float(partials, group_ids[idx], value, skip_nans);
                });
            }
        }
    );
    Ok(())
}

fn accumulate_grouped_bool(
    partials: &mut [SumPartial],
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
    valid_true.for_each_set_index(|idx| {
        accumulate_grouped_unsigned(partials, group_ids[idx], 1);
    });
    Ok(())
}

fn accumulate_grouped_decimal(
    partials: &mut [SumPartial],
    decimals: &DecimalArray,
    group_ids: &[u32],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let validity = decimals
        .as_ref()
        .validity()?
        .execute_mask(decimals.as_ref().len(), ctx)?;
    let output_dtype = sum_decimal_dtype(&decimals.decimal_dtype());
    let output_type = DecimalType::smallest_decimal_value_type(&output_dtype);
    match_each_decimal_value_type!(decimals.values_type(), |T| {
        match_each_decimal_value_type!(output_type, |I| {
            accumulate_grouped_decimal_values::<T, I>(
                partials,
                decimals.buffer::<T>(),
                group_ids,
                &validity,
            );
        })
    });
    Ok(())
}

fn accumulate_grouped_decimal_values<T, I>(
    partials: &mut [SumPartial],
    values: Buffer<T>,
    group_ids: &[u32],
    validity: &Mask,
) where
    T: NativeDecimalType + AsPrimitive<I>,
    I: NativeDecimalType,
    DecimalValue: From<I>,
{
    for_each_valid_idx(validity, values.len(), |idx| {
        let partial = &mut partials[group_ids[idx] as usize];
        let Some(SumState::Decimal { value, dtype }) = partial.current.as_mut() else {
            return;
        };
        let operand = DecimalValue::from(values[idx].as_());
        let Some(result) = value.checked_add(&operand) else {
            partial.current = None;
            return;
        };
        if result.fits_in_precision(*dtype) {
            *value = result;
        } else {
            partial.current = None;
        }
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
    use crate::aggregate_fn::NumericalAggregateOpts;
    use crate::aggregate_fn::fns::sum::Sum;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::arrays::DecimalArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DecimalDType;
    use crate::dtype::i256;
    use crate::scalar::DecimalValue;
    use crate::validity::Validity;

    fn run_grouped_sum(
        values: &crate::ArrayRef,
        ids: impl IntoIterator<Item = u32>,
        num_groups: usize,
        options: NumericalAggregateOpts,
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
        let actual = run_grouped_sum(
            &values,
            [2, 0, 2, 0, 2, 0],
            4,
            NumericalAggregateOpts::default(),
        )?;
        let expected =
            PrimitiveArray::from_option_iter([Some(10i64), Some(0), Some(9), Some(0)]).into_array();
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
            NumericalAggregateOpts::default(),
        )?;
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(
            &actual,
            &PrimitiveArray::from_option_iter([Some(1u64), Some(2)]).into_array(),
            &mut ctx
        );

        let values =
            PrimitiveArray::new(buffer![i64::MAX, 1, 2, 3], Validity::NonNullable).into_array();
        let actual = run_grouped_sum(&values, [0, 0, 1, 1], 2, NumericalAggregateOpts::default())?;
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
        let skipped = run_grouped_sum(&values, [0, 0, 1, 1], 2, NumericalAggregateOpts::default())?;
        let included = run_grouped_sum(
            &values,
            [0, 0, 1, 1],
            2,
            NumericalAggregateOpts::include_nans(),
        )?;
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
        let actual = run_grouped_sum(
            &values,
            [2, 0, 2, 0, 2],
            4,
            NumericalAggregateOpts::default(),
        )?;
        let output_dtype = DecimalDType::new(20, 2);
        let expected =
            DecimalArray::new(buffer![200i64, 0, 450, 0], output_dtype, Validity::AllValid)
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
        let actual = run_grouped_sum(&values, [0, 1, 0], 2, NumericalAggregateOpts::default())?;
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
}
