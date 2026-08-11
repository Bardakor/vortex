// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::GenericByteArray;
use arrow_array::types::ByteArrayType;
use arrow_buffer::ArrowNativeType;
use arrow_buffer::Buffer as ArrowBuffer;
use arrow_buffer::OffsetBuffer;
use arrow_schema::DataType;
use num_traits::AsPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBin;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::arrays::varbin::VarBinArraySlotsExt;
use vortex_array::builders::VarBinBuilder;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::OffsetBuilderPType;
use vortex_array::match_each_integer_ptype;
use vortex_array::vtable::VarBinExportable;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::executor::validity::to_arrow_null_buffer;

/// Convert a Vortex array into an Arrow GenericBinaryArray.
pub(super) fn to_arrow_byte_array<T: ByteArrayType>(
    array: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef>
where
    T::Offset: OffsetBuilderPType,
    usize: AsPrimitive<T::Offset>,
{
    if !matches!(array.dtype(), DType::Utf8(_) | DType::Binary(_)) {
        vortex_bail!(
            "Cannot convert Vortex array with dtype {} to Arrow byte array type {}",
            array.dtype(),
            T::DATA_TYPE
        );
    }

    // A logical dtype mismatch changes nothing about the physical export except that a `Binary`
    // source exported to `Utf8` has to have its bytes validated.
    let source_is_utf8 = matches!(array.dtype(), DType::Utf8(_));
    let target_is_utf8 = matches!(T::DATA_TYPE, DataType::Utf8 | DataType::LargeUtf8);
    let validate_utf8 = target_is_utf8 && !source_is_utf8;

    // Stop at an encoding that appends into the builder itself: operators like Filter and Slice
    // still run and push down, but executing further would only reach a `VarBinView` that the
    // builder has to re-lay out.
    let array = array.execute_until::<dyn VarBinExportable>(ctx)?;

    // A VarBin array already has the Arrow layout, so its buffers can be handed over as they are.
    if let Some(array) = array.as_opt::<VarBin>() {
        return varbin_to_byte_array::<T>(array, validate_utf8, ctx);
    }

    // The builder's offset type matches the Arrow target, so `varbin_to_byte_array` hands the
    // offsets buffer straight to Arrow without a cast.
    let mut builder = VarBinBuilder::<T::Offset>::with_capacity(array.dtype().clone(), array.len());
    array.append_to_builder(&mut builder, ctx)?;
    varbin_to_byte_array::<T>(builder.finish_into_varbin().as_view(), validate_utf8, ctx)
}

/// Convert a Vortex VarBinArray into an Arrow GenericBinaryArray.
///
/// `validate_utf8` must be set when the Vortex dtype does not already guarantee the bytes are
/// valid for `T`, i.e. a `Binary` source exported to a `Utf8` target.
fn varbin_to_byte_array<T: ByteArrayType>(
    array: ArrayView<'_, VarBin>,
    validate_utf8: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef>
where
    T::Offset: NativePType,
{
    let offsets = to_arrow_offsets::<T::Offset>(array.offsets(), ctx)?;

    let data = array.bytes().clone().into_arrow_buffer();

    if validate_utf8 {
        validate_values_utf8::<T>(&offsets, &data)?;
    }
    let null_buffer = to_arrow_null_buffer(array.validity()?, array.len(), ctx)?;
    // SAFETY: `T::DATA_TYPE` is a `Utf8` variant only when the source dtype already guarantees the
    // bytes are UTF-8, or when `validate_utf8` made us prove it above.
    Ok(Arc::new(unsafe {
        GenericByteArray::<T>::new_unchecked(offsets, data, null_buffer)
    }))
}

/// Converts a `VarBin` array's offsets into an Arrow offsets buffer.
///
/// `VarBin` offsets are usually unsigned whereas Arrow's are signed. Casting between the two is a
/// reinterpret, but it scans the whole buffer to prove every value fits; offsets are non-decreasing,
/// so checking the last one proves the same thing.
fn to_arrow_offsets<O: NativePType + ArrowNativeType>(
    offsets: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<OffsetBuffer<O>> {
    let offsets = offsets.clone().execute::<PrimitiveArray>(ctx)?;
    let offsets = if offsets.ptype() == O::PTYPE {
        offsets
    } else if offsets.ptype() == O::PTYPE.to_unsigned() {
        let last = last_offset(&offsets);
        vortex_ensure!(
            <O as ArrowNativeType>::from_usize(last).is_some(),
            "Offset {last} does not fit in the Arrow offset type {}",
            O::PTYPE
        );
        offsets.reinterpret_cast(O::PTYPE)
    } else {
        offsets
            .into_array()
            .cast(DType::Primitive(O::PTYPE, Nullability::NonNullable))?
            .execute::<PrimitiveArray>(ctx)?
    };

    Ok(offsets.to_buffer::<O>().into_arrow_offset_buffer())
}

/// Reads the last offset, which bounds every other offset since offsets are non-decreasing.
fn last_offset(offsets: &PrimitiveArray) -> usize {
    match_each_integer_ptype!(offsets.ptype(), |O| {
        offsets.to_buffer::<O>().last().map_or(0, |last| last.as_())
    })
}

/// Checks that every value spanned by `offsets` is valid UTF-8, null slots included.
///
/// Vortex only requires non-null values to be UTF-8, but [`GenericByteArray::value`] reinterprets
/// any slot's bytes unchecked, so skipping null slots would hand out a malformed `&str`.
///
/// Every byte the offsets span belongs to some slot, so one pass validates all of them. The offsets
/// then only have to land on character boundaries, which stops `value(i)` from slicing a multi-byte
/// character in half.
fn validate_values_utf8<T: ByteArrayType>(
    offsets: &OffsetBuffer<T::Offset>,
    values: &ArrowBuffer,
) -> VortexResult<()> {
    let (Some(first), Some(last)) = (offsets.first(), offsets.last()) else {
        return Ok(());
    };
    let start = first.as_usize();
    let end = last.as_usize();
    let bytes = values
        .get(start..end)
        .ok_or_else(|| vortex_err!("Offsets {start}..{end} are out of bounds"))?;
    let validated =
        utf8_from_bytes(bytes).map_err(|err| vortex_err!("Encountered non UTF-8 data: {err}"))?;
    for (index, offset) in offsets.iter().enumerate() {
        let boundary = offset
            .as_usize()
            .checked_sub(start)
            .filter(|boundary| validated.is_char_boundary(*boundary));
        vortex_ensure!(
            boundary.is_some(),
            "Offset {} at index {index} does not fall on a UTF-8 character boundary",
            offset.as_usize()
        );
    }
    Ok(())
}

/// Validates `bytes` as UTF-8, re-running the slower checker only to describe a failure.
fn utf8_from_bytes(bytes: &[u8]) -> Result<&str, simdutf8::compat::Utf8Error> {
    simdutf8::basic::from_utf8(bytes).map_err(|_| {
        #[expect(clippy::unwrap_used)]
        // Run validation again using the `compat` module to get a more detailed error message.
        simdutf8::compat::from_utf8(bytes).unwrap_err()
    })
}

#[cfg(test)]
mod tests {
    use arrow_array::Array;
    use arrow_array::cast::AsArray;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar_fn::EmptyOptions;
    use vortex_array::scalar_fn::fns::mask::Mask as MaskFn;
    use vortex_array::validity::Validity;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use crate::ArrowSessionExt;

    #[test]
    fn mask_wrapped_varbin_exports() -> VortexResult<()> {
        let session = array_session();
        let mut ctx = session.create_execution_ctx();

        let varbin = VarBinArray::from_vec(
            vec!["hello", "world", "vortex"],
            DType::Utf8(Nullability::NonNullable),
        );
        let mask = BoolArray::from_iter([true, false, true]);
        let masked =
            MaskFn.try_new_array(3, EmptyOptions, [varbin.into_array(), mask.into_array()])?;

        let field = Field::new("s", DataType::Utf8, true);
        let arrow = session
            .arrow()
            .execute_arrow(masked, Some(&field), &mut ctx)?;

        let strings = arrow.as_string::<i32>();
        assert_eq!(strings.len(), 3);
        assert!(!strings.is_null(0));
        assert!(strings.is_null(1));
        assert_eq!(strings.value(0), "hello");
        assert_eq!(strings.value(2), "vortex");
        Ok(())
    }

    fn make_utf8_array() -> VarBinViewArray {
        VarBinViewArray::from_iter_str(["hello", "world", "this is a longer string for testing"])
    }

    fn make_binary_array() -> VarBinViewArray {
        VarBinViewArray::from_iter_bin([
            b"hello".as_slice(),
            b"world".as_slice(),
            b"this is a longer string for testing".as_slice(),
        ])
    }

    #[rstest]
    // Utf8 source can convert to all string types and binary types
    #[case::utf8_to_binary(make_utf8_array(), DataType::Binary)]
    #[case::utf8_to_large_binary(make_utf8_array(), DataType::LargeBinary)]
    #[case::utf8_to_utf8(make_utf8_array(), DataType::Utf8)]
    #[case::utf8_to_large_utf8(make_utf8_array(), DataType::LargeUtf8)]
    #[case::utf8_to_utf8_view(make_utf8_array(), DataType::Utf8View)]
    // Binary source can convert to all binary types and string types
    #[case::binary_to_binary(make_binary_array(), DataType::Binary)]
    #[case::binary_to_large_binary(make_binary_array(), DataType::LargeBinary)]
    #[case::binary_to_utf8(make_binary_array(), DataType::Utf8)]
    #[case::binary_to_large_utf8(make_binary_array(), DataType::LargeUtf8)]
    #[case::binary_to_binary_view(make_binary_array(), DataType::BinaryView)]
    // Note: utf8 → binary_view and binary → utf8_view require Vortex cast kernels that don't exist
    fn test_vortex_string_binary_to_arrow(
        #[case] vortex_array: VarBinViewArray,
        #[case] target_dtype: DataType,
    ) {
        let session = array_session();
        let mut ctx = session.create_execution_ctx();
        let field = Field::new("test_field", target_dtype.clone(), true);
        let arrow = session
            .arrow()
            .execute_arrow(vortex_array.into_array(), Some(&field), &mut ctx)
            .unwrap();

        assert_eq!(arrow.data_type(), &target_dtype);
        assert_eq!(arrow.len(), 3);
        assert_eq!(arrow.null_count(), 0);

        // Verify the actual values by converting back to bytes
        let expected: Vec<&[u8]> = vec![b"hello", b"world", b"this is a longer string for testing"];

        for (i, expected_bytes) in expected.iter().enumerate() {
            let actual_bytes: &[u8] = match &target_dtype {
                DataType::Binary => arrow.as_binary::<i32>().value(i),
                DataType::LargeBinary => arrow.as_binary::<i64>().value(i),
                DataType::Utf8 => arrow.as_string::<i32>().value(i).as_bytes(),
                DataType::LargeUtf8 => arrow.as_string::<i64>().value(i).as_bytes(),
                DataType::BinaryView => arrow.as_binary_view().value(i),
                DataType::Utf8View => arrow.as_string_view().value(i).as_bytes(),
                _ => unreachable!(),
            };
            assert_eq!(actual_bytes, *expected_bytes, "Mismatch at index {i}");
        }
    }

    #[rstest]
    #[case::utf8_to_binary(DType::Utf8(Nullability::Nullable), DataType::Binary)]
    #[case::utf8_to_large_binary(DType::Utf8(Nullability::Nullable), DataType::LargeBinary)]
    #[case::utf8_to_utf8(DType::Utf8(Nullability::Nullable), DataType::Utf8)]
    #[case::utf8_to_large_utf8(DType::Utf8(Nullability::Nullable), DataType::LargeUtf8)]
    #[case::utf8_to_utf8_view(DType::Utf8(Nullability::Nullable), DataType::Utf8View)]
    #[case::binary_to_binary(DType::Binary(Nullability::Nullable), DataType::Binary)]
    #[case::binary_to_large_binary(DType::Binary(Nullability::Nullable), DataType::LargeBinary)]
    #[case::binary_to_utf8(DType::Binary(Nullability::Nullable), DataType::Utf8)]
    #[case::binary_to_large_utf8(DType::Binary(Nullability::Nullable), DataType::LargeUtf8)]
    #[case::binary_to_binary_view(DType::Binary(Nullability::Nullable), DataType::BinaryView)]
    fn test_nullable_vortex_string_binary_to_arrow(
        #[case] vortex_dtype: DType,
        #[case] target_dtype: DataType,
    ) {
        let vortex_array = VarBinViewArray::from_iter(
            [Some(b"hello".as_slice()), None, Some(b"world".as_slice())],
            vortex_dtype,
        );

        let session = array_session();
        let mut ctx = session.create_execution_ctx();
        let field = Field::new("test_field", target_dtype.clone(), true);
        let arrow = session
            .arrow()
            .execute_arrow(vortex_array.into_array(), Some(&field), &mut ctx)
            .unwrap();

        assert_eq!(arrow.data_type(), &target_dtype);
        assert_eq!(arrow.len(), 3);
        assert_eq!(arrow.null_count(), 1);
        assert!(!arrow.is_null(0));
        assert!(arrow.is_null(1));
        assert!(!arrow.is_null(2));
    }

    #[rstest]
    #[case(DataType::Utf8)]
    #[case(DataType::LargeUtf8)]
    fn binary_with_invalid_utf8_to_string_returns_error(#[case] target_dtype: DataType) {
        let vortex_array = VarBinViewArray::from_iter_bin([
            b"hello".as_slice(),
            b"\xff\xfe invalid utf8".as_slice(),
        ]);

        let session = array_session();
        let mut ctx = session.create_execution_ctx();
        let field = Field::new("test_field", target_dtype, true);
        let result =
            session
                .arrow()
                .execute_arrow(vortex_array.into_array(), Some(&field), &mut ctx);

        assert!(result.is_err());
    }

    /// The whole values buffer is valid UTF-8, but each individual value slices a two-byte
    /// character in half. Arrow's `value(i)` reinterprets the bytes unchecked, so the export has
    /// to reject this rather than hand out a malformed `&str`.
    #[rstest]
    #[case(DataType::Utf8)]
    #[case(DataType::LargeUtf8)]
    fn binary_to_string_rejects_offsets_splitting_a_character(#[case] target_dtype: DataType) {
        let array = VarBinArray::new(
            buffer![0i32, 1, 2].into_array(),
            ByteBuffer::copy_from("é".as_bytes()),
            DType::Binary(Nullability::NonNullable),
            Validity::NonNullable,
        );

        let session = array_session();
        let mut ctx = session.create_execution_ctx();
        let field = Field::new("test_field", target_dtype, false);
        let result = session
            .arrow()
            .execute_arrow(array.into_array(), Some(&field), &mut ctx);

        assert!(result.is_err());
    }

    /// Multi-byte characters that do line up with the offsets must still export.
    #[rstest]
    #[case(DataType::Utf8)]
    #[case(DataType::LargeUtf8)]
    fn binary_to_string_accepts_multibyte_characters(
        #[case] target_dtype: DataType,
    ) -> VortexResult<()> {
        let array = VarBinArray::new(
            buffer![0i32, 2, 5].into_array(),
            ByteBuffer::copy_from("é☃".as_bytes()),
            DType::Binary(Nullability::NonNullable),
            Validity::NonNullable,
        );

        let session = array_session();
        let mut ctx = session.create_execution_ctx();
        let field = Field::new("test_field", target_dtype.clone(), false);
        let arrow = session
            .arrow()
            .execute_arrow(array.into_array(), Some(&field), &mut ctx)?;

        let values: Vec<&str> = match target_dtype {
            DataType::Utf8 => (0..2).map(|i| arrow.as_string::<i32>().value(i)).collect(),
            _ => (0..2).map(|i| arrow.as_string::<i64>().value(i)).collect(),
        };
        assert_eq!(values, ["é", "☃"]);
        Ok(())
    }

    /// Arrow's `value(i)` reinterprets a null slot's bytes just like any other slot's, so a
    /// `Binary` to `Utf8` export has to reject invalid bytes even where the slot is null.
    #[rstest]
    #[case(DataType::Utf8)]
    #[case(DataType::LargeUtf8)]
    fn varbin_binary_to_string_validates_null_slots(#[case] target_dtype: DataType) {
        // Slot 1 is null and its offsets span two bytes that are not valid UTF-8.
        let array = VarBinArray::new(
            buffer![0i32, 5, 7, 12].into_array(),
            ByteBuffer::copy_from(b"hello\xff\xfeworld".as_slice()),
            DType::Binary(Nullability::Nullable),
            Validity::from_mask(Mask::from_iter([true, false, true]), Nullability::Nullable),
        );

        let session = array_session();
        let mut ctx = session.create_execution_ctx();
        let field = Field::new("test_field", target_dtype, true);
        let result = session
            .arrow()
            .execute_arrow(array.into_array(), Some(&field), &mut ctx);

        assert!(result.is_err());
    }

    /// The `VarBin` fast path hands Arrow the untrimmed bytes buffer, so a `Binary` to `Utf8`
    /// export must not validate bytes that lie outside the range the offsets span.
    #[rstest]
    #[case(DataType::Utf8)]
    #[case(DataType::LargeUtf8)]
    fn varbin_binary_to_string_ignores_bytes_outside_offsets(
        #[case] target_dtype: DataType,
    ) -> VortexResult<()> {
        let array = VarBinArray::new(
            buffer![0i32, 5, 10].into_array(),
            ByteBuffer::copy_from(b"helloworld\xff\xfe".as_slice()),
            DType::Binary(Nullability::NonNullable),
            Validity::NonNullable,
        );

        let session = array_session();
        let mut ctx = session.create_execution_ctx();
        let field = Field::new("test_field", target_dtype.clone(), false);
        let arrow = session
            .arrow()
            .execute_arrow(array.into_array(), Some(&field), &mut ctx)?;

        let values: Vec<&str> = match target_dtype {
            DataType::Utf8 => (0..2).map(|i| arrow.as_string::<i32>().value(i)).collect(),
            _ => (0..2).map(|i| arrow.as_string::<i64>().value(i)).collect(),
        };
        assert_eq!(values, ["hello", "world"]);
        Ok(())
    }

    /// Unsigned `VarBin` offsets are reinterpreted rather than cast, so the values must survive it.
    #[rstest]
    #[case::utf8(DataType::Utf8)]
    #[case::large_utf8(DataType::LargeUtf8)]
    #[case::binary(DataType::Binary)]
    #[case::large_binary(DataType::LargeBinary)]
    fn unsigned_varbin_offsets_export(#[case] target_dtype: DataType) -> VortexResult<()> {
        let array = VarBinArray::new(
            buffer![0u32, 5, 5, 10].into_array(),
            ByteBuffer::copy_from(b"helloworld".as_slice()),
            DType::Utf8(Nullability::Nullable),
            Validity::from_mask(Mask::from_iter([true, false, true]), Nullability::Nullable),
        );

        let session = array_session();
        let mut ctx = session.create_execution_ctx();
        let field = Field::new("test_field", target_dtype.clone(), true);
        let arrow = session
            .arrow()
            .execute_arrow(array.into_array(), Some(&field), &mut ctx)?;

        assert_eq!(arrow.len(), 3);
        assert!(arrow.is_null(1));
        let values: Vec<&[u8]> = match target_dtype {
            DataType::Utf8 => vec![
                arrow.as_string::<i32>().value(0).as_bytes(),
                arrow.as_string::<i32>().value(2).as_bytes(),
            ],
            DataType::LargeUtf8 => vec![
                arrow.as_string::<i64>().value(0).as_bytes(),
                arrow.as_string::<i64>().value(2).as_bytes(),
            ],
            DataType::Binary => vec![
                arrow.as_binary::<i32>().value(0),
                arrow.as_binary::<i32>().value(2),
            ],
            _ => vec![
                arrow.as_binary::<i64>().value(0),
                arrow.as_binary::<i64>().value(2),
            ],
        };
        assert_eq!(values, [b"hello".as_slice(), b"world".as_slice()]);
        Ok(())
    }

    #[rstest]
    #[case(DataType::Utf8)]
    #[case(DataType::LargeUtf8)]
    #[case(DataType::Binary)]
    #[case(DataType::LargeBinary)]
    fn incompatible_vortex_dtype_returns_error(#[case] target_dtype: DataType) {
        let session = array_session();
        let mut ctx = session.create_execution_ctx();
        let field = Field::new("test_field", target_dtype, true);
        let result = session.arrow().execute_arrow(
            PrimitiveArray::from_iter([1i32, 2, 3]).into_array(),
            Some(&field),
            &mut ctx,
        );

        assert!(result.is_err());
    }

    #[test]
    fn filtered_utf8_view_export_does_not_retain_unselected_buffers() -> VortexResult<()> {
        let unselected = "x".repeat(1 << 20);
        let array =
            VarBinViewArray::from_iter_str(["selected", unselected.as_str(), unselected.as_str()]);
        let filtered = array
            .into_array()
            .filter(Mask::from_iter([true, false, false]))?;

        let session = array_session();
        let mut ctx = session.create_execution_ctx();
        let arrow = session
            .arrow()
            .execute_arrow(filtered.into_array(), None, &mut ctx)?;

        assert_eq!(arrow.as_string_view().value(0), "selected");
        assert!(
            arrow.get_array_memory_size() < unselected.len(),
            "filtered export retained unselected payload: {} bytes",
            arrow.get_array_memory_size()
        );
        Ok(())
    }
}
