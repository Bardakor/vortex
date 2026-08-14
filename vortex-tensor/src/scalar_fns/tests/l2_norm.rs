// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::EmptyMetadata;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::MaskedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayPlugin;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;

use crate::encodings::normalized::Normalized;
use crate::scalar_fns::l2_norm::L2Norm;
use crate::tests::SESSION;
use crate::types::vector::Vector;
use crate::utils::test_helpers::assert_close;
use crate::utils::test_helpers::literal_vector_array;
use crate::utils::test_helpers::tensor_array;
use crate::utils::test_helpers::vector_array;
use crate::utils::test_helpers::zero_width_vector_array;

/// Evaluates L2 norm on a tensor/vector array and returns the result as `Vec<f64>`.
fn eval_l2_norm(input: ArrayRef) -> VortexResult<Vec<f64>> {
    let scalar_fn = L2Norm.bind(EmptyOptions);
    let result = ScalarFnArray::try_new(scalar_fn, vec![input])?;
    let mut ctx = SESSION.create_execution_ctx();
    let prim: PrimitiveArray = result.into_array().execute(&mut ctx)?;
    Ok(prim.as_slice::<f64>().to_vec())
}

#[test]
fn inherent_constructors_remain_available() -> VortexResult<()> {
    let _scalar_fn = L2Norm::new();
    let array = L2Norm::try_new_array(tensor_array(&[1], &[3.0])?)?;

    assert_eq!(array.len(), 1);
    Ok(())
}

#[test]
fn zero_width_and_empty_inputs() -> VortexResult<()> {
    assert_close(
        &eval_l2_norm(zero_width_vector_array::<f64>(3)?)?,
        &[0.0, 0.0, 0.0],
    );
    assert!(eval_l2_norm(vector_array(2, &[] as &[f64])?)?.is_empty());

    let constant = Vector::constant_array::<f64>(&[], 3)?;
    assert_close(&eval_l2_norm(constant)?, &[0.0, 0.0, 0.0]);
    Ok(())
}

#[rstest]
#[case::three_four_five(&[2], &[3.0, 4.0], &[5.0])]
#[case::zero_vector(&[3], &[0.0, 0.0, 0.0], &[0.0])]
#[case::single_element(&[1], &[7.0], &[7.0])]
#[case::negative_elements(&[2], &[-3.0, -4.0], &[5.0])]
fn known_norms(
    #[case] shape: &[usize],
    #[case] elements: &[f64],
    #[case] expected: &[f64],
) -> VortexResult<()> {
    let arr = tensor_array(shape, elements)?;
    assert_close(&eval_l2_norm(arr)?, expected);
    Ok(())
}

#[test]
fn multiple_rows() -> VortexResult<()> {
    let arr = tensor_array(
        &[3],
        &[
            3.0, 4.0, 0.0, // norm = 5.0
            0.0, 0.0, 0.0, // norm = 0.0
            1.0, 1.0, 1.0, // norm = sqrt(3)
        ],
    )?;
    assert_close(&eval_l2_norm(arr)?, &[5.0, 0.0, 3.0_f64.sqrt()]);
    Ok(())
}

#[test]
fn vector_multiple_rows() -> VortexResult<()> {
    let arr = vector_array(
        3,
        &[
            1.0, 0.0, 0.0, // norm = 1.0
            3.0, 4.0, 0.0, // norm = 5.0
        ],
    )?;
    assert_close(&eval_l2_norm(arr)?, &[1.0, 5.0]);
    Ok(())
}

#[test]
fn null_input_row() -> VortexResult<()> {
    // 2 rows of dim-2 vectors. Row 1 is masked as null.
    let arr = tensor_array(&[2], &[3.0, 4.0, 0.0, 0.0])?;
    let arr = MaskedArray::try_new(arr, Validity::from_iter([true, false]))?.into_array();

    let scalar_fn = L2Norm.bind(EmptyOptions);
    let result = ScalarFnArray::try_new(scalar_fn, vec![arr])?;
    let mut ctx = SESSION.create_execution_ctx();
    let prim: PrimitiveArray = result.into_array().execute(&mut ctx)?;

    // Row 0: norm = 5.0, row 1: null.
    assert!(prim.is_valid(0, &mut ctx)?);
    assert!(!prim.is_valid(1, &mut ctx)?);
    assert_close(&[prim.as_slice::<f64>()[0]], &[5.0]);
    Ok(())
}

/// A constant input whose scalar is a non-null tensor should short-circuit to a
/// [`ConstantArray`] output whose scalar is the precomputed norm. Uses [`execute_until`] so
/// execution stops at the [`Constant`] encoding instead of canonicalizing into a
/// [`PrimitiveArray`].
#[test]
fn constant_non_null_input_yields_constant_output() -> VortexResult<()> {
    let input = literal_vector_array(&[3.0f64, 4.0], 4);

    let scalar_fn = L2Norm.bind(EmptyOptions);
    let result = ScalarFnArray::try_new(scalar_fn, vec![input])?.into_array();
    let mut ctx = SESSION.create_execution_ctx();
    let output = result.execute_until::<Constant>(&mut ctx)?;

    let constant = output
        .as_opt::<Constant>()
        .expect("L2Norm over a constant input must produce a constant output");
    assert_eq!(constant.len(), 4);
    let norm = constant
        .scalar()
        .as_primitive()
        .as_::<f64>()
        .expect("norm scalar must be a non-null primitive");
    assert_close(&[norm], &[5.0]);
    Ok(())
}

/// An extension array over constant storage is folded just like a top-level constant instead of
/// recomputing the same norm once per row.
#[test]
fn extension_backed_constant_yields_constant_output() -> VortexResult<()> {
    let input = Vector::constant_array(&[3.0f64, 4.0], 4)?;

    let scalar_fn = L2Norm.bind(EmptyOptions);
    let result = ScalarFnArray::try_new(scalar_fn, vec![input])?.into_array();
    let mut ctx = SESSION.create_execution_ctx();
    let output = result.execute_until::<Constant>(&mut ctx)?;

    let constant = output
        .as_opt::<Constant>()
        .expect("L2Norm over constant-backed extension storage must produce a constant output");
    assert_eq!(constant.len(), 4);
    let norm = constant
        .scalar()
        .as_primitive()
        .as_::<f64>()
        .expect("norm scalar must be a non-null primitive");
    assert_close(&[norm], &[5.0]);
    Ok(())
}

/// A constant input whose scalar is null should short-circuit to a null [`ConstantArray`] of
/// the correct primitive dtype and length.
#[test]
fn constant_null_input_yields_null_constant_output() -> VortexResult<()> {
    let storage_dtype = DType::FixedSizeList(
        DType::Primitive(PType::F64, Nullability::NonNullable).into(),
        2,
        Nullability::Nullable,
    );
    let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, storage_dtype)?.erased();
    let null_scalar = Scalar::null(DType::Extension(ext_dtype));
    let input = ConstantArray::new(null_scalar, 3).into_array();

    let scalar_fn = L2Norm.bind(EmptyOptions);
    let result = ScalarFnArray::try_new(scalar_fn, vec![input])?.into_array();
    let mut ctx = SESSION.create_execution_ctx();
    let output = result.execute_until::<Constant>(&mut ctx)?;

    let constant = output
        .as_opt::<Constant>()
        .expect("null constant input must produce a constant output");
    assert_eq!(constant.len(), 3);
    assert!(constant.scalar().is_null());
    assert_eq!(
        constant.dtype(),
        &DType::Primitive(PType::F64, Nullability::Nullable)
    );
    Ok(())
}

/// An `f32` column must dispatch at `f32` and produce an `f32` result, which is the property that
/// makes width polymorphism load-bearing rather than decorative.
#[rstest]
#[case::f32(&[3.0f32, 4.0], PType::F32)]
#[case::f64(&[3.0f64, 4.0], PType::F64)]
fn dispatches_at_input_width<T: vortex_array::dtype::NativePType>(
    #[case] elements: &[T],
    #[case] expected: PType,
) -> VortexResult<()> {
    let arr = tensor_array(&[2], elements)?;
    let mut ctx = SESSION.create_execution_ctx();
    let prim: PrimitiveArray = L2Norm
        .try_new_array(arr.len(), EmptyOptions, [arr])?
        .execute(&mut ctx)?;
    assert_eq!(prim.ptype(), expected);
    Ok(())
}

/// `L2Norm(Normalized(normalized, norms))` reads back the authoritative stored norms rather than
/// recomputing over decoded coordinates. The normalized child here is deliberately *not*
/// unit-norm, mimicking lossy storage, so readthrough and recompute disagree: row 0 decodes to
/// `[6, 8]` (norm `10`) and row 1 to `[6, 0]` (norm `6`), while the stored norms are `5` and `2`.
#[test]
fn normalized_readthrough_returns_stored_norms() -> VortexResult<()> {
    let normalized = tensor_array(&[2], &[1.2, 1.6, 3.0, 0.0])?;
    let norms = PrimitiveArray::from_iter([5.0f64, 2.0]).into_array();
    // SAFETY: A focused test of the lossy storage contract: the stored norms are authoritative
    // even though this normalized child violates the unit-norm invariant.
    let denorm =
        unsafe { Normalized::new_unchecked(normalized, norms, Validity::NonNullable) }.into_array();

    assert_close(&eval_l2_norm(denorm)?, &[5.0, 2.0]);
    Ok(())
}

/// The readthrough must survive a partially-null column.
///
/// This pins the dense policy the row contract derives. Filtering could hand `reduce_encoded` a
/// filtered input, which is no longer an `ExactScalarFn<Normalized>`, silently falling back to
/// decode-and-recompute. For a lossy child that changes the answer: row 0 below would come back as
/// `10` (recomputed from `[6, 8]`) instead of the authoritative stored `5`.
#[test]
fn normalized_readthrough_survives_null_rows() -> VortexResult<()> {
    let normalized = tensor_array(&[2], &[1.2, 1.6, 3.0, 0.0])?;
    let norms = PrimitiveArray::from_iter([5.0f64, 2.0]).into_array();
    // SAFETY: Intentionally lossy, as in `normalized_readthrough_returns_stored_norms`, so that
    // a recompute fallback is observable.
    let denorm =
        unsafe { Normalized::new_unchecked(normalized, norms, Validity::from_iter([true, false])) }
            .into_array();

    let scalar_fn = L2Norm.bind(EmptyOptions);
    let result = ScalarFnArray::try_new(scalar_fn, vec![denorm])?;
    let mut ctx = SESSION.create_execution_ctx();
    let prim: PrimitiveArray = result.into_array().execute(&mut ctx)?;

    assert!(prim.is_valid(0, &mut ctx)?);
    assert!(!prim.is_valid(1, &mut ctx)?);
    assert_close(&[prim.as_slice::<f64>()[0]], &[5.0]);
    Ok(())
}

/// The readthrough must still propagate validity carried by the `Normalized` parent.
#[test]
fn normalized_readthrough_propagates_parent_validity() -> VortexResult<()> {
    let normalized = tensor_array(&[2], &[0.6, 0.8, 1.0, 0.0])?;
    let norms = PrimitiveArray::from_iter([5.0f64, 1.0]).into_array();
    let mut ctx = SESSION.create_execution_ctx();
    let denorm = Normalized::try_new(
        normalized,
        norms,
        Validity::from_iter([true, false]),
        &mut ctx,
    )?
    .into_array();

    let scalar_fn = L2Norm.bind(EmptyOptions);
    let result = ScalarFnArray::try_new(scalar_fn, vec![denorm])?;
    let prim: PrimitiveArray = result.into_array().execute(&mut ctx)?;

    assert!(prim.is_valid(0, &mut ctx)?);
    assert!(!prim.is_valid(1, &mut ctx)?);
    assert_close(&[prim.as_slice::<f64>()[0]], &[5.0]);
    Ok(())
}

#[rstest]
#[case::fixed_shape_tensor(l2_norm_tensor_child())]
#[case::vector(l2_norm_vector_child())]
fn serde_round_trip(#[case] child: ArrayRef) -> VortexResult<()> {
    let original = L2Norm.try_new_array(child.len(), EmptyOptions, [child.clone()])?;

    let plugin = ScalarFnArrayPlugin::new(L2Norm);
    let metadata = plugin
        .serialize(&original, &SESSION)?
        .expect("L2Norm serialize must produce metadata");

    let children = vec![child];
    let recovered = plugin.deserialize(
        original.dtype(),
        original.len(),
        &metadata,
        &[],
        &children,
        &SESSION,
    )?;

    assert_eq!(recovered.dtype(), original.dtype());
    assert_eq!(recovered.len(), original.len());
    assert_eq!(recovered.encoding_id(), original.encoding_id());
    Ok(())
}

fn l2_norm_tensor_child() -> ArrayRef {
    tensor_array(&[3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("valid tensor array")
}

fn l2_norm_vector_child() -> ArrayRef {
    vector_array(3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("valid vector array")
}
