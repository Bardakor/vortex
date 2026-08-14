// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::MaskedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayPlugin;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;

use crate::encodings::normalized::Normalized;
use crate::scalar_fns::inner_product::InnerProduct;
use crate::tests::SESSION;
use crate::types::vector::Vector;
use crate::utils::test_helpers::assert_close;
use crate::utils::test_helpers::normalized_array;
use crate::utils::test_helpers::tensor_array;
use crate::utils::test_helpers::vector_array;
use crate::utils::test_helpers::zero_width_vector_array;

/// Evaluates inner product between two tensor arrays and returns the result as `Vec<f64>`.
fn eval_inner_product(lhs: ArrayRef, rhs: ArrayRef) -> VortexResult<Vec<f64>> {
    let scalar_fn = InnerProduct.bind(EmptyOptions);
    let result = ScalarFnArray::try_new(scalar_fn, vec![lhs, rhs])?;
    let mut ctx = SESSION.create_execution_ctx();
    let prim: PrimitiveArray = result.into_array().execute(&mut ctx)?;
    Ok(prim.as_slice::<f64>().to_vec())
}

#[test]
fn inherent_constructors_remain_available() -> VortexResult<()> {
    let _scalar_fn = InnerProduct::new();
    let lhs = tensor_array(&[1], &[2.0])?;
    let rhs = tensor_array(&[1], &[3.0])?;
    let array = InnerProduct::try_new_array(lhs, rhs)?;

    assert_eq!(array.len(), 1);
    Ok(())
}

#[test]
fn zero_width_and_empty_inputs() -> VortexResult<()> {
    let lhs = zero_width_vector_array::<f64>(3)?;
    let rhs = zero_width_vector_array::<f64>(3)?;
    assert_close(&eval_inner_product(lhs, rhs)?, &[0.0, 0.0, 0.0]);

    let lhs = vector_array(2, &[] as &[f64])?;
    let rhs = vector_array(2, &[] as &[f64])?;
    assert!(eval_inner_product(lhs, rhs)?.is_empty());

    let lhs = Vector::constant_array::<f64>(&[], 3)?;
    let rhs = zero_width_vector_array::<f64>(3)?;
    assert_close(&eval_inner_product(lhs, rhs)?, &[0.0, 0.0, 0.0]);
    Ok(())
}

/// Single-row inner product for various vector pairs.
#[rstest]
// Orthogonal: [1, 0] . [0, 1] = 0.
#[case::orthogonal(&[2], &[1.0, 0.0], &[0.0, 1.0], &[0.0])]
// Parallel: [3, 4] . [3, 4] = 9 + 16 = 25.
#[case::parallel(&[2], &[3.0, 4.0], &[3.0, 4.0], &[25.0])]
// Antiparallel: [1, 2] . [-1, -2] = -1 + -4 = -5.
#[case::antiparallel(&[2], &[1.0, 2.0], &[-1.0, -2.0], &[-5.0])]
// Scaled: [2, 0] . [3, 0] = 6.
#[case::scaled(&[2], &[2.0, 0.0], &[3.0, 0.0], &[6.0])]
fn single_row(
    #[case] shape: &[usize],
    #[case] lhs_elems: &[f64],
    #[case] rhs_elems: &[f64],
    #[case] expected: &[f64],
) -> VortexResult<()> {
    let lhs = tensor_array(shape, lhs_elems)?;
    let rhs = tensor_array(shape, rhs_elems)?;
    assert_close(&eval_inner_product(lhs, rhs)?, expected);
    Ok(())
}

#[test]
fn multiple_rows() -> VortexResult<()> {
    let lhs = tensor_array(
        &[3],
        &[
            1.0, 0.0, 0.0, // tensor 0
            3.0, 4.0, 0.0, // tensor 1
            1.0, 1.0, 1.0, // tensor 2
        ],
    )?;
    let rhs = tensor_array(
        &[3],
        &[
            0.0, 1.0, 0.0, // tensor 0: dot = 0
            3.0, 4.0, 0.0, // tensor 1: dot = 25
            2.0, 2.0, 2.0, // tensor 2: dot = 6
        ],
    )?;
    assert_close(&eval_inner_product(lhs, rhs)?, &[0.0, 25.0, 6.0]);
    Ok(())
}

#[test]
fn vector_inner_product() -> VortexResult<()> {
    let lhs = vector_array(
        2,
        &[
            3.0, 4.0, // vector 0
            1.0, 0.0, // vector 1
        ],
    )?;
    let rhs = vector_array(
        2,
        &[
            3.0, 4.0, // vector 0: dot = 25
            0.0, 1.0, // vector 1: dot = 0
        ],
    )?;
    assert_close(&eval_inner_product(lhs, rhs)?, &[25.0, 0.0]);
    Ok(())
}

#[test]
fn null_input_row() -> VortexResult<()> {
    // 3 rows of dim-2 vectors. Row 1 of lhs is masked as null.
    let lhs = tensor_array(&[2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let rhs = tensor_array(&[2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0])?;
    let lhs = MaskedArray::try_new(lhs, Validity::from_iter([true, false, true]))?.into_array();

    let scalar_fn = InnerProduct.bind(EmptyOptions);
    let result = ScalarFnArray::try_new(scalar_fn, vec![lhs, rhs])?;
    let mut ctx = SESSION.create_execution_ctx();
    let prim: PrimitiveArray = result.into_array().execute(&mut ctx)?;

    // Row 0: 1*7 + 2*8 = 23, row 1: null, row 2: 5*11 + 6*12 = 127.
    assert!(prim.is_valid(0, &mut ctx)?);
    assert!(!prim.is_valid(1, &mut ctx)?);
    assert!(prim.is_valid(2, &mut ctx)?);
    assert_close(&[prim.as_slice::<f64>()[0]], &[23.0]);
    assert_close(&[prim.as_slice::<f64>()[2]], &[127.0]);
    Ok(())
}

#[test]
fn rejects_non_extension_dtype() {
    let lhs = PrimitiveArray::from_iter([1.0_f64, 2.0]).into_array();
    let rhs = PrimitiveArray::from_iter([3.0_f64, 4.0]).into_array();
    let result = InnerProduct.try_new_array(lhs.len(), EmptyOptions, [lhs, rhs]);
    assert!(result.is_err());
}

#[test]
fn rejects_mismatched_dtypes() -> VortexResult<()> {
    let lhs = tensor_array(&[2], &[1.0_f64, 2.0])?;
    let rhs = vector_array(2, &[3.0_f64, 4.0])?;
    let result = InnerProduct.try_new_array(lhs.len(), EmptyOptions, [lhs, rhs]);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn both_normalized() -> VortexResult<()> {
    // LHS: [3.0, 4.0] = Normalized([0.6, 0.8], 5.0).
    // RHS: [1.0, 0.0] = Normalized([1.0, 0.0], 1.0).
    // dot([3.0, 4.0], [1.0, 0.0]) = 3.0.
    let mut ctx = SESSION.create_execution_ctx();
    let lhs = normalized_array(&[2], &[0.6, 0.8], &[5.0], &mut ctx)?;
    let rhs = normalized_array(&[2], &[1.0, 0.0], &[1.0], &mut ctx)?;

    // Expected: 5.0 * 1.0 * dot([0.6, 0.8], [1.0, 0.0]) = 5.0 * 0.6 = 3.0.
    assert_close(&eval_inner_product(lhs, rhs)?, &[3.0]);
    Ok(())
}

#[test]
fn both_normalized_multiple_rows() -> VortexResult<()> {
    // Row 0: [3.0, 4.0] dot [3.0, 4.0] = 25.0.
    // Row 1: [1.0, 0.0] dot [0.0, 1.0] = 0.0.
    let mut ctx = SESSION.create_execution_ctx();
    let lhs = normalized_array(&[2], &[0.6, 0.8, 1.0, 0.0], &[5.0, 1.0], &mut ctx)?;
    let rhs = normalized_array(&[2], &[0.6, 0.8, 0.0, 1.0], &[5.0, 1.0], &mut ctx)?;

    assert_close(&eval_inner_product(lhs, rhs)?, &[25.0, 0.0]);
    Ok(())
}

#[test]
fn one_side_normalized_lhs() -> VortexResult<()> {
    // LHS: Normalized([0.6, 0.8], 5.0) representing [3.0, 4.0].
    // RHS: plain [1.0, 2.0].
    // dot([3.0, 4.0], [1.0, 2.0]) = 3.0 + 8.0 = 11.0.
    let mut ctx = SESSION.create_execution_ctx();
    let lhs = normalized_array(&[2], &[0.6, 0.8], &[5.0], &mut ctx)?;
    let rhs = tensor_array(&[2], &[1.0, 2.0])?;

    assert_close(&eval_inner_product(lhs, rhs)?, &[11.0]);
    Ok(())
}

#[test]
fn one_side_normalized_rhs() -> VortexResult<()> {
    // LHS: plain [1.0, 2.0].
    // RHS: Normalized([0.6, 0.8], 5.0) representing [3.0, 4.0].
    // dot([1.0, 2.0], [3.0, 4.0]) = 3.0 + 8.0 = 11.0.
    let mut ctx = SESSION.create_execution_ctx();
    let lhs = tensor_array(&[2], &[1.0, 2.0])?;
    let rhs = normalized_array(&[2], &[0.6, 0.8], &[5.0], &mut ctx)?;

    assert_close(&eval_inner_product(lhs, rhs)?, &[11.0]);
    Ok(())
}

#[test]
fn both_normalized_null_rows() -> VortexResult<()> {
    let normalized_l = tensor_array(&[2], &[0.6, 0.8, 1.0, 0.0])?;
    let norms_l = PrimitiveArray::from_iter([5.0f64, 1.0]).into_array();
    let mut ctx = SESSION.create_execution_ctx();

    let validity = Validity::from_iter([true, false]);
    let lhs = Normalized::try_new(normalized_l, norms_l, validity, &mut ctx)?.into_array();
    let rhs = normalized_array(&[2], &[0.6, 0.8, 1.0, 0.0], &[5.0, 1.0], &mut ctx)?;

    let scalar_fn = InnerProduct.bind(EmptyOptions);
    let result = ScalarFnArray::try_new(scalar_fn, vec![lhs, rhs])?;
    let prim: PrimitiveArray = result.into_array().execute(&mut ctx)?;

    // Row 0: 5.0 * 5.0 * dot([0.6, 0.8], [0.6, 0.8]) = 25.0, row 1: null.
    assert!(prim.is_valid(0, &mut ctx)?);
    assert!(!prim.is_valid(1, &mut ctx)?);
    assert_close(&[prim.as_slice::<f64>()[0]], &[25.0]);
    Ok(())
}

#[rstest]
#[case::vector(inner_product_vector_lhs(), inner_product_vector_rhs())]
#[case::fixed_shape_tensor(inner_product_tensor_lhs(), inner_product_tensor_rhs())]
fn serde_round_trip(#[case] lhs: ArrayRef, #[case] rhs: ArrayRef) -> VortexResult<()> {
    let original =
        InnerProduct.try_new_array(lhs.len(), EmptyOptions, [lhs.clone(), rhs.clone()])?;

    let plugin = ScalarFnArrayPlugin::new(InnerProduct);
    let metadata = plugin
        .serialize(&original, &SESSION)?
        .expect("InnerProduct serialize must produce metadata");

    let children = vec![lhs, rhs];
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

fn inner_product_vector_lhs() -> ArrayRef {
    vector_array(3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("valid vector array")
}

fn inner_product_vector_rhs() -> ArrayRef {
    vector_array(3, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).expect("valid vector array")
}

fn inner_product_tensor_lhs() -> ArrayRef {
    tensor_array(&[2], &[1.0, 2.0, 3.0, 4.0]).expect("valid tensor array")
}

fn inner_product_tensor_rhs() -> ArrayRef {
    tensor_array(&[2], &[5.0, 6.0, 7.0, 8.0]).expect("valid tensor array")
}
