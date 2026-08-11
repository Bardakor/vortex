// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::MaskedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayPlugin;
use vortex_array::assert_arrays_eq;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;

use crate::encodings::normalized::Normalized;
use crate::scalar_fns::cosine_similarity::CosineSimilarity;
use crate::scalar_fns::row::probe;
use crate::tests::SESSION;
use crate::types::vector::Vector;
use crate::utils::test_helpers::assert_close;
use crate::utils::test_helpers::constant_tensor_array;
use crate::utils::test_helpers::literal_vector_array;
use crate::utils::test_helpers::normalized_array;
use crate::utils::test_helpers::tensor_array;
use crate::utils::test_helpers::vector_array;
use crate::utils::test_helpers::zero_width_vector_array;

/// Evaluates cosine similarity between two tensor arrays and returns the result as `Vec<f64>`.
fn eval_cosine_similarity(lhs: ArrayRef, rhs: ArrayRef) -> VortexResult<Vec<f64>> {
    let scalar_fn = CosineSimilarity.bind(EmptyOptions);
    let result = ScalarFnArray::try_new(scalar_fn, vec![lhs, rhs])?;
    let mut ctx = SESSION.create_execution_ctx();
    let prim: PrimitiveArray = result.into_array().execute(&mut ctx)?;
    Ok(prim.as_slice::<f64>().to_vec())
}

#[test]
fn inherent_constructors_remain_available() -> VortexResult<()> {
    let _scalar_fn = CosineSimilarity::new();
    let lhs = tensor_array(&[1], &[2.0])?;
    let rhs = tensor_array(&[1], &[3.0])?;
    let array = CosineSimilarity::try_new_array(lhs, rhs)?;

    assert_eq!(array.len(), 1);
    Ok(())
}

#[test]
fn zero_width_and_empty_inputs() -> VortexResult<()> {
    let lhs = zero_width_vector_array::<f64>(3)?;
    let rhs = zero_width_vector_array::<f64>(3)?;
    assert_close(&eval_cosine_similarity(lhs, rhs)?, &[0.0, 0.0, 0.0]);

    let lhs = vector_array(2, &[] as &[f64])?;
    let rhs = vector_array(2, &[] as &[f64])?;
    assert!(eval_cosine_similarity(lhs, rhs)?.is_empty());

    let lhs = Vector::constant_array::<f64>(&[], 3)?;
    let rhs = zero_width_vector_array::<f64>(3)?;
    assert_close(&eval_cosine_similarity(lhs, rhs)?, &[0.0, 0.0, 0.0]);
    Ok(())
}

/// Like [`eval_cosine_similarity`], but returns the executed array for exact array comparisons.
fn eval_cosine_similarity_array(
    lhs: ArrayRef,
    rhs: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let scalar_fn = CosineSimilarity.bind(EmptyOptions);
    let result = ScalarFnArray::try_new(scalar_fn, vec![lhs, rhs])?;
    Ok(result
        .into_array()
        .execute::<PrimitiveArray>(ctx)?
        .into_array())
}

#[test]
fn unit_vectors_1d() -> VortexResult<()> {
    let lhs = tensor_array(
        &[3],
        &[
            1.0, 0.0, 0.0, // Tensor 1
            0.0, 1.0, 0.0, // Tensor 2
        ],
    )?;
    let rhs = tensor_array(
        &[3],
        &[
            1.0, 0.0, 0.0, // Tensor 1
            1.0, 0.0, 0.0, // Tensor 2
        ],
    )?;

    // Row 0: identical -> 1.0, row 1: orthogonal -> 0.0.
    assert_close(&eval_cosine_similarity(lhs, rhs)?, &[1.0, 0.0]);
    Ok(())
}

/// Single-row cosine similarity for various vector pairs.
#[rstest]
// Antiparallel -> -1.0.
#[case::opposite(&[3], &[1.0, 0.0, 0.0],  &[-1.0, 0.0, 0.0], &[-1.0])]
// dot=24, both magnitudes=5 -> 24/25 = 0.96.
#[case::non_unit(&[2], &[3.0, 4.0],        &[4.0, 3.0],       &[0.96])]
// Zero vector -> guarded to 0.0.
#[case::zero_norm(&[2], &[0.0, 0.0],       &[1.0, 0.0],       &[0.0])]
fn single_row(
    #[case] shape: &[usize],
    #[case] lhs_elems: &[f64],
    #[case] rhs_elems: &[f64],
    #[case] expected: &[f64],
) -> VortexResult<()> {
    let lhs = tensor_array(shape, lhs_elems)?;
    let rhs = tensor_array(shape, rhs_elems)?;
    assert_close(&eval_cosine_similarity(lhs, rhs)?, expected);
    Ok(())
}

/// Self-similarity across various tensor shapes should always produce 1.0.
#[rstest]
// 2x3 matrix, flattened to 6 elements.
#[case::matrix_2d(
    &[2, 3],
    &[
        1.0, 0.0, 0.0, // row 0
        0.0, 0.0, 0.0, // row 1
    ],
)]
// 2x2x2 tensor, 8 elements.
#[case::tensor_3d(&[2, 2, 2], &[1.0; 8])]
fn self_similarity(#[case] shape: &[usize], #[case] elements: &[f64]) -> VortexResult<()> {
    let lhs = tensor_array(shape, elements)?;
    let rhs = tensor_array(shape, elements)?;
    assert_close(&eval_cosine_similarity(lhs, rhs)?, &[1.0]);
    Ok(())
}

#[test]
fn scalar_0d() -> VortexResult<()> {
    // 0-dimensional tensor: each "tensor" is a single scalar value.
    let lhs = tensor_array(&[], &[5.0, 3.0])?;
    let rhs = tensor_array(&[], &[5.0, -3.0])?;

    // Same sign -> 1.0, opposite sign -> -1.0.
    assert_close(&eval_cosine_similarity(lhs, rhs)?, &[1.0, -1.0]);
    Ok(())
}

#[test]
fn many_rows() -> VortexResult<()> {
    // 5 tensors of shape [4] compared against themselves -> all 1.0.
    let lhs = tensor_array(
        &[4],
        &[
            1.0, 2.0, 3.0, 4.0, // tensor 0
            0.0, 1.0, 0.0, 0.0, // tensor 1
            5.0, 0.0, 5.0, 0.0, // tensor 2
            1.0, 1.0, 1.0, 1.0, // tensor 3
            0.0, 0.0, 0.0, 7.0, // tensor 4
        ],
    )?;
    let rhs = lhs.clone();

    assert_close(
        &eval_cosine_similarity(lhs, rhs)?,
        &[1.0, 1.0, 1.0, 1.0, 1.0],
    );
    Ok(())
}

#[test]
fn constant_query_tensor() -> VortexResult<()> {
    // Compare 4 tensors of shape [3] against a single constant query tensor [1,0,0].
    let data = tensor_array(
        &[3],
        &[
            1.0, 0.0, 0.0, // tensor 0
            0.0, 1.0, 0.0, // tensor 1
            0.0, 0.0, 1.0, // tensor 2
            1.0, 0.0, 0.0, // tensor 3
        ],
    )?;
    let query = constant_tensor_array(&[3], &[1.0, 0.0, 0.0], 4)?;

    assert_close(&eval_cosine_similarity(data, query)?, &[1.0, 0.0, 0.0, 1.0]);
    Ok(())
}

#[test]
fn vector_unit_vectors() -> VortexResult<()> {
    let lhs = vector_array(
        3,
        &[
            1.0, 0.0, 0.0, // vector 0
            0.0, 1.0, 0.0, // vector 1
        ],
    )?;
    let rhs = vector_array(
        3,
        &[
            1.0, 0.0, 0.0, // vector 0
            1.0, 0.0, 0.0, // vector 1
        ],
    )?;

    // Row 0: identical -> 1.0, row 1: orthogonal -> 0.0.
    assert_close(&eval_cosine_similarity(lhs, rhs)?, &[1.0, 0.0]);
    Ok(())
}

#[test]
fn vector_constant_query() -> VortexResult<()> {
    let data = vector_array(
        3,
        &[
            1.0, 0.0, 0.0, // vector 0
            0.0, 1.0, 0.0, // vector 1
            0.0, 0.0, 1.0, // vector 2
            1.0, 0.0, 0.0, // vector 3
        ],
    )?;
    let query = Vector::constant_array(&[1.0, 0.0, 0.0], 4)?;

    assert_close(&eval_cosine_similarity(data, query)?, &[1.0, 0.0, 0.0, 1.0]);
    Ok(())
}

#[test]
fn null_input_row() -> VortexResult<()> {
    // 2 rows of dim-2 vectors. Row 1 of rhs is masked as null.
    let lhs = tensor_array(&[2], &[3.0, 4.0, 1.0, 0.0])?;
    let rhs = tensor_array(&[2], &[3.0, 4.0, 0.0, 1.0])?;
    let rhs = MaskedArray::try_new(rhs, Validity::from_iter([true, false]))?.into_array();

    let scalar_fn = CosineSimilarity.bind(EmptyOptions);
    let result = ScalarFnArray::try_new(scalar_fn, vec![lhs, rhs])?;
    let mut ctx = SESSION.create_execution_ctx();
    let prim: PrimitiveArray = result.into_array().execute(&mut ctx)?;

    // Row 0: self-similarity = 1.0, row 1: null.
    assert!(prim.is_valid(0, &mut ctx)?);
    assert!(!prim.is_valid(1, &mut ctx)?);
    assert_close(&[prim.as_slice::<f64>()[0]], &[1.0]);
    Ok(())
}

#[test]
fn both_normalized_self_similarity() -> VortexResult<()> {
    // [3.0, 4.0] has norm 5.0, normalized [0.6, 0.8].
    // [1.0, 0.0] has norm 1.0, normalized [1.0, 0.0].
    let mut ctx = SESSION.create_execution_ctx();
    let lhs = normalized_array(&[2], &[0.6, 0.8, 1.0, 0.0], &[5.0, 1.0], &mut ctx)?;
    let rhs = normalized_array(&[2], &[0.6, 0.8, 1.0, 0.0], &[5.0, 1.0], &mut ctx)?;

    // Self-similarity should always be 1.0.
    assert_close(&eval_cosine_similarity(lhs, rhs)?, &[1.0, 1.0]);
    Ok(())
}

#[test]
fn both_normalized_orthogonal() -> VortexResult<()> {
    // [3.0, 0.0] normalized [1.0, 0.0], norm 3.0.
    // [0.0, 4.0] normalized [0.0, 1.0], norm 4.0.
    let mut ctx = SESSION.create_execution_ctx();
    let lhs = normalized_array(&[2], &[1.0, 0.0], &[3.0], &mut ctx)?;
    let rhs = normalized_array(&[2], &[0.0, 1.0], &[4.0], &mut ctx)?;

    assert_close(&eval_cosine_similarity(lhs, rhs)?, &[0.0]);
    Ok(())
}

#[test]
fn both_normalized_zero_norm() -> VortexResult<()> {
    // Zero-norm row: normalized is [0.0, 0.0], norm is 0.0.
    let mut ctx = SESSION.create_execution_ctx();
    let lhs = normalized_array(&[2], &[0.6, 0.8, 0.0, 0.0], &[5.0, 0.0], &mut ctx)?;
    let rhs = normalized_array(&[2], &[0.6, 0.8, 1.0, 0.0], &[5.0, 1.0], &mut ctx)?;

    // Row 0: dot([0.6, 0.8], [0.6, 0.8]) = 1.0, row 1: dot([0,0], [1,0]) = 0.0.
    assert_close(&eval_cosine_similarity(lhs, rhs)?, &[1.0, 0.0]);
    Ok(())
}

#[test]
fn one_side_normalized_lhs() -> VortexResult<()> {
    // LHS is Normalized([0.6, 0.8], 5.0) representing [3.0, 4.0].
    // RHS is plain [3.0, 4.0].
    // cosine_similarity([3.0, 4.0], [3.0, 4.0]) = 1.0.
    let mut ctx = SESSION.create_execution_ctx();
    let lhs = normalized_array(&[2], &[0.6, 0.8], &[5.0], &mut ctx)?;
    let rhs = tensor_array(&[2], &[3.0, 4.0])?;

    assert_close(&eval_cosine_similarity(lhs, rhs)?, &[1.0]);
    Ok(())
}

#[test]
fn one_side_normalized_rhs() -> VortexResult<()> {
    // LHS is plain [1.0, 0.0], RHS is Normalized([0.6, 0.8], 5.0) representing [3.0, 4.0].
    // cosine_similarity([1.0, 0.0], [3.0, 4.0]) = 3.0 / (1.0 * 5.0) = 0.6.
    let mut ctx = SESSION.create_execution_ctx();
    let lhs = tensor_array(&[2], &[1.0, 0.0])?;
    let rhs = normalized_array(&[2], &[0.6, 0.8], &[5.0], &mut ctx)?;

    assert_close(&eval_cosine_similarity(lhs, rhs)?, &[0.6]);
    Ok(())
}

#[test]
fn both_normalized_null_norms() -> VortexResult<()> {
    // Row 0: valid, row 1: null (via nullable norms on rhs).
    let mut ctx = SESSION.create_execution_ctx();
    let lhs = normalized_array(&[2], &[0.6, 0.8, 1.0, 0.0], &[5.0, 1.0], &mut ctx)?;

    let normalized_r = tensor_array(&[2], &[0.6, 0.8, 1.0, 0.0])?;
    let norms_r = PrimitiveArray::from_option_iter([Some(5.0f64), None]).into_array();
    let rhs = Normalized::try_new(normalized_r, norms_r, &mut ctx)?.into_array();

    let scalar_fn = CosineSimilarity.bind(EmptyOptions);
    let result = ScalarFnArray::try_new(scalar_fn, vec![lhs, rhs])?;
    let prim: PrimitiveArray = result.into_array().execute(&mut ctx)?;

    assert!(prim.is_valid(0, &mut ctx)?);
    assert!(!prim.is_valid(1, &mut ctx)?);
    assert_close(&[prim.as_slice::<f64>()[0]], &[1.0]);
    Ok(())
}

#[test]
fn both_normalized_lossy_zero_stored_norm_returns_zero() -> VortexResult<()> {
    // Mimics a lossy encoding where the stored norm is authoritative but
    // the decoded normalized child is physically nonzero. With a stored norm of `0.0`, cosine
    // similarity for that row must be `0.0` even though the dot product of the normalized
    // children is nonzero.
    let normalized_l = tensor_array(&[2], &[0.6, 0.8])?;
    let norms_l = PrimitiveArray::from_iter([0.0f64]).into_array();
    // SAFETY: This is a focused test that intentionally violates the unit-norm invariant by
    // pairing a nonzero normalized row with a stored norm of `0.0`, mimicking lossy storage.
    let lhs = unsafe { Normalized::new_unchecked(normalized_l, norms_l) }.into_array();

    let normalized_r = tensor_array(&[2], &[0.6, 0.8])?;
    let norms_r = PrimitiveArray::from_iter([0.0f64]).into_array();
    // SAFETY: Same as above for the rhs operand.
    let rhs = unsafe { Normalized::new_unchecked(normalized_r, norms_r) }.into_array();

    // `dot(normalized_l, normalized_r) = 1.0`, but the authoritative stored norms are both
    // `0.0`, so cosine similarity must be `0.0`.
    assert_close(&eval_cosine_similarity(lhs, rhs)?, &[0.0]);
    Ok(())
}

#[test]
fn one_side_normalized_lossy_zero_stored_norm_returns_zero() -> VortexResult<()> {
    // Mimics a lossy encoding where the stored norm is authoritative but
    // the decoded normalized child is physically nonzero. The plain side is a normal nonzero
    // tensor with positive norm. cosine similarity must still be `0.0` because the
    // authoritative stored norm on the denorm side is `0.0`.
    let normalized = tensor_array(&[2], &[0.6, 0.8])?;
    let norms = PrimitiveArray::from_iter([0.0f64]).into_array();
    // SAFETY: This is a focused test that intentionally pairs a nonzero normalized row with a
    // stored norm of `0.0`, mimicking lossy storage where the stored norm is authoritative.
    let denorm = unsafe { Normalized::new_unchecked(normalized, norms) }.into_array();

    let plain = tensor_array(&[2], &[1.0, 0.0])?;

    // Denorm on the lhs: `One { denorm: lhs, plain: rhs }`.
    assert_close(
        &eval_cosine_similarity(denorm.clone(), plain.clone())?,
        &[0.0],
    );

    // Denorm on the rhs: `One { denorm: rhs, plain: lhs }`. The same zero-norm guard must
    // fire regardless of operand order.
    assert_close(&eval_cosine_similarity(plain, denorm)?, &[0.0]);
    Ok(())
}

#[test]
fn constant_lhs_matches_plain_tensor() -> VortexResult<()> {
    // The constant query `[1, 2, 2]` has norm 3, so its normalized form is `[1/3, 2/3, 2/3]`.
    // Expected cosine similarity against each row is `dot([1, 2, 2], row) / (3 * ||row||)`.
    let lhs = constant_tensor_array(&[3], &[1.0, 2.0, 2.0], 4)?;
    let rhs = tensor_array(
        &[3],
        &[
            1.0, 0.0, 0.0, // dot=1, ||rhs||=1, expected=1/3
            1.0, 2.0, 2.0, // dot=9, ||rhs||=3, expected=1
            0.0, 0.0, 1.0, // dot=2, ||rhs||=1, expected=2/3
            2.0, 1.0, 2.0, // dot=8, ||rhs||=3, expected=8/9
        ],
    )?;
    assert_close(
        &eval_cosine_similarity(lhs, rhs)?,
        &[1.0 / 3.0, 1.0, 2.0 / 3.0, 8.0 / 9.0],
    );
    Ok(())
}

#[test]
fn constant_rhs_matches_plain_tensor() -> VortexResult<()> {
    // Mirror of `constant_lhs_matches_plain_tensor` with the constant on the right.
    let lhs = tensor_array(
        &[3],
        &[
            1.0, 0.0, 0.0, //
            1.0, 2.0, 2.0, //
            0.0, 0.0, 1.0, //
            2.0, 1.0, 2.0, //
        ],
    )?;
    let rhs = constant_tensor_array(&[3], &[1.0, 2.0, 2.0], 4)?;
    assert_close(
        &eval_cosine_similarity(lhs, rhs)?,
        &[1.0 / 3.0, 1.0, 2.0 / 3.0, 8.0 / 9.0],
    );
    Ok(())
}

#[test]
fn both_constant_tensors() -> VortexResult<()> {
    // `[1, 0, 0]` vs `[1, 1, 0]`. dot=1, ||lhs||=1, ||rhs||=sqrt(2), expected=1/sqrt(2).
    let lhs = constant_tensor_array(&[3], &[1.0, 0.0, 0.0], 3)?;
    let rhs = constant_tensor_array(&[3], &[1.0, 1.0, 0.0], 3)?;
    let expected = 1.0 / 2.0_f64.sqrt();
    assert_close(
        &eval_cosine_similarity(lhs, rhs)?,
        &[expected, expected, expected],
    );
    Ok(())
}

#[test]
fn constant_zero_norm_query() -> VortexResult<()> {
    // A zero-norm constant query must produce `0.0` through the prepared row kernel's
    // zero-denominator guard.
    let lhs = constant_tensor_array(&[3], &[0.0, 0.0, 0.0], 3)?;
    let rhs = tensor_array(
        &[3],
        &[
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0, //
        ],
    )?;
    assert_close(&eval_cosine_similarity(lhs, rhs)?, &[0.0, 0.0, 0.0]);
    Ok(())
}

#[test]
fn constant_self_similarity_nonunit() -> VortexResult<()> {
    // The prepared path hoists both norms and computes the same dot product for every row.
    let lhs = constant_tensor_array(&[3], &[3.0, 4.0, 0.0], 5)?;
    let rhs = constant_tensor_array(&[3], &[3.0, 4.0, 0.0], 5)?;
    assert_close(&eval_cosine_similarity(lhs, rhs)?, &[1.0; 5]);
    Ok(())
}

/// An extension array over constant storage (what [`Vector::constant_array`] builds) is a batch
/// constant like any other. The row layer sees through the wrapper, so `prepare` hoists its norm.
#[test]
fn vector_constant_matches_plain() -> VortexResult<()> {
    let lhs = Vector::constant_array(&[1.0, 2.0, 2.0], 4)?;
    let rhs = vector_array(
        3,
        &[
            1.0, 0.0, 0.0, //
            1.0, 2.0, 2.0, //
            0.0, 0.0, 1.0, //
            2.0, 1.0, 2.0, //
        ],
    )?;

    assert_close(
        &eval_cosine_similarity(lhs, rhs)?,
        &[1.0 / 3.0, 1.0, 2.0 / 3.0, 8.0 / 9.0],
    );
    assert_eq!(
        probe::SEEN_CONSTANTS.get(),
        0b01,
        "the extension-over-constant lhs must reach prepare as a batch constant",
    );
    Ok(())
}

/// Both literal and extension-wrapped constant storage reach the prepared row path. The probe
/// ensures that the literal query remains a batch constant instead of becoming a per-row column.
///
/// [`ConstantArray`]: vortex_array::arrays::ConstantArray
#[test]
fn literal_constant_rhs_matches_expanded_column() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let lhs = vector_array(
        3,
        &[
            1.0, 0.0, 0.0, //
            1.0, 2.0, 2.0, //
            0.0, 0.0, 1.0, //
            2.0, 1.0, 2.0, //
        ],
    )?;
    let query = [1.0, 2.0, 2.0];

    let from_constant =
        eval_cosine_similarity_array(lhs.clone(), literal_vector_array(&query, 4), &mut ctx)?;
    let from_expanded =
        eval_cosine_similarity_array(lhs, vector_array(3, &query.repeat(4))?, &mut ctx)?;

    assert_arrays_eq!(from_constant, from_expanded, &mut ctx);
    Ok(())
}

/// The mirror of [`literal_constant_rhs_matches_expanded_column`], exercising the hoisted-lhs arm
/// of the prepared kernel.
#[test]
fn literal_constant_lhs_matches_expanded_column() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let rhs = vector_array(
        3,
        &[
            1.0, 0.0, 0.0, //
            1.0, 2.0, 2.0, //
            0.0, 0.0, 1.0, //
            2.0, 1.0, 2.0, //
        ],
    )?;
    let query = [1.0, 2.0, 2.0];

    let from_constant =
        eval_cosine_similarity_array(literal_vector_array(&query, 4), rhs.clone(), &mut ctx)?;
    let from_expanded =
        eval_cosine_similarity_array(vector_array(3, &query.repeat(4))?, rhs, &mut ctx)?;

    assert_arrays_eq!(from_constant, from_expanded, &mut ctx);
    Ok(())
}

/// A zero-norm literal constant query must be guarded to `0.0` on every row by the prepared row
/// kernel, exactly as the unprepared kernel guards it.
#[test]
fn literal_constant_zero_norm_query_yields_zero() -> VortexResult<()> {
    let lhs = vector_array(3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let rhs = literal_vector_array(&[0.0f64, 0.0, 0.0], 2);
    assert_close(&eval_cosine_similarity(lhs, rhs)?, &[0.0, 0.0]);
    Ok(())
}

/// Two literal constants are folded to a single-row execution by the row lifting, and that row
/// still runs the prepared kernel with both norms hoisted.
#[test]
fn both_literal_constants() -> VortexResult<()> {
    let lhs = literal_vector_array(&[1.0f64, 0.0, 0.0], 3);
    let rhs = literal_vector_array(&[1.0f64, 1.0, 0.0], 3);
    let expected = 1.0 / 2.0_f64.sqrt();
    assert_close(&eval_cosine_similarity(lhs, rhs)?, &[expected; 3]);
    Ok(())
}

#[rstest]
#[case::vector(cosine_vector_lhs(), cosine_vector_rhs())]
#[case::fixed_shape_tensor(cosine_tensor_lhs(), cosine_tensor_rhs())]
fn serde_round_trip(#[case] lhs: ArrayRef, #[case] rhs: ArrayRef) -> VortexResult<()> {
    let original =
        CosineSimilarity.try_new_array(lhs.len(), EmptyOptions, [lhs.clone(), rhs.clone()])?;

    let plugin = ScalarFnArrayPlugin::new(CosineSimilarity);
    let metadata = plugin
        .serialize(&original, &SESSION)?
        .expect("CosineSimilarity serialize must produce metadata");

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

fn cosine_vector_lhs() -> ArrayRef {
    vector_array(3, &[1.0, 0.0, 0.0, 3.0, 4.0, 0.0]).expect("valid vector array")
}

fn cosine_vector_rhs() -> ArrayRef {
    vector_array(3, &[0.0, 1.0, 0.0, 3.0, 4.0, 0.0]).expect("valid vector array")
}

fn cosine_tensor_lhs() -> ArrayRef {
    tensor_array(&[2], &[1.0, 0.0, 3.0, 4.0]).expect("valid tensor array")
}

fn cosine_tensor_rhs() -> ArrayRef {
    tensor_array(&[2], &[0.0, 1.0, 3.0, 4.0]).expect("valid tensor array")
}
