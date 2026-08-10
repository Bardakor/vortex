// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::Float;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::MaskedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::match_each_float_ptype;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::InitializedElement;
use vortex_array::scalar_fn::RowFn;
use vortex_array::scalar_fn::RowVisitor;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::UninitElementSink;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_session::registry::CachedId;

use crate::scalar_fns::row::TensorRow;
use crate::scalar_fns::row::tensor_element_ptype;
use crate::utils::test_helpers::assert_close;
use crate::utils::test_helpers::tensor_array;

/// The marginal cost of a new tensor scalar function is this entire definition. Everything else
/// (null propagation, constants, validity, f16/f32/f64 dispatch, dtype checks, and constructors) is
/// derived.
#[derive(Clone, Debug, Default)]
struct L1Norm;

impl RowFn for L1Norm {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["input"];

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.test.l1_norm");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        match_each_float_ptype!(tensor_element_ptype(args)?, |T| {
            visitor.visit_into::<(TensorRow<T>,), UninitElementSink<T>, _>(|(row,), output| {
                // SAFETY: `output` is the `UninitElementSink` row supplied for this callback.
                unsafe { InitializedElement::write(output, l1_norm_row(row)) }
            })
        })
    }
}

fn l1_norm_row<T: Float + NativePType>(row: &[T]) -> T {
    row.iter().fold(T::zero(), |acc, &x| acc + x.abs())
}

#[test]
fn derived_fn_executes_with_nulls() -> VortexResult<()> {
    let arr = tensor_array(&[2], &[3.0, -4.0, 1.0, 1.0])?;
    let arr = MaskedArray::try_new(arr, Validity::from_iter([true, false]))?.into_array();

    let mut ctx = crate::tests::SESSION.create_execution_ctx();
    let prim: PrimitiveArray = L1Norm
        .try_new_array(arr.len(), EmptyOptions, [arr])?
        .execute(&mut ctx)?;

    assert!(prim.is_valid(0, &mut ctx)?);
    assert!(!prim.is_valid(1, &mut ctx)?);
    assert_close(&[prim.as_slice::<f64>()[0]], &[7.0]);
    Ok(())
}

/// A kernel written once serves every float width.
#[test]
fn derived_fn_dispatches_at_input_width() -> VortexResult<()> {
    let mut ctx = crate::tests::SESSION.create_execution_ctx();

    let f32_result: PrimitiveArray = L1Norm
        .try_new_array(1, EmptyOptions, [tensor_array(&[2], &[3.0f32, -4.0])?])?
        .execute(&mut ctx)?;
    assert_eq!(f32_result.ptype(), PType::F32);
    assert_eq!(f32_result.as_slice::<f32>(), &[7.0f32]);

    let f64_result: PrimitiveArray = L1Norm
        .try_new_array(1, EmptyOptions, [tensor_array(&[2], &[3.0f64, -4.0])?])?
        .execute(&mut ctx)?;
    assert_eq!(f64_result.ptype(), PType::F64);
    Ok(())
}
