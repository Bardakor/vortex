// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Inner product expression for tensor-like types.

use num_traits::Float;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::scalar_fn::ScalarFnArrayView;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayParts;
use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayVTable;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_float_ptype;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::TypedScalarFnInstance;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::scalar_fn::unstable::row::InitializedElement;
use vortex_array::scalar_fn::unstable::row::RowExecution;
use vortex_array::scalar_fn::unstable::row::RowFn;
use vortex_array::scalar_fn::unstable::row::RowVisitor;
use vortex_array::scalar_fn::unstable::row::UninitElementSink;
use vortex_array::serde::ArrayChildren;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::encodings::normalized::NormalizedOrientation;
use crate::scalar_fns::row::TensorRow;
use crate::scalar_fns::row::tensor_element_ptype;
use crate::utils::BinaryTensorOpMetadata;
use crate::utils::extract_normalized_children;

/// Inner product (dot product) between two columns.
///
/// Computes `sum(a_i * b_i)` over the flat backing buffer of each tensor or vector. For vectors
/// this is the standard dot product; for higher-rank ([`FixedShapeTensor`]) arrays this is the
/// Frobenius inner product.
///
/// Both inputs must be tensor-like extension arrays ([`FixedShapeTensor`] or [`Vector`]) with the
/// same dtype and a float element type. The output is a float column of the same float type.
///
/// [`FixedShapeTensor`]: crate::fixed_shape_tensor::FixedShapeTensor
/// [`Vector`]: crate::vector::Vector
#[derive(Clone, Debug, Default)]
pub struct InnerProduct;

impl InnerProduct {
    /// Creates a new [`TypedScalarFnInstance`] wrapping the inner product operation.
    pub fn new() -> TypedScalarFnInstance<Self> {
        TypedScalarFnInstance::new(Self, EmptyOptions)
    }

    /// Constructs a [`ScalarFnArray`] that lazily computes the inner product between `lhs` and
    /// `rhs`.
    ///
    /// # Errors
    ///
    /// Returns an error if the array cannot be constructed, such as when the input dtypes are
    /// unsupported.
    pub fn try_new_array(lhs: ArrayRef, rhs: ArrayRef) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(Self::new().erased(), vec![lhs, rhs])
    }
}

impl RowFn for InnerProduct {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.tensor.inner_product");
        *ID
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        Ok(EmptyOptions)
    }

    fn dispatch<V: RowVisitor<Self::Options>>(
        &self,
        _options: &Self::Options,
        args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        match_each_float_ptype!(tensor_element_ptype(args)?, |T| {
            visitor.visit_into::<(TensorRow<T>, TensorRow<T>), UninitElementSink<T>, _>(
                |(lhs, rhs), output| {
                    // SAFETY: `output` is the `UninitElementSink` row supplied for this callback.
                    unsafe { InitializedElement::write(output, inner_product_row(lhs, rhs)) }
                },
            )
        })
    }

    /// [`Normalized`]-encoded operands factor through their stored norms: with `D(x, s)` denoting
    /// `x * s` rowwise, `dot(D(x, s), D(y, t)) = s * t * dot(x, y)` and
    /// `dot(D(x, s), y) = s * dot(x, y)`. The rewrite is expressed with lazy [`Operator::Mul`]
    /// arrays over the (much smaller) norm columns, so no denormalized coordinates are decoded.
    ///
    /// [`Normalized`]: crate::encodings::normalized::Normalized
    fn reduce_encoded(
        &self,
        _options: &Self::Options,
        args: &[ArrayRef],
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<RowExecution>> {
        let len = args[0].len();

        Ok(match NormalizedOrientation::classify(&args[0], &args[1]) {
            NormalizedOrientation::Both { lhs, rhs } => {
                let (normalized_l, norms_l) = extract_normalized_children(lhs);
                let (normalized_r, norms_r) = extract_normalized_children(rhs);
                let dot =
                    InnerProduct.try_new_array(len, EmptyOptions, [normalized_l, normalized_r])?;
                Some(
                    dot.binary(norms_l, Operator::Mul)?
                        .binary(norms_r, Operator::Mul)?,
                )
            }
            NormalizedOrientation::One {
                normalized_array,
                plain,
            } => {
                let (normalized, norms) = extract_normalized_children(normalized_array);
                let dot =
                    InnerProduct.try_new_array(len, EmptyOptions, [normalized, plain.clone()])?;
                Some(dot.binary(norms, Operator::Mul)?)
            }
            NormalizedOrientation::Neither => None,
        }
        .map(RowExecution::Output))
    }
}

impl ScalarFnArrayVTable for InnerProduct {
    fn serialize(
        &self,
        view: &ScalarFnArrayView<Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(BinaryTensorOpMetadata::encode_from_view(view)?))
    }

    fn deserialize(
        &self,
        _dtype: &DType,
        len: usize,
        metadata: &[u8],
        children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<ScalarFnArrayParts<Self>> {
        let reconstructed =
            BinaryTensorOpMetadata::decode_children(metadata, len, children, session)?;
        Ok(ScalarFnArrayParts {
            options: EmptyOptions,
            children: reconstructed,
        })
    }
}

/// Computes the inner product (dot product) of two equal-length float slices.
///
/// Returns `sum(a_i * b_i)`.
fn inner_product_row<T: Float + NativePType>(lhs: &[T], rhs: &[T]) -> T {
    lhs.iter()
        .zip(rhs)
        .map(|(&lhs_element, &rhs_element)| lhs_element * rhs_element)
        .fold(T::zero(), |sum, product| sum + product)
}
