// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar function expressions defined on tensor and tensor-like extension types.

/// Adopt the standard scalar-function behavior for a row function defined in this module.
macro_rules! impl_row_fn_scalar_vtable {
    ($function:ty) => {
        impl vortex_array::scalar_fn::ScalarFnVTable for $function {
            type Options = <$function as vortex_array::scalar_fn::RowFn>::Options;

            fn id(&self) -> vortex_array::scalar_fn::ScalarFnId {
                vortex_array::scalar_fn::RowFn::id(self)
            }

            fn serialize(
                &self,
                options: &Self::Options,
            ) -> vortex_error::VortexResult<Option<Vec<u8>>> {
                vortex_array::scalar_fn::RowFn::serialize(self, options)
            }

            fn deserialize(
                &self,
                metadata: &[u8],
                session: &vortex_session::VortexSession,
            ) -> vortex_error::VortexResult<Self::Options> {
                vortex_array::scalar_fn::RowFn::deserialize(self, metadata, session)
            }

            fn arity(&self, _options: &Self::Options) -> vortex_array::scalar_fn::Arity {
                vortex_array::scalar_fn::Arity::Exact(
                    <$function as vortex_array::scalar_fn::RowFn>::ARG_NAMES.len(),
                )
            }

            fn child_name(
                &self,
                _options: &Self::Options,
                child_index: usize,
            ) -> vortex_array::scalar_fn::ChildName {
                vortex_array::scalar_fn::ChildName::from(
                    <$function as vortex_array::scalar_fn::RowFn>::ARG_NAMES[child_index],
                )
            }

            fn return_dtype(
                &self,
                options: &Self::Options,
                args: &[vortex_array::dtype::DType],
            ) -> vortex_error::VortexResult<vortex_array::dtype::DType> {
                vortex_array::scalar_fn::row_fn_return_dtype(self, options, args)
            }

            fn execute(
                &self,
                options: &Self::Options,
                args: &dyn vortex_array::scalar_fn::ExecutionArgs,
                ctx: &mut vortex_array::ExecutionCtx,
            ) -> vortex_error::VortexResult<vortex_array::ArrayRef> {
                vortex_array::scalar_fn::execute_rows(self, options, args, ctx)
            }

            fn validity(
                &self,
                _options: &Self::Options,
                expression: &vortex_array::expr::Expression,
            ) -> vortex_error::VortexResult<Option<vortex_array::expr::Expression>> {
                vortex_array::expr::union_child_validities(expression)
            }

            fn is_strict(&self, _options: &Self::Options) -> bool {
                true
            }

            fn is_fallible(&self, _options: &Self::Options) -> bool {
                <$function as vortex_array::scalar_fn::RowFn>::FALLIBLE
            }
        }
    };
}

pub mod cosine_similarity;
pub mod inner_product;
pub mod l2_norm;
pub(crate) mod row;

#[cfg(test)]
mod tests;
