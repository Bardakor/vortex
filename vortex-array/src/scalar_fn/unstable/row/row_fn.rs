// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar functions computed one row at a time.

use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use super::visitor::RowVisitor;
use crate::dtype::DType;
use crate::scalar_fn::ScalarFnId;

/// A scalar function computed one row at a time.
///
/// Declare the argument names and use [`dispatch`](Self::dispatch) to choose concrete element and
/// sink types for each accepted dtype combination.
///
/// Every `RowFn` receives the standard [`ScalarFnVTable`] implementation. A function that needs
/// custom scalar-function hooks instead implements `ScalarFnVTable` on its public type and
/// delegates row execution to a private `RowFn` kernel with [`row_fn_return_dtype`] and
/// [`execute_rows`]. Implement only `ScalarFnVTable` when the natural kernel is columnar.
///
/// [`ScalarFnVTable`]: crate::scalar_fn::ScalarFnVTable
/// [`execute_rows`]: crate::scalar_fn::unstable::row::execute_rows
/// [`row_fn_return_dtype`]: crate::scalar_fn::unstable::row::row_fn_return_dtype
pub trait RowFn: 'static + Sized + Clone + Send + Sync {
    /// Options for this function, or [`EmptyOptions`](crate::scalar_fn::EmptyOptions) for none.
    type Options: 'static + Send + Sync + Clone + Debug + Display + PartialEq + Eq + Hash;

    /// The arguments in display order. Its length is the function's exact arity.
    const ARG_NAMES: &'static [&'static str];

    /// Whether any legal dispatch can raise a semantic error.
    ///
    /// The framework checks this at compile time for every fallible dispatched element or result.
    /// A conservative `true` is allowed when only some dtype choices are fallible.
    ///
    /// [`OutputSink`](crate::scalar_fn::unstable::row::OutputSink) lifecycle methods only report
    /// incidental execution failures. Semantic sink errors must come from the row callback.
    ///
    /// Semantic errors are defined by
    /// [`ScalarFnVTable::is_fallible`](crate::scalar_fn::ScalarFnVTable::is_fallible).
    const FALLIBLE: bool = false;

    /// Returns the ID of the scalar function.
    fn id(&self) -> ScalarFnId;

    /// Serialize this function's options, or return `None` when the function is not serializable.
    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        _ = options;
        Ok(None)
    }

    /// Restore options written by [`serialize`](Self::serialize).
    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        vortex_bail!("Expression {} is not deserializable", self.id())
    }

    /// Choose element types for these input dtypes and visit the framework with them.
    ///
    /// Plan time and run time both call this method, so the choice **must** be a pure function of
    /// `options` and `args`. The framework rejects a change to the derived nullable execution
    /// policy before row execution. It cannot compare the remaining types, preparation values, or
    /// closure behavior, so those must also remain stable. Cross-argument dtype validation belongs
    /// here.
    fn dispatch<V: RowVisitor<Self::Options>>(
        &self,
        options: &Self::Options,
        args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult>;
}
