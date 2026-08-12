// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Visits that plan or execute the concrete row signature selected by [`RowFn::dispatch`].
//!
//! [`RowFn::dispatch`]: crate::scalar_fn::unstable::row::RowFn::dispatch

mod check;

mod plan;
pub(super) use plan::PlanRows;

mod row_visitor;
pub use row_visitor::RowVisitor;
