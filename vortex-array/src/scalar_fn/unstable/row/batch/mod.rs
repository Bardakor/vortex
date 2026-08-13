// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Batch execution around a non-null row kernel.
//!
//! A row kernel handles typed values for one row. This module adds the columnar concerns around it:
//! planning the output and null strategy, preserving batch constants and encodings, propagating
//! strict validity, selecting an execution strategy, and validating the finished output.
//!
//! [`BatchPlan`] carries the nullable execution strategy selected by a concrete dispatch. [`Batch`]
//! applies that strategy, and [`BorrowedExecutionArgs`] pairs each kernel invocation with its
//! planning metadata.

mod args;
pub(super) use args::BorrowedExecutionArgs;

mod execution;
pub(super) use execution::Batch;
pub(super) use execution::finalize_kernel_output;

pub(super) use super::visitor::BatchPlan;
pub(super) use super::visitor::RowPolicy;

#[cfg(test)]
mod tests;
