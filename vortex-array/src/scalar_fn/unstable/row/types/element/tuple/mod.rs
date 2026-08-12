// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Argument lists built from [`InputElement`](super::InputElement)s and their per-argument decode.

mod element_tuple;
pub use element_tuple::ElementTuple;

mod indexed;
pub use indexed::IndexedElementTuple;

#[cfg(test)]
mod tests;
