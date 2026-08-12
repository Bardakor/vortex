// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The element types a row function can read and produce.
//!
//! [`InputElement::Elem`] may borrow from its decoded column. [`OutputElement`] is returned by an
//! owned row computation; runtime-shaped output uses an
//! [`OutputSink`](crate::scalar_fn::unstable::row::OutputSink).

mod bool;

mod input;
pub use input::InputElement;

mod output;
pub use output::OutputElement;

mod primitive;

mod tuple;
pub use tuple::ElementTuple;
pub use tuple::IndexedElementTuple;
