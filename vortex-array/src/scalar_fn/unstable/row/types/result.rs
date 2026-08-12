// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! What a sink-writing row closure may return.

use vortex_error::VortexResult;

use super::InitializedElement;

/// The result of writing one row: success or an immediate error.
///
/// This trait is sealed; row functions choose one of its supplied implementations.
pub trait SinkResult: 'static + private::Sealed {
    /// The [`OutputSink::WriteToken`](super::OutputSink::WriteToken) carried by a success.
    type WriteToken: 'static;

    /// Whether this return type can carry an error.
    const FALLIBLE: bool;

    /// Convert this row's outcome into immediate success or failure.
    fn into_result(self) -> VortexResult<()>;
}

impl private::Sealed for () {}

impl SinkResult for () {
    type WriteToken = ();
    const FALLIBLE: bool = false;

    fn into_result(self) -> VortexResult<()> {
        Ok(())
    }
}

impl private::Sealed for InitializedElement {}

impl SinkResult for InitializedElement {
    type WriteToken = InitializedElement;
    const FALLIBLE: bool = false;

    fn into_result(self) -> VortexResult<()> {
        Ok(())
    }
}

impl private::Sealed for VortexResult<()> {}

impl SinkResult for VortexResult<()> {
    type WriteToken = ();
    const FALLIBLE: bool = true;

    fn into_result(self) -> VortexResult<()> {
        self
    }
}

impl private::Sealed for VortexResult<InitializedElement> {}

impl SinkResult for VortexResult<InitializedElement> {
    type WriteToken = InitializedElement;
    const FALLIBLE: bool = true;

    fn into_result(self) -> VortexResult<()> {
        self.map(|_| ())
    }
}

mod private {
    pub trait Sealed {}
}
