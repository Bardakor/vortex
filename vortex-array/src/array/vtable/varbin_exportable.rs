// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayRef;
use crate::matcher::Matcher;

/// Capability for encodings whose [`append_to_builder`](super::VTable::append_to_builder) writes
/// `Utf8`/`Binary` values straight into a [`VarBinBuilder`](crate::builders::VarBinBuilder).
///
/// Callers filling a `VarBinBuilder` should `execute_until::<dyn VarBinExportable>`: executing past
/// these encodings reaches a canonical `VarBinView` that the builder has to re-lay out.
///
/// Claim it only where stopping is the better trade. `Dict` does not: it appends canonical values,
/// but executing through it lets kernels such as FSST's `Dict` parent kernel decode through the
/// dictionary instead.
pub trait VarBinExportable: 'static + Send + Sync {}

/// Matches every array whose encoding offers [`VarBinExportable`].
impl Matcher for dyn VarBinExportable {
    type Match<'a> = &'a ArrayRef;

    #[inline]
    fn try_match(array: &ArrayRef) -> Option<Self::Match<'_>> {
        array
            .has_capability::<dyn VarBinExportable>()
            .then_some(array)
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use super::*;
    use crate::IntoArray;
    use crate::arrays::DictArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::VarBinViewArray;

    #[test]
    fn encodings_report_the_capability_through_their_vtable() -> VortexResult<()> {
        let values = VarBinViewArray::from_iter_str(["a", "b"]).into_array();
        assert!(values.has_capability::<dyn VarBinExportable>());
        assert!(values.is::<dyn VarBinExportable>());

        // Dict gathers canonical values into the builder but deliberately does not claim the
        // capability, so execution continues through it.
        let codes = PrimitiveArray::from_iter([0u8, 1, 0]).into_array();
        let dict = DictArray::try_new(codes, values)?.into_array();
        assert!(!dict.has_capability::<dyn VarBinExportable>());
        Ok(())
    }
}
