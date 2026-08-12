// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;

use super::element_tuple::batch_constant;
use crate::IntoArray;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::ExtensionArray;
use crate::arrays::MaskedArray;
use crate::dtype::Nullability;
use crate::extension::datetime::TimeUnit;
use crate::extension::datetime::Timestamp;
use crate::validity::Validity;

#[test]
fn test_batch_constant_unwraps_filtered_masked_constant() -> VortexResult<()> {
    let child = ConstantArray::new(7_i64, 3).into_array();
    let masked =
        MaskedArray::try_new(child, Validity::from_iter([true, false, true]))?.into_array();
    let filtered = masked.filter(Mask::from_iter([true, true, false]))?;

    let Some(constant) = batch_constant(&filtered) else {
        vortex_bail!("filtered masked constant must remain batch-constant");
    };

    assert!(constant.is::<Constant>());
    Ok(())
}

#[test]
fn test_batch_constant_preserves_filtered_extension() -> VortexResult<()> {
    let ext_dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased();
    let extension =
        ExtensionArray::new(ext_dtype, ConstantArray::new(7_i64, 3).into_array()).into_array();
    let filtered = extension.filter(Mask::from_iter([true, false, true]))?;

    let Some(constant) = batch_constant(&filtered) else {
        vortex_bail!("filtered extension storage must remain batch-constant");
    };

    assert_eq!(constant.dtype(), extension.dtype());
    Ok(())
}
