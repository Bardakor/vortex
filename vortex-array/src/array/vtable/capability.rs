// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::TypeId;

/// Whether `capability` names the capability trait `C`.
///
/// Called once per capability from [`VTable::has_capability`](super::VTable::has_capability).
/// Passing the vtable as `&C` is what proves it implements `C`, so a vtable cannot claim a
/// capability it does not have.
pub fn has_capability<C: ?Sized + 'static>(_vtable: &C, capability: TypeId) -> bool {
    capability == TypeId::of::<C>()
}

#[cfg(test)]
mod tests {
    use super::*;

    trait Greet {}

    trait Absent {}

    struct Encoding;

    impl Greet for Encoding {}

    #[test]
    fn has_capability_matches_only_the_requested_trait() {
        assert!(has_capability::<dyn Greet>(
            &Encoding,
            TypeId::of::<dyn Greet>()
        ));
        assert!(!has_capability::<dyn Greet>(
            &Encoding,
            TypeId::of::<dyn Absent>()
        ));
    }
}
