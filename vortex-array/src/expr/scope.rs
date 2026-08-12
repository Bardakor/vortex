// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_utils::aliases::hash_map::HashMap;

use crate::dtype::DType;
use crate::expr::variable::Variable;

/// The context an [`Expression`](crate::expr::Expression) is bound against.
///
/// A scope is the dtype that [`root`](crate::expr::root) resolves to, plus named bindings.
/// Binding names are unique for the whole scope: introducing the same name twice is an error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Scope {
    root: DType,
    bindings: HashMap<Variable, DType>,
}

impl Scope {
    /// Create a scope in which `root` resolves to the given dtype, with no bindings.
    pub fn new(root: DType) -> Self {
        Self {
            root,
            bindings: HashMap::new(),
        }
    }

    /// The dtype that `root` resolves to.
    pub fn root(&self) -> &DType {
        &self.root
    }

    /// Add a named binding to this scope.
    ///
    /// Returns an error if `variable` is already bound. Names are unique across the complete
    /// scope, so nested binders cannot shadow an outer binding.
    pub fn bind(&mut self, variable: Variable, dtype: DType) -> VortexResult<()> {
        if self.bindings.contains_key(&variable) {
            vortex_bail!("variable '{variable}' is already bound");
        }
        self.bindings.insert(variable, dtype);
        Ok(())
    }

    /// Return this scope extended with named bindings.
    ///
    /// Returns an error if any name is already bound, including a duplicate in `bindings`.
    pub fn with_bindings(
        mut self,
        bindings: impl IntoIterator<Item = (Variable, DType)>,
    ) -> VortexResult<Self> {
        for (variable, dtype) in bindings {
            self.bind(variable, dtype)?;
        }
        Ok(self)
    }

    /// Resolve `name` to its bound dtype.
    pub fn resolve(&self, name: &Variable) -> Option<&DType> {
        self.bindings.get(name)
    }
}

impl From<DType> for Scope {
    fn from(root: DType) -> Self {
        Self::new(root)
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use super::*;
    use crate::dtype::Nullability;
    use crate::dtype::PType;

    fn i32_() -> DType {
        DType::Primitive(PType::I32, Nullability::NonNullable)
    }

    fn utf8() -> DType {
        DType::Utf8(Nullability::NonNullable)
    }

    #[test]
    fn root_round_trips() {
        let dtype = DType::Bool(Nullability::Nullable);
        assert_eq!(Scope::new(dtype.clone()).root(), &dtype);
        assert_eq!(Scope::from(dtype.clone()).root(), &dtype);
    }

    #[test]
    fn an_empty_scope_resolves_nothing() {
        let scope = Scope::new(i32_());
        assert!(scope.resolve(&Variable::new("x")).is_none());
    }

    #[test]
    fn bindings_resolve_by_name() -> VortexResult<()> {
        let scope = Scope::new(i32_())
            .with_bindings([(Variable::new("a"), i32_()), (Variable::new("b"), utf8())])?;

        assert_eq!(scope.resolve(&Variable::new("a")), Some(&i32_()));
        assert_eq!(scope.resolve(&Variable::new("b")), Some(&utf8()));
        Ok(())
    }

    #[test]
    fn duplicate_names_are_rejected() -> VortexResult<()> {
        let mut scope = Scope::new(i32_());
        scope.bind(Variable::new("x"), i32_())?;
        assert!(scope.bind(Variable::new("x"), utf8()).is_err());
        assert!(
            Scope::new(i32_())
                .with_bindings([(Variable::new("x"), i32_()), (Variable::new("x"), utf8())])
                .is_err()
        );
        Ok(())
    }
}
