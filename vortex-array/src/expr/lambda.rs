// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_utils::aliases::hash_set::HashSet;

use crate::expr::Expression;
use crate::expr::variable::Variable;

/// A body evaluated with named bindings for `params`.
///
/// A lambda is **not a value**: its parameter dtypes are determined by whatever applies it, so it
/// has no dtype of its own and cannot be bound by [`bind_scope`](Expression::bind_scope). The
/// higher-order function that applies it supplies its parameter types and binds its body.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Lambda {
    params: Arc<Vec<Variable>>,
    body: Arc<Expression>,
}

impl Lambda {
    /// Create a lambda binding `params` over `body`.
    ///
    /// Returns an error when a parameter name is repeated.
    pub fn try_new(
        params: impl IntoIterator<Item = impl Into<Variable>>,
        body: Expression,
    ) -> VortexResult<Self> {
        let mut vars = Vec::new();
        let mut seen = HashSet::new();

        for param in params {
            let var: Variable = param.into();
            if !seen.insert(var.clone()) {
                vortex_bail!("duplicate parameter");
            }

            vars.push(var)
        }

        Ok(Self {
            params: Arc::new(vars),
            body: Arc::new(body),
        })
    }

    /// The variables this lambda binds, in declaration order.
    pub fn params(&self) -> &[Variable] {
        &self.params
    }

    /// The expression evaluated under the parameter bindings.
    pub fn body(&self) -> &Expression {
        &self.body
    }

    /// Take the body if this lambda holds the only reference to it.
    pub(crate) fn take_unique_body(&mut self) -> Option<Expression> {
        Arc::get_mut(&mut self.body).map(std::mem::take)
    }
}

impl Display for Lambda {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "({}) -> {}", self.params.iter().join(", "), self.body)
    }
}

impl From<Lambda> for Expression {
    fn from(lambda: Lambda) -> Self {
        Expression::Lambda(lambda)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_parameters_are_rejected() {
        assert!(Lambda::try_new(["x", "x"], Expression::Root).is_err());
    }
}
