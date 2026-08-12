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

/// A body evaluated under a frame binding `params`.
///
/// A lambda is **not a value**: its parameter dtypes are determined by whatever applies it, so it
/// has no dtype of its own and cannot be bound by [`bind_scope`](Expression::bind_scope). The
/// higher-order function that applies it supplies its parameter types and binds its body.
///
/// It is a struct rather than only an enum variant so that an API expecting a lambda — a
/// higher-order function, for instance — can say so in its signature and reject anything else at
/// compile time. It is a scope boundary: generic expression passes leave its body to the
/// higher-order function that establishes the parameter frame.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Lambda {
    params: Box<[Variable]>,
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
        let params: Box<[Variable]> = params.into_iter().map(Into::into).collect();
        {
            let mut seen = HashSet::with_capacity(params.len());
            for parameter in &params {
                if !seen.insert(parameter) {
                    vortex_bail!("duplicate lambda parameter '{parameter}'");
                }
            }
        }

        Ok(Self {
            params,
            body: Arc::new(body),
        })
    }

    /// The variables this lambda binds, in declaration order.
    pub fn params(&self) -> &[Variable] {
        &self.params
    }

    /// The expression evaluated under the parameter frame.
    pub fn body(&self) -> &Expression {
        &self.body
    }

    /// Take the body if this lambda holds the only reference to it.
    ///
    /// Used by `Expression`'s iterative [`Drop`] to drain a lambda chain onto a worklist instead of
    /// recursing through it, which would overflow the stack on a deeply nested chain.
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
