// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use itertools::Itertools;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use crate::dtype::DType;
use crate::expr::Expression;
use crate::expr::display::DisplayTreeExpr;
use crate::expr::scope::Frame;
use crate::expr::scope::Scope;
use crate::expr::scope::VariableRef;
use crate::expr::traversal::TraversalOrder;
use crate::expr::traversal::pre_order_visit_down;
use crate::expr::variable::Variable;
use crate::scalar_fn::ScalarFnRef;
use crate::scalar_fn::ScalarFnVTable;
use crate::stats::rewrite::StatsRewriteCtx;

/// An [`Expression`] that has been type-checked against a [`Scope`].
///
/// Every node carries its own dtype, so reading one is a field access rather than a walk of the
/// subtree. Holding a `BoundExpression` is proof that the whole tree type-checked.
///
/// Binding is purely logical: it deals only in [`DType`]s and never sees an array, a length, or an
/// encoding.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum BoundExpression {
    /// A scalar function applied to bound children.
    Scalar {
        /// The dtype this node evaluates to.
        dtype: DType,
        /// The scalar function for this node.
        scalar_fn: ScalarFnRef,
        /// The bound children, in argument order.
        ///
        /// Sharing keeps clones cheap even though the iterative [`Drop`] implementation prevents
        /// consumers from destructuring a `BoundExpression` by value.
        children: Arc<Vec<BoundExpression>>,
    },
    /// The scope itself. Its dtype is the scope's root dtype.
    Root {
        /// The dtype this node evaluates to.
        dtype: DType,
    },
    /// A resolved reference to a bound variable.
    Variable {
        /// The dtype this node evaluates to.
        dtype: DType,
        /// The variable that was resolved.
        variable: Variable,
        /// The lexical frame and slot it resolved to.
        reference: VariableRef,
    },
}

/// A bound lambda.
///
/// A higher-order function's type-checked lambda argument.
///
/// This is deliberately separate from [`BoundExpression`]: a lambda is not a value and has no
/// dtype. The higher-order function establishes the parameter frame, binds the body, and stores
/// the resulting `BoundLambda` in its own state.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BoundLambda {
    /// The parameters and their dtypes: the argument side of the function type.
    ///
    /// Storing the [`Frame`] the body was bound under, rather than two parallel arrays, makes the
    /// name/dtype pairing and the duplicate-name rejection structural instead of maintained by
    /// hand.
    frame: Frame,
    body: Arc<BoundExpression>,
}

impl BoundLambda {
    /// The frame this lambda's body was bound under.
    pub fn frame(&self) -> &Frame {
        &self.frame
    }

    /// The parameters paired with their dtypes, in declaration order.
    pub fn bindings(&self) -> &[(Variable, DType)] {
        self.frame.bindings()
    }

    /// The variables this lambda binds, in declaration order.
    ///
    /// An iterator rather than a slice, because the names and dtypes are interleaved in the frame.
    pub fn params(&self) -> impl ExactSizeIterator<Item = &Variable> {
        self.frame.bindings().iter().map(|(variable, _)| variable)
    }

    /// The dtypes of the parameters, in declaration order.
    pub fn param_dtypes(&self) -> impl ExactSizeIterator<Item = &DType> {
        self.frame.bindings().iter().map(|(_, dtype)| dtype)
    }

    /// The bound body.
    pub fn body(&self) -> &BoundExpression {
        &self.body
    }

    /// The dtype the body evaluates to — the result side of the function type.
    pub fn body_dtype(&self) -> &DType {
        self.body.dtype()
    }
}

/// A bound-expression wrapper that compares shared tree identity instead of structure.
#[derive(Clone, Debug)]
pub struct ExactBoundExpr(pub BoundExpression);

impl PartialEq for ExactBoundExpr {
    fn eq(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (
                BoundExpression::Root { dtype: lhs_dtype },
                BoundExpression::Root { dtype: rhs_dtype },
            ) => lhs_dtype == rhs_dtype,
            (
                BoundExpression::Scalar {
                    dtype: lhs_dtype,
                    scalar_fn: lhs_fn,
                    children: lhs_children,
                },
                BoundExpression::Scalar {
                    dtype: rhs_dtype,
                    scalar_fn: rhs_fn,
                    children: rhs_children,
                },
            ) => {
                lhs_fn == rhs_fn
                    && Arc::ptr_eq(lhs_children, rhs_children)
                    && lhs_dtype == rhs_dtype
            }
            // No catch-all: a new variant must state its own identity rather than silently
            // comparing unequal, which would put `eq` out of step with `hash`.
            (
                BoundExpression::Variable {
                    dtype: lhs_dtype,
                    variable: lhs_var,
                    reference: lhs_reference,
                },
                BoundExpression::Variable {
                    dtype: rhs_dtype,
                    variable: rhs_var,
                    reference: rhs_reference,
                },
            ) => lhs_var == rhs_var && lhs_reference == rhs_reference && lhs_dtype == rhs_dtype,
            // No catch-all: a new variant must state its own identity, or `eq` drifts out of step
            // with `hash` and keys stop equalling themselves.
            (BoundExpression::Root { .. }, _)
            | (BoundExpression::Scalar { .. }, _)
            | (BoundExpression::Variable { .. }, _) => false,
        }
    }
}

impl Eq for ExactBoundExpr {}

impl Hash for ExactBoundExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // DType differences are resolved by equality. Omitting the potentially lazy dtype keeps
        // identity-keyed cache lookups from deserializing an entire schema just to compute a hash.
        match &self.0 {
            BoundExpression::Root { .. } => state.write_u8(0),
            BoundExpression::Variable {
                variable,
                reference,
                ..
            } => {
                state.write_u8(2);
                variable.hash(state);
                reference.hash(state);
            }
            BoundExpression::Scalar {
                scalar_fn,
                children,
                ..
            } => {
                state.write_u8(1);
                scalar_fn.hash(state);
                Arc::as_ptr(children).hash(state);
            }
        }
    }
}

impl BoundExpression {
    /// Create a bound root expression with the given dtype.
    pub fn new_root(dtype: DType) -> Self {
        Self::Root { dtype }
    }

    /// Create a bound scalar node from a scalar function and already-bound children.
    pub fn try_new(
        scalar_fn: ScalarFnRef,
        children: impl IntoIterator<Item = BoundExpression>,
    ) -> VortexResult<Self> {
        let children = Vec::from_iter(children);
        vortex_ensure!(
            scalar_fn.signature().arity().matches(children.len()),
            "Expression arity mismatch: expected {} children but got {}",
            scalar_fn.signature().arity(),
            children.len()
        );

        let arg_dtypes = children
            .iter()
            .map(|child| child.dtype().clone())
            .collect_vec();
        let dtype = scalar_fn.return_dtype(&arg_dtypes)?;

        Ok(Self::Scalar {
            dtype,
            scalar_fn,
            children: children.into(),
        })
    }

    /// Rebuild this node with new bound children, recomputing its dtype.
    pub fn with_children(
        self,
        children: impl IntoIterator<Item = BoundExpression>,
    ) -> VortexResult<Self> {
        let children = Vec::from_iter(children);
        match &self {
            BoundExpression::Scalar { scalar_fn, .. } => Self::try_new(scalar_fn.clone(), children),
            BoundExpression::Root { .. } | BoundExpression::Variable { .. } => {
                vortex_ensure!(
                    children.is_empty(),
                    "{self} cannot have {} children",
                    children.len()
                );
                Ok(self)
            }
        }
    }

    /// The dtype this expression evaluates to.
    pub fn dtype(&self) -> &DType {
        match self {
            Self::Scalar { dtype, .. } | Self::Root { dtype } | Self::Variable { dtype, .. } => {
                dtype
            }
        }
    }

    /// The bound children of this node, in argument order. Empty for [`BoundExpression::Root`].
    pub fn children(&self) -> &[BoundExpression] {
        match self {
            Self::Scalar { children, .. } => children.as_slice(),
            Self::Root { .. } | Self::Variable { .. } => &[],
        }
    }

    /// Return the child at `index`.
    pub fn child(&self, index: usize) -> &BoundExpression {
        &self.children()[index]
    }

    /// The scalar function for this node, or `None` if it is the scope root.
    pub fn as_scalar(&self) -> Option<&ScalarFnRef> {
        match self {
            Self::Scalar { scalar_fn, .. } => Some(scalar_fn),
            Self::Root { .. } | Self::Variable { .. } => None,
        }
    }

    /// Return whether this node uses the given scalar-function vtable.
    pub fn is<V: ScalarFnVTable>(&self) -> bool {
        self.as_scalar().is_some_and(ScalarFnRef::is::<V>)
    }

    /// Return whether this expression tree contains a node using the given scalar-function vtable.
    pub fn contains<V: ScalarFnVTable>(&self) -> VortexResult<bool> {
        let mut contains = false;
        pre_order_visit_down(self, |node| {
            if node.is::<V>() {
                contains = true;
                return Ok(TraversalOrder::Stop);
            }
            Ok(TraversalOrder::Continue)
        })?;
        Ok(contains)
    }

    /// Return the typed scalar-function options when this node uses the given vtable.
    pub fn as_opt<V: ScalarFnVTable>(&self) -> Option<&V::Options> {
        self.as_scalar().and_then(ScalarFnRef::as_opt::<V>)
    }

    /// Return the typed scalar-function options for this node.
    ///
    /// # Panics
    ///
    /// Panics when this node is the scope root or uses a different scalar-function vtable.
    pub fn as_<V: ScalarFnVTable>(&self) -> &V::Options {
        self.as_opt::<V>()
            .vortex_expect("Bound expression options type mismatch")
    }

    /// The variable and lexical location it resolves to, if this node is a variable reference.
    pub fn as_variable(&self) -> Option<(&Variable, VariableRef)> {
        match self {
            Self::Variable {
                variable,
                reference,
                ..
            } => Some((variable, *reference)),
            Self::Scalar { .. } | Self::Root { .. } => None,
        }
    }

    /// Whether this node is the scope root.
    pub fn is_root(&self) -> bool {
        matches!(self, Self::Root { .. })
    }

    /// Return whether every scope root in this expression has `dtype`.
    ///
    /// Expressions without a scope root, such as literals, match every dtype.
    pub fn is_root_bound_to(&self, dtype: &DType) -> bool {
        let mut is_bound_to = true;
        pre_order_visit_down(self, |node| {
            if node.is_root() && node.dtype() != dtype {
                is_bound_to = false;
                return Ok(TraversalOrder::Stop);
            }
            Ok(TraversalOrder::Continue)
        })
        .vortex_expect("bound expression traversal cannot not fail");
        is_bound_to
    }

    /// Return an expression that proves this predicate is definitely false from statistics.
    pub fn falsify(&self, session: &VortexSession) -> VortexResult<Option<BoundExpression>> {
        StatsRewriteCtx::new(session).falsify(self)
    }

    /// Return an expression that proves this predicate is definitely true from statistics.
    pub fn satisfy(&self, session: &VortexSession) -> VortexResult<Option<BoundExpression>> {
        StatsRewriteCtx::new(session).satisfy(self)
    }

    /// Display the bound expression as a formatted tree structure.
    pub fn display_tree(&self) -> impl Display {
        DisplayTreeExpr(self)
    }
}

impl Display for BoundExpression {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scalar { scalar_fn, .. } => scalar_fn.fmt_sql(self, f),
            Self::Root { .. } => f.write_str("$"),
            Self::Variable { variable, .. } => write!(f, "${variable}"),
        }
    }
}

impl Expression {
    /// Bind this expression against a root dtype, type-checking every node in a single walk.
    ///
    /// The returned tree carries a dtype on each node, so callers needing types at more than one
    /// node should bind once and read fields rather than calling
    /// [`return_dtype`](Expression::return_dtype) repeatedly.
    pub fn bind(&self, dtype: &DType) -> VortexResult<BoundExpression> {
        self.bind_scope(&Scope::new(dtype.clone()))
    }

    /// Bind this expression against an explicit [`Scope`].
    pub fn bind_scope(&self, scope: &Scope) -> VortexResult<BoundExpression> {
        match self {
            Expression::Root => Ok(BoundExpression::new_root(scope.root().clone())),
            Expression::Variable(variable) => {
                let Some((dtype, reference)) = scope.resolve(variable) else {
                    vortex_bail!(
                        "unbound variable '{variable}'; the scope binds {} frame(s)",
                        scope.depth()
                    );
                };
                Ok(BoundExpression::Variable {
                    dtype: dtype.clone(),
                    variable: variable.clone(),
                    reference,
                })
            }
            Expression::Lambda(_) => {
                vortex_bail!("a lambda must be bound by the higher-order function that applies it")
            }
            Expression::Scalar {
                scalar_fn,
                children,
            } => {
                let children: Vec<_> = children
                    .iter()
                    .map(|child| child.bind_scope(scope))
                    .try_collect()?;
                BoundExpression::try_new(scalar_fn.clone(), children)
            }
        }
    }
}

/// Iterative drop to avoid stack overflows on deep trees.
impl Drop for BoundExpression {
    fn drop(&mut self) {
        let Self::Scalar { children, .. } = self else {
            return;
        };
        let Some(children) = Arc::get_mut(children) else {
            return;
        };

        let mut to_drop = std::mem::take(children);
        while let Some(mut child) = to_drop.pop() {
            if let BoundExpression::Scalar { children, .. } = &mut child
                && let Some(grandchildren) = Arc::get_mut(children)
            {
                to_drop.append(grandchildren);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use super::*;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::Frame;
    use crate::expr::checked_add;
    use crate::expr::col;
    use crate::expr::eq;
    use crate::expr::lambda;
    use crate::expr::lit;
    use crate::expr::root;
    use crate::expr::test_harness::struct_dtype;
    use crate::expr::var;
    use crate::scalar_fn::fns::is_not_null::IsNotNull;

    fn scope() -> Scope {
        Scope::new(struct_dtype())
    }

    #[test]
    fn root_binds_to_the_scope() -> VortexResult<()> {
        let bound = root().bind_scope(&scope())?;
        assert!(bound.is_root());
        assert_eq!(bound.dtype(), &struct_dtype());
        assert_eq!(bound, BoundExpression::new_root(struct_dtype()));
        Ok(())
    }

    #[test]
    fn every_node_carries_its_dtype() -> VortexResult<()> {
        let expr = eq(col("a"), lit(1_i32));
        let bound = expr.bind_scope(&scope())?;

        assert_eq!(bound.dtype(), &DType::Bool(Nullability::NonNullable));

        let lhs = &bound.children()[0];
        assert_eq!(
            lhs.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );
        assert_eq!(lhs.children()[0].dtype(), &struct_dtype());
        Ok(())
    }

    #[test]
    fn bind_agrees_with_return_dtype() -> VortexResult<()> {
        for expr in [root(), col("a"), eq(col("a"), lit(1_i32)), lit(true)] {
            assert_eq!(
                expr.bind(&struct_dtype())?.dtype(),
                &expr.return_dtype(&struct_dtype())?,
                "disagreement for {expr}"
            );
        }
        Ok(())
    }

    #[test]
    fn contains_scalar_function() -> VortexResult<()> {
        let bound = eq(col("a"), lit(1_i32)).bind_scope(&scope())?;
        assert!(bound.contains::<Literal>()?);
        assert!(!root().bind_scope(&scope())?.contains::<Literal>()?);
        Ok(())
    }

    #[test]
    fn bound_to_checks_every_root() -> VortexResult<()> {
        let dtype = struct_dtype();
        let bound = eq(col("a"), col("a")).bind(&dtype)?;
        assert!(bound.is_root_bound_to(&dtype));
        assert!(!bound.is_root_bound_to(&DType::Bool(Nullability::NonNullable)));
        assert!(
            lit(true)
                .bind(&dtype)?
                .is_root_bound_to(&DType::Bool(Nullability::NonNullable))
        );
        Ok(())
    }

    #[test]
    fn bound_display_matches_unbound() -> VortexResult<()> {
        for expr in [root(), col("a"), eq(col("a"), lit(1_i32)), lit(true)] {
            let bound = expr.bind_scope(&scope())?;
            assert_eq!(bound.to_string(), expr.to_string());
            assert_eq!(
                bound.display_tree().to_string(),
                expr.display_tree().to_string()
            );
        }
        Ok(())
    }

    #[test]
    fn variable_binds_to_its_lexical_frame() -> VortexResult<()> {
        let scope = scope().push_frame(Frame::try_new([(
            Variable::new("value"),
            DType::Primitive(PType::I64, Nullability::Nullable),
        )])?);

        let bound = var("value").bind_scope(&scope)?;
        let (variable, reference) = bound
            .as_variable()
            .vortex_expect("variable must remain bound");

        assert_eq!(variable, &Variable::new("value"));
        assert_eq!(reference.frame(), 0);
        assert_eq!(reference.slot(), 0);
        assert_eq!(
            bound.dtype(),
            &DType::Primitive(PType::I64, Nullability::Nullable)
        );
        Ok(())
    }

    #[test]
    fn variable_validity_is_deferred_to_its_bound_array() -> VortexResult<()> {
        let scope = scope().push_frame(Frame::try_new([(
            Variable::new("value"),
            DType::Primitive(PType::I32, Nullability::Nullable),
        )])?);
        let expression = checked_add(var("value"), lit(1_i32));

        let validity = expression.validity()?;
        assert!(validity.contains::<IsNotNull>()?);
        assert_eq!(
            validity.bind_scope(&scope)?.dtype(),
            &DType::Bool(Nullability::NonNullable)
        );
        Ok(())
    }

    #[test]
    fn binding_lambda_syntax_returns_an_error() -> VortexResult<()> {
        assert!(lambda(["x"], var("x"))?.bind_scope(&scope()).is_err());
        assert!(
            eq(lambda(["x"], var("x"))?, lit(1_i32))
                .bind_scope(&scope())
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn duplicate_lambda_parameters_are_rejected() {
        assert!(lambda(["x", "x"], var("x")).is_err());
    }

    #[test]
    fn structural_and_exact_equality_are_distinct() -> VortexResult<()> {
        let expr = eq(col("a"), lit(1_i32));
        let bound = expr.bind_scope(&scope())?;
        let independently_bound = expr.bind_scope(&scope())?;

        assert_eq!(bound, independently_bound);
        assert_eq!(ExactBoundExpr(bound.clone()), ExactBoundExpr(bound.clone()));
        assert_ne!(ExactBoundExpr(bound), ExactBoundExpr(independently_bound));
        Ok(())
    }

    #[test]
    fn binding_reports_a_type_error() {
        assert!(eq(col("a"), lit("nope")).bind_scope(&scope()).is_err());
    }
}
