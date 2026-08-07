// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_proto::expr as pb;
use vortex_session::VortexSession;

use crate::expr::Expression;
use crate::expr::Lambda;
use crate::expr::Variable;
use crate::scalar_fn::ForeignScalarFnVTable;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::session::ScalarFnSessionExt;

pub trait ExprSerializeProtoExt {
    /// Serialize the expression to its protobuf representation.
    fn serialize_proto(&self) -> VortexResult<pb::Expr>;
}

/// The wire id for [`Expression::Root`], retained from when `Root` was a scalar function so that
/// already-serialized expressions keep round-tripping.
pub(crate) const ROOT_ID: &str = "vortex.root";

/// The wire id for [`Expression::Variable`].
pub(crate) const VARIABLE_ID: &str = "vortex.var";

/// The wire id for [`Expression::Lambda`]. Its body is the message's single child.
pub(crate) const LAMBDA_ID: &str = "vortex.lambda";

impl ExprSerializeProtoExt for Expression {
    fn serialize_proto(&self) -> VortexResult<pb::Expr> {
        let scalar_fn = match self {
            Expression::Root => {
                return Ok(pb::Expr {
                    id: ROOT_ID.to_string(),
                    children: vec![],
                    metadata: Some(vec![]),
                });
            }
            Expression::Variable(variable) => {
                return Ok(pb::Expr {
                    id: VARIABLE_ID.to_string(),
                    children: vec![],
                    metadata: Some(
                        pb::VariableOpts {
                            name: variable.name().to_string(),
                        }
                        .encode_to_vec(),
                    ),
                });
            }
            Expression::Lambda(lambda) => {
                return Ok(pb::Expr {
                    id: LAMBDA_ID.to_string(),
                    children: vec![lambda.body().serialize_proto()?],
                    metadata: Some(
                        pb::LambdaOpts {
                            params: lambda
                                .params()
                                .iter()
                                .map(|v| v.name().to_string())
                                .collect(),
                        }
                        .encode_to_vec(),
                    ),
                });
            }
            Expression::Scalar { scalar_fn, .. } => scalar_fn,
        };

        let children = self
            .children()
            .iter()
            .map(|child| child.serialize_proto())
            .try_collect()?;

        let metadata = scalar_fn.options().serialize()?.ok_or_else(|| {
            vortex_err!(
                "Expression '{}' is not serializable: {}",
                scalar_fn.id(),
                self
            )
        })?;

        Ok(pb::Expr {
            id: scalar_fn.id().to_string(),
            children,
            metadata: Some(metadata),
        })
    }
}

impl Expression {
    pub fn from_proto(expr: &pb::Expr, session: &VortexSession) -> VortexResult<Expression> {
        // These are language primitives rather than registered scalar fns, so they must be
        // resolved before the registry lookup below.
        if expr.id == ROOT_ID {
            vortex_ensure!(
                expr.children.is_empty(),
                "root expression must have no children, got {}",
                expr.children.len()
            );
            return Ok(Expression::Root);
        }

        if expr.id == VARIABLE_ID {
            vortex_ensure!(
                expr.children.is_empty(),
                "a variable must have no children, got {}",
                expr.children.len()
            );
            let opts = pb::VariableOpts::decode(expr.metadata())?;
            return Ok(Expression::Variable(Variable::new(opts.name)));
        }

        if expr.id == LAMBDA_ID {
            vortex_ensure!(
                expr.children.len() == 1,
                "a lambda must have exactly one child, its body, got {}",
                expr.children.len()
            );
            let opts = pb::LambdaOpts::decode(expr.metadata())?;
            let body = Expression::from_proto(&expr.children[0], session)?;
            return Ok(Expression::from(Lambda::new(
                opts.params.into_iter().map(Variable::new),
                body,
            )));
        }

        #[expect(clippy::disallowed_methods, reason = "interning a dynamic id")]
        let expr_id = ScalarFnId::new(expr.id.as_str());
        let children = expr
            .children
            .iter()
            .map(|e| Expression::from_proto(e, session))
            .collect::<VortexResult<Vec<_>>>()?;

        let scalar_fn = if let Some(vtable) = session.scalar_fns().registry().get(&expr_id) {
            vtable.deserialize(expr.metadata(), session)?
        } else if session.allows_unknown() {
            ForeignScalarFnVTable::make_scalar_fn(expr_id, expr.metadata().to_vec(), children.len())
        } else {
            return Err(vortex_err!("unknown expression id: {}", expr_id));
        };

        Expression::try_new(scalar_fn, children)
    }
}

/// Deserialize a [`Expression`] from the protobuf representation.
#[deprecated(note = "Use Expression::from_proto instead")]
pub fn deserialize_expr_proto(
    expr: &pb::Expr,
    session: &VortexSession,
) -> VortexResult<Expression> {
    Expression::from_proto(expr, session)
}

#[cfg(test)]
mod tests {
    use prost::Message;
    use vortex_error::VortexResult;
    use vortex_proto::expr as pb;
    use vortex_session::VortexSession;

    use super::ExprSerializeProtoExt;
    use crate::array_session;
    use crate::expr::Expression;
    use crate::expr::and;
    use crate::expr::between;
    use crate::expr::eq;
    use crate::expr::get_item;
    use crate::expr::lit;
    use crate::expr::or;
    use crate::expr::root;
    use crate::scalar_fn::fns::between::BetweenOptions;
    use crate::scalar_fn::fns::between::StrictComparison;
    use crate::scalar_fn::session::ScalarFnSession;

    #[test]
    fn expression_serde() {
        let expr: Expression = or(
            and(
                between(
                    lit(1),
                    root(),
                    get_item("a", root()),
                    BetweenOptions {
                        lower_strict: StrictComparison::Strict,
                        upper_strict: StrictComparison::Strict,
                    },
                ),
                lit(1),
            ),
            eq(lit(1), root()),
        );

        let s_expr = expr.serialize_proto().unwrap();
        let buf = s_expr.encode_to_vec();
        let s_expr = pb::Expr::decode(buf.as_slice()).unwrap();
        let deser_expr = Expression::from_proto(&s_expr, &array_session()).unwrap();

        assert_eq!(&deser_expr, &expr);
    }

    /// Variables and lambdas round-trip on reserved ids, like `Root`. The body travels as the
    /// lambda message's single child, so a generic walk of the wire format still sees it.
    #[test]
    fn variables_and_lambdas_round_trip() -> VortexResult<()> {
        use crate::expr::checked_add;
        use crate::expr::lambda;
        use crate::expr::var;

        for expr in [
            var("x"),
            Expression::from(lambda(["x"], var("x"))),
            Expression::from(lambda(
                ["x", "y"],
                checked_add(var("x"), checked_add(var("y"), lit(1_i32))),
            )),
            // Nested, so the inner lambda travels inside the outer one's body.
            Expression::from(lambda(["x"], Expression::from(lambda(["y"], var("y"))))),
        ] {
            let bytes = expr.serialize_proto()?.encode_to_vec();
            let decoded =
                Expression::from_proto(&pb::Expr::decode(bytes.as_slice())?, &array_session())?;
            assert_eq!(decoded, expr, "round trip changed {expr}");
        }
        Ok(())
    }

    /// A lambda carries its body as a child, so a malformed message must be rejected rather than
    /// silently producing a lambda with the wrong arity.
    #[test]
    fn a_lambda_without_a_body_is_rejected() {
        let malformed = pb::Expr {
            id: "vortex.lambda".to_string(),
            children: vec![],
            metadata: Some(
                pb::LambdaOpts {
                    params: vec!["x".to_string()],
                }
                .encode_to_vec(),
            ),
        };
        assert!(Expression::from_proto(&malformed, &array_session()).is_err());
    }

    #[test]
    fn unknown_expression_id_allow_unknown() {
        let session = VortexSession::empty().with::<ScalarFnSession>();
        session.allow_unknown();

        let expr_proto = pb::Expr {
            id: "vortex.test.foreign_scalar_fn".to_string(),
            metadata: Some(vec![1, 2, 3, 4]),
            children: vec![root().serialize_proto().unwrap()],
        };

        let expr = Expression::from_proto(&expr_proto, &session).unwrap();
        assert_eq!(
            expr.as_scalar().map(|f| f.id().as_ref().to_string()),
            Some("vortex.test.foreign_scalar_fn".to_string())
        );

        let roundtrip = expr.serialize_proto().unwrap();
        assert_eq!(roundtrip.id, expr_proto.id);
        assert_eq!(roundtrip.metadata(), expr_proto.metadata());
        assert_eq!(roundtrip.children.len(), 1);
    }
}
