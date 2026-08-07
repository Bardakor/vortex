// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Expression-level type coercion pass.

use vortex_error::VortexResult;

use crate::dtype::DType;
use crate::expr::Expression;
use crate::expr::cast;
use crate::scalar_fn::fns::literal::Literal;

/// Rewrite an expression tree to insert casts where a scalar function's `coerce_args` demands
/// a different type than what the child currently produces.
///
/// The rewrite is bottom-up: children are coerced first, then each parent node checks whether
/// its children match the coerced argument types.
pub fn coerce_expression(expr: Expression, scope: &DType) -> VortexResult<Expression> {
    // A lambda is a coercion boundary. Its body types against a parameter frame that this pass
    // does not carry, so descending into one would try to type a variable against the root dtype
    // and fail. Leave it for whoever binds the lambda and knows the parameter types. Recursing
    // explicitly rather than using `transform_up` is what makes skipping the body possible.
    fn coerce_node(node: Expression, scope: &DType) -> VortexResult<Expression> {
        if node.as_lambda().is_some() {
            return Ok(node);
        }

        let coerced_children = node
            .children()
            .iter()
            .cloned()
            .map(|child| coerce_node(child, scope))
            .collect::<VortexResult<Vec<_>>>()?;
        let node = node.with_children(coerced_children)?;

        coerce_one(node, scope)
    }

    fn coerce_one(node: Expression, scope: &DType) -> VortexResult<Expression> {
        let scope = scope.clone();
        {
            // Leaf nodes (Root, Literal) have no children to coerce.
            if node.is_root() || node.is::<Literal>() || node.children().is_empty() {
                return Ok(node);
            }

            // Compute the current child return types.
            let child_dtypes: Vec<DType> = node
                .children()
                .iter()
                .map(|c| c.return_dtype(&scope))
                .collect::<VortexResult<_>>()?;

            // Ask the scalar function what types it wants.
            let Some(scalar_fn) = node.as_scalar() else {
                return Ok(node);
            };
            let coerced_dtypes = scalar_fn.coerce_args(&child_dtypes)?;

            // If nothing changed, skip.
            if child_dtypes == coerced_dtypes {
                return Ok(node);
            }

            // Build new children, inserting casts where needed.
            let new_children: Vec<Expression> = node
                .children()
                .iter()
                .zip(coerced_dtypes.iter())
                .map(|(child, target)| {
                    let child_dtype = child.return_dtype(&scope)?;
                    if child_dtype.eq_ignore_nullability(target)
                        && child_dtype.nullability() == target.nullability()
                    {
                        Ok(child.clone())
                    } else {
                        Ok(cast(child.clone(), target.clone()))
                    }
                })
                .collect::<VortexResult<_>>()?;

            node.with_children(new_children)
        }
    }

    coerce_node(expr, scope)
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::expr::col;
    use crate::expr::lit;
    use crate::expr::transform::coerce::coerce_expression;
    use crate::scalar::Scalar;
    use crate::scalar_fn::ScalarFnVTableExt;
    use crate::scalar_fn::fns::binary::Binary;
    use crate::scalar_fn::fns::cast::Cast;
    use crate::scalar_fn::fns::operators::Operator;

    fn test_scope() -> DType {
        DType::Struct(
            StructFields::new(
                ["x", "y"].into(),
                vec![
                    DType::Primitive(PType::I32, NonNullable),
                    DType::Primitive(PType::I64, NonNullable),
                ],
            ),
            NonNullable,
        )
    }

    #[test]
    fn mixed_type_comparison_inserts_cast() -> VortexResult<()> {
        let scope = test_scope();
        // x (I32) < y (I64) => should cast x to I64
        let expr = Binary.new_expr(Operator::Lt, [col("x"), col("y")]);
        let coerced = coerce_expression(expr, &scope)?;

        // The LHS child should now be a cast expression
        assert!(coerced.child(0).is::<Cast>());
        // The coerced LHS should return I64
        assert_eq!(
            coerced.child(0).return_dtype(&scope)?,
            DType::Primitive(PType::I64, NonNullable)
        );
        // The RHS should be unchanged
        assert!(!coerced.child(1).is::<Cast>());
        Ok(())
    }

    #[test]
    fn same_type_comparison_no_cast() -> VortexResult<()> {
        let scope = test_scope();
        // x (I32) < x (I32) => no cast needed
        let expr = Binary.new_expr(Operator::Lt, [col("x"), col("x")]);
        let coerced = coerce_expression(expr, &scope)?;

        // Neither child should be a cast
        assert!(!coerced.child(0).is::<Cast>());
        assert!(!coerced.child(1).is::<Cast>());
        Ok(())
    }

    #[test]
    fn mixed_type_arithmetic_coerces_both() -> VortexResult<()> {
        let scope = DType::Struct(
            StructFields::new(
                ["a", "b"].into(),
                vec![
                    DType::Primitive(PType::U8, NonNullable),
                    DType::Primitive(PType::I32, NonNullable),
                ],
            ),
            NonNullable,
        );
        // a (U8) + b (I32) => both should be coerced to I32
        // U8 + I32: unsigned_signed_supertype(U8, I32) => max(1,4)=4 => I64
        let expr = Binary.new_expr(Operator::Add, [col("a"), col("b")]);
        let coerced = coerce_expression(expr, &scope)?;

        // LHS (U8) should be cast
        assert!(coerced.child(0).is::<Cast>());
        // Both should return the same supertype
        let lhs_dt = coerced.child(0).return_dtype(&scope)?;
        let rhs_dt = coerced.child(1).return_dtype(&scope)?;
        assert_eq!(lhs_dt, rhs_dt);
        Ok(())
    }

    #[test]
    fn decimal_arithmetic_coerces_precision_and_scale() -> VortexResult<()> {
        let common_dtype = DType::Decimal(DecimalDType::new(4, 2), NonNullable);
        let result_dtype = DType::Decimal(DecimalDType::new(5, 2), NonNullable);
        let scope = DType::Struct(
            StructFields::new(
                ["a", "b"].into(),
                vec![
                    DType::Decimal(DecimalDType::new(3, 1), NonNullable),
                    common_dtype,
                ],
            ),
            NonNullable,
        );
        let expr = Binary.new_expr(Operator::Add, [col("a"), col("b")]);

        let coerced = coerce_expression(expr, &scope)?;

        assert!(coerced.child(0).is::<Cast>());
        assert!(!coerced.child(1).is::<Cast>());
        assert_eq!(coerced.return_dtype(&scope)?, result_dtype);
        Ok(())
    }

    #[test]
    fn boolean_operators_no_coercion() -> VortexResult<()> {
        let scope = DType::Struct(
            StructFields::new(
                ["p", "q"].into(),
                vec![DType::Bool(NonNullable), DType::Bool(NonNullable)],
            ),
            NonNullable,
        );
        let expr = Binary.new_expr(Operator::And, [col("p"), col("q")]);
        let coerced = coerce_expression(expr, &scope)?;

        assert!(!coerced.child(0).is::<Cast>());
        assert!(!coerced.child(1).is::<Cast>());
        Ok(())
    }

    #[test]
    fn literal_coercion() -> VortexResult<()> {
        let scope = DType::Struct(
            StructFields::new(
                ["x"].into(),
                vec![DType::Primitive(PType::I64, NonNullable)],
            ),
            NonNullable,
        );
        // x (I64) + 1i32 => literal should be cast to I64
        let expr = Binary.new_expr(Operator::Add, [col("x"), lit(Scalar::from(1i32))]);
        let coerced = coerce_expression(expr, &scope)?;

        // The RHS (literal) should be cast to I64
        assert!(coerced.child(1).is::<Cast>());
        assert_eq!(
            coerced.child(1).return_dtype(&scope)?,
            DType::Primitive(PType::I64, NonNullable)
        );
        Ok(())
    }
}

#[cfg(test)]
mod lambda_tests {
    use vortex_error::VortexResult;

    use super::*;
    use crate::expr::Expression;
    use crate::expr::checked_add;
    use crate::expr::col;
    use crate::expr::lambda;
    use crate::expr::lit;
    use crate::expr::test_harness::struct_dtype;
    use crate::expr::var;

    /// A lambda body types against a parameter frame, which this pass does not carry. Descending
    /// into one would try to type the variable against the root dtype and fail, so a lambda is a
    /// coercion boundary and is returned untouched.
    #[test]
    fn a_lambda_is_a_coercion_boundary() -> VortexResult<()> {
        let l = Expression::from(lambda(["x"], checked_add(var("x"), lit(1i32))));
        assert_eq!(coerce_expression(l.clone(), &struct_dtype())?, l);
        Ok(())
    }

    /// The boundary must not stop coercion of everything around it.
    #[test]
    fn coercion_still_applies_outside_a_lambda() -> VortexResult<()> {
        let expr = checked_add(col("a"), lit(1i64));
        let coerced = coerce_expression(expr.clone(), &struct_dtype())?;
        assert_ne!(
            coerced, expr,
            "an i32 column against an i64 literal should coerce"
        );
        Ok(())
    }
}
