// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::expr::Expression;
use crate::expr::analysis::BooleanLabels;
use crate::expr::label_tree;

pub fn label_is_fallible(expr: &Expression) -> BooleanLabels<'_> {
    label_tree(
        expr,
        |expr| match expr {
            Expression::Scalar { scalar_fn, .. } => scalar_fn.signature().is_fallible(),
            Expression::Lambda(lambda) => is_fallible(&lambda.body()),
            Expression::Root | Expression::Variable(_) => false,
        },
        |acc, &child| acc | child,
    )
}

fn is_fallible(expr: &Expression) -> bool {
    match expr {
        Expression::Scalar {
            scalar_fn,
            children,
        } => scalar_fn.signature().is_fallible() || children.iter().any(is_fallible),
        Expression::Lambda(lambda) => is_fallible(lambda.body()),
        Expression::Root | Expression::Variable(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::checked_add;
    use crate::expr::col;
    use crate::expr::eq;
    use crate::expr::is_null;
    use crate::expr::lambda;
    use crate::expr::lit;
    use crate::expr::merge_opts;
    use crate::expr::not;
    use crate::expr::var;
    use crate::scalar_fn::fns::merge::DuplicateHandling;

    #[test]
    fn not_is_not_fallible() {
        let expr = not(col("x"));
        let labels = label_is_fallible(&expr);
        assert_eq!(labels.get(&expr), Some(&false));
    }

    #[test]
    fn checked_add_defaults_to_fallible() {
        let expr = checked_add(col("a"), col("b"));
        let labels = label_is_fallible(&expr);
        assert_eq!(labels.get(&expr), Some(&true));
    }

    #[test]
    fn eq_not_fallible() {
        let expr = eq(col("a"), lit(5));
        let labels = label_is_fallible(&expr);
        assert_eq!(labels.get(&expr), Some(&false));
    }

    #[test]
    fn merge_with_error_handling_is_fallible() {
        let expr = merge_opts([col("a"), col("b")], DuplicateHandling::Error);
        let labels = label_is_fallible(&expr);
        assert_eq!(labels.get(&expr), Some(&true));
    }

    #[test]
    fn merge_with_rightmost_handling_is_not_fallible() {
        let expr = merge_opts([col("a"), col("b")], DuplicateHandling::RightMost);
        let labels = label_is_fallible(&expr);
        assert_eq!(labels.get(&expr), Some(&false));
    }

    #[test]
    fn nested_with_fallible_child() {
        let child = checked_add(col("a"), col("b"));
        let expr = not(child.clone());
        let labels = label_is_fallible(&expr);
        assert_eq!(labels.get(&child), Some(&true));
        assert_eq!(labels.get(&expr), Some(&true));
    }

    #[test]
    fn nested_without_fallible_child() {
        let child = is_null(col("x"));
        let expr = not(child.clone());
        let labels = label_is_fallible(&expr);
        assert_eq!(labels.get(&child), Some(&false));
        assert_eq!(labels.get(&expr), Some(&false));
    }

    #[test]
    fn lambda_fallibility_from_body() -> vortex_error::VortexResult<()> {
        let fallible = lambda(["x"], checked_add(var("x"), lit(1i32)))?;
        assert_eq!(label_is_fallible(&fallible).get(&fallible), Some(&true));

        let infallible = lambda(["x"], var("x"))?;
        assert_eq!(
            label_is_fallible(&infallible).get(&infallible),
            Some(&false)
        );
        Ok(())
    }
}
