// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Nullable execution strategies derived from a concrete row dispatch.

use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::scalar_fn::ElementTuple;
use crate::scalar_fn::SinkResult;

/// The execution policy and output dtype selected by a planning visit.
pub struct BatchPlan {
    /// The non-nullable dtype built by the selected output capability.
    pub output_dtype: DType,

    /// How this concrete dispatch executes nullable rows.
    pub policy: RowPolicy,
}

impl BatchPlan {
    /// Return the output dtype widened with strict input nullability.
    pub fn result_dtype(&self, args: &[DType]) -> DType {
        let nullability = self.output_dtype.nullability()
            | Nullability::from(args.iter().any(DType::is_nullable));

        self.output_dtype.with_nullability(nullability)
    }
}

/// The nullable execution policy derived from one concrete dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowPolicy {
    /// Evaluate all rows and mask the result.
    Dense,

    /// Evaluate all rows, retrying only valid rows if a deferred error is raised.
    DenseWithRetry,

    /// Execute only valid rows, trying skip-invalid execution before filtering.
    ValidOnly,
}

impl RowPolicy {
    /// The policy for an infallible owned output.
    pub const fn for_owned_output<Args: ElementTuple>() -> Self {
        if Args::DENSE_SAFE && !Args::DECODE_FALLIBLE {
            Self::Dense
        } else {
            Self::ValidOnly
        }
    }

    /// The policy for an owned output carrying batch-deferred failure evidence.
    pub const fn for_deferred_output<Args: ElementTuple>() -> Self {
        if Args::DENSE_SAFE && !Args::DECODE_FALLIBLE {
            Self::DenseWithRetry
        } else {
            Self::ValidOnly
        }
    }

    /// The policy one concrete dispatch executes nullable rows under.
    ///
    /// Batch execution always tries [`reduce_encoded`](crate::scalar_fn::RowFn::reduce_encoded)
    /// against the original arrays before it tries the sink or filters the inputs. Skipping that
    /// probe can change the result of an encoding-aware function.
    pub const fn for_sink<Args: ElementTuple, ApplyResult: SinkResult>() -> Self {
        if Args::DENSE_SAFE && !Args::DECODE_FALLIBLE && !ApplyResult::FALLIBLE {
            if ApplyResult::DEFERRED {
                Self::DenseWithRetry
            } else {
                Self::Dense
            }
        } else {
            Self::ValidOnly
        }
    }
}
