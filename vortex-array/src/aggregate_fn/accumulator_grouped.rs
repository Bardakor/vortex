// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;
use std::sync::OnceLock;

use num_traits::ToPrimitive;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateFn;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::DynAccumulator;
use crate::aggregate_fn::session::AggregateFnSessionExt;
use crate::array::ArrayId;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
use crate::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use crate::arrays::listview::ListViewArraySlotsExt;
use crate::builders::builder_with_capacity;
use crate::builtins::ArrayBuiltins;
use crate::columnar::AnyColumnar;
use crate::columnar::Columnar;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::executor::max_iterations;
use crate::match_each_integer_ptype;
use crate::scalar::Scalar;
use crate::validity::Validity;

/// Reference-counted type-erased grouped accumulator.
pub type GroupedAccumulatorRef = Box<dyn DynGroupedAccumulator>;

/// A canonical list representation used to adapt list-shaped groups to dense group IDs.
pub enum GroupedArray {
    /// Groups represented as a list-view array with per-group offsets and sizes.
    ListView(ListViewArray),
    /// Groups represented as a fixed-size list array.
    FixedSizeList(FixedSizeListArray),
}

impl From<ListViewArray> for GroupedArray {
    fn from(groups: ListViewArray) -> Self {
        Self::ListView(groups)
    }
}

impl From<FixedSizeListArray> for GroupedArray {
    fn from(groups: FixedSizeListArray) -> Self {
        Self::FixedSizeList(groups)
    }
}

impl GroupedArray {
    /// Return the inner element array shared by all groups.
    pub fn elements(&self) -> &ArrayRef {
        match self {
            Self::ListView(groups) => groups.elements(),
            Self::FixedSizeList(groups) => groups.elements(),
        }
    }

    /// Return the physical element ranges for each group.
    pub fn group_ranges(&self, ctx: &mut ExecutionCtx) -> VortexResult<GroupRanges> {
        match self {
            Self::ListView(groups) => list_view_group_ranges(groups, ctx),
            Self::FixedSizeList(groups) => Ok(fixed_size_list_group_ranges(groups)),
        }
    }

    /// Return the per-group validity mask.
    pub fn group_validity(&self, ctx: &mut ExecutionCtx) -> VortexResult<Mask> {
        match self {
            Self::ListView(groups) => groups.validity()?.execute_mask(groups.len(), ctx),
            Self::FixedSizeList(groups) => groups.validity()?.execute_mask(groups.len(), ctx),
        }
    }

    /// Return the number of groups.
    pub fn len(&self) -> usize {
        match self {
            Self::ListView(groups) => groups.len(),
            Self::FixedSizeList(groups) => groups.len(),
        }
    }

    /// Return whether there are no groups.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Convert list-shaped groups into a values array and parallel dense group IDs.
    ///
    /// Null groups contribute no values. Contiguous, all-valid groups reuse the original element
    /// array; arbitrary list views are gathered into group order.
    pub fn dense_input(&self, ctx: &mut ExecutionCtx) -> VortexResult<(ArrayRef, GroupIds)> {
        let num_groups = self.len();
        validate_num_groups(num_groups)?;
        let ranges = self.group_ranges(ctx)?;
        let validity = self.group_validity(ctx)?;
        let mut rows = Vec::new();
        let mut ids = Vec::new();
        let mut identity = true;

        for (group, ((offset, size), valid)) in ranges.iter().zip(validity.iter()).enumerate() {
            if !valid {
                identity = false;
                continue;
            }
            let group = u32::try_from(group)?;
            for row in offset..offset + size {
                identity &= row == rows.len();
                rows.push(u64::try_from(row)?);
                ids.push(group);
            }
        }

        identity &= rows.len() == self.elements().len();
        let values = if identity {
            self.elements().clone()
        } else {
            self.elements()
                .clone()
                .take(Buffer::from_iter(rows).into_array())?
        };
        Ok((values, GroupIds::from_iter(ids, num_groups)?))
    }
}

/// The physical element ranges of a canonical grouped list array.
pub enum GroupRanges {
    /// Explicit ranges extracted from a list-view array.
    ListView {
        /// The `(offset, size)` ranges.
        ranges: Vec<(usize, usize)>,
    },
    /// Uniform ranges derived from a fixed-size list array.
    FixedSizeList {
        /// The number of groups.
        len: usize,
        /// The number of elements in each group.
        size: usize,
    },
}

impl GroupRanges {
    /// Return the number of groups described by these ranges.
    pub fn len(&self) -> usize {
        match self {
            Self::ListView { ranges } => ranges.len(),
            Self::FixedSizeList { len, .. } => *len,
        }
    }

    /// Return whether no groups are described.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn range(&self, index: usize) -> (usize, usize) {
        match self {
            Self::ListView { ranges } => ranges[index],
            Self::FixedSizeList { len, size } => {
                assert!(index < *len, "group range index out of bounds");
                (index * *size, *size)
            }
        }
    }

    /// Iterate over `(offset, size)` ranges.
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..self.len()).map(|index| self.range(index))
    }
}

/// Encoded group ids parallel to a grouped aggregate input batch.
///
/// The array must contain non-null `u32` ordinals. The ordinals are dense state slots in
/// `0..num_groups`, not raw group keys. Range validation may require executing the encoded array,
/// so kernels that can prove the invariant from encoded metadata should avoid materializing and
/// otherwise call [`Self::validated_ids`] before indexing group state.
#[derive(Clone, Debug)]
pub struct GroupIds {
    ids: ArrayRef,
    num_groups: usize,
    validated: Arc<OnceLock<Buffer<u32>>>,
}

impl GroupIds {
    /// Create group ids from an encoded non-null `u32` array.
    pub fn new(ids: ArrayRef, num_groups: usize) -> VortexResult<Self> {
        validate_num_groups(num_groups)?;
        vortex_ensure!(
            ids.dtype() == &DType::Primitive(PType::U32, Nullability::NonNullable),
            "Group ids must be non-nullable u32, got {}",
            ids.dtype()
        );
        Ok(Self {
            ids,
            num_groups,
            validated: Arc::new(OnceLock::new()),
        })
    }

    /// Create group ids from a materialized buffer, validating the dense-id invariant once.
    pub fn from_buffer(ids: Buffer<u32>, num_groups: usize) -> VortexResult<Self> {
        validate_group_ids(ids.as_ref(), num_groups)?;
        Ok(Self {
            ids: PrimitiveArray::new(ids.clone(), Validity::NonNullable).into_array(),
            num_groups,
            validated: Arc::new(OnceLock::from(ids)),
        })
    }

    /// Create group ids from materialized values.
    pub fn from_iter(ids: impl IntoIterator<Item = u32>, num_groups: usize) -> VortexResult<Self> {
        Self::from_buffer(Buffer::from_iter(ids), num_groups)
    }

    /// Return the encoded ids array.
    pub fn ids(&self) -> &ArrayRef {
        &self.ids
    }

    /// Return the number of dense group state slots.
    pub fn num_groups(&self) -> usize {
        self.num_groups
    }

    /// Return the number of ids.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Return whether there are no ids.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Return the encoding id for kernel dispatch.
    pub fn encoding_id(&self) -> ArrayId {
        self.ids.encoding_id()
    }

    /// Execute the ids to a native buffer and validate every id is in range.
    ///
    /// The validated buffer is cached and shared by clones, so multiple aggregate functions over
    /// the same group ids pay materialization and validation at most once.
    pub fn validated_ids(&self, ctx: &mut ExecutionCtx) -> VortexResult<Buffer<u32>> {
        if let Some(ids) = self.validated.get() {
            return Ok(ids.clone());
        }

        let ids = self.ids.clone().execute::<Buffer<u32>>(ctx)?;
        validate_group_ids(ids.as_ref(), self.num_groups)?;
        drop(self.validated.set(ids.clone()));
        Ok(ids)
    }
}

/// Aggregate-owned dense state used by grouped accumulation.
pub trait GroupedState: 'static + Send {
    /// Expose the concrete state container to a typed grouped kernel.
    fn as_any_mut(&mut self) -> &mut dyn Any;

    /// Return the number of allocated group slots.
    fn len(&self) -> usize;

    /// Return whether no group slots are allocated.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Ensure that at least `num_groups` state slots exist.
    fn ensure_groups(&mut self, num_groups: usize) -> VortexResult<()>;

    /// Return whether one group has reached a terminal state.
    fn is_saturated(&self, group_id: usize) -> bool;

    /// Combine one scalar partial into a group.
    fn combine_scalar(&mut self, group_id: usize, partial: Scalar) -> VortexResult<()>;

    /// Read one group's partial state.
    fn partial_scalar(&self, group_id: usize) -> VortexResult<Scalar>;

    /// Fold an array of partial states into dense groups.
    fn accumulate_partials(
        &mut self,
        partials: &ArrayRef,
        group_ids: &[u32],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        for (row_idx, &group_id) in group_ids.iter().enumerate() {
            self.combine_scalar(group_id as usize, partials.execute_scalar(row_idx, ctx)?)?;
        }
        Ok(())
    }

    /// Flush all partial states into an array and reset the container.
    fn flush_partials(&mut self, num_groups: usize) -> VortexResult<ArrayRef>;
}

/// Default grouped state backed by one aggregate partial value per group.
pub(crate) struct DefaultGroupedState<V: AggregateFnVTable> {
    vtable: V,
    options: V::Options,
    input_dtype: DType,
    partial_dtype: DType,
    partials: Vec<V::Partial>,
}

impl<V: AggregateFnVTable> DefaultGroupedState<V> {
    pub(crate) fn new(
        vtable: V,
        options: V::Options,
        input_dtype: DType,
        partial_dtype: DType,
    ) -> Self {
        Self {
            vtable,
            options,
            input_dtype,
            partial_dtype,
            partials: Vec::new(),
        }
    }
}

impl<V: AggregateFnVTable> GroupedState for DefaultGroupedState<V> {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn len(&self) -> usize {
        self.partials.len()
    }

    fn ensure_groups(&mut self, num_groups: usize) -> VortexResult<()> {
        self.partials
            .reserve(num_groups.saturating_sub(self.partials.len()));
        while self.partials.len() < num_groups {
            self.partials.push(
                self.vtable
                    .empty_partial(&self.options, &self.input_dtype)?,
            );
        }
        Ok(())
    }

    fn is_saturated(&self, group_id: usize) -> bool {
        self.vtable.is_saturated(&self.partials[group_id])
    }

    fn combine_scalar(&mut self, group_id: usize, partial: Scalar) -> VortexResult<()> {
        self.vtable
            .combine_partials(&mut self.partials[group_id], partial)
    }

    fn partial_scalar(&self, group_id: usize) -> VortexResult<Scalar> {
        if let Some(partial) = self.partials.get(group_id) {
            self.vtable.to_scalar(partial)
        } else {
            let partial = self
                .vtable
                .empty_partial(&self.options, &self.input_dtype)?;
            self.vtable.to_scalar(&partial)
        }
    }

    fn flush_partials(&mut self, num_groups: usize) -> VortexResult<ArrayRef> {
        vortex_ensure!(
            num_groups >= self.partials.len(),
            "Cannot flush {} groups after accumulating {} groups",
            num_groups,
            self.partials.len()
        );
        self.ensure_groups(num_groups)?;

        if let Some(states) = self
            .vtable
            .partials_to_array(&self.partials, &self.partial_dtype)?
        {
            vortex_ensure!(
                states.dtype() == &self.partial_dtype,
                "Partial array DType mismatch: expected {}, got {}",
                self.partial_dtype,
                states.dtype()
            );
            self.partials.clear();
            return Ok(states);
        }

        let mut states = builder_with_capacity(&self.partial_dtype, num_groups);
        for partial in &self.partials {
            states.append_scalar(&self.vtable.to_scalar(partial)?)?;
        }
        self.partials.clear();
        Ok(states.finish())
    }
}

/// An accumulator used for computing aggregates over group ids.
///
/// Group ids are caller-assigned `u32` ordinals in the dense range `0..num_groups`. Input batches
/// may repeat, omit, and reorder those ids, but every id must identify a state slot rather than a
/// raw group key. The accumulator keeps one partial state per slot, so ordered and unordered
/// grouping only differ in how the caller assigns ids.
pub struct GroupedAccumulator<V: AggregateFnVTable> {
    /// The vtable of the aggregate function.
    vtable: V,
    /// The options of the aggregate function.
    options: V::Options,
    /// Type-erased aggregate function used for kernel dispatch.
    aggregate_fn: AggregateFnRef,
    /// The DType of the input.
    dtype: DType,
    /// The DType of the aggregate.
    return_dtype: DType,
    /// The DType of the partial accumulator state.
    partial_dtype: DType,
    /// Aggregate-owned dense per-group state.
    state: Box<dyn GroupedState>,
}

impl<V: AggregateFnVTable> GroupedAccumulator<V> {
    pub fn try_new(vtable: V, options: V::Options, dtype: DType) -> VortexResult<Self> {
        let aggregate_fn = AggregateFn::new(vtable.clone(), options.clone()).erased();
        let return_dtype = vtable.return_dtype(&options, &dtype).ok_or_else(|| {
            vortex_err!(
                "Aggregate function {} cannot be applied to dtype {}",
                vtable.id(),
                dtype
            )
        })?;
        let partial_dtype = vtable.partial_dtype(&options, &dtype).ok_or_else(|| {
            vortex_err!(
                "Aggregate function {} cannot be applied to dtype {}",
                vtable.id(),
                dtype
            )
        })?;
        let state = vtable.grouped_state(&options, &dtype, &partial_dtype)?;

        Ok(Self {
            vtable,
            options,
            aggregate_fn,
            dtype,
            return_dtype,
            partial_dtype,
            state,
        })
    }

    fn ensure_groups(&mut self, num_groups: usize) -> VortexResult<()> {
        validate_num_groups(num_groups)?;

        self.state.ensure_groups(num_groups)
    }

    fn try_accumulate_kernel(
        &mut self,
        batch: &ArrayRef,
        group_ids: &GroupIds,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        let session = ctx.session().clone();

        if let Some(kernel) = session.aggregate_fns().find_grouped_kernel(
            self.aggregate_fn.id(),
            batch.encoding_id(),
            group_ids.encoding_id(),
        ) && kernel.grouped_accumulate(
            &self.aggregate_fn,
            batch,
            group_ids,
            self.state.as_any_mut(),
            ctx,
        )? {
            return Ok(true);
        }

        Ok(false)
    }

    fn accumulate_fallback(
        &mut self,
        batch: &ArrayRef,
        group_ids: &[u32],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let Some((&first, rest)) = group_ids.split_first() else {
            return Ok(());
        };
        let mut first = first;
        let mut last = first;
        for &group_id in rest {
            first = first.min(group_id);
            last = last.max(group_id);
        }

        let first = first as usize;
        let span = last as usize - first + 1;
        // Stable counting-sort the rows so every group becomes a slice of one gathered array.
        let mut offsets = vec![0usize; span + 1];
        for &group_id in group_ids {
            offsets[group_id as usize - first + 1] += 1;
        }
        for idx in 1..offsets.len() {
            offsets[idx] += offsets[idx - 1];
        }

        let mut cursors = offsets.clone();
        let mut permutation = vec![0u64; group_ids.len()];
        for (row_idx, &group_id) in group_ids.iter().enumerate() {
            let cursor = &mut cursors[group_id as usize - first];
            permutation[*cursor] = row_idx as u64;
            *cursor += 1;
        }

        let batch = batch.clone().execute::<Columnar>(ctx)?.into_array();
        let gathered = batch.take(Buffer::from_iter(permutation).into_array())?;
        for group_offset in 0..span {
            let start = offsets[group_offset];
            let end = offsets[group_offset + 1];
            if start == end {
                continue;
            }

            let group = first + group_offset;
            if self.state.is_saturated(group) {
                continue;
            }

            let mut accumulator = Accumulator::try_new(
                self.vtable.clone(),
                self.options.clone(),
                self.dtype.clone(),
            )?;
            accumulator.accumulate(&gathered.slice(start..end)?, ctx)?;
            let partial = accumulator.flush()?;
            self.state.combine_scalar(group, partial)?;
        }
        Ok(())
    }
}

fn validate_num_groups(num_groups: usize) -> VortexResult<()> {
    vortex_ensure!(
        num_groups == 0 || u32::try_from(num_groups - 1).is_ok(),
        "num_groups {} exceeds dense u32 group id capacity",
        num_groups
    );
    Ok(())
}

fn validate_group_ids(group_ids: &[u32], num_groups: usize) -> VortexResult<()> {
    validate_num_groups(num_groups)?;
    for &group_id in group_ids {
        vortex_ensure!(
            (group_id as usize) < num_groups,
            "Group id {} out of range for {} groups",
            group_id,
            num_groups
        );
    }
    Ok(())
}

fn list_view_group_ranges(
    groups: &ListViewArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<GroupRanges> {
    let offsets = groups.offsets();
    let sizes = groups.sizes().cast(offsets.dtype().clone())?;
    let ranges = match_each_integer_ptype!(offsets.dtype().as_ptype(), |O| {
        let offsets = offsets.clone().execute::<Buffer<O>>(ctx)?;
        let sizes = sizes.execute::<Buffer<O>>(ctx)?;
        offsets
            .as_ref()
            .iter()
            .zip(sizes.as_ref().iter())
            .map(|(offset, size)| {
                (
                    offset.to_usize().vortex_expect("Offset value is not usize"),
                    size.to_usize().vortex_expect("Size value is not usize"),
                )
            })
            .collect::<Vec<_>>()
    });
    Ok(GroupRanges::ListView { ranges })
}

fn fixed_size_list_group_ranges(groups: &FixedSizeListArray) -> GroupRanges {
    GroupRanges::FixedSizeList {
        len: groups.len(),
        size: groups.list_size() as usize,
    }
}

/// A trait object for type-erased grouped accumulators, used for dynamic dispatch when the
/// aggregate function is not known at compile time.
pub trait DynGroupedAccumulator: 'static + Send {
    /// Accumulate a values batch into dense group state.
    ///
    /// `group_ids` is parallel to `batch`. Each id must be a caller-assigned group ordinal in
    /// `0..group_ids.num_groups()`; ids may repeat, appear out of order, or be absent from a
    /// given batch.
    fn accumulate(
        &mut self,
        batch: &ArrayRef,
        group_ids: &GroupIds,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()>;

    /// Fold columnar partial states into dense group state.
    ///
    /// `group_ids` is parallel to `partials` and follows the same dense ordinal contract as
    /// [`Self::accumulate`].
    fn accumulate_partials(
        &mut self,
        partials: &ArrayRef,
        group_ids: &GroupIds,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()>;

    /// Merge one group from another grouped accumulator into this accumulator.
    fn merge_group(
        &mut self,
        into: u32,
        other: &dyn DynGroupedAccumulator,
        from: u32,
    ) -> VortexResult<()>;

    /// Return this accumulator's partial dtype.
    fn partial_dtype(&self) -> &DType;

    /// Read one group's current partial state.
    fn partial_scalar(&self, group_id: u32) -> VortexResult<Scalar>;

    /// Finish the accumulation and return partial aggregate results for all groups.
    ///
    /// Resets the accumulator state for the next round of accumulation.
    fn flush_partials(&mut self, num_groups: usize) -> VortexResult<ArrayRef>;

    /// Finish the accumulation and return final aggregate results for all groups.
    ///
    /// Resets the accumulator state for the next round of accumulation.
    fn finish(&mut self, num_groups: usize) -> VortexResult<ArrayRef>;
}

impl<V: AggregateFnVTable> DynGroupedAccumulator for GroupedAccumulator<V> {
    fn accumulate(
        &mut self,
        batch: &ArrayRef,
        group_ids: &GroupIds,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let num_groups = group_ids.num_groups();
        vortex_ensure!(
            batch.dtype() == &self.dtype,
            "Input DType mismatch: expected {}, got {}",
            self.dtype,
            batch.dtype()
        );
        vortex_ensure!(
            batch.len() == group_ids.len(),
            "Grouped aggregate input length mismatch: {} values, {} group ids",
            batch.len(),
            group_ids.len()
        );

        self.ensure_groups(num_groups)?;

        if self.try_accumulate_kernel(batch, group_ids, ctx)? {
            return Ok(());
        }

        let input = batch.clone();
        let mut batch = batch.clone();
        let mut tried_current = true;
        for _ in 0..max_iterations() {
            if batch.is::<AnyColumnar>() {
                break;
            }

            if !tried_current && self.try_accumulate_kernel(&batch, group_ids, ctx)? {
                return Ok(());
            }

            batch = batch.execute(ctx)?;
            tried_current = false;
        }

        if !tried_current && self.try_accumulate_kernel(&batch, group_ids, ctx)? {
            return Ok(());
        }

        let group_ids = group_ids.validated_ids(ctx)?;
        self.accumulate_fallback(&input, group_ids.as_ref(), ctx)
    }

    fn accumulate_partials(
        &mut self,
        partials: &ArrayRef,
        group_ids: &GroupIds,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let num_groups = group_ids.num_groups();
        vortex_ensure!(
            partials.dtype() == &self.partial_dtype,
            "Partial DType mismatch: expected {}, got {}",
            self.partial_dtype,
            partials.dtype()
        );
        vortex_ensure!(
            partials.len() == group_ids.len(),
            "Grouped aggregate partial length mismatch: {} partials, {} group ids",
            partials.len(),
            group_ids.len()
        );

        let group_ids = group_ids.validated_ids(ctx)?;
        self.ensure_groups(num_groups)?;
        self.state
            .accumulate_partials(partials, group_ids.as_ref(), ctx)
    }

    fn merge_group(
        &mut self,
        into: u32,
        other: &dyn DynGroupedAccumulator,
        from: u32,
    ) -> VortexResult<()> {
        vortex_ensure!(
            other.partial_dtype() == &self.partial_dtype,
            "Partial DType mismatch: expected {}, got {}",
            self.partial_dtype,
            other.partial_dtype()
        );
        self.ensure_groups((into as usize) + 1)?;
        self.state
            .combine_scalar(into as usize, other.partial_scalar(from)?)
    }

    fn partial_dtype(&self) -> &DType {
        &self.partial_dtype
    }

    fn partial_scalar(&self, group_id: u32) -> VortexResult<Scalar> {
        self.state.partial_scalar(group_id as usize)
    }

    fn flush_partials(&mut self, num_groups: usize) -> VortexResult<ArrayRef> {
        self.state.flush_partials(num_groups)
    }

    fn finish(&mut self, num_groups: usize) -> VortexResult<ArrayRef> {
        let states = self.flush_partials(num_groups)?;
        let results = self.vtable.finalize(states)?;

        vortex_ensure!(
            results.dtype() == &self.return_dtype,
            "Return DType mismatch: expected {}, got {}",
            self.return_dtype,
            results.dtype()
        );

        Ok(results)
    }
}
