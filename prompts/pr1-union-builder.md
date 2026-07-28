# PR 1: `UnionBuilder` and union canonicalization plumbing

You are implementing the `UnionBuilder` for the Vortex columnar toolkit, plus the canonicalization
plumbing that depends on it. This is one of two parallel PRs; a sibling agent is implementing
`UnionArray` take compute at the same time. Read the "Coordination" section before starting.

Work on a new branch based on `develop` (suggested name: `ct/union-builder`). Target the PR at
`develop`.

## Context

Vortex recently gained a `DType::Union` and a canonical sparse `UnionArray`. Read these before
writing any code:

- Tracking issue: https://github.com/vortex-data/vortex/issues/7882
- Nullability semantics (settled, closed): https://github.com/vortex-data/vortex/issues/8769
- `vortex-array/src/arrays/union/mod.rs` and `union/array.rs`: the array layout.
- `vortex-array/src/arrays/union/vtable/mod.rs`: the vtable, including the `append_to_builder`
  `todo!` you will replace.
- `vortex-array/src/builders/mod.rs`: the `ArrayBuilder` trait (line ~88) and
  `builder_with_capacity` (the `DType::Union` arm at line ~306 is `todo!`). Also check
  `builder_with_capacity_in` just below it.
- `vortex-array/src/builders/struct_.rs`: the closest existing builder; study how it handles
  validity, `set_validity`, `finish`, and nested child builders.
- `vortex-array/src/scalar/scalar_impl.rs` lines ~105-160: `Scalar::default_value` and
  `Scalar::zero_value` semantics for unions (documented in detail there).

Key layout facts:

- `UnionArray` slots: slot 0 is `type_ids` (a `u8` primitive array, row aligned), slots 1.. are one
  row-aligned sparse child per variant. Child dtypes must exactly equal the variant dtypes
  (validated in `union/vtable/validate.rs`).
- Outer (top-level) union nulls live in the `type_ids` child validity: a null type ID is an outer
  null. This is independent of nulls inside a variant child (inner nulls). The union dtype
  nullability equals the `type_ids` nullability.
- Inactive slots of a sparse child (rows where the type ID selects a different variant) are
  placeholders: they must be addressable and dtype-valid, but their values are never observed.
- `UnionVariants` maps between data-level tags (`u8`) and child indices via
  `tag_to_child_index`. Tags are not necessarily `0..N`. Find the inverse accessor on
  `UnionVariants` for going from child index to tag.
- `Scalar::union(variants, tag, child_scalar, nullability)` constructs union scalars;
  `UnionScalar` exposes the tag and inner value.

## The critical performance constraint

A naive union builder that appends a placeholder value to every non-selected child per row does
O(number of variants) work per row. That is unacceptable; this builder must be O(1) amortized per
append. The design:

- **While building**: maintain a `u8` type IDs builder (with validity for outer nulls) and one
  *dense* value builder per variant (created lazily or up front from the variant dtype via
  `builder_with_capacity`). `append_scalar` on a union scalar pushes the tag onto the type IDs
  builder and the inner value onto the *selected* variant's dense builder only. Nothing is written
  to the other variants. `append_null` pushes a null type ID and touches no variant builder.
- **At `finish()`**: the dense builders are not row aligned, so the finish step assembles each
  sparse child from its dense values plus the row positions that selected it. One O(len) pass over
  the finished type IDs recovers, for each variant `c`, the sorted list of row positions
  `positions[c]`. Then each child is assembled as:
  - No rows selected the variant: `ConstantArray::new(Scalar::zero_value(variant_dtype), len)`.
    For a `DType::Null` variant use `Scalar::null` (there is no zero value for `Null`; see the
    `zero_value` docs).
  - Otherwise: patch the dense values over that same constant fill at `positions[c]`. Use
    `Patched::from_array_and_patches` (`vortex-array/src/arrays/patched/array.rs:178`) with a
    `Patches` built from the positions and the finished dense values (`vortex-array/src/patches.rs`).
    Sparse union children do not need to be canonical, so a `PatchedArray` child is a valid
    canonical union child.

This "(sorted positions, dense values) to row-aligned sparse child" assembly is a natural helper
function. Keep it private to the builder for now (see "Coordination").

## Subtlety: `set_validity` versus dense alignment

The bucketing at `finish` must know, for each row, which variant (if any) consumed a dense slot.
That information is fixed at append time. If `ArrayBuilder::set_validity` is later used to change
the validity mask, the append-time record must not be disturbed, or the dense values will
misalign with the recovered positions.

Recommended resolution: bucket from the append-time tags and append-time validity; apply a
post-hoc `set_validity` only to the outer validity of the emitted `type_ids` array. Marking a row
valid when it never had an appended variant value cannot produce a meaningful union row; make that
case a documented error (or panic consistent with the builder trait's conventions; check what
`StructBuilder` does for the analogous case and stay consistent). Document the chosen semantics on
the builder.

## Scope of this PR

1. **`UnionBuilder`** in `vortex-array/src/builders/union.rs`, implementing `ArrayBuilder` with the
   design above. Implement all trait methods; note `append_zero` / `append_default` must follow the
   documented union semantics of `Scalar::zero_value` / `Scalar::default_value`.
2. **Wire `builder_with_capacity`**: replace the `todo!` arm at `builders/mod.rs:306` (and the
   `_in` variant if it also dispatches on dtype).
3. **`append_to_builder` for `Union`**: replace the `todo!` in `union/vtable/mod.rs`. Do not
   iterate rows with per-element `execute_scalar` (see the hot-loop guidance in the repo
   `CLAUDE.md`). Efficient approach: canonicalize the source array's `type_ids` once, bucket row
   positions per variant in one pass, then for each variant filter the source child down to its
   selected rows and delegate to that child's own `append_to_builder` into the corresponding dense
   builder; append the type IDs (with outer validity) into the type IDs builder in bulk.
4. **Constant union canonicalization**: replace the `todo!` at
   `vortex-array/src/arrays/constant/vtable/canonical.rs:167`. A constant union of length `n` is:
   type IDs = constant tag of length `n` (or all-null type IDs for a null union scalar), the
   selected child = constant of the inner value, and every other child = constant zero-value
   placeholder. This unblocks paths in the sibling take PR, so please do not drop it from scope.
5. **Tests**, including at minimum:
   - Builder round trips: mixed variants, outer nulls, inner nulls (a null value inside a nullable
     variant is distinct from an outer null), non-nullable unions, empty builder.
   - A union where one variant receives zero rows (its child must be a valid placeholder).
   - `append_zero`, `append_default`, and the `set_validity` semantics you defined.
   - End-to-end: build a `ChunkedArray` of unions and `execute::<Canonical>` it. This currently
     hits both `todo!`s and is the motivating use case (chunked take of unions canonicalizes
     through builders; see `chunked/compute/take.rs:97`).
   - Constant union canonicalization, including a null union constant.

## Optimization calibration

We want the O(1)-per-append and O(len)-finish structure described above; that is a required
asymptotic property, not premature optimization. Beyond that, keep it simple: a plain loop over a
materialized `u8` slice plus validity mask is fine for the finish pass, no SIMD heroics, no
speculative caching, no extra configuration knobs. Follow the "Performance: avoid hidden-cost
accessors in hot loops" section of the repo `CLAUDE.md` (materialize masks once; never call
`is_valid(i)` or `execute_scalar(i)` in a loop).

## Repo conventions that matter here

- Tests: prefer `rstest` for parameterized cases, return `VortexResult<()>` and use `?` instead of
  `unwrap`, use `assert_arrays_eq!`, and create one `ExecutionCtx` per test with
  `SESSION.create_execution_ctx()` (or `array_session().create_execution_ctx()`) and reuse it.
- Comments: full sentences with periods, 100-column limit, backtick code items, link items in doc
  comments with square brackets. Do not use em dashes in comments or documentation.
- Every new public API needs a doc comment.
- Before finishing: `cargo +nightly fmt --all`, `cargo clippy --all-targets --all-features`, and
  `cargo nextest run -p vortex-array`.
- Commits must be signed off: `Signed-off-by: "NAME" <EMAIL>` (DCO).

## Coordination with the parallel take PR

A sibling agent is implementing union take (`TakeReduce` fast paths, a partitioned `TakeExecute`
kernel, and the dictionary-decode arm) on a separate branch at the same time. To avoid coupling:

- Do NOT depend on any of their code, and do not implement take compute in this PR.
- They also need a "(positions, values) to sparse child" assembly. Duplicating that small helper
  across the two PRs is expected and fine; unifying it is a follow-up after both merge.
- Expected (trivial) merge conflict points: `union/vtable/mod.rs` (you replace
  `append_to_builder`; they add kernel registration) and possibly `arrays/mod.rs`. Keep your
  changes minimal in shared files.
