# PR 2: `UnionArray` take compute

You are implementing take for the canonical sparse `UnionArray` in the Vortex columnar toolkit.
This is one of two parallel PRs; a sibling agent is implementing `UnionBuilder` at the same time.
Read the "Coordination" section before starting.

Work on a new branch based on `develop` (suggested name: `ct/union-take`). Target the PR at
`develop`.

## Context

Vortex recently gained a `DType::Union` and a canonical sparse `UnionArray`. Read these before
writing any code:

- Tracking issue: https://github.com/vortex-data/vortex/issues/7882 (this PR is the "take with
  null indices" portion of the unchecked null-semantics step).
- Nullability semantics (settled, closed): https://github.com/vortex-data/vortex/issues/8769
- `vortex-array/src/arrays/union/`: the array (`array.rs`), vtable (`vtable/`), existing compute
  (`compute/mask.rs`, `compute/slice.rs`), and the `TODO` in `compute/rules.rs` about take.
- `vortex-array/src/arrays/dict/take.rs`: the `TakeReduce` and `TakeExecute` traits, the shared
  `precondition` helper, the adaptors, and `propagate_take_stats`.
- `vortex-array/src/arrays/struct_/compute/take.rs`: the struct `TakeReduce` (useful for the
  fill-null-indices idiom, but see below for why the union version must NOT copy its shape).
- `vortex-array/src/arrays/extension/compute/take.rs` plus `extension/compute/rules.rs` and
  `extension/vtable/kernel.rs`: precedent for an encoding registering BOTH a `TakeReduce`
  optimizer rule and a `TakeExecute` execution kernel.
- `vortex-array/src/arrays/dict/execute.rs`: `take_canonical`, whose `Canonical::Union` arm is a
  `todo!` you will replace.

Key facts:

- `UnionArray` slots: slot 0 is `type_ids` (a `u8` primitive array, row aligned), slots 1.. are one
  row-aligned sparse child per variant. Child dtypes must exactly equal the variant dtypes
  (validated in `union/vtable/validate.rs`), which is why you can never let nullable take indices
  leak nullability into the children.
- Outer union nulls live in the `type_ids` validity: a null type ID is an outer null. The union
  dtype nullability equals the `type_ids` nullability, and `make_union_parts` derives it from
  `type_ids` automatically.
- Inactive slots of a sparse child are placeholders: addressable and dtype-valid, values never
  observed. This is the property the whole design exploits.
- `array.take(indices)` wraps the array in a `DictArray` (codes = indices, values = array) and
  calls `.optimize()` (`array/erased.rs:256`). `TakeReduce` rules fire during optimization via the
  encoding's `PARENT_RULES`; `TakeExecute` kernels fire at execution time via
  `register_execute_parent_kernel(Dict.id(), Encoding, TakeExecuteAdaptor(Encoding))`.
- `UnionVariants::tag_to_child_index(u8)` maps data-level tags to child indices. Tags are not
  necessarily `0..N`. An undeclared tag on a valid row is allowed to panic on access (consistent
  with `scalar_at` in `union/vtable/operations.rs`).

## Why the struct-style take is wrong for unions

`Struct` takes every field with the full index array because every field carries real data at
every row. Union children are sparse: across ALL children only one value per valid row is
meaningful, so the total useful gather work is (number of valid taken rows), independent of the
variant count. Taking every child with the full indices does O(variants x indices) work. If one
variant covers 99% of rows, N-1 of those full gathers are pure waste. The design below keeps total
child-gather work O(indices length) regardless of variant count, plus the metadata needed to
represent each observed child.

## Design: layered, multiple paths for different data shapes

### Layer 0: `TakeReduce` fast paths (lazy, metadata and stats only, zero buffer reads)

Implement `TakeReduce for Union` handling only shapes decidable without reading buffers, and
return `Ok(None)` for everything else so the `DictArray` survives to execution:

- **All rows outer-null**: `validity()` is `Validity::AllInvalid`. Result:
  `ConstantArray::new(Scalar::null(union dtype as nullable), M)`. Check this before inspecting
  constant type IDs, because an all-null constant has no active tag.
- **Single possible variant**: the union has exactly one variant, or `type_ids` is a
  `ConstantArray`, or its statistics prove a single tag (exact `Stat::Min == Stat::Max`; only
  trust `Precision::Exact`). Result: `type_ids` taken lazily (`type_ids.take(indices)`; the
  constant encoding's own take rule reduces it), the single active child taken lazily
  (`child.take(fill_null(indices, 0))` so the child encoding's own take rules and kernels exploit
  its structure), and every other child replaced by
  `ConstantArray::new(Scalar::default_value(variant_dtype), M)`. The lazy child take must use
  non-nullable indices (`fill_null` with zero, as struct take does) so the child dtype is
  preserved; the taken `type_ids` uses the original possibly-nullable indices so outer nulls
  appear. Use `default_value`, not `zero_value`: inactive placeholders may be null when their
  exact dtype permits it, and `zero_value` is not defined for `Null` or for every nested dtype.

Register it by adding `ParentRuleSet::lift(&TakeReduceAdaptor(Union))` to `PARENT_RULES` in
`union/compute/rules.rs`, replacing the existing `TODO(connor)` comment with a short comment
explaining the layering (reduce handles metadata-decidable shapes; the general shape needs to read
type IDs, so it lives in the execution kernel).

### Layer 1: `TakeExecute` partitioned kernel (the general shape)

Implement `TakeExecute for Union` as a thin wrapper over a shared helper (the dictionary decode
path reuses it, see below):

1. Canonicalize the indices once. Reuse this canonical array for both taking the type IDs and
   reading source row numbers during bucketing.
2. Execute `type_ids.take(canonical_indices)` to a canonical `u8` primitive plus validity mask.
   This is the result's type IDs child; null indices become null type IDs, which are exactly the
   outer nulls, and the result union dtype nullability follows automatically.
3. Do one pass over the taken type IDs to bucket rows by variant:
   for each valid output row `j` with tag `t`, push `j` onto `positions[c]` and `indices[j]` onto
   `source_rows[c]` where `c = tag_to_child_index(t)`. Skip invalid rows entirely; no `fill_null`
   is needed on this path because a null row belongs to no variant. Follow the "Performance: avoid
   hidden-cost accessors in hot loops" section of the repo `CLAUDE.md`: materialize the validity
   mask once and iterate it word-at-a-time alongside the raw `u8` slice; never call `is_valid(i)`
   or `execute_scalar(i)` per element. `positions[c]` comes out sorted for free.
4. Choose at most one dominant child: the child with the largest bucket, provided its count is at
   least `DOMINANT_TAKE_FRACTION` of the full output length `M`. Make this a named const with a
   short comment and start at `1/2`. Measure against `M`, not only the valid-row count, because the
   dense take gathers placeholders for outer-null rows too. Resolve ties deterministically, for
   example by retaining the first maximum.
5. Build each result child by observed shape:
   - **Absent** (`positions[c]` empty):
     `ConstantArray::new(Scalar::default_value(variant_dtype), M)`.
   - **Dominant** (the one child selected above): take that child densely with the full index array
     (`child.take(fill_null(canonical_indices, 0))`). The threshold now bounds the inactive gather
     work. This remains a lazy `DictArray`, but its codes are the ordinary full take indices rather
     than a sparse patched representation, which is friendlier when executed downstream.
   - **Sparse** (every other observed child): patch the child's TAKE CODES, not its values:
     1. Create non-nullable constant-zero codes of length `M`, using the canonical indices' integer
        `PType` without nullability.
     2. Build `Patches` (`vortex-array/src/patches.rs`) with patch indices = `positions[c]` and
        patch values = the non-nullable `source_rows[c]`.
     3. Use `Patched::from_array_and_patches` (`arrays/patched/array.rs:178`) to create the codes
        array, then return `child.take(patched_codes)`.

   `Patched` currently supports only primitive, all-valid patch values and eagerly transposes
   them. Using it for integer take codes satisfies those constraints; using it for gathered child
   values would fail for `Bool`, nullable primitives, strings, and nested children, and would also
   make the child gather eager. With patched non-nullable codes, `child.take(...)` stays lazy,
   supports arbitrary child dtypes, and preserves the child's exact dtype. Never canonicalize the
   input children wholesale.
6. Assemble with `UnionArray::try_new(taken_type_ids, variants, children)`.

Register the kernel in a new `union/vtable/kernel.rs` with a `pub(crate) fn initialize(session)`
calling `register_execute_parent_kernel(Dict.id(), Union, TakeExecuteAdaptor(Union))`, mirroring
`extension/vtable/kernel.rs`. Union is currently missing from the per-encoding initialize list in
`arrays/mod.rs:132`; add a `union::initialize` (or wire it through `union/vtable/mod.rs` the way
`struct_` does) and call it there.

### Dictionary decode

Replace the `Canonical::Union` `todo!` in `dict/execute.rs::take_canonical` with a call to the
shared helper (mirroring how `take_struct` delegates to the struct implementation). Note this path
receives codes that may carry validity, and it bypasses the adaptors' shared `precondition`, so
the helper must handle empty values arrays itself (all indices are then necessarily null; produce
all-null type IDs and constant placeholder children).

## Edge cases and known limitations

- Empty indices and empty arrays are short-circuited by `precondition` in `dict/take.rs:60` for
  the adaptor paths; only the dictionary-decode path needs explicit handling (see above).
- The `precondition` empty-array fast path returns a `ConstantArray` of a null union scalar.
  Canonicalizing THAT constant currently hits a `todo!` at `constant/vtable/canonical.rs:167`; the
  parallel builder PR is implementing it. Write your empty-array test to assert dtype and length
  of the returned array without canonicalizing it, and leave a brief comment that end-to-end
  coverage arrives when the builder PR merges.
- Definitely-all-null take codes are also short-circuited to a constant union by
  `dict/vtable/mod.rs`. Test all-null indices through dtype, length, and scalar access without
  requiring canonicalization or assuming a particular output encoding until the builder PR
  merges.
- Take over a `ChunkedArray` of unions requires union builders (chunked take canonicalizes through
  `builder_with_capacity`, see `chunked/compute/take.rs:97`); that is the parallel PR's scope, not
  yours. Do not add chunked-of-union tests here.
- An undeclared tag encountered during bucketing should panic with the same style of message as
  `scalar_at` ("Unknown UnionArray type ID {t}").

## Tests

- Unit tests (in `union/tests.rs` or a `tests` module in `compute/take.rs`, following the existing
  patterns there): non-nullable indices; nullable indices introducing outer nulls; the outer
  null versus inner null distinction surviving take (take a row that is an inner null of a
  nullable variant and a row that is an outer null, and assert they come back distinct); an
  already-nullable union; repeated and reversed indices; a variant absent from the taken rows
  (assert the result is still valid and the absent child is a constant); a sparse non-primitive
  child such as `Bool` or `Utf8`; a sparse nullable child whose selected value is null; a
  dominant-variant case that inspects the dense codes shape; a null-heavy case proving dominance
  is measured against total output length; a single-variant / constant type IDs case asserting the
  Layer 0 reduce fired (inspect the optimized array, mirroring how `struct_/compute/rules.rs` tests
  assert on encodings, rather than only checking values); all-null indices without canonicalizing
  the constant result.
- Conformance: hook `test_take_conformance` up for union arrays the way
  `struct_/compute/mod.rs` does (multiple array shapes via `rstest`).
- Dictionary decode: a `DictArray` whose values are a union, executed to canonical.

## Optimization calibration

The layering and the O(indices) child-gather and patch-count property are required; they are the
point of this PR, not premature optimization. Per-child encoding metadata is still expected. Beyond
that, keep it simple: a plain loop over the materialized `u8` slice with a word-at-a-time validity
mask is fine for bucketing, no SIMD heroics, no extra configuration, and no speculative fast paths
beyond the ones listed. Keep `DOMINANT_TAKE_FRACTION` a simple const; tuning it with a criterion
benchmark is an explicitly out-of-scope follow-up (a small `vortex-array/benches` benchmark
comparing dominant versus uniform tag distributions is a welcome stretch goal if it stays small,
following `vortex-array/benches/validity_is_valid.rs` as a template).

## Repo conventions that matter here

- Tests: prefer `rstest` for parameterized cases, return `VortexResult<()>` and use `?` instead of
  `unwrap`, use `assert_arrays_eq!`, and create one `ExecutionCtx` per test with
  `array_session().create_execution_ctx()` and reuse it.
- Comments: full sentences with periods, 100-column limit, backtick code items, link items in doc
  comments with square brackets. Do not use em dashes in comments or documentation.
- Every new public API needs a doc comment.
- Before finishing: `cargo +nightly fmt --all`, `cargo clippy --all-targets --all-features`, and
  `cargo nextest run -p vortex-array`.
- Commits must be signed off: `Signed-off-by: "NAME" <EMAIL>` (DCO).

## Coordination with the parallel builder PR

A sibling agent is implementing `UnionBuilder`, `append_to_builder` for unions, and constant union
canonicalization on a separate branch at the same time. To avoid coupling:

- Do NOT depend on any of their code and do not implement builders or `append_to_builder` here.
- The take implementation's patched-code assembly is specific to lazy gathers. Do not try to
  share it with or move it into the builder implementation in this PR.
- Expected (trivial) merge conflict points: `union/vtable/mod.rs` (they replace
  `append_to_builder`; you add kernel registration) and possibly `arrays/mod.rs`. Keep your
  changes minimal in shared files.
