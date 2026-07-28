# Pseudocode: `UnionBuilder` and `UnionArray` take

Companion to the PR prompts. Rust-flavored pseudocode; error handling, exact accessor names, and
trait plumbing are elided. `M` is the output row count. `variants` is the `UnionVariants` schema;
`c` ranges over child indices and `tag(c)` / `tag_to_child_index(t)` convert between data-level
tags and child indices.

## Shared building block: row-aligned sparse child from dense values

Both algorithms need to turn "a dense array of `count` values plus the sorted output rows they
occupy" into a row-aligned sparse child of length `M`, without materializing the placeholders and
without restricting the child dtype.

Key constraint discovered in review: `Patched::from_array_and_patches` only accepts NON-NULL
PRIMITIVE patch values (see its `is_primitive` and `all_valid` ensures) and eagerly transposes
them. Therefore we never patch the child's values; we patch the integer TAKE CODES that select
into the dense values, which always satisfy those constraints.

```text
fn sparse_child_from_dense(dense: Array, positions: Buffer<u64>, m: usize) -> Array {
    // dense.len() == positions.len() == count, count > 0, positions sorted ascending.
    // Rows not in `positions` are placeholders; they all point at dense[0], which is a valid
    // value of the exact child dtype. Their values are never observed.

    codes = ConstantArray(0_u64, m)                        // non-nullable primitive zeros
    patch = Patches(indices: positions, values: [0, 1, ..., count - 1])
    patched_codes = Patched::from_array_and_patches(codes, patch)   // primitive, all valid: OK

    return dense.take(patched_codes)                       // lazy DictArray, any child dtype
}

fn placeholder_child(variant_dtype: DType, m: usize) -> Array {
    // `default_value`, not `zero_value`: defined for `Null` and nested dtypes, and placeholders
    // may be null when the dtype permits it.
    return ConstantArray(Scalar::default_value(variant_dtype), m)
}
```

## `UnionBuilder`

Required property: O(1) amortized per append (never touch the non-selected variants per row),
O(M) finish. State:

```text
struct UnionBuilder {
    variants: UnionVariants
    nullability: Nullability
    type_ids: PrimitiveBuilder<u8>          // tags + append-time validity (outer nulls)
    dense: [Box<dyn ArrayBuilder>; N]       // one per variant, holds ONLY that variant's values
    // Bucketing at finish uses the append-time record (tags + validity). A post-hoc
    // set_validity only tightens the emitted outer validity; it must not disturb alignment
    // between tags and dense slots (see the prompt's subtlety section).
}
```

Appends:

```text
fn append_scalar(union_scalar) {
    c = tag_to_child_index(union_scalar.tag)
    type_ids.append(tag(c))                 // valid row
    dense[c].append_scalar(union_scalar.inner_value)
    // Nothing appended to any other variant. O(1) amortized.
}

fn append_null() {
    type_ids.append_null()                  // outer null; no variant consumes a dense slot
}
```

Finish:

```text
fn finish() -> UnionArray {
    (tags, validity) = type_ids.finish()    // u8 buffer + outer validity, length m
    m = tags.len()

    // One O(m) pass recovers each variant's sorted output positions. Iterate the validity mask
    // word-at-a-time next to the raw u8 slice; never call is_valid(i) per element.
    positions: [Vec<u64>; N] = bucket by tag over VALID rows only
        // Invalid rows never consumed a dense slot (append_null touches no variant), so
        // skipping them keeps positions[c].len() == dense[c].len().

    for c in 0..N {
        if positions[c].is_empty() {
            children[c] = placeholder_child(variant_dtype(c), m)
        } else {
            children[c] = sparse_child_from_dense(dense[c].finish(), positions[c], m)
        }
    }

    return UnionArray(type_ids: (tags, validity), variants, children)
}
```

Bulk append of an existing union array (`append_to_builder` for `Union`):

```text
fn append_union_array(src: UnionArray, builder: UnionBuilder) {
    src_tags = canonicalize(src.type_ids)               // u8 slice + validity mask, once
    src_positions[c] = bucket valid rows by tag         // same O(len) pass as finish
    for c with src_positions[c] non-empty {
        // Filter the sparse child down to its selected rows, then delegate to that child's own
        // append_to_builder into dense[c]. No per-row scalar access.
        src.child(c).filter(mask_from(src_positions[c])).append_to_builder(builder.dense[c])
    }
    builder.type_ids.extend(src_tags)                   // tags + outer validity, in bulk
}
```

## `UnionArray` take

`array.take(indices)` wraps a `DictArray(codes = indices, values = array)` and optimizes.
Layer 0 runs at optimize time; Layer 1 runs at execute time if Layer 0 declined.

### Layer 0: `TakeReduce` (lazy; metadata and exact stats only, zero buffer reads)

```text
fn take_reduce(union: UnionArray, indices: Array) -> Option<Array> {
    m = indices.len()

    // Order matters: an all-null constant type_ids has no active tag, so check nulls first.
    if union.validity() is AllInvalid {
        return Some(ConstantArray(Scalar::null(union.dtype.as_nullable()), m))
    }

    single_tag =
        if variants.len() == 1                        -> tag(0)
        else if type_ids is ConstantArray             -> its value
        else if exact_stat(Min) == exact_stat(Max)    -> that value   // Precision::Exact only
        else                                          -> return None  // fall through to Layer 1

    c = tag_to_child_index(single_tag)
    taken_type_ids = type_ids.take(indices)           // lazy; constant take reduces it; carries
                                                      // outer nulls from nullable indices
    for i in 0..N {
        children[i] = if i == c
            then union.child(c).take(fill_null(indices, 0))   // lazy; child's own rules fire;
                                                              // non-nullable indices preserve
                                                              // the child dtype
            else placeholder_child(variant_dtype(i), m)
    }
    return Some(UnionArray(taken_type_ids, variants, children))
}
```

### Layer 1: `TakeExecute` (the general shape; shared helper reused by dictionary decode)

```text
fn take_execute(union: UnionArray, indices: Array, ctx) -> Array {
    canonical_indices = canonicalize(indices, ctx)    // once; reused below
    m = canonical_indices.len()

    // O(m) u8 gather. Null indices become null type IDs = the outer nulls of the result; the
    // union dtype nullability follows from type_ids automatically.
    taken_type_ids = execute(type_ids.take(canonical_indices), ctx)   // u8 slice + validity mask

    // One O(m) bucketing pass over VALID rows (word-at-a-time mask iteration, raw u8 slice):
    //   positions[c].push(j); source_rows[c].push(canonical_indices[j])
    // Invalid rows are skipped entirely: a null row belongs to no variant, so no fill_null is
    // needed on this path. An undeclared tag panics ("Unknown UnionArray type ID {t}").
    (positions, source_rows) = bucket(taken_type_ids)

    // At most one dominant child: largest bucket, and only if it covers enough of the FULL
    // output (the dense take also gathers placeholders at outer-null rows). First maximum wins.
    dominant = argmax_c positions[c].len()
        if positions[dominant].len() < DOMINANT_TAKE_FRACTION * m { dominant = none }

    for c in 0..N {
        children[c] = match shape(c) {
            absent   (positions[c].is_empty())
                     => placeholder_child(variant_dtype(c), m)
            dominant (c == dominant)
                     // Bounded waste; plain full-take codes are friendlier downstream than a
                     // patched representation. Still lazy.
                     => union.child(c).take(fill_null(canonical_indices, 0))
            sparse   (otherwise)
                     // Patched CODES select directly into the ORIGINAL child: placeholders point
                     // at child[0], real rows at their source row. Gathers only what it touches
                     // when executed; works for any child dtype; stays lazy.
                     => union.child(c).take(
                            Patched::from_array_and_patches(
                                ConstantArray(0, m),                       // non-nullable codes
                                Patches(indices: positions[c], values: source_rows[c])))
        }
    }

    return UnionArray(taken_type_ids, variants, children)
}
```

Note the take path needs no `sparse_child_from_dense`: there is no dense intermediate, so the
patched codes point straight into the original child with `source_rows[c]` instead of `0..count`.
The builder's dense accumulators are what make the two variants of the same idea differ.

### Dictionary decode arm (`take_canonical` in `dict/execute.rs`)

```text
Canonical::Union(u) => {
    // Bypasses the adaptors' shared precondition, so handle the degenerate case here.
    if u.is_empty() {
        // All codes are necessarily null: all-null type IDs plus placeholder children.
        return all_null_union(u.variants, codes.len())
    }
    take_execute(u, codes, ctx)   // same shared helper
}
```

## Cost summary

| Shape | Work |
| --- | --- |
| Layer 0 shapes | O(1) plus one lazy child take |
| Layer 1, per call | O(M) type IDs gather + O(M) bucketing |
| Absent variant | O(1) |
| Sparse variant | patch metadata O(bucket size); child data gathered lazily, once per real row |
| Dominant variant | one full-length lazy take, waste bounded by `1 - DOMINANT_TAKE_FRACTION` |
| Builder append | O(1) amortized per row |
| Builder finish | O(M) bucketing + per-variant assembly as above |
