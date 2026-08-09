<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# RowFn regression and compiler-configuration research

This document records the follow-up performance investigation for `ct/row-fn`. It covers the
benchmarks requested in the [original issue comment], comparison with the [CodSpeed report], four
compiler configurations, commit bisection, source ablations, and the selected optimization.

The main result is narrow but important. Commit `5c02036a2` moved the `Args::varying` result and its
length check out of the branch that consumes the result. That source-only refactor made mixed
constant primitive operations about 4x slower with the default bench profile and more than 6x
slower with AVX2. Restoring the branch-local view and check recovers the performance. No algorithm
changed.

The remaining spatial `envelope` regression is separate. It first appears when numeric RowFn code
is linked into the benchmark, even before the spatial functions use RowFn. The experiments below
show code-generation sensitivity, but they do not identify a specific compiler pass or source-level
cause.

## Revisions and host

- Candidate before the selected fix: `892717f30` (`ct/row-fn`).
- Develop baseline: `66d096b5d` (`origin/develop`).
- Last fast revision before the regression: `89fd28bc1`.
- First slow revision: `5c02036a2`.
- Rust: 1.91.0, LLVM 21.1.2.
- Host: AMD Ryzen 9 7950X, 16 physical cores and 32 hardware threads.
- Timed process: pinned to logical CPU 4.
- CPU governor: `powersave`; energy-performance preference: `power`.

The governor could not be changed without elevated host privileges. Every comparison in a table
uses the same host and settings, so ratios are useful. Absolute times should not be compared
directly with the original performance-governor runs.

The normal repository bench profile already matches two important CodSpeed settings:

```toml
[profile.bench]
codegen-units = 16
lto = false
```

CodSpeed also supplies `RUSTFLAGS=-C target-feature=+avx2`. Both the default target and this AVX2
target were measured.

## What `Args::varying` represents

RowFn decodes each argument into an `ArgColumn`. An argument is either:

- `Varying`, with one stored value for every logical row.
- `Constant`, with one stored value reused for every logical row.

For a tuple, `Args::varying(&columns)` returns `Some` only when _every_ argument is varying. The
tuple implementation uses `?` for each column:

```rust
fn varying(columns: &Self::Columns) -> Option<Self::VaryingColumns<'_>> {
    Some(($($t::varying(columns.$idx.varying()?),)+))
}
```

One constant therefore makes the whole result `None`. This is not a statement about validity. It
classifies the physical row-addressing shape of the decoded arguments.

The two results select different access mechanisms:

1. `Some(varying)` contains a tuple of typed contiguous views. After one length check,
   `indexed_source` and `map_checked_into` can use unchecked lane reads without per-row shape
   dispatch.
2. `None` means at least one argument is constant. `Args::get(&columns, index)` then reads index
   zero for each constant column and `index` for each varying column.

The second mechanism sounds expensive, but it was already present in `89fd28bc1`, where constant
add and subtract took about 9.2 microseconds. The 4x regression was therefore not caused by
introducing the mixed-shape loop.

The regression came from changing the optimizer-visible data flow around that loop. The slow form
first materialized `Option<Args::VaryingColumns<'_>>`, passed `Option<&...>` to a separate generic
validation helper, and later consumed the original option in a branch:

```rust
let varying = Args::varying(&columns);
ensure_decoded_lengths::<Args>(&columns, varying.as_ref(), row_count)?;

if let Some(varying) = varying {
    // All-varying execution.
} else {
    // Mixed constant and varying execution.
}
```

The fast form constructs and validates the typed view only in the selected branch:

```rust
if let Some(varying) = Args::varying(&columns) {
    vortex_ensure!(Args::varying_len_matches(&varying, row_count), ...);
    // All-varying execution.
} else {
    vortex_ensure!(Args::decoded_lens_match(&columns, row_count), ...);
    // Mixed constant and varying execution.
}
```

On Rust 1.91.0 and LLVM 21.1.2, this placement determines whether the mixed-constant monomorphs are
well specialized. Source ablation and repeated benchmarks prove the relationship. They do not
prove which LLVM pass makes the poor decision. This should be treated as a measured compiler
workaround, not a general Rust rule.

The code does need to retain this specific placement for the measured toolchain. The varying view,
its matching length proof, and its consumer should remain in one control-flow branch. Moving them
through the shared helper is semantically equivalent, but currently changes generated-code quality.
The sink executors still use the shared helper because moving their checks did not improve the
cosine or spatial benchmarks.

## Commit bisection

The large constant-input regression first appears in `5c02036a2`.

| Revision | Add constant | Subtract constant | Multiply constant | Add varying | Multiply varying |
| --- | ---: | ---: | ---: | ---: | ---: |
| `89fd28bc1` | 9.219 us | 9.229 us | 18.94 us | 9.379 us | 26.68 us |
| `5c02036a2` | 30.46 us | 31.11 us | 37.73 us | 9.439 us | 26.61 us |

That commit deduplicated five decoded-length checks into `ensure_decoded_lengths`. Reverting only
the owned executor to branch-local checks recovers constant inputs. Keeping the helper in the sink
executors preserves the useful deduplication where no regression was measured.

Two other controls did not fix the regression:

- Reverting the `BorrowedExecutionArgs` move and delegation.
- Adding `#[inline(never)]` to the spatial `box_corners` helper.

## Selected optimization

The selected change is confined to `row/execute/owned.rs`:

- Call `Args::varying` in the `if let` condition.
- Validate `VaryingColumns` inside the all-varying branch.
- Validate the decoded `ArgColumn` tuple inside the mixed branch.
- Keep both validations before their loops so bounds-check elimination remains possible.
- Leave sink and valid-row execution unchanged.

This is a control-flow and proof-placement change. It adds no per-row work and does not change
null, failure, constant, or output semantics.

### Primitive binary results

Default bench profile, median time:

| Benchmark | Candidate | Fixed | Develop | Fixed/develop |
| --- | ---: | ---: | ---: | ---: |
| `add_i64_constant` | 35.4 us | 9.26 us | 8.38 us | 1.10x |
| `sub_i64_constant` | 36.2 us | 9.15 us | 8.23 us | 1.11x |
| `mul_i32_constant` | 41.9 us | 18.89 us | 26.45 us | 0.71x |
| `add_i64_nonnull` | 9.44 us | 9.44 us | approximately 9 us | approximately 1x |
| `mul_i32_nonnull` | 26.66 us | 26.66 us | approximately 26 us | approximately 1x |

AVX2, median time:

| Benchmark | Candidate | Fixed | Develop | Fixed/develop |
| --- | ---: | ---: | ---: | ---: |
| `add_i64_constant` | 29.70 us | 6.099 us | 4.919 us | 1.24x |
| `sub_i64_constant` | 30.65 us | 6.249 us | 4.959 us | 1.26x |
| `mul_i32_constant` | 37.97 us | 6.519 us | 5.689 us | 1.15x |

The fix removes the major regression. Small constant add and subtract gaps remain, especially with
AVX2, but they are not the same failure mode.

### RowFn executor microbenchmarks

Default-profile medians before and after the selected fix:

| Benchmark | Before | After |
| --- | ---: | ---: |
| Handwritten wrapping | approximately 127 ns | approximately 127 ns |
| RowFn sink wrapping | approximately 128 ns | approximately 128.5 ns |
| RowFn wrapping | approximately 128.5 ns | approximately 128.5 ns |
| RowFn checked | approximately 141.7 ns | approximately 141.7 ns |
| RowFn wrapping constant | approximately 62.1 ns | approximately 11.91 ns |
| RowFn checked constant | approximately 67.8 ns | approximately 35.69 ns |
| RowFn wrapping nullable | approximately 129.6 ns | approximately 130 ns |
| RowFn checked nullable | approximately 143.4 ns | approximately 143.6 ns |

Only the mixed-constant cases move materially, which matches the source-level diagnosis.

## Tensor results

### Squared L2 distance

Candidate and develop medians in microseconds:

| Width | Candidate nonnull | Develop nonnull | Candidate nullable | Develop nullable |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 17.29 | 31.77 | 18.47 | 32.45 |
| 32 | 6.77 | 7.26 | 7.95 | 8.01 |
| 256 | 10.15 | 10.00 | 11.28 | 10.72 |

The candidate is about 1.83x faster at nonnull width 2 and 1.75x faster at nullable width 2. It is
about 7% and 1% faster at width 32. At width 256 it is about 1.5% slower for nonnull input and 5.2%
slower for nullable input.

### Cosine similarity

Candidate and develop medians in microseconds:

| Shape and width | Candidate | Develop | Candidate speedup |
| --- | ---: | ---: | ---: |
| Column-column, 2 | 4.47 | 18.28 | 4.1x |
| Column-column, 32 | 2.44 | 4.97 | 2.0x |
| Column-column, 256 | 2.37 | 5.74 | 2.4x |
| Column-constant, 2 | 6.33 | 56.45 | 8.9x |
| Column-constant, 32 | 6.52 | 49.25 | 7.5x |
| Column-constant, 256 | 26.45 | 67.73 | 2.6x |
| Extension constant, 2 | 6.65 | 16.36 | 2.5x |
| Extension constant, 32 | 6.84 | 9.91 | 1.4x |
| Extension constant, 256 | 26.85 | 41.49 | 1.5x |

The owned-executor optimization does not affect cosine similarity because that implementation uses
prepared sink execution. Moving the sink length proof into its selected branch was tested and did
not materially change these results.

## Spatial results

Most predicate benchmarks remain close to the handwritten kernels:

- Column-column cases are generally 1% to 6% slower.
- Constant-input cases are generally 7% to 17% slower.
- Inputs with 90% nulls are about 4% faster.
- Dual-nullable inputs are about 2% slower.
- Polygon-column against constant-point cases are approximately equal.
- Constant-input `intersects` cases are about 4% to 9% slower.
- Exact and bounding-box diagnostic cases are approximately equal.
- The disjoint bounding-box diagnostic is slightly faster on the candidate.

Moving the sink proof into its selected branch did not materially change these predicate results.

### `envelope`

`envelope` has a separate, reproducible regression. Default-profile multipolygon results in
microseconds were:

| Input | Candidate before fix | Candidate after fix | Develop |
| --- | ---: | ---: | ---: |
| Mixed | 66.0 | 57.61 | 42.3 |
| Nonnull | 68.36 | 59.11 | 43.94 |
| Random | 54.28 | 48.40 | 33.63 |

The owned-executor change removes part of the final branch's loss, but the remaining regression is
about 34% to 45%.

Commit history isolates when it appears:

| Revision | Mixed | Nonnull | Random |
| --- | ---: | ---: | ---: |
| Framework only, `fef191df5` | 42.52 us | 44.52 us | 33.73 us |
| Numeric RowFn port, `b324f3e26` | 58.02 us | 59.72 us | 49.11 us |
| Before geo RowFn, `aebe3ca` | 58.43 us | 59.99 us | 49.50 us |

The regression therefore predates the geo visitor conversion. The `envelope.rs` source is
unchanged. It appears when numeric RowFn code is linked into the benchmark binary.

The generated candidate `envelope_array` function was smaller than develop, not larger:

| Revision | Instructions | Calls | Jumps |
| --- | ---: | ---: | ---: |
| Candidate | 1,725 | 115 | 175 |
| Develop | 1,811 | 122 | 189 |

This rules out the simple explanation that the candidate executes a visibly larger function. It
does not rule out placement, inlining, alignment, cache, or compiler phase-order effects elsewhere
in the linked binary. `perf` was unavailable on this host. LLVM-MCA was available, but no isolated
hot loop that retained the end-to-end regression was found.

## `list_sum` and unrelated code-generation sensitivity

`list_sum` does not call the RowFn owned executor, but it changed at the same source-shape commit.
This is evidence that generic code placement can perturb other monomorphs in the benchmark binary.

Default-profile progression:

| Revision | Large | Medium |
| --- | ---: | ---: |
| Framework only, `0a0ad0db1` | 13.84 ms | 59.71 us |
| Numeric RowFn, `89fd28bc1` | 13.49 ms | 61.8 us |
| Shared proof, `5c02036a2` | 14.99 ms | 77.82 us |
| Same revision with branch-local owned proof | 13.65 ms | 63.83 us |
| Final candidate with fix | 13.43 ms | 60.10 us |
| Develop | 13.59 ms | 60.48 us |

With AVX2, the fixed candidate measured 12.96 ms and 61.75 us; develop measured 13.26 ms and
58.75 us. The large case is about 2% faster, while the medium case is about 5% slower.

Because `list_sum` does not execute this RowFn path, the exact compiler mechanism remains an
inference. The commit bisection and one-change source ablation establish correlation and
reversibility, not a specific LLVM pass.

## Compact-slice control

The `compact_sliced(16384, 10)` benchmark did not reproduce the 26% CodSpeed loss:

| Configuration | Candidate | Develop | Difference |
| --- | ---: | ---: | ---: |
| Default | 107.45 us | 105.7 us | Candidate 1.7% slower |
| One CGU | 105.7 us | 104.9 us | Candidate 0.8% slower |
| AVX2 | 64.21 us | 66.08 us | Candidate 2.8% faster |
| Thin LTO | 107.3 us | 107.5 us | Approximately equal |

This result is consistent with simulation noise or linked-code layout sensitivity in CodSpeed. It
does not reproduce a durable algorithmic regression on this host.

## Compiler-configuration matrix

Changing codegen units, LTO, or AVX2 did not remove the two main regressions before the selected
fix.

| Configuration | Constant operands | `list_sum` | `envelope` | Compact slice |
| --- | --- | --- | --- | --- |
| 16 CGUs, no LTO | About 4x slower | Medium 33% slower | 55% to 62% slower | 1.7% slower |
| 1 CGU, no LTO | Add/sub 3.9x; mul 1.33x | 13% / 25% slower | 53% to 60% slower | 0.8% slower |
| 16 CGUs, AVX2 | 6x to 6.7x slower | 9% / 26% slower | 58% to 63% slower | 2.8% faster |
| 16 CGUs, Thin LTO | Similar large loss | Large 9%; medium 32% slower | 52% to 58% slower | Equal |

The repository's default of 16 CGUs and no LTO does not create the problem. One CGU and Thin LTO
also do not fix it. AVX2 amplifies the mixed-constant gap before the branch-local change.

## Confirmed findings

- `Args::varying` returns `Some` only when every decoded argument varies by row.
- Its `Some` value enables a typed indexed lane source; `None` selects mixed-shape row access.
- The mixed-shape loop itself was fast before `5c02036a2`.
- Hoisting the option and its proof through a generic helper causes the large mixed-constant loss on
  Rust 1.91.0 and LLVM 21.1.2.
- Restoring branch-local construction and validation recovers the loss without new per-row work.
- All-varying numeric benchmarks are unchanged by the selected fix.
- Prepared-sink cosine and geo cases do not benefit from the analogous source change.
- `list_sum` tracks the source ablation even though it does not use owned RowFn execution.
- The `envelope` regression begins with the numeric RowFn port, before geo adopts RowFn.
- CGU count, Thin LTO, and AVX2 do not remove the unfixed regressions.
- The compact-slice CodSpeed regression does not reproduce materially on this host.

## Inferences and unresolved questions

- The mixed-constant result is likely an LLVM phase-order or specialization-quality problem. The
  benchmark and source ablation do not identify the responsible pass.
- `list_sum` and `envelope` are likely sensitive to linked-code placement, inlining, alignment, or
  another whole-program code-generation effect. No single mechanism has been proven.
- Smaller `envelope_array` assembly does not imply faster execution. The relevant difference may
  be outside that symbol or may involve front-end behavior rather than instruction count.
- A compiler reduction should preserve both the timing delta and the production monomorph before
  filing an LLVM or rustc issue.

## Benchmark coverage and limitations

The durable current-tree replacements for the original issue comment were run:

- Primitive binary operations.
- RowFn executor microbenchmarks.
- Tensor L2 and cosine similarity.
- Geo predicates, bounding-box diagnostics, and envelope.
- `list_sum`.
- Compact sliced arrays.

The old experimental `BytesLen` and forced null-strategy benchmarks no longer exist in the current
tree, so they could not be rerun. No substitute result is presented as if it were the removed
benchmark.

Representative commands were:

```bash
taskset -c 4 cargo bench -p vortex-array --bench binary_ops -- <filters>
taskset -c 4 cargo bench -p vortex-array --bench row_fn_executor -- <filters>
taskset -c 4 cargo bench -p vortex-array --bench list_sum -- <filters>
RUSTFLAGS='-C target-feature=+avx2' taskset -c 4 cargo bench ...
CARGO_PROFILE_BENCH_CODEGEN_UNITS=1 taskset -c 4 cargo bench ...
CARGO_PROFILE_BENCH_LTO=thin taskset -c 4 cargo bench ...
```

Compilations used separate target directories before timed runs when configurations differed. Timed
runs were serialized on one logical CPU.

[original issue comment]: https://github.com/vortex-data/vortex/issues/9128#issuecomment-5151831802
[CodSpeed report]: https://github.com/vortex-data/vortex/pull/9255#issuecomment-5211040550
