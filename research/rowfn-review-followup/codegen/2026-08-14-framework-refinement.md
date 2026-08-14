<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# RowFn framework refinement and stack gate

Date: 2026-08-14.

Platform: Apple M4 Max (`aarch64-apple-darwin`), rustc 1.97.1, LLVM 22.1.6. The compiler-output
gate used the bench profile, 16 codegen units, no LTO, `-C target-cpu=native`, loop-vectorizer
remarks, and line-table debug information. The timing evidence is local Apple Silicon evidence and
does not establish x86 performance.

## Final design decisions

- The framework is stricter than an ordinary strict scalar function. Strict input validity is
  propagated by batch execution, while a row kernel must return a valid value for every valid input
  row. A function that can create null from valid inputs cannot use this row-kernel contract.
- `skip-invalid` means that the executor initializes invalid output rows, invokes the row kernel only
  for valid rows, then publishes the completed sink. `filter-and-scatter` means that batch execution
  filters every input to the valid rows, runs the ordinary row kernel, then scatters those results
  into their original positions with nulls elsewhere.
- Documentation uses _partially valid_ for a mask with both valid and invalid rows. It uses
  _batch-constant_ for a decoded input with one addressable value and _non-constant_ for the
  opposite. Vague uses of _mixed_ and _varying_ were removed.
- `Args::views_no_constants` names the fast path that exists only when no input is batch-constant.
  The partial-constant path keeps representation dispatch visible to LLVM so loop unswitching can
  specialize the runtime constant orientation and broadcast the constant operand.
- Decoded batch constants are validated when `ArgColumn` is constructed. Both ordinary and
  null-tolerant decoding call `ArgColumn::try_from_constant`, so a malformed zero-length decode is
  rejected before row zero can be read. The private constant representation therefore has exactly
  one addressable row.
- `RowExecution` remains public and re-exported because the tensor L2 branch consumes it through
  `RowFn::reduce_encoded`. Encoding-aware reductions remain on the first branch that consumes them
  rather than expanding the framework PR in advance.
- Masked-array cleanup was kept out of the framework change. It belongs with the separate work
  tracked by issue #9403.

## Source structure

`Batch` now lives in `batch/mod.rs`. Planning lives in `batch/planning.rs`. `batch/execute/mod.rs`
owns the high-level router and universal fast paths, while named leaves own constant, dense,
valid-only, filter-and-scatter, and output behavior. This keeps the central state and reader-facing
control flow in their parent modules without rebuilding a monolithic execution file.

Owned and sink executors separate algorithm comments from compiler-sensitive source constraints.
The local constraints point to the relevant accessors and describe the observed non-LTO behavior
without embedding machine-specific timing numbers in production code. Safety comments are attached
to the exact unsafe operation and use named intermediate values when that makes the proof readable.

`BitBuffer::try_for_each_set_index` generalizes fallible set-bit traversal in `vortex-buffer`. It
retains the word-oriented and all-ones traversal paths and returns immediately after the callback
fails. `execute_sink_valid_rows` uses it instead of storing a deferred row error and continuing the
scan.

## Compiler-output experiments

The final owning-library build was generated with:

```text
CARGO_TARGET_DIR=/private/tmp/rowfn-ir-batch-split-stacked-candidate \
CARGO_PROFILE_BENCH_CODEGEN_UNITS=16 \
CARGO_PROFILE_BENCH_LTO=false \
RUSTFLAGS='-C target-cpu=native -C remark=loop-vectorize -C debuginfo=line-tables-only' \
cargo rustc -p vortex-array --lib --profile bench \
  --features _test-harness,table-display,unstable_row_fns -- \
  --emit=llvm-ir,asm,link
```

The module split was compared with the same executable source before the split. NumericBinary
functions emitted out of line fell from 147 to 94 because 36 owned, eight dense-sink, and eight
valid-row-sink helpers inlined into their visitors. End-to-end NumericBinary AArch64 instructions
fell from 44,986 to 44,179. Dense and owned paths fell from 30,190 to 29,300 instructions; sparse
paths fell from 10,082 to 9,995.

Hot-loop structure was preserved:

- Owned paths retained 152 vector-body references, 164 splat references, and the same vector-load
  widths.
- Sparse paths retained 16 vector-body references, 48 splat references, and the same widths.
- Dense paths had no vector-body references in either build.
- Planning retained 19 functions and 1,191 AArch64 instructions.

Splitting `Args::indexed_source` into a named local produced equivalent optimized LLVM IR and
identical AArch64 instructions for all 60 inspected owned NumericBinary monomorphizations. Removing
the owned lexical scope likewise changed only IR numbering and debug placement; the machine
instructions and vector, broadcast, and bounds-check metrics were identical.

Moving bindings at the start of owned execution preserved every vectorized and unswitched hot path.
It also reduced the inspected bounds-failure references from six to four per monomorphization and
reduced representative executor instruction counts. This remains target- and compiler-specific
evidence, not a language-level guarantee.

The first sink source rewrite exposed an important measurement trap. Bounds checks reported inside
each outlined sparse callback were not newly created checks. The baseline called a shared
`ElementTuple::get` that contained the same checks, while the candidate inlined that getter into
each callback. The confirmed difference was CGU placement, inlining, and code duplication. Reports
must compare the same logical work across callers and callees rather than count references inside
one symbol.

## Partial-constant execution

The final valid-row setup hoists the no-constant versus partial-constant dispatch outside set-bit
traversal. This duplicates both optimized word traversals in each specialization, growing the
static sparse assembly from 6,275 to 8,778 instructions and the full valid-row symbols from 8,000
to 10,576. Static size alone was not treated as a regression because the relevant question was hot
execution and instruction-cache behavior.

Five warmed, alternating A/B runs with 5,000 iterations per case compared the hoisted dispatch with
the earlier single traversal:

| Case | Change in median |
| --- | ---: |
| Dense, left batch-constant | -57% |
| Dense, right batch-constant | -23% |
| Sparse, left batch-constant | -31% |
| Sparse, right batch-constant | -24% |
| Dense, no constants | -14.5% |
| Sparse, no constants | +1.7% |
| Owned control | -1.9% |

The partial-constant improvements are material. The sparse no-constant and owned controls are within
run noise. A standalone dense-division control was noisy at +7.8% by its median, but focused pairs
overlapped and its hot instruction sequence was identical, so it was not defensible regression
evidence.

Moving sink allocation after decoding, validation, and preparation was isolated with five
alternating 100,000-iteration runs. The retained ordering had a 22.28 microsecond median with a
0.28 microsecond median absolute deviation. Allocation before decoding had a 23.82 microsecond
median with a 1.17 microsecond deviation, 6.9% slower and noisier in this experiment.

## Rejected alternatives

- `set_indices().try_for_each` made the sparse code smaller but lost the vectorized remainder paths
  and the all-ones fast path.
- New unchecked `ArgColumn` and `ElementTuple` access removed 16 panic sites but recovered only 398
  instructions from the 8,778-instruction sparse result. The unsafe API was not justified.
- A generic `#[inline(never)]` traversal worsened the full valid-row code to 12,982 instructions.
- One shared dynamic-callback traversal reduced static code to 6,079 instructions plus a shared
  435-instruction traversal, but introduced two indirect `blr` calls in the set-bit loops. That
  per-valid-row dispatch was rejected.
- Moving constant-length error formatting to a cold, no-inline helper preserved the hot paths but
  did not reliably reduce whole-binary text across consumers. The simpler construction-time check
  was retained.

## Stack integration

The focused stack order is framework, primitive numeric operators, primitive comparisons, tensor
L2, tensor products, spatial distance, spatial predicates, and benchmark tools. The already merged
RowFn types PR is below `develop` and was not rewritten.

| Branch | Final commit |
| --- | --- |
| `ct/row-fn-framework` | `8d7edf34dd445e152fc91be8c9b73d027d0131ad` |
| `ct/row-fn-numeric-operators` | `413796dd61aae2470d1139cf014bc8844d9ca792` |
| `ct/row-fn-primitive-comparisons` | `c057706c943eacef3b930e9f86bc3a5ff53c8b9d` |
| `ct/row-fn-tensor-l2` | `b655d61815351d284cdf6f575cfa9c60d0a313af` |
| `ct/row-fn-tensor-products` | `2f55638bcdcc2876eab193c85464d4aff74313a0` |
| `ct/row-fn-spatial-distance` | `2a2d8ae9da281aa39480fdcc61d6cbad430b6924` |
| `ct/row-fn-spatial-predicates` | `b37103bfe10defe9f80cc8bc4d3e1d92f78a0700` |
| `ct/row-fn-benchmark-tools` | `97b5a990abb4c9828dbc711c23a0a0f065140d78` |

The framework split required one conflict resolution in tensor L2. The encoded-reduction hook moved
from the deleted monolithic execution file into the new high-level router, output reconciliation,
and valid-only modules. Every other dependent commit replayed without a semantic adaptation. Each
rewritten commit retained its DCO trailer, and every layer was checked with `git range-diff`.

The final focused checks passed:

```text
cargo test -p vortex-buffer set_index
  11 passed

cargo test -p vortex-array scalar_fn::unstable::row
  40 passed

cargo test -p vortex-tensor scalar_fns
  73 passed

cargo test -p vortex-spatial scalar_fn
  178 passed

cargo +nightly fmt --all -- --check
  passed

PYO3_PYTHON=/Users/connor/spiral/vortex-data/vortex1/.venv/bin/python3 \
  cargo clippy -p vortex-buffer -p vortex-array -p vortex-tensor -p vortex-spatial \
  --all-targets --all-features
  passed
```

The historical `ct/row-fn` branch is synchronized with a two-parent merge. Its first parent keeps
the umbrella investigation history and this report. Its second parent is the final focused stack
tip, and the merge tree exactly equals that second parent.
