<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# RowFn review follow-up

Platform: Apple Silicon `arm64`, macOS 15.7.3, Rust 1.91.0, LLVM 21.1.2. This machine uses
128-bit NEON and cannot reproduce or compare the pinned Ryzen wall-clock results. This
investigation uses optimized LLVM IR and correctness tests only.

Baseline: `833632aaa3cab59fb4a7d4f001df26975b2267a1` on `ct/row-fn`.

## Result

| # | Verdict | Evidence and status |
| ---: | --- | --- |
| 1 | Fixed, not upstreamed | `44457b7654` makes `OutputSink::finish` unsafe. The zero-unsafe external reproduction changes from reading allocator contents to `E0133`. All gate items pass, but the local `ct/row-fn-api` ref contains divergent unpushed history and was not overwritten. |
| 2 | Fixed, not upstreamed | `InputElement` is an unsafe trait with local safety proofs. `ElementTuple` and `IndexedElementTuple` remain sealed framework traits. Same upstream status as #1. |
| 3 | Fixed, not upstreamed | The existing unsafe `InitializedElement::write` token remains unchanged. Unsafe publication now belongs to `OutputSink::finish`, and the generic executor owns the proof. Same upstream status as #1. |
| 4 | Fixed, not upstreamed | `initialize_skipped_rows` now returns its capability result. A separate support constant cannot disagree with a no-op default. Same upstream status as #1. |
| 5 | Fixed, not upstreamed | `5204fb2be5` documents totality and probes original inputs once. Gate item 3 fails because the batch refactor adds an owned mixed-path bounds edge. |
| 6 | Not started | The requested redesign was out of scope. A temporary manual implementation reproduced `E0119`. The options analysis is below. |
| 7 | Fixed, not upstreamed | `44457b7654` documents why private `NumericBinary` borrows the registered `Binary` ID and makes privacy the registration guard. Same upstream status as #1. |
| 8 | Fixed, not upstreamed | `5204fb2be5` validates every input length in `Batch::new`. The regression test fails before the fix. Gate item 3 fails as described for #5. |
| 9 | Refuted | `MaskedArray::try_new` enforces an all-valid child, and null `ConstantArray` validity is `AllInvalid`. Both couplings are constructor invariants. |
| 10 | Fixed, not upstreamed | `5204fb2be5` probes once before retry. The regression test fails before the fix and passes after it. Gate item 3 fails as described for #5. |
| 11 | Investigated, needs x86 | The path is unreachable because no sink sets `ERRORS_ARE_DEFERRED`. Deleting it perturbed arithmetic IR, so no change remains. |
| 12 | Refuted | Filtering preserves every constant shape recognized by `batch_constant`: literal `Constant`, constant `Masked` child, and constant `Extension` storage. `ConstElems` stays consistent. |
| 13 | Fixed, not upstreamed | `5204fb2be5` runs the encoding probe before one-row broadcast, against the original arrays. Gate item 3 fails as described for #5. |
| 14 | Investigated, needs x86 | The five deferred word implementations are unreachable. Their deletion changed codegen-unit placement and owned-loop IR, so the machinery remains. |
| 15 | Refuted | Mixed `i64` add/sub and `i32` multiply contain broadcast vector loops on `arm64`. They do not fall back to scalar row execution. |
| 16 | Investigated, needs x86 | Full-row skipped initialization and non-breaking mask traversal are independent costs. The proposed API and early-exit split are below. |
| 17 | Investigated, needs x86 | `scatter_valid` allocates one `u64` per original row. The cited `vortex-spatial/src/scalar_fn/execute/geo_types.rs` path does not exist at this revision. |
| 18 | Refuted | No consumer requires row output to retain 256-byte physical alignment. Alignment-sensitive consumers call `ensure_aligned`. Any performance change still needs x86 evidence. |

## Reproductions

### Uninitialized sink publication

A separate crate used no unsafe code. It freed a `Vec<u64>` filled with
`0xd1775eedd1775eed`, created `UninitElementSink::<u64>::with_capacity(4096, ...)`, called
`finish`, and read the result. The baseline returned the recognizable value in all 4,096 rows.

After `44457b7654`, the unchanged call fails to compile:

```text
error[E0133]: call to unsafe function `OutputSink::finish` is unsafe and requires unsafe block
```

The sink keeps the existing write-token mechanism. The generic executor makes one documented
unsafe `finish` call after successful traversal or successful skipped-row initialization and
traversal.

### Blanket vtable implementation

A temporary `impl ScalarFnVTable for LazyDouble` in `strict_validity.rs` fails with `E0119` because
the blanket `impl<F: RowFn> ScalarFnVTable for F` already applies. The temporary change was removed.

### Batch failures

`test_batch_rejects_input_length_mismatch` fails before the length check. The baseline reaches a
strategy-specific panic, error, or wrong-length result instead of rejecting the batch boundary.

`test_dense_retry_does_not_reduce_filtered_inputs` uses a reducer that answers only for compacted
one-row inputs. The baseline retries after a deferred error, re-probes the filtered inputs, and
suppresses the valid-row failure. The candidate probes the originals once and preserves the error.

The first proposed reproduction required every filtered input to become `ConstantArray`. Filtering
a one-row `PrimitiveArray` keeps it primitive, so that version did not exercise the defect. The
one-row encoding probe establishes the same retry violation without assuming a filter encoding.

## API options for the blanket vtable

### `RowFnAdaptor<F>`

An adaptor can own the blanket `ScalarFnVTable` implementation while `F` remains a `RowFn`.
Registration and expression construction must wrap every function. Existing call sites that name
the concrete function type also change. Arithmetic can delegate to the same generic row executor,
but the wrapper changes monomorph identities and requires the full IR gate.

### Public `execute_rows`

A public free function lets each adopter implement `ScalarFnVTable` and delegate only execution.
It preserves all five vtable hooks but repeats arity, child naming, return dtype, strictness, and
fallibility boilerplate in each implementation. The arithmetic row loop can remain the same helper
monomorph. The surrounding vtable implementation still requires IR verification.

### Hooks on `RowFn`

`RowFn` can redeclare `coerce_args`, `simplify`, `simplify_untyped`, `reduce`, and `fmt_sql`.
The blanket implementation can forward them. This keeps call sites unchanged but duplicates the
`ScalarFnVTable` surface and creates two contracts for each hook. Default hook forwarding remains
outside the arithmetic loop, but the public trait change still requires the full IR gate.

No option is implemented here.

## Performance findings

### Mixed constants

The optimized `arm64` IR contains fixed-width vector loops for mixed `i64` add/sub and `i32`
multiply. The constant operand is broadcast with `insertelement` and `shufflevector`. Failure is a
loop-carried vector phi and rich error construction remains outside the loop. See
`codegen/final-batch-summary.md`.

### Skipped rows

`UninitElementSink::initialize_skipped_rows` writes `T::default()` to every row because its API has
no mask. A mask-aware API can accept `&Mask` and initialize only unset positions. This is a public
sink API change and needs x86 measurement for sparse and dense masks.

The early-exit problem is separable. A fallible or breakable mask iterator can stop after the first
row error without changing `OutputSink`. `Mask::indices()` is not an equivalent production fix
because it can materialize indices. No change is made here.

### Scatter

`scatter_valid` allocates `vec![0u64; valid.len()]`, fills ranks for set-bit runs, performs `take`,
and applies validity. A run-based scatter can copy dense value ranges directly into a pre-sized
output. That design is encoding-sensitive and needs a focused x86 benchmark. The spatial precedent
named in the finding is absent from this revision.

### Alignment

`OutputElement::build(Vec<T>)` reports `Alignment::of::<T>()` and does not retain the 256-byte
physical over-alignment from `BufferMut`. Repository consumers that need a stronger alignment use
`BufferHandle::ensure_aligned` in IO, serialization, Zstd, and benchmark paths. No numeric consumer
assumes 256-byte alignment. This is not a correctness requirement.

## Code generation gate

The API-only commit preserves the six owned arithmetic monomorphs and both `i64` division sink
paths. No new closure call, loop bounds check, vector loss, failure spill, or in-loop error
construction appears. See `codegen/api-contract-summary.md`.

The batch correctness commit adds one `panic_bounds_check` site for `output[index]` in the mixed
owned branch even though `owned.rs` is unchanged. The vector loops remain, but gate item 3 fails.
The commit stays on `ct/row-fn` and needs an x86 rerun before any upstream attempt.

## Do not do this

Do not delete the unreachable deferred sink machinery as semantically inert cleanup. The deletion
changed codegen-unit placement and the optimized owned arithmetic IR. It was reverted.

Do not move the `reduce_encoded` probe without checking the owned mixed-path bounds edge. The
correctness fix is valid, but the current source shape does not pass the arithmetic IR gate.

The investigation did not challenge any guardrail in the task. `owned.rs`, numeric primitive
inlining policy, the numeric `Vec<ArrayRef>`, `BorrowedExecutionArgs`, and LLVM loop flags remain
unchanged.

## Verification

```text
cargo nextest run -p vortex-array -p vortex-spatial -p vortex-tensor
  3805 passed, 1 skipped

cargo test --doc -p vortex-array
  73 passed, 13 ignored

cargo +nightly fmt --all
  passed

PYO3_PYTHON=.venv/bin/python cargo clippy --all-targets --all-features
  passed
```

The first clippy run selected `/usr/bin/python3` 3.9 and stopped because `abi3-py311` requires
Python 3.11. Rerunning with the repository virtual environment completed cleanly.
