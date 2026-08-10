<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# RowFn review follow-up

Platform: Apple Silicon `arm64`, macOS 15.7.3, Rust 1.91.0, LLVM 21.1.2. This machine uses
128-bit NEON and cannot reproduce or compare the pinned Ryzen wall-clock results. This
investigation uses optimized LLVM IR and correctness tests only.

Round 1 baseline: `833632aaa3cab59fb4a7d4f001df26975b2267a1` on `ct/row-fn`.

Round 2 target baselines: `84837ad36f` on `origin/ct/row-fn-api` and `32ad0bf3b7` on
`origin/ct/row-fn-numeric`.

## Result

| # | Verdict | Gate branch | Evidence and status |
| ---: | --- | --- | --- |
| 1 | Fixed, not upstreamed | API target plus downstream numeric target | `OutputSink::finish(self)` is unsafe. The zero-unsafe external reproduction changes from reading allocator contents to `E0133`. Publication awaits approval for the combined API commit. |
| 2 | Fixed, not upstreamed | API target plus downstream numeric target | `InputElement` is an unsafe trait with local safety proofs. `ElementTuple` and `IndexedElementTuple` remain sealed framework traits. Publication awaits approval for the combined API commit. |
| 3 | Mitigated | API target plus downstream numeric target | Unsafe publication belongs to `OutputSink::finish`, and each successful callback must return the sink's write token. The token is not type-tied to the exact row handle; misuse now requires violating an unsafe sink/input contract rather than safe client code. |
| 4 | Fixed, not upstreamed | API target plus downstream numeric target | `SKIPPED_ROWS_INITIALIZER: Option<fn>` makes capability and operation one fact while restoring decline before decode and allocation. Publication awaits approval for the combined API commit. |
| 5 | Fixed, not upstreamed | API target plus downstream numeric target | `reduce_encoded` probes original inputs once and returns `RowExecution`, so encoded reductions can defer errors behind nulls through the same validity path as row execution. Publication awaits approval for the combined API commit. |
| 6 | Investigated | Design only | A temporary manual implementation still reproduces `E0119`. A public `execute_rows` free function is the recommended redesign; it is not implemented in this pass. |
| 7 | Fixed, not upstreamed | Numeric target | Private `NumericBinary` deliberately borrows the registered `Binary` ID and is guarded by privacy. Publication awaits the API commit. |
| 8 | Fixed, not upstreamed | API target plus downstream numeric target | `Batch::new` validates each input length directly. The regression test fails before the fix. Publication awaits approval for the combined API commit. |
| 9 | Refuted | API target spot-check | `MaskedArray::try_new` enforces an all-valid child, and null `ConstantArray` validity is `AllInvalid`. Both couplings remain constructor invariants. |
| 10 | Fixed, not upstreamed | API target plus downstream numeric target | The encoding probe runs once before retry. The regression test fails before the fix and passes after it. Publication awaits approval for the combined API commit. |
| 11 | Fixed, not upstreamed | API target plus downstream numeric target | No sink can defer errors. The unreachable `finish_sink` retry classification and deferred finish argument are deleted. Publication awaits explicit approval for this deletion. |
| 12 | Refuted | API target tests and source spot-check | Filtering preserves literal, masked-child, and extension-storage constants recognized by `batch_constant`; new masked and extension tests pin the behavior. |
| 13 | Fixed, not upstreamed | API target plus downstream numeric target | The encoding probe runs before one-row broadcast and sees the original arrays. Publication awaits approval for the combined API commit. |
| 14 | Fixed, not upstreamed | API target plus downstream numeric target | The five unreachable deferred `SinkResult` word implementations and their capability pairing are deleted. Publication awaits explicit approval for this deletion. |
| 15 | Refuted | Downstream numeric target IR | Mixed `i64` add/sub and `i32` multiply contain broadcast vector loops on `arm64`; they are not scalar fallbacks. |
| 16 | Investigated, needs x86 | Analysis only | Full-row skipped initialization and non-breaking mask traversal remain independent costs. No performance change is made. |
| 17 | Investigated, needs x86 | API target source | `scatter_valid` still allocates one `u64` per original row. Whether replacing the gather pays for itself is a standalone benchmark question. |
| 18 | Refuted | API target source spot-check | No consumer requires row output to retain 256-byte physical alignment. Alignment-sensitive consumers call `ensure_aligned`; a performance change would still need x86 evidence. |

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

Round 2 adds `test_reduce_encoded_defers_errors_behind_nulls`. An encoded reducer reports a
`RowExecution::DeferredError`; mixed validity reruns only observable rows, all-valid validity makes
the error fatal, and all-invalid validity produces the declared all-null result.

The early-decline regression uses a sink whose `with_capacity` returns an error. Before the round 2
fix, `execute_sink_valid_rows` reaches that allocation before declining. The candidate observes the
absent initializer and returns `None` before decoding inputs or constructing the sink.

## API options for the blanket vtable

### `RowFnAdaptor<F>`

An adaptor can own the blanket `ScalarFnVTable` implementation while `F` remains a `RowFn`.
Registration and expression construction must wrap every function. Existing call sites that name
the concrete function type also change. Arithmetic can delegate to the same generic row executor,
but the wrapper changes monomorph identities and requires the full IR gate.

### Public `execute_rows` (recommended)

A public free function lets each adopter keep its `ScalarFnVTable` implementation and delegate only
execution. `Binary` keeps `coerce_args`, both simplifiers, and `fmt_sql`; `Between` and `Like` keep
their SQL formatting; the other concrete functions keep their existing encoded and validity hooks.
This is the least disruptive path for functions that already have a vtable. Its call-site cost is
one explicit `execute` delegation per adopter. Simple RowFn-only functions lose the blanket
one-line adoption story unless a separate opt-in adaptor is also provided. The arithmetic row loop
can remain the same helper monomorph, but the surrounding delegation still needs the full IR gate.

### Hooks on `RowFn`

`RowFn` can redeclare `coerce_args`, `simplify`, `simplify_untyped`, `reduce`, and `fmt_sql`.
The blanket implementation can forward them. This keeps call sites unchanged but duplicates the
`ScalarFnVTable` surface and creates two contracts for each hook. Default hook forwarding remains
outside the arithmetic loop, but the public trait change still requires the full IR gate.

`RowFnAdaptor<F>` changes registration and expression construction to name a wrapper and changes
monomorph identities. Re-declaring the hooks on `RowFn` duplicates the `ScalarFnVTable` contract.
Neither cost is justified merely to preserve the blanket implementation. No redesign is implemented
here.

## Coverage added in round 2

The new tests cover array-backed validity resolution, filter/scatter finalization, one-row constant
broadcast, output dtype and length validation, masked and extension constant unwrapping, all three
`RowPolicy` constructors, early decline for non-skipping sinks, dense-retry suppression, and all
three prepared visitor forms with both constant and varying inputs. `prepare` is asserted to run
once per batch.

## Performance findings

### Mixed constants

The optimized `arm64` IR contains fixed-width vector loops for mixed `i64` add/sub and `i32`
multiply. The constant operand is broadcast with `insertelement` and `shufflevector`. Failure is a
loop-carried vector phi and rich error construction remains outside the loop. See
`codegen/final-batch-summary.md`.

### Skipped rows

`UninitElementSink::SKIPPED_ROWS_INITIALIZER` writes `T::default()` to every row because the
initializer receives no mask. A mask-aware API can accept `&Mask` and initialize only unset
positions. This is a public sink API change and needs x86 measurement for sparse and dense masks.

The function-pointer option restores a compile-time capability fact without creating a second
boolean that can disagree with the operation. `RowPolicy::for_sink` does not need it: policy decides
whether dense execution is semantically legal, while skip support decides whether the later
valid-row sink attempt proceeds or falls back to filter-and-scatter.

The early-exit problem is separable. A fallible or breakable mask iterator can stop after the first
row error without changing `OutputSink`. `Mask::indices()` is not an equivalent production fix
because it can materialize indices. No change is made here.

### Scatter

`scatter_valid` allocates `vec![0u64; valid.len()]`, fills ranks for set-bit runs, performs `take`,
and applies validity. The cited spatial helper does exist on the target branches, but its typed
primitive/bool output construction is not a reusable implementation for arbitrary `ArrayRef`
encodings. A run-based scatter can copy dense value ranges directly into a pre-sized output, but
that design is encoding-sensitive. Whether avoiding the eight-byte-per-row gather index matters
needs a focused x86 benchmark.

### Alignment

`OutputElement::build(Vec<T>)` reports `Alignment::of::<T>()` and does not retain the 256-byte
physical over-alignment from `BufferMut`. Repository consumers that need a stronger alignment use
`BufferHandle::ensure_aligned` in IO, serialization, Zstd, and benchmark paths. No numeric consumer
assumes 256-byte alignment. This is not a correctness requirement.

## Code generation gate

Round 1's gate was measured on `ct/row-fn`, not on either branch that would receive the change.
Round 2 starts from exact `origin/ct/row-fn-api` and `origin/ct/row-fn-numeric` tips. The API branch
does not instantiate the production arithmetic monomorphs, so its candidate is compiled through
the numeric branch after rebasing the two numeric commits onto it.

On that target lineage, the `5204fb2` `output[index]` bounds regression does not reproduce. Owned
arithmetic has two bounds sites after the candidate rather than three at the numeric baseline. The
candidate and the separate deferred-sink deletion preserve inlining, vector factors, loop-carried
failure phis, overflow checks, and out-of-loop error construction. The deletion changes codegen-unit
placement but none of the stated loop properties. See
`codegen/target-branch-round2-summary.md`.

Staged capture found that the `RowExecution` reducer signature alone restores the third bounds
edge. The required coverage and final safety comment change placement again and remove it in the
final combined tree. Do not upstream or benchmark the reducer-signature commit in isolation.

The exact tested trees were consolidated without changing their contents:

- API candidate `f759998dcefa69b11d11b281e3bbebb6b88584e4`, based on `84837ad36f`.
- Numeric candidate `b4900072c1300238d246d395d986466db21583f6`, based on the API candidate.

Both remote tips still matched the recorded baselines after the final fetch. The candidates remain
unpublished because publishing the combined API commit, including the broad deferred-sink
deletion, requires explicit approval.

## Do not do this

Do not use a `ct/row-fn` IR result as the upstream gate. Every hot-path file and the linked numeric
code differ on the target branches; round 1's bounds regression does not reproduce there.

Do not call the deferred-sink deletion byte-identical. It changes codegen-unit placement on the
numeric target even though the arithmetic loop structure passes the gate.

Do not cherry-pick the `RowExecution` reducer-signature change without its required coverage and
the final target-tree gate. Its isolated target build reintroduces the owned mixed-path bounds edge.

The investigation did not challenge any guardrail in the task. `owned.rs`, numeric primitive
inlining policy, the numeric `Vec<ArrayRef>`, `BorrowedExecutionArgs`, and LLVM loop flags remain
unchanged.

## Round 1 verification

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

## Round 2 verification

API candidate:

```text
cargo nextest run -p vortex-array -p vortex-spatial -p vortex-tensor
  3757 passed, 1 skipped

cargo test --doc -p vortex-array
  73 passed, 13 ignored

cargo +nightly fmt --all
  passed

PYO3_PYTHON=/Users/connor/spiral/vortex-data/vortex1/.venv/bin/python \
  cargo clippy --all-targets --all-features
  passed
```

Downstream numeric candidate:

```text
cargo nextest run -p vortex-array
  3401 passed, 1 skipped

cargo test --doc -p vortex-array
  73 passed, 13 ignored

cargo +nightly fmt --all
  passed

PYO3_PYTHON=.venv/bin/python cargo clippy -p vortex-array --all-targets --all-features
  passed
```
