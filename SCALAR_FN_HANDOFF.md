<!-- SPDX-License-Identifier: Apache-2.0 -->
<!--SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Handoff: the row scalar-function framework

This is the concise source of truth for the branch. `STRICT_SCALAR_FN_RESEARCH.md` keeps the full
design history, rejected alternatives, measurements, and generated-code evidence.
`NUMERIC_ROWFN_PLAN.md` records the numeric-binary migration and its narrower performance boundary.
`research/rowfn-x86-2026-08-07/README.md` records the later x86 regression reproduction, the
owned-output and indexed-source experiments, raw benchmark logs, and exact production IR/assembly.
All three are branch-only working notes for agents. They are not intended to land with the API.

The public design lives in these tracking issues, which now match the implementation:

- [#9128, Row-oriented scalar functions](https://github.com/vortex-data/vortex/issues/9128)
- [#9129, Define the `RowFn` API](https://github.com/vortex-data/vortex/issues/9129)
- [#9130, Execute `RowFn` over Vortex arrays](https://github.com/vortex-data/vortex/issues/9130)

The branch is `ct/row-fn`, and draft PR #9255 remains the integration and research branch. Its
history was rewritten at `ea58061b5d` to remove the regressing broadcast-index-mask experiment.
Do not use the draft PR as the first mergeable change. Cut the first PR from the latest
`origin/develop`, and keep this branch as the source for later tensor and spatial ports. Push or
rewrite either branch only when explicitly requested.

## Next action: cut the vortex-array PR

The first mergeable PR must stay within `vortex-array` and contain:

1. the `RowFn` API, lifting, executor, and focused behavioral tests.
2. the primitive `NumericBinary` port as its production consumer.
3. only the executor and numeric benchmarks needed to support its performance claim.

Do not include the tensor or spatial ports, these branch-only working notes, the unrelated `like`
benchmark additions, or the fixed-size-list test. `NumericBinary` is the only `RowFn` consumer in
`vortex-array` on this branch. It already exercises varying and constant inputs, all-constant
folding, null constants, nullable execution, deferred overflow evidence, and the valid-row retry.
Do not add another consumer only to make the PR appear broader.

The numeric commit deletes the now-unused `vortex-compute::lane_kernels::map_into` helper. Leave
that helper in place for a strictly `vortex-array`-only PR, and remove it in a separate cleanup.

The first PR must establish parity against the latest `origin/develop`, not the integration
branch's old merge base. Run the public `binary_ops` benchmark on x86 with identical build flags at
both revisions. Cover varying x varying, varying x constant, constant x varying, and nullable plus
constant inputs. Run each revision at least twice in alternating order. Record the exact commits,
CPU, timer, pinning, fastest values, and medians. Inspect optimized LLVM IR or assembly for every
stable regression before changing the row API.

The production benchmark commands across the staged work are:

```bash
cargo bench -p vortex-array --bench binary_ops
cargo bench -p vortex-array --bench like
cargo bench -p vortex-tensor --bench l2_norm
cargo bench -p vortex-tensor --bench inner_product
cargo bench -p vortex-tensor --bench cosine_similarity
cargo bench -p vortex-tensor --bench normalized
cargo bench -p vortex-spatial --bench binary_predicates
cargo bench -p vortex-spatial --bench distance
cargo bench -p vortex-spatial --bench envelope
cargo bench -p vortex-spatial --bench predicate_bbox
```

The public benchmark names are shared with `develop`, so cross-revision comparisons do not need a
frozen benchmark-local implementation as their primary control.

## The API in one screen

`RowFn` is the author-facing function trait. A function gives the framework its exact argument
names, a conservative fallibility declaration, function-owned persistence, and a value-blind
dispatch over concrete input and sink types:

```rust
impl RowFn for Example {
    type Options = ExampleOptions;

    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.example");
        *ID
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(encode(options)?))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        decode(metadata, session)
    }

    fn dispatch<V: RowVisitor>(
        &self,
        options: &Self::Options,
        args: &[DType],
        visitor: V,
    ) -> VortexResult<V::Out> {
        validate_options(options, args)?;
        visitor.visit_prepared_into::<(InputA, InputB), ElementSink<Output>, _, _>(
            |_| (),
            |&(), (lhs, rhs), output| {
                *output = compute(lhs, rhs);
            },
        )
    }
}
```

There are no argument or return witness types. The dispatched tuple is the argument declaration,
the sink owns the output representation, and the row result names the error behavior. Planning
runs the same dispatch as execution and checks the selected types against the function constants.

## The extension boundary

The framework is deliberately not sealed wholesale. Function authors need to add decode and output
primitives for their own scalar functions. Only the executor mechanics are closed.

| API | Boundary | Why |
| --- | --- | --- |
| `RowFn` | open | Defines a scalar function and selects concrete execution types. |
| `InputElement` | open | Adds a new scalar decode primitive, including crate-local domain types. |
| `OutputElement` | open | Adds an ordinary one-value-per-row output primitive. |
| `OutputSink` | open | Adds a custom output representation or builder. |
| `RowVisitor` | sealed | Executor-owned dispatch mechanism with one supported implementation. |
| `ElementTuple` | sealed | Executor-owned tuple recursion, with built-ins through arity 12. |
| `SinkResult` | sealed | Executor-owned loop and error facts trusted by the blanket vtable. |

`ElementTuple` being sealed does not prevent a function from adding a decode primitive. Implement
`InputElement` and use it inside one of the supplied tuples. Likewise, a function with two logical
outputs should define one `OutputSink` whose state has two fields. The framework does not need a
second tuple or composite-sink abstraction.

The supplied `SinkResult` forms are:

- `()` for infallible rows;
- `VortexResult<()>` for an error that must stop immediately; and
- `bool`, `u8`, `u16`, `u32`, or `u64` for error evidence OR-reduced after the loop.

The unsigned evidence widths let each kernel choose a word no wider than its element type. That is
load-bearing for vectorization, particularly for checked unsigned multiplication.

## Function-owned persistence

Persistence belongs to the function ID, not to the Rust options type. `RowFn::Options` has no
serialization supertrait. The `RowFn::serialize` and `RowFn::deserialize` hooks have conservative
defaults, and registered functions override them when their existing wire contract requires it.

This has three useful consequences:

- two functions may reuse an options type while choosing different formats;
- a function may deliberately be nonserializable even if another function serializes the same
  options type; and
- an unregistered helper such as `NumericBinary` needs no dummy persistence implementation.

Tensor and geo functions keep their explicit existing formats. Do not introduce a blanket options
wire format or infer serializability from `Options`.

## One sink abstraction

`OutputSink` is the complete output contract. It owns the output dtype, allocation, row storage,
row lookup, length proof, and final array construction. `ElementSink<T>` covers the common case. Its
row type is `&mut T`, so the closure writes with ordinary assignment.

Custom sinks remain available for a real output shape that cannot use `ElementSink`. The unused
public `TensorSink` was removed. No current tensor row function returns tensor-valued rows, and a
90-line public runtime-shaped sink was not justified without a user. Add a custom sink when a real
function needs one, using one sink struct even when it owns several builders.

Every current sink produces an all-valid child column. The blanket vtable can therefore derive the
function result validity from the input validities. Nullable row outputs remain out of scope. A
sink that emits its own nulls must change that derivation in the same change.

`OutputSink::sink_dtype` must return a non-nullable dtype. `SUPPORTS_SKIPPED_ROWS` says whether
branch-and-skip may leave placeholder rows behind the result validity. `ERRORS_ARE_DEFERRED` says
whether the sink accepts accumulated error evidence at `finish`.

## Dispatch and fallibility

`dispatch` must be pure in `(options, args)`. It sees dtypes, not array values. Planning and
execution both call it, so value-dependent preparation belongs inside `visit_prepared_into`.

The executor statically checks each dispatched visit:

- the tuple arity equals `ARG_NAMES.len()`;
- a fallible decoder, early-error result, or deferred result implies `RowFn::FALLIBLE`;
- deferred evidence requires both `RowFn::FALLIBLE` and a sink with
  `ERRORS_ARE_DEFERRED = true`; and
- the sink and result agree about their error contract.

The implications are intentionally one-way. `FALLIBLE = true` is a conservative function-level
claim, while a particular dtype dispatch arm may be infallible.

`prepare` must not be load-bearing for validation. Empty batches may bypass value preparation, and
the executor needs its safety and fallibility facts before it runs the closure.

## Null execution policy

The old public `NullHandling` enum and argument witness were removed. Authors do not select an
execution mechanism. The executor derives a private row policy from the dispatched input and result
types:

- `Dense` may execute over garbage behind nulls and masks afterward;
- `DenseWithRetry` may execute densely, then retry valid rows when deferred evidence reports an
  error; and
- `ValidOnly` guarantees that the row closure sees only valid rows.

An early-failing row or a decoder that is not dense-safe must use valid-only execution. A deferred
kernel may use dense execution because it writes a legal provisional value for every row. If only
garbage behind nulls reports an error, the valid-row retry discards it.

Valid-only execution first calls `reduce_encoded` on the original arrays. If reduction declines,
the executor tries branch-and-skip on the original batch. This path decodes values behind nulls and
visits the set bits from the conjoined validity mask. It requires null-tolerant input decoding and a
sink that supports skipped rows.

If branch-and-skip declines, filter-and-scatter shrinks the inputs before decoding. It then scatters
the output into a full-length nullable array. Authors declare local safety through their input and
result types. They do not select the mechanism or provide a decode-cost estimate.

## Performance and generated-code evidence

The older Ryzen 9 7950X AVX-512 measurements remain the production-performance record in the
[#9128 follow-up](https://github.com/vortex-data/vortex/issues/9128#issuecomment-5151831802).

The final API cleanup was checked separately against its parent, `53c51d803c`, by cross-compiling
the optimized `row_fn_executor` benchmark for `x86_64-apple-darwin` with `target-cpu=x86-64-v3`.
After normalizing symbol names and metadata, the vector and reduction blocks were identical for all
three executor shapes:

- ordinary wrapping add through `ElementSink`;
- checked add with deferred evidence; and
- wrapping add through a custom sink.

The wrapping loops retain 256-bit `<4 x i64>` loads, adds, and stores. The checked loop retains the
same vector loads and adds, derives overflow with vector xor/and/compare operations, accumulates
`<4 x i1>` with vector OR, and reduces after the loop. None of the vector bodies contains a call or
panic path. Scalar tails are unchanged.

The production tensor benchmarks were also cross-compiled before and after the cleanup. Normalized
arithmetic sequences and counts match for `l2_norm`, inner product, and cosine similarity. Their
ordered floating-point reductions are scalar-unrolled in both revisions because LLVM preserves the
strict reduction order. The cleanup did not remove vectorization because those reductions were not
vectorized before it.

Native Apple M4 Max timings used 65,536 rows, two alternating before/after runs, 100 samples, and a
0.5-second minimum per arm. RowFn median deltas ranged from 1.11% faster to 0.94% slower. Fastest
deltas stayed within about 0.17%, while specialized controls drifted by as much as 3.7% in their
medians. There is no measurable native regression from the API cleanup.

This does not replace the required x86 runtime run above. Cross-target IR proves that the hot loop
shape survived, not that the revised null selector has the expected branch-predictor behavior on
x86.

## Current implementation and checks

The implementation includes production users in `vortex-array`, `vortex-tensor`, and
`vortex-spatial`.
`NumericBinary` is an unregistered `RowFn` used only for primitive arithmetic execution. Decimal
arithmetic keeps its existing path. The stable public-path benchmark baseline landed as #9136.

The checks recorded for the final API state are:

- 67 focused RowFn tests;
- 179 `vortex-tensor` tests;
- 230 `vortex-spatial` tests;
- `cargo +nightly fmt --all`; and
- full workspace clippy, with `PYO3_NO_PYTHON=1 PYO3_BUILD_EXTENSION_MODULE=1` because the host
  `/usr/bin/python3` is 3.9 while the workspace requires the Python 3.11 stable ABI.

The generated-code comparison and native timing evidence are described above and in the final
section of `STRICT_SCALAR_FN_RESEARCH.md`.

## Review pass: what changed and what was deliberately left

A review of the three parts (API, execution, implementations). **The author-facing API is
unchanged**: every proposal that would have altered it was backed out, for the reasons below, and
what landed is cleanup, corrected documentation, and test coverage. The emitted IR of every
`visit_prepared_into` monomorph is identical to the pre-review commit.

API:

- `InputElement::decode_null_tolerant` overrides that only restated the default were deleted from
  the primitive, bool and `TensorRow` elements. `GeometryRow`'s override is the only real one. The
  doc now says a dense-safe element should *not* override.
- `ElementTuple` now records why it carries arities past the widest function in tree: it is sealed,
  so a downstream crate cannot add the one it needs, and an uninstantiated arity costs only its own
  macro expansion.

Execution:

- `execute_filtered` and the forced-strategy test seam now share `resolve_validity`, so the mask
  materialization and the all-true/all-false shortcuts cannot drift apart between them.
- The dense-retry path's comment was wrong and is corrected. It filters unconditionally because
  `execute_dense` is not handed the `branch` closure, **not** because a deferred sink cannot skip
  rows: `ERRORS_ARE_DEFERRED` and `SUPPORTS_SKIPPED_ROWS` are independent consts and a sink may
  legally set both.

Implementations:

- `l2_norm_row` had two copies, in `l2_norm.rs` and `cosine_similarity.rs`. Cosine's prepared and
  per-row arms must agree bit for bit, which only holds while both accumulate in the same order, so
  the duplicate was an invitation to break exactly the property the comments defend. One copy now
  lives in `utils.rs` beside the other shared tensor helpers.
- `CosineSimilarity::reduce_encoded` zips its three slices instead of indexing `0..len` three times
  per row, and documents why it materializes where `InnerProduct::reduce_encoded` stays lazy (the
  zero-norm guard is a conditional, not an arithmetic factor).
- `IndexedSourceExt::map_checked_into` was deleted from vortex-compute. `CheckedSink` replaced the
  split value/evidence pass it served, and it had no caller left.
- `contains_route` and the workspace `geo` dependency both record that the table transcribes geo's
  `impl_contains_from_relate!` and must be re-verified on a version bump. `geo` is pinned to
  `=0.31.0`: a caret requirement would admit 0.31.x patches, which `cargo update` (or automated
  lockfile maintenance) takes with no diff to review, and a patch is free to reshuffle the dispatch
  without any API change. The agreement tests stay green wherever relate and the direct algorithm
  agree, so the pin, not the suite, is what makes the coupling break only deliberately.

Split out onto `develop` instead of landing here:

- **The checked-arithmetic macro collapse.** `primitive.rs` on this branch and on `develop` both
  carry four near-identical `CheckedArithmetic` bodies that differ only in `mul_failure`, so the
  collapse into one `impl_checked_integer!` belongs on `develop` where every caller benefits. It is
  on `claude/collapse-checked-arith-macros`. This branch's `primitive.rs` keeps its four bodies
  until `develop` is merged, at which point the collapse arrives with it and the merge conflict is
  a member deletion rather than two competing macro structures.
- **The `mul_failure` kernel tests.** The exhaustive 8-bit sweep and the 64-bit probe grid already
  exist on `develop` from vortex-data/vortex#9210 and arrive with the same merge.

Deliberately **not** done:

- **No `DeferredElementSink`.** `CheckedSink` exists largely because `ElementSink` cannot name an
  error at `finish`. A framework sink combining an element output with a type-level message would
  remove ~100 lines per function, but there is exactly one deferred-error function. Build it when a
  second appears, rather than copying `CheckedSink`.
- **No change to `reduce_encoded`'s probe semantics.** Hoisting the probe out of the strategy paths
  and masking a full-length result looks like a simplification and is not one:
  `normalized_readthrough_survives_null_rows` pins that a filtered input is no longer `Normalized`,
  so which arrays reach `reduce_encoded` is load-bearing and differs per strategy.
- **No mixed-constant specialization without a failing benchmark.** The broadcast-index-mask
  experiment regressed numeric constants by 4x to 7x on x86 and was removed. Keep the current
  executor for the first PR. Add a specialized varying x constant or constant x varying loop only
  after the clean branch has a stable regression against the current merge base.

### Three API changes proposed, and why none of them landed

All three were implemented, run against the suite, and backed out. None prevents a bug, and this
branch's open work is *settling* the API rather than churning it, so they belong in #9129 as
questions decided alongside the rest of the surface:

- **Should `reduce_encoded` take an explicit `row_count`?** The filtered-count requirement is real
  and easy to miss, but `args` are filtered to match, so `args[0].len()` is already both the natural
  thing to write and correct. The parameter is documentation, and it costs every implementor a
  signature change. What survived is the test:
  `reduce_encoded_is_probed_before_and_after_filtering` pins that the rewrite is offered the
  original arrays at full length and then the filtered ones at the surviving count.
- **Should `OutputSink::row_count_matches` become `rows_len`?** A length reads cleaner and lets the
  executor name what it found. Against that, `row_count_matches` lets a sink fold in its own
  invariants, which `SpreadSink` uses for its width check; narrowing it turns that into a panic.
  Neither spelling prevents a bug.
- **Should the nullary path go?** A function with no inputs has no validity to lift, which is the
  lifting's whole job. But `RowFn` would still give it sink allocation and dtype derivation, so
  `random()` or `now()` is not obviously better hand-written, and the path is ~70 lines and tested.

Trimming `ElementTuple` to arity four was proposed on the same reasoning and backed out for a
stronger one: the trait is sealed, so the arities are the only ones a downstream crate can ever
have.

### Two changes this pass made and then reverted

Both were proposed, implemented, reviewed, and backed out on evidence. They are recorded because
each is an attractive idea that a later reader will have again.

**Making `CheckedSink` safe with `BufferMut::zeroed` costs 1.65 to 1.71x.** Replacing the
`MaybeUninit` storage removes an `unsafe set_len` and reads as a clear win, and `ElementSink`'s own
comment appears to bless it by routing a zeroable placeholder to `alloc_zeroed`. Measured, it is
not: allocate-zeroed-then-fill against allocate-then-fill, interleaved in one process over `u64`
outputs, ran **1.221x** slower at 8 KiB, **1.71x** at 64 KiB, **1.66x** at 512 KiB and **1.71x** at
2 MiB, stable to within 2% across two runs. `alloc_zeroed` does not avoid the write: below glibc's
mmap threshold `calloc` recycles a dirty chunk and memsets it, and above it every fresh page faults
on first touch. The row loop overwrites every slot regardless, so this is a duplicated pass over
the output of the hottest kernel in the system.

Note the corollary, which is a real optimization nobody has taken: `ElementSink::with_capacity`
pays exactly this on every batch, and only branch-and-skip ever reads a placeholder back. A sink
that allocated uninitialized on the dense and filter paths would recover it.

**Hoisting `OutputSink::SUPPORTS_SKIPPED_ROWS` into the plan is not sound as an optimization.**
#9130 records "avoid probing `reduce_encoded` twice when branch execution is unsupported" as a
follow-up. It reads as free, and is not, because the branch path probes `reduce_encoded` against
the _original_ arrays before it consults the sink, and that is the only probe that ever sees them
still encoded. Skipping the path early leaves such a function with only the filtered probe, whose
canonical arrays match no encoding fast path. For a function whose reduction is _defined_ to answer
differently from its row loop, which is exactly what `L2Norm` over `Normalized` is, that is a wrong
answer rather than a slow one. Nothing in tree is reachable today only because every `ValidOnly`
dispatch happens to use `ElementSink`. **#9130's follow-up should be struck, not implemented.**
`reduce_encoded_is_probed_before_and_after_filtering` now pins the two probes and their row
counts.

### On measurement, and what the IR gate does and does not cover

Wall-clock benchmarking of the row loops was attempted first and abandoned on evidence. Two runs of
the *same* baseline binary, pinned with `taskset -c 2`, 100 samples, disagreed by up to 4x
(`row_wrapping_add_nullable`: 198.8 us then 52.9 us median; `specialized_checked_add`: 185.5 us then
34.4 us). The 4-vCPU shared VM drifts more within a session than any effect being measured, which is
the same conclusion this branch already reached on a dedicated 7950X.

The gate used instead is the emitted optimized IR of every `visit_prepared_into` monomorph in
`vortex-array`, profiled by vector width, reduction count, overflow-intrinsic survival and bounds
checks, then compared as a multiset before and after. Reproduce with:

```bash
RUSTFLAGS="--emit=llvm-ir -C codegen-units=1" cargo rustc -p vortex-array --release --lib
```

**Its blind spot is worth stating, because it nearly landed a regression.** The IR of a row loop
cannot show an allocator call outside it, so the `BufferMut::zeroed` substitution above passed this
gate cleanly while costing 1.7x. An allocation-strategy change needs its own targeted A/B, which is
cheap to write and immune to the host drift above because both arms run interleaved in one process.
Use the IR gate for loop shape and a focused microbenchmark for anything the loop does not contain.

## Remaining boundaries

- Keep nullable outputs separate until the first real function can define the validity contract.
- Do not add another sink composition abstraction. Put multiple builders in one custom sink.
- Do not add a general runtime-shaped sink until a production function needs one.
- Keep pattern compilation and other state shared across rows outside `RowFn` when it cannot be
  represented as batch preparation.
- Use emitted optimized IR as a gate for numeric changes near LLVM's vectorization boundary, then
  use the stable #9136 benchmark names for runtime confirmation.

## Repository rules for the next agent

Follow `AGENTS.md`. Keep public APIs small, run narrow checks before workspace-wide checks, and
report blocked checks separately from passing ones. Preserve unrelated working-tree and staging
state. Every commit must include the required `Signed-off-by` trailer.
