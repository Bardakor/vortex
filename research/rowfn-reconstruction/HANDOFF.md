<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# RowFn investigation handoff

This file records the current state of the 2026-08-09 investigation. Start with this file, then
read [`DESIGN.md`](DESIGN.md), [`OPTIMIZATION.md`](OPTIMIZATION.md), and
[`REPRODUCE.md`](REPRODUCE.md).

## Branch state

- Branch: `ct/row-fn`.
- Last RowFn code commit: `4c936447a`.
- Documentation head before the offsets fix: `bdf95a77e`.
- Comparison revision: develop at `66d096b5d`.
- Direct offsets fix: `61410ef21`.
- Numeric helper ID fix: `f9dfde730` on this branch and `df8fcbe1a` on `ct/row-fn-api`.
- Masked tensor decode fix: `7baa9fab7`.
- Cleaned `ct/row-fn-api` head: `6dd500f59`.
- PR #9299 head measured locally: `d97e53e66`.

The API branch was rewritten with an exact force-with-lease from seven commits to five:

1. `71c3e7a58` adds the framework, refined contracts, and self-contained arguments.
2. `6e864bf8b` moves primitive numeric operators to RowFn and reuses `Binary`'s ID.
3. `41bb10143` adds focused executor benchmarks.
4. `266350488` removes validated input bounds checks.
5. `6dd500f59` restores mixed-constant performance.

Three temporary remote refs exist for the CodSpeed ablation:

- `ct/row-fn-codspeed-framework` points to `0a0ad0db1`.
- `ct/row-fn-codspeed-numeric` points to `89fd28bc1`.
- `ct/row-fn-codspeed-take-filter` is the head of temporary draft PR #9298.

The first two refs contain exact historical code. The third ref adds a PR-only workflow that runs
only `cargo codspeed run --bench take_filter`.

## Corrected CodSpeed history

The latest push did not bring back the `take_filter_list_*` regressions.

- The [CodSpeed check at `892717f30`] already reports the cases as about 15% to 16% slower.
- The [CodSpeed check at `4c936447a`] reports the same cases as about 14% to 16% slower.
- Most take/filter simulated times improve by less than 2% between those checks.
- `4c936447a` fixes the much larger constant add, subtract, and multiply regressions. This moves the
  persistent take/filter entries higher in the ordered list of the 20 largest changes.
- Every retained RowFn CodSpeed summary from `0e5c19c00` through `4c936447a` that has a performance
  table also contains take/filter regressions.

The PR bot edits one current comment, and GitHub displays only the 20 largest changes. These two
details can make a persistent regression appear to leave and return.

## Verified take/filter cause

The list, filter, and take source files are identical between develop and `4c936447a`. The
benchmark still reaches RowFn through an indirect call:

```text
take_filter
  -> list_view_from_list
    -> ListArrayExt::reset_offsets
      -> binary(Sub) on offsets and the first offset
        -> numeric RowFn
```

The differential profile therefore corrects the earlier claim that the benchmark does not execute
RowFn. `reset_offsets` creates a constant array and runs generic numeric subtraction. Numeric RowFn
adds batch planning, dispatch, argument decoding, and output reconciliation to this small operation.

The representative benchmark is
`take_filter_list_small_uncached_random_mask_random_indices[256, 10]`. The current PR report gives
233.737 microseconds for develop and 280.793 microseconds for `bdf95a77e`. This is a 16.76%
regression.

CodSpeed creates the downloadable callgraph in a separate profiling execution. Its total can
differ slightly from the aggregate report. The callgraph components are:

| Revision | Instructions | Cache | Memory | Total |
| --- | ---: | ---: | ---: | ---: |
| Develop `66d096b5d` | 21.312 us | 83.443 us | 133.294 us | 238.050 us |
| RowFn `bdf95a77e` | 26.210 us | 104.293 us | 155.172 us | 285.675 us |
| Increase | 4.898 us | 20.850 us | 21.878 us | 47.626 us |

The extra instructions and the changed stack rule out a cache-only layout explanation. Cache and
memory costs also increase, but they occur on newly executed RowFn work.

The largest changed functions in the focused numeric profile are:

| Function | Base self / total | Head self / total |
| --- | ---: | ---: |
| Old `execute_numeric_primitive` | 0.741 / 18.639 us | absent |
| RowFn `execute_numeric_primitive` | absent | 0.430 / 71.156 us |
| `Batch::execute` | absent | 1.033 / 49.972 us |
| `Batch::execute_dense` | absent | 0.634 / 45.781 us |
| `NumericBinary::dispatch` | absent | 1.316 / 45.736 us |
| `(A, B)::decode` | absent | 0.539 / 37.501 us |
| `ArgColumn<T>::decode` | absent | 0.968 / 36.254 us |
| `list_view_from_list` | 3.543 / 79.144 us | 2.592 / 108.951 us |
| `Batch::new` | absent | 1.797 / 10.794 us |

These totals are inclusive callgraph costs. A function can appear in more than one caller stack.

The linked AVX2 benchmark binaries still differ. Native inspection found:

- The main filter-take function has the same `0x41cc` byte size on develop, `892717f30`, and
  `4c936447a`.
- The main list `TakeExecute::take` function has the same `0x40ac` byte size.
- Normalized list-take disassembly has the same instructions.
- Function addresses, relative call targets, and linked layout differ.

That native inspection covered the large take and filter functions. It missed the changed numeric
callee reached during list offset normalization.

CodSpeed documents [function alignment] as a reason unchanged microbenchmarks can move after a
rebuild. That warning remains useful, but alignment is not the cause of this simulation regression.

## Native measurements are separate evidence

For the rest of this investigation, pinned local x86 wall time is the primary acceptance signal.
CodSpeed remains useful for finding changed call paths and separating instruction, cache, and
memory costs, but a simulated microbenchmark movement is not by itself a reason to reject code
that has native parity or an improvement. Keep the two measurements labeled; neither predicts the
other.

Pinned AVX2 wall-time runs on an AMD Ryzen 9 7950X found both `892717f30` and `4c936447a` about 25%
to 31% slower than develop for the tested take/filter list cases. The final push changes those
native medians by only 0% to 2%.

Changing the bench profile from 16 codegen units to one did not remove that native gap. One
representative median pair was:

| Profile | `4c936447a` | Develop |
| --- | ---: | ---: |
| 16 codegen units | 8.25 us | 6.41 us |
| One codegen unit | 7.86 us | 6.21 us |

These measurements do not explain the CodSpeed simulation result. Do not use local wall time as a
proxy for CodSpeed CPU simulation.

### Primitive numeric matrix

The cleaned API branch was compared with develop on an AMD Ryzen 9 7950X. Each Divan binary was
pinned to logical CPU 2 and used the TSC timer, 100 samples, and a 250-millisecond minimum time.
Five alternating runs covered 26 shared `binary_ops` cases.

Before the mixed-constant fix, the varying cases were generally within 0% to 8.5% of develop. The
constant cases exposed a separate source-placement regression:

| Benchmark | Develop | Before fix | Difference |
| --- | ---: | ---: | ---: |
| `add_i64_constant` | 8.369 us | 35.42 us | +323.2% |
| `sub_i64_constant` | 8.319 us | 36.19 us | +335.0% |
| `mul_i32_constant` | 26.43 us | 41.91 us | +58.6% |

Commit `6dd500f59` keeps each length proof in the branch that consumes it. After the fix,
`add_i64_constant` measures 9.269 microseconds, `sub_i64_constant` measures 9.199 microseconds, and
`mul_i32_constant` measures 18.91 microseconds. The first two retain about 11% overhead; multiply
is 28.5% faster than develop.

### `mul_u16_nonnull` code placement

Ten one-second alternating runs isolate a stable native regression:

| Binary | Median | Observed range |
| --- | ---: | ---: |
| Develop `66d096b5d` | 2.229 us | 2.229 to 2.239 us |
| Clean API `6dd500f59` | 2.809 us | 2.799 to 2.829 us |
| `-C llvm-args=-align-loops=64` diagnostic | 2.449 us | 2.439 to 2.499 us |

The develop and RowFn steady-state loops have the same normalized instruction sequence: two
128-bit loads, `pmullw`, `pmulhuw`, failure accumulation, one store, and the loop branch. Both are
vectorized. Develop's loop starts 16 bytes into a cache line and fits in that line. The ordinary
RowFn loop starts 32 bytes into a line and crosses the boundary.

The LLVM diagnostic did not force this loop to a 64-byte boundary. It changed the linked layout so
the loop starts 19 bytes into a line and fits. That recovers 0.360 microseconds of the 0.580
microsecond gap, leaving the diagnostic binary 9.9% slower than develop. This is evidence that code
placement matters, but it is not a complete cause or a suitable global compiler flag. Do not add
padding or enable the hidden LLVM option as a production fix.

Samply could not record this benchmark because `perf_event_paranoid` is 2 and the machine requires
1 or lower. The assembly comparison is available evidence; there is no sampled native profile.

Outlining the validated all-varying lane kernel behind `#[inline(never)]` did not change the result.
Ten runs measured 2.799 to 2.829 microseconds, the same range as the ordinary cleaned API binary.
Do not add this code movement; it does not isolate the residual cost.

Compiling both revisions with `-C target-cpu=native` reduces the gap. Ten alternating runs measure
2.259 to 2.269 microseconds on develop and 2.479 to 2.489 microseconds on the API branch. The
native difference is about 9.7%, not 26%.

Both native loops use AVX-512. Develop handles 64 `u16` lanes per iteration with two ZMM vectors.
RowFn handles 128 lanes with four ZMM vectors. Both compute `vpmullw`, `vpmulhuw`, the failure OR,
and the output stores. The remaining difference is not lost autovectorization.

Five alternating native runs across all 27 shared `binary_ops` cases give this shape:

- Decimal arithmetic, integer division, comparisons, and nullable wide arithmetic are within 1%.
- Varying narrow integer operations are generally 4% to 12% slower.
- `mul_i64_nonnull` is 2.8% faster and `mul_u64_nonnull` is at parity.
- Constant `i64` add and subtract are 19.5% and 22.9% slower.
- Constant `i32` multiply is 22.5% slower.

The mixed-constant native loops also use AVX-512 broadcasts and packed arithmetic. Their remaining
regressions are not scalar fallbacks.

Replacing numeric dispatch's two-element `Vec<ArrayRef>` with a stack-backed borrowed view removes
an allocation but does not improve the repeated matrix. Do not keep that change without a smaller
benchmark that shows the allocation itself matters.

Skipping `Array::validity` for inputs whose dtype is non-nullable is also not a measured fast path.
Five focused native comparisons move non-nullable and constant cases by less than 1%. Nullable
controls move by a similar amount even though their executed logic is unchanged. Treat those
differences as linked-layout noise and keep the uniform validity fold.

A 32-times-larger batch separates fixed setup from loop throughput. The benchmark-only ablation
changes `LEN` from 32,768 to 1,048,576 and keeps `target-cpu=native`:

| Benchmark | Develop | RowFn | Difference |
| --- | ---: | ---: | ---: |
| `add_i64_constant` | 162.3 us | 163.0 us | +0.4% |
| `sub_i64_constant` | 162.5 us | 162.8 us | +0.2% |
| `mul_i32_constant` | 94.84 us | 92.99 us | -2.0% |
| `mul_u16_nonnull` | 61.04 us | 61.53 us | +0.8% |
| `add_i32_nonnull` | 121.8 us | 122.1 us | +0.2% |
| `mul_i64_nonnull` | 778.1 us | 749.4 us | -3.7% |

The per-element loops have native parity or better at scale. The visible percentages at 32,768
rows come primarily from fixed RowFn batch planning, dispatch, decode, and reconciliation costs.
Do not attribute them to failed autovectorization or slower arithmetic throughput.

The framework control reaches the same conclusion without the numeric wrapper. Five
`target-cpu=native` runs of `row_fn_executor` compare 65,536-row loops in one linked binary. The
hand-written sink median is 137.4 microseconds. Infallible owned RowFn execution is 138.8
microseconds, and sink RowFn execution is 138.5 microseconds, both within 1%. Checked owned
execution is 141.9 microseconds, or 3.3% slower. The shared executor does not impose a large
steady-state throughput cost.

## Focused CodSpeed ablation

Two `workflow_dispatch` runs were started and then canceled:

- Framework only: [run `31289620637`].
- Numeric RowFn: [run `31289622392`].

This approach was not sufficient. A workflow-dispatch run has no pull-request context, so it does
not create the needed comparison. Do not use either run as performance evidence.

Draft PR [#9298] provides the required pull-request context. Its workflow builds and runs only the
`take_filter` benchmark.

- [Focused framework check] at `0a0ad0db1`: 232.542 microseconds against 233.737 microseconds for
  develop. This is a 0.51% improvement and CodSpeed classifies it as no change.
- [Focused numeric check] at `89fd28bc1`: 279.491 microseconds against 233.737 microseconds for
  develop. This is a 16.37% regression.

`89fd28bc1` is the first bad revision. It is the direct child of clean revision `0a0ad0db1`.

The numeric revision's callgraph totals are 25.835 microseconds for instructions, 103.531
microseconds for cache, and 154.728 microseconds for memory. Develop's totals are 21.312, 83.443,
and 133.294 microseconds. The total increases from 238.050 to 284.093 microseconds.

## Focused fix

`ListArrayExt::reset_offsets` now decodes offsets once and subtracts the first offset in a typed
loop. It no longer allocates a constant array or invokes the generic scalar-function path.

The AVX2 release binary auto-vectorizes every supported integer width. Each unrolled iteration has
two 128-bit packed subtracts:

- `psubb` handles 32 `i8` or `u8` offsets.
- `psubw` handles 16 `i16` or `u16` offsets.
- `psubd` handles 8 `i32` or `u32` offsets.
- `psubq` handles 4 `i64` or `u64` offsets.

Signed and unsigned monomorphs share machine code. This is code-generation evidence, not a local
timing result.

A new test covers nonzero `u16` offsets. The existing list and list-view tests cover other offset
types and conversion behavior.

The [offsets fix check] validates the change in CodSpeed CPU simulation. The representative case
measures 176.524 microseconds, compared with 233.737 microseconds on develop and 280.793
microseconds before the fix. It changes from a 16.76% regression to a 32.41% improvement against
develop. All 14 `take_filter_list_*` cases improve by 25.61% to 35.54% against develop.

The representative callgraph components after the fix are:

| Revision | Instructions | Cache | Memory | Total |
| --- | ---: | ---: | ---: | ---: |
| Develop `66d096b5d` | 21.312 us | 83.443 us | 133.294 us | 238.050 us |
| Before fix `bdf95a77e` | 26.210 us | 104.293 us | 155.172 us | 285.675 us |
| Offsets fix `61410ef21` | 15.462 us | 59.031 us | 103.767 us | 178.259 us |

The generic scalar-function stack is absent after the fix. The typed `reset_offsets` function
costs 0.933 microseconds self and 7.629 microseconds total. On develop, the old primitive numeric
function alone costs 0.741 microseconds self and 18.639 microseconds total. The larger reduction in
`list_view_from_list`, from 79.144 to 29.634 microseconds total, includes the lazy scalar-function
array and optimizer work removed by the direct operation.

PR [#9299] originally extracted this direct typed subtraction at `fa54891b`. Five alternating
native AVX2 runs found that superseded revision 27.9% to 33.3% faster than its exact develop base.
Do not attribute those numbers to the current PR implementation.

The current PR head, `d97e53e66`, keeps the generic lazy subtraction in `reset_offsets`. It executes
the normalized offsets once in `list_view_from_list`, then uses the same primitive array to build
sizes and output offsets. A fresh pinned AVX2 comparison used separate binaries, logical CPU 2, the
TSC timer, 100 samples, and a 500-millisecond minimum time. Five alternating runs covered all 14
list benchmarks. Every median-of-run-medians improves:

- The range is 17.2% to 19.3% faster.
- `take_filter_list_small_uncached_random_mask_random_indices[256, 10]` improves from 5.909 to
  4.879 microseconds, or 17.4%.
- The matching 768 case improves from 6.169 to 5.109 microseconds, or 17.2%.
- The largest improvement is the small random 256 case, from 5.659 to 4.569 microseconds, or 19.3%.

This is native wall-time evidence that executing and reusing the normalized offsets is worthwhile
independently of the CodSpeed result. It does not measure the same implementation as the direct
typed fix on `ct/row-fn`.

The same five-run comparison with `-C target-cpu=native` improves every case by 16.0% to 19.6%.
The small uncached cases move from 6.159 to 4.979 microseconds and from 6.389 to 5.209
microseconds. The improvement therefore survives the host's AVX-512 code generation.

## Numeric helper ID

The focused numeric profile also found 6.820 microseconds of new inclusive cost in
`CachedId::deref`. The new `vortex.numeric_binary` ID initializes during the measured call.
Develop's ID lookup costs 0.702 microseconds total. The numeric RowFn revision costs 7.522
microseconds.

`NumericBinary` is an internal helper for the registered `Binary` function. Commit `df8fcbe1a` on
`ct/row-fn-api` reuses `Binary`'s ID. This removes the second interner initialization and gives
errors the public function's name. It does not change the arithmetic loop or the public API.

This is a first-execution cost, not a per-row cost. The [numeric ID check] validates it:

- `sub_i64_constant` improves from 675.849 to 670.968 microseconds.
- `CachedId::deref` drops from 5.327 to 0.376 microseconds total.
- `Id::new_static`, previously 3.723 microseconds total, disappears from the callgraph.
- CodSpeed still classifies the complete benchmark as no change against develop. The fixed 4.881
  microseconds is less than 1% of this operation.

The take/filter control remains improved by 33.93% against develop.

## Nullable tensor decode

The current report has two remaining nullable tensor regressions at width 256. The differential
profile for `inner_product::nullable[256]` records these component increases:

| Component | Develop | RowFn | Increase |
| --- | ---: | ---: | ---: |
| Instructions | 13.146 us | 14.378 us | 1.232 us |
| Cache | 62.165 us | 71.844 us | 9.679 us |
| Memory | 158.567 us | 186.622 us | 28.056 us |
| Total | 233.878 us | 272.845 us | 38.966 us |

The floating-point row work is approximately unchanged. Before the fix, `TensorRow::decode` costs
33.553 microseconds total. It spends 25.638 microseconds canonicalizing the masked extension. The
`ArrayRef::mask` node in this profile is Batch's expected output mask, not input decode.

Dense RowFn execution owns input validity and restores it on the output. `TensorRow::decode` now
reads a `Masked` tensor's child values directly. The [masked tensor check] validates the change:

- `inner_product::nullable[256]` improves from 270.674 to 247.710 microseconds. It changes from a
  14.77% regression to a 6.87% no-change result against develop.
- `l2_norm::nullable[256]` improves from 271.115 to 249.766 microseconds. It changes from a 12.46%
  regression to a 4.98% no-change result against develop.
- `TensorRow::decode` drops from 33.553 to 6.801 microseconds total.
- Extension canonicalization under that decoder drops from 25.638 to 0.439 microseconds total.

The post-fix inner-product callgraph totals are 12.537 microseconds for instructions, 64.096
microseconds for cache, and 172.833 microseconds for memory. Its total is 249.466 microseconds.
The remaining difference from develop is memory cost, not extra executed instructions.

The local `f64` inner-product loop is scalar-unrolled by four. It emits `mulsd` and `addsd` in the
source fold order, not packed floating-point SIMD. Reassociating this reduction could enable wider
SIMD, but it would change floating-point results. It is not a free RowFn code-generation change.

## Remaining `mul_u8_nonnull` regression

The [numeric ID check] still reports `mul_u8_nonnull` as 12.74% slower than develop. Its callgraph
components increase by 1.149 microseconds for instructions, 6.147 microseconds for cache, and
18.411 microseconds for memory. The indexed loop's self cost is 69.973 microseconds on both sides.

The RowFn run enters `mi_page_fresh_alloc`, which is absent on develop. Inclusive `__rust_alloc`
cost increases from 7.221 to 22.449 microseconds. The evidence points to allocator state or
benchmark-order sensitivity around the output allocation. It does not show a slower arithmetic
loop. Do not change the loop or add layout padding without an isolated allocator experiment.

## Recommended next steps

1. Use pinned, alternating local x86 runs for performance decisions and retain the raw per-run
   medians.
2. Reduce the remaining `mul_u16_nonnull` native gap without relying on incidental padding.
3. Isolate allocator state before changing the `mul_u8_nonnull` loop.
4. Keep local wall time separate from CodSpeed CPU simulation.

## Mixed-constant optimization

Keep `4c936447a`. It fixes a real RowFn regression.

For two varying inputs, `Args::varying` returns typed slices and selects the indexed lane source.
For an array plus a constant, one argument returns `None`, so the tuple returns `None`. Here,
`None` means "not every input varies," not "no input varies." The mixed loop reads the array at
`index` and the one-row constant at zero.

The measured compiler requires the varying match and its length proof to remain inside the selected
owned-executor branch. Moving the proof through one shared `Option` helper made constant add and
subtract about 3.3 times slower. The branch-local form restored them. The semantic reason for the
source-placement sensitivity remains unknown.

[CodSpeed check at `892717f30`]: https://github.com/vortex-data/vortex/runs/93169735961
[CodSpeed check at `4c936447a`]: https://github.com/vortex-data/vortex/runs/93181527671
[function alignment]: https://codspeed.io/docs/instruments/cpu/regression-causes#function-alignment
[#9298]: https://github.com/vortex-data/vortex/pull/9298
[Focused framework check]: https://github.com/vortex-data/vortex/actions/runs/31316492455
[Focused numeric check]: https://github.com/vortex-data/vortex/actions/runs/31316710479
[offsets fix check]: https://github.com/vortex-data/vortex/actions/runs/31317322594
[numeric ID check]: https://github.com/vortex-data/vortex/actions/runs/31318131466
[masked tensor check]: https://github.com/vortex-data/vortex/actions/runs/31318825883
[#9299]: https://github.com/vortex-data/vortex/pull/9299
[run `31289620637`]: https://github.com/vortex-data/vortex/actions/runs/31289620637
[run `31289622392`]: https://github.com/vortex-data/vortex/actions/runs/31289622392
