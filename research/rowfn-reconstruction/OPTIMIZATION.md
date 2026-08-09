<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# RowFn optimization guide

## Performance model

RowFn is fast when the hot loop contains only work that changes for each row. These operations can
stay outside the loop:

- Array and dtype dispatch.
- Decoding and downcasts.
- Batch-constant detection.
- Input length validation.
- Output allocation and array construction.
- Validity policy.
- Rich error construction.
- Work derived only from constant operands.

The design also gives LLVM concrete types and independent indexed lanes. A short row closure is not
enough by itself. The generic plumbing must disappear after monomorphization.

## Optimization history

### Stage 0: sink-only output

The first shared executor required every row function to write through a sink. This model supported
runtime-shaped output, but it hid the independence of primitive output values.

The checked primitive loop was slower for several wide integer types. Signed `i64` multiply was
about 29% slower than the baseline. Unsigned `u64` multiply was about 59% slower.

### Stage 1: owned output

The next design let the row closure return `(Output, Failure)`. Shared execution owned the final
store and reduced failure evidence.

This change improved wide integer multiplication, but it did not give LLVM a simple input source.
For example, `i32` multiplication remained about 18% slower in the measured matrix.

This stage proved that output ownership mattered. It also proved that output ownership alone was
not sufficient.

### Stage 2: typed indexed input

`IndexedElementTuple` added an all-varying source. A primitive pair becomes
`LaneZip<&[Left], &[Right]>`. Shared execution validates both lengths once and calls
`map_checked_into`.

This stage restored varying and nullable multiplication to approximately baseline performance. It
also removed hot bounds checks from the inspected production monomorphs.

The trait is separate from `ElementTuple`. Many element types do not have a contiguous source.
Stable Rust cannot combine a blanket fallback with a more specific primitive implementation
without specialization.

### Stage 3: remove the `Output: Copy` bound

The executor needs only one property from owned output: abandoning initialized spare capacity on
unwind must not leak a required destructor. `Output: Copy` was stronger than this property.

On Rust 1.91.0 and LLVM 21.1.2, adding the public `Copy` bound changed the production `i32` checked
multiply monomorph from about 18.7 microseconds to about 29.9 microseconds. An inert marker bound
did not cause the loss. One codegen unit did not remove it.

The selected design uses a compile-time `!needs_drop::<Output>()` assertion. It does not expose a
`Copy` bound that the executor does not need.

The exact compiler mechanism remains unknown. Standalone reduced loops did not reproduce the
effect. The real trait, closure, vector, and monomorphization context was necessary.

### Stage 4: preserve mixed-constant code placement

Commit `5c02036a2` deduplicated length validation:

```rust
let varying = Args::varying(&columns);
ensure_decoded_lengths(&columns, varying.as_ref(), row_count)?;

if let Some(varying) = varying {
    // All-varying loop.
} else {
    // Mixed loop.
}
```

This source-only change made constant add and subtract about 3.3 times slower at that revision. It
did not change the all-varying cases.

The selected form keeps the view and proof in the selected branch:

```rust
if let Some(varying) = Args::varying(&columns) {
    validate_varying_lengths(&varying, row_count)?;
    // All-varying loop.
} else {
    validate_mixed_lengths(&columns, row_count)?;
    // Mixed loop.
}
```

This change restored constant add and subtract to about 9.2 microseconds. Constant `i32` multiply
returned to about 18.9 microseconds. The all-varying controls did not move.

The source placement is a measured constraint for the current toolchain. Rust semantics do not
require it. The source ablation proves the performance relationship, but it does not identify the
LLVM pass that causes it.

Pinned local x86 measurements on an AMD Ryzen 9 7950X confirm that this is not only a CodSpeed
effect. Before the fix, constant `i64` add and subtract were 3.23 and 3.35 times slower than
develop. After the fix, they are about 11% slower. Constant `i32` multiply changes from 58.6%
slower than develop to 28.5% faster.

The sink executors retain the shared validator. Moving their proof into each branch did not improve
the cosine or spatial benchmarks.

### Stage 5: typed tensor rows

The old tensor row accessor repeated a ptype check and buffer downcast for every output row. The
new `TensorRows<T>` representation performs these operations once during decode.

Each row access uses a typed flat buffer, width, and stride. A constant-backed tensor uses stride
zero, so `index * stride` selects row zero without a branch.

This representation makes the tensor inner loop ordinary slice arithmetic. It also keeps constant
input storage compact.

### Stage 6: prepared tensor and spatial constants

Prepared visits expose batch constants before the loop. Cosine similarity computes a constant norm
once. Spatial predicates compute constant bounding boxes and relation helpers once.

This optimization does not require a new array kernel. The same row declaration handles both
constant and varying operands.

## Source-placement constraints

### Decode before the loop

The `InputElement::decode` method must contain dtype checks, array execution, downcasts, and buffer
extraction. Calling these operations through `get` makes the loop pay batch work for every row.

### Prepare before the loop

`Args::constants` and the prepare closure run once after decode. The prepared value is borrowed by
the row closure. It must not be rebuilt for each row.

### Validate lengths before the loop

Unchecked input reads are sound only after each varying source proves that it contains
`row_count` rows. The output slice must also contain `row_count` slots.

The validations must execute before the loop. A check in the loop keeps bounds control flow in the
hot path and can prevent bounds-check elimination.

### Keep the owned varying proof in its branch

The owned executor must not pass `Option<&VaryingColumns>` through the shared generic helper on the
measured toolchain. The option construction, proof, and consumer stay in one branch.

This rule is intentionally narrow. Applying it to every executor adds duplication without measured
benefit.

### Borrow sink rows once

`sink.rows()` runs before the loop. The loop receives a stable row view instead of repeatedly
borrowing the sink object. This keeps the buffer descriptor and output shape invariant.

### Keep rich errors cold

The row closure computes a small failure word. A `#[cold]` and `#[inline(never)]` helper creates the
`VortexError` after the loop or on the immediate failure path.

This arrangement prevents formatting, allocation, and error branches from entering successful
checked-arithmetic loops.

### Use inlining evidence, not a blanket attribute

The public wrappers use ordinary `#[inline]` only where a caller must see captured constants or a
small adapter. The implementation does not apply `#[inline(always)]` to checked arithmetic.

The lane-kernel module contains small internal chunk helpers with stronger attributes. Those
helpers were measured as part of the pre-existing lane-kernel work. A new strong inlining attribute
requires separate assembly or benchmark evidence.

## Why the loop can autovectorize

The optimized all-varying primitive loop presents these facts to LLVM:

1. The element types are concrete because `dispatch` selected `T` before execution.
2. The input sources are typed slices or a typed `LaneZip`.
3. Input and output lengths match.
4. Unchecked reads follow one pre-loop proof.
5. Each iteration reads and writes an independent row.
6. Failure combines with bitwise OR.
7. The closure is concrete and can inline into the loop.
8. Error construction and validity are outside the loop.

The generated loop can use SIMD when LLVM has a legal and profitable lowering. Checked add and
small-width arithmetic often fit this model.

The word _autovectorize_ must not describe every result. The inspected `i64` and `u64` widened
multiply loops remained scalar on x86. They recovered performance because RowFn matched the
handwritten scalar loop, not because LLVM found SIMD.

The tensor outer loop returns one scalar for each tensor row. SIMD commonly appears in the inner
loop over each tensor slice. The outer RowFn loop does not need to vectorize across variable slice
references.

## Rejected or incomplete alternatives

### Keep every output behind a sink

This model supports more output shapes, but it loses the independent owned-value contract that
primitive code generation needs.

### Add a numeric `reduce_encoded` fast path

This path recovered speed by duplicating shared null and constant policy inside the numeric
function. It made RowFn a slow fallback instead of making shared execution fast.

### Add a numeric-specific visitor seam

This design moved the same specialization into generic execution under a different name. It did
not establish a reusable capability for nonnumeric row functions.

### Use safe zipped iterators

The tested iterator forms caused 3x to 9x losses for narrow integer types. They did not preserve the
same indexed source shape across all monomorphs.

### Depend on per-row bounds checks

Unchecked access improved some cases, but it did not solve the original output and source-shape
problems. It also regressed some `u8` cases when applied without the final indexed design.

### Scan output for failures

The selected loop returns failure evidence directly. Scanning a finished output adds another pass
and cannot represent every error condition.

### Use `Copy` as the no-drop proof

`Copy` is stronger than required and triggered a measured compiler regression. The compile-time
no-drop assertion expresses the actual safety condition.

### Apply branch-local validation to sinks

This change did not improve cosine or spatial performance. The shared helper remains in those
paths.

### Outline the all-varying kernel

Moving the validated all-varying lane kernel into a private `#[inline(never)]` helper does not
improve `mul_u16_nonnull`. Ten pinned runs remain between 2.799 and 2.829 microseconds, the same as
the ordinary cleaned API binary. The larger Rust function containing both argument shapes is not
by itself the residual cause.

## Unrelated benchmark movement

An unrelated benchmark can move after a RowFn source edit even when it never calls RowFn. The
source edit rebuilds `vortex-array` and the benchmark executable. This rebuild can change:

- Codegen-unit partitioning.
- Inlining decisions in affected monomorphs.
- Function order and address alignment.
- Instruction-cache and decoded-instruction-cache set placement.
- Branch target placement.
- Linker layout of code that remains reachable through the shared session.

These are code-generation dependencies, not semantic dependencies.

[CodSpeed CPU simulation] measures executed instructions and models cache and memory access. It
can therefore report a different result when the instruction sequence or binary layout changes.
Local wall time can differ from the simulated ratio because it uses a real AMD processor instead
of the CodSpeed CPU model.

CodSpeed documents [function alignment] as one reason an unchanged microbenchmark can move after
a rebuild. The correct diagnostic is the simulated instruction and cache counts in the
differential flame graph.

An unrelated recovery does not prove that an algorithmic problem was fixed. The result is stable
only after source ablation, machine-code inspection, and repeated measurements agree on a cause.

### Native benchmark policy

Pinned local x86 wall time is the primary performance acceptance signal for the remaining RowFn
work. Run separate copied binaries on the same logical CPU, alternate revision order, and report
the median of repeated run medians. Use enough minimum time to make a narrow result stable.

CodSpeed simulation remains a diagnostic tool. Its instruction, cache, and memory components can
expose a changed stack that local wall time cannot explain. A CodSpeed-only movement does not
override native parity or improvement, and local wall time must not be presented as a prediction
of CodSpeed simulation.

### Identical vector loops can retain a native gap

`mul_u16_nonnull` is a useful counterexample to treating autovectorization as the end of the
investigation. Ten alternating one-second runs measure 2.229 microseconds on develop and 2.809
microseconds on the cleaned API branch, a 26.0% native regression.

Both hot loops contain the same normalized vector instructions. They load two 128-bit vectors,
execute `pmullw` and `pmulhuw`, combine the overflow evidence, store one vector, and branch. The
develop loop fits in one 64-byte cache line. The ordinary RowFn loop crosses a line boundary.

A diagnostic build with `-C llvm-args=-align-loops=64` measures 2.449 microseconds. The option did
not align this loop to 64 bytes, but the resulting linked layout moved it wholly inside one cache
line. This recovers 62% of the gap while leaving a 9.9% difference from develop.

This experiment supports front-end and code-placement sensitivity. It does not prove that line
crossing explains the complete regression. A hidden global LLVM option and source padding are not
stable remedies. The RowFn monomorph also contains all-varying and mixed shape branches in one
larger function, so entry and setup code remain candidates for the residual cost.

With `-C target-cpu=native` on the Ryzen 9 7950X, develop measures 2.259 to 2.269 microseconds and
RowFn measures 2.479 to 2.489 microseconds. Native CPU targeting reduces the gap from 26.0% to
about 9.7%.

Both native loops use AVX-512. Develop processes two ZMM vectors, or 64 `u16` lanes, per iteration.
RowFn processes four ZMM vectors, or 128 lanes. Both use packed low- and high-half multiply,
failure reduction, and packed stores. Autovectorization is intact; LLVM chose a different unroll
factor and the shared RowFn path retains additional batch setup.

The complete native matrix shows the same distinction. Decimal arithmetic, integer division, and
most nullable wide cases are within 1%. Narrow varying integer cases are generally 4% to 12%
slower. Mixed-constant add, subtract, and multiply remain 19% to 23% slower even though their hot
loops use AVX-512 broadcasts and packed arithmetic.

`mul_u8_nonnull` retains a stable 10.8% gap when run alone: 1.939 microseconds on develop and 2.149
microseconds with RowFn across ten alternating runs. Both hot loops process 64 lanes with the same
normalized AVX-512 instructions. Develop's loop target is 64-byte aligned, while RowFn's is seven
bytes into a line. The global `-align-loops=64` diagnostic did not align this loop and did not
change the timing. This rules out local benchmark-order allocator state, but it does not establish
an alignment cause.

Changing numeric dispatch from a two-element `Vec<ArrayRef>` to a stack-backed borrowed argument
view does not improve repeated timings. Removing that allocation is not a measured remedy.

Skipping each encoding's validity function when its dtype is non-nullable also moves focused
native cases by less than 1%. Nullable controls move by a similar amount without a call-path
change. This is linked-layout noise, not evidence for a second batch-planning path.

A benchmark-only 1,048,576-row ablation reduces the remaining differences to within 1% for
`mul_u16_nonnull`, `add_i32_nonnull`, and constant `i64` add and subtract. Constant `i32` multiply
is 2.0% faster than develop, and varying `i64` multiply is 3.7% faster.

The large-batch result shows that RowFn preserves native per-element throughput. The percentages
in the 32,768-row microbenchmarks primarily measure fixed batch planning, dispatch, decode, and
output reconciliation. Optimize those costs as batch overhead; do not rewrite the vector loops.

The `row_fn_executor` control isolates the framework in one linked binary. Across five
`target-cpu=native` runs at 65,536 rows, the hand-written sink median is 137.4 microseconds.
Infallible owned RowFn execution is 138.8 microseconds, and sink RowFn execution is 138.5
microseconds. Checked owned execution is 141.9 microseconds. The infallible executor variants are
within 1% of the hand-written loop, while deferred overflow reduction retains about 3.3% overhead.

## `take_filter_list` regression

The [CodSpeed check at `4c936447a`] reports 31 regressions. Several `take_filter_list_*`
benchmarks are 14% to 16% slower than develop in CPU simulation.

The [CodSpeed check at `892717f30`] already reported the same benchmarks as 15% to 16% slower. The
final mixed-constant fix did not bring them back. Most of their simulated times improved by less
than 2% between the two checks. The fix removed larger constant-arithmetic regressions, so the
unchanged take/filter entries became more prominent in the ordered report.

Every retained RowFn CodSpeed summary from `0e5c19c00` through `4c936447a` that contains a
performance table also contains `take_filter_list_*` regressions. Some GitHub views show only the
20 largest changes, and the bot edits one current PR comment. Either behavior can make a persistent
regression appear to leave and return.

The compared list, filter, and take source files are identical between develop and the branch.
However, the benchmark reaches RowFn through code outside those files:

```text
take_filter
  -> list_view_from_list
    -> ListArrayExt::reset_offsets
      -> binary(Sub) on offsets and the first offset
        -> numeric RowFn
```

The old implementation of `reset_offsets` used generic binary subtraction. It created a constant
array from the first offset. The numeric RowFn migration changed that generic call's implementation.

### Differential simulation evidence

For `take_filter_list_small_uncached_random_mask_random_indices[256, 10]`, the current PR report
measures 233.737 microseconds on develop and 280.793 microseconds on `bdf95a77e`. This is a 16.76%
regression.

CodSpeed creates the downloadable callgraph during a separate profiling execution. Its absolute
total can differ slightly from the report aggregate. The component totals are:

| Revision | Instructions | Cache | Memory | Total |
| --- | ---: | ---: | ---: | ---: |
| Develop `66d096b5d` | 21.312 us | 83.443 us | 133.294 us | 238.050 us |
| RowFn `bdf95a77e` | 26.210 us | 104.293 us | 155.172 us | 285.675 us |
| Increase | 4.898 us | 20.850 us | 21.878 us | 47.626 us |

The profile contains extra executed instructions and a new call path. It does not support a
cache-only or alignment-only explanation.

The focused numeric profile shows these self and inclusive function costs:

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

These inclusive costs overlap when functions call each other. They identify the changed stack.

### First bad revision

Temporary draft PR [#9298] runs only `cargo codspeed run --bench take_filter` in a pull-request
context.

- The [focused framework check] at `0a0ad0db1` measures 232.542 microseconds. Develop measures
  233.737 microseconds, so CodSpeed classifies the 0.51% improvement as no change.
- The [focused numeric check] at `89fd28bc1` measures 279.491 microseconds. This is 16.37% slower
  than develop.

The two revisions are parent and child. Therefore, `89fd28bc1` is the first bad revision.

The numeric revision's callgraph totals are 25.835 microseconds for instructions, 103.531
microseconds for cache, and 154.728 microseconds for memory. Its total is 284.093 microseconds.

### Focused remedy

`ListArrayExt::reset_offsets` now decodes its offsets once. A typed loop subtracts the first offset
and builds the replacement primitive array. This removes the constant allocation, batch planning,
dispatch, argument decoding, and output reconciliation from this small internal operation.

The AVX2 release binary auto-vectorizes every integer width. Each unrolled iteration contains two
128-bit packed subtracts. `psubb` handles 32 offsets, `psubw` handles 16, `psubd` handles 8, and
`psubq` handles 4. Signed and unsigned monomorphs share their machine code.

This fix targets the measured changed call path. It does not add padding or unrelated structural
changes.

The [offsets fix check] validates the result in CodSpeed CPU simulation. The representative case
measures 176.524 microseconds, compared with 233.737 microseconds on develop and 280.793
microseconds before the fix. It changes from a 16.76% regression to a 32.41% improvement against
develop. All 14 `take_filter_list_*` cases improve by 25.61% to 35.54% against develop.

The representative post-fix callgraph totals are 15.462 microseconds for instructions, 59.031
microseconds for cache, and 103.767 microseconds for memory. Its total is 178.259 microseconds.
The generic scalar-function stack is absent. The typed `reset_offsets` path costs 0.933
microseconds self and 7.629 microseconds total. `list_view_from_list` drops from 79.144 to 29.634
microseconds total.

This result is larger than a recovery to develop because develop also uses generic scalar-function
subtraction for this internal offset adjustment. The direct typed operation removes that older
overhead as well as the additional RowFn work.

PR [#9299] first extracted the direct typed offset fix at `fa54891b`. Five alternating native AVX2
runs against its exact develop base improved all 14 list cases by 27.9% to 33.3%. That commit is no
longer the PR head, so those results describe only the superseded implementation.

The current PR head, `d97e53e66`, leaves the generic lazy subtraction in `reset_offsets`. It
materializes that result once in `list_view_from_list`, then reuses the primitive offsets for both
sizes and output offsets. Five fresh alternating runs improve all 14 cases by 17.2% to 19.3%. The
small uncached 256 case moves from 5.909 to 4.879 microseconds, and its 768 counterpart moves from
6.169 to 5.109 microseconds. This implementation is also a native win, but it is distinct from the
direct typed fix measured in CodSpeed and retained on `ct/row-fn`.

With `-C target-cpu=native`, five more alternating runs improve every case by 16.0% to 19.6%. The
small uncached cases move from 6.159 to 4.979 microseconds and from 6.389 to 5.209 microseconds.
The optimization therefore remains effective under this host's AVX-512 code generation.

### Avoid a second ID for an internal helper

The focused numeric profile shows another fixed cost. `CachedId::deref` increases from 0.702 to
7.522 microseconds inclusive. The new `vortex.numeric_binary` ID initializes inside the measured
call.

`NumericBinary` is not registered. It executes the registered `Binary` operation's primitive path.
Commit `df8fcbe1a` on `ct/row-fn-api` therefore reuses `Binary`'s existing ID. This removes a second
interner initialization and makes internal errors name the public function.

This change does not alter dispatch or the row loop. The cost occurs on first execution, so it is
separate from per-row vectorization. The [numeric ID check] validates the result:

- `sub_i64_constant` improves from 675.849 to 670.968 microseconds.
- `CachedId::deref` drops from 5.327 to 0.376 microseconds total.
- `Id::new_static`, previously 3.723 microseconds total, disappears from the callgraph.
- CodSpeed still classifies the complete benchmark as no change against develop. The fixed 4.881
  microseconds is less than 1% of this operation.

The take/filter control remains improved by 33.93% against develop.

### Decode masked tensor values directly

The report also shows 14.77% and 12.46% regressions for nullable width-256 inner product and L2
norm. For `inner_product::nullable[256]`, the callgraph components are:

| Component | Develop | RowFn | Increase |
| --- | ---: | ---: | ---: |
| Instructions | 13.146 us | 14.378 us | 1.232 us |
| Cache | 62.165 us | 71.844 us | 9.679 us |
| Memory | 158.567 us | 186.622 us | 28.056 us |
| Total | 233.878 us | 272.845 us | 38.966 us |

The floating-point row work is approximately unchanged. Before the fix, `TensorRow::decode` costs
33.553 microseconds total. It spends 25.638 microseconds canonicalizing the masked extension. The
`ArrayRef::mask` node in this profile is Batch's expected output mask, not input decode.

Dense RowFn execution owns input validity and restores it on the result. The tensor decoder now
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

The linked `f64` inner-product loop is scalar-unrolled by four. It uses `mulsd` and `addsd` in the
source fold order, not packed floating-point SIMD. LLVM cannot reassociate the strict reduction.
Changing that order could enable wider SIMD, but it would change floating-point results and needs
an explicit numerical contract.

### `mul_u8_nonnull` allocator path

The [numeric ID check] still reports `mul_u8_nonnull` as 12.74% slower than develop. Its callgraph
components increase by 1.149 microseconds for instructions, 6.147 microseconds for cache, and
18.411 microseconds for memory. The indexed loop's self cost is 69.973 microseconds on both sides.

The RowFn run enters `mi_page_fresh_alloc`, which is absent on develop. Inclusive `__rust_alloc`
cost increases from 7.221 to 22.449 microseconds. This points to allocator state or benchmark-order
sensitivity around the output allocation. It does not show a slower arithmetic loop. A focused
allocator-state experiment must precede any code or benchmark change.

AVX2 wall-time runs on CPU 4 provide a separate native-runtime observation:

| Revision | Typical list/filter median | Difference from develop |
| --- | ---: | ---: |
| Develop `66d096b5d` | 6.2 to 7.0 us | Baseline |
| Before latest push `892717f30` | 7.9 to 8.8 us | About 25% to 31% slower |
| Latest push `4c936447a` | 8.0 to 8.9 us | About 25% to 31% slower |

The latest push changes most local cases by only 0% to 2%. The branch already contains a native
wall-time gap before that push. This result does not explain the CodSpeed simulation result.

Changing the bench profile from 16 codegen units to one did not remove the native gap. For one
representative case, the candidate and develop medians were 7.86 and 6.21 microseconds. The same
case measured 8.25 and 6.41 microseconds with 16 codegen units.

The main filter-take and list-take function sizes are identical across the three AVX2 binaries.
Normalized disassembly of the list-take function has the same instructions. Relative addresses and
link layout differ. The earlier inspection did not include the numeric callee in `reset_offsets`.

Do not fix unrelated movement with arbitrary padding or an unrelated source edit. Such a change can
move a report without removing a measured cause.

### Reuse zero-based list offsets

The review follow-up adds the remaining fast path from the old `reset_offsets` TODO. When the
executed primitive offsets start at zero, `reset_offsets` now reuses that array. It does not copy
the complete offsets buffer to subtract zero.

The native control contains every other review edit and removes only the early return. Three
alternating runs used `-C target-cpu=native`, CPU 2, the TSC timer, 100 samples, and a 0.5-second
minimum per case. The early return improves all 14 `take_filter_list_*` cases by 1.66% to 3.29%.
The small uncached 256 case moves from 4.149 to 4.049 microseconds. The matching 768 case moves
from 4.379 to 4.299 microseconds.

This isolated result supports the code change, but it remains native wall-time evidence. It does
not provide CodSpeed instruction, cache, or memory counters.

### Select primitive comparison output by measured code generation

Primitive comparisons expose a second output trade-off. The owned RowFn path writes one `bool` per
row, then `OutputElement for bool` packs the values into a `BitBuffer`. The old columnar path fuses
the predicate and bit-packing loop.

The separate pack is cheap on the current x86 host. Packing 65,536 values takes 570 nanoseconds.
The comparison loop determines the larger differences:

- RowFn improves measured `u8`, `i32`, `f32`, equality, and varying `u64` cases by 10% to 92%.
- The fused path remains faster for ordered `i64`, ordered `f64`, and constant ordered `u64`.
- A direct RowFn port regresses those cases by 11% to 34%.

Dispatch each operator to a separate RowFn closure. This keeps the operator match outside the row
loop and gives LLVM one predicate per monomorph. Do not move the operator match into the closure.

On x86, select the fused path before RowFn planning for the measured wide ordered cases. A
`reduce_encoded` prototype recovered the loop but repeated planning and validity work. Nullable
`i64` remained 5.7% slower. Selecting at the primitive entry point restores parity.

Keep the fallback instantiation set narrow. Only `i64`, `u64`, and `f64` can reach it, so a full
`match_each_native_ptype!` adds unused columnar monomorphs. Explicit dispatch avoids that code-size
cost.

This pruning moved the local `u8` median from approximately 3.06 to 3.42 microseconds without
changing its selected source path. Treat this as layout sensitivity, not a loop regression, until
a normalized machine-code comparison shows otherwise.

The benchmark source, commands, and representative medians are in `HANDOFF.md`. These results use
local wall time, not CodSpeed CPU simulation.

The completed CodSpeed run for `9bed9c9` moved many benchmarks outside this comparison path. Its PR
report has 36 improvements and 45 regressions, including expression, FastLanes, compact, and file
benchmarks. It also fell back to `66d096b` rather than the newer develop head. Without the simulated
instruction, cache, and memory counters, this broad movement cannot distinguish changed work from
linked-layout costs. Do not use it to override the focused native A/B above.

## Current unresolved work

- Reduce the mixed-constant LLVM sensitivity while preserving the production monomorph.
- Reduce fixed RowFn batch overhead if 32,768-row numeric calls are latency-critical.
- Profile the isolated `mul_u8_nonnull` case with native performance counters.
- Choose between the direct typed offset fix and PR #9299's materialize-once design based on API
  maintenance and correctness. Both are native wins, but they are different implementations.
- Identify the spatial `envelope` regression that begins when numeric RowFn code enters the linked
  binary.
- Repeat the key local results on a second x86 machine and compiler version before filing a
  compiler issue.

[CodSpeed check at `4c936447a`]: https://github.com/vortex-data/vortex/runs/93181527671
[CodSpeed check at `892717f30`]: https://github.com/vortex-data/vortex/runs/93169735961
[CodSpeed CPU simulation]: https://codspeed.io/docs/instruments/cpu
[function alignment]: https://codspeed.io/docs/instruments/cpu/regression-causes#function-alignment
[#9298]: https://github.com/vortex-data/vortex/pull/9298
[focused framework check]: https://github.com/vortex-data/vortex/actions/runs/31316492455
[focused numeric check]: https://github.com/vortex-data/vortex/actions/runs/31316710479
[offsets fix check]: https://github.com/vortex-data/vortex/actions/runs/31317322594
[numeric ID check]: https://github.com/vortex-data/vortex/actions/runs/31318131466
[masked tensor check]: https://github.com/vortex-data/vortex/actions/runs/31318825883
[#9299]: https://github.com/vortex-data/vortex/pull/9299
