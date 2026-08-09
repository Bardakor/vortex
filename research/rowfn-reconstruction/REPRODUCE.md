<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# RowFn reproduction guide

This guide gives a new contributor enough information to rebuild RowFn and repeat its main
performance experiments. Read [`DESIGN.md`](DESIGN.md) before implementing the API. Read
[`OPTIMIZATION.md`](OPTIMIZATION.md) before changing a hot loop.

## Recorded environment

The final x86 measurements used this environment:

- Candidate: `ct/row-fn` at `4c936447a`.
- Baseline: develop at `66d096b5d`.
- Rust: `rustc 1.91.0 (f8297e351 2025-10-28)`.
- LLVM: 21.1.2, as reported by `rustc -vV`.
- Host: AMD Ryzen 9 7950X, 16 cores and 32 hardware threads.
- Local benchmark CPU: hardware thread 4, selected with `taskset -c 4`.
- CodSpeed-compatible target feature: `RUSTFLAGS='-C target-feature=+avx2'`.
- Default bench profile: 16 codegen units and no LTO.

Record the exact revisions, compiler, CPU, governor, and flags for every new run. A percentage
without this context is not reproducible.

## Build order

Implement the framework in this order. Each step has a correctness or performance control before
the next step adds another capability.

### 1. Define decoded element types

Create an `InputElement` trait with these associated types:

- `Array`: the supported decoded array representation.
- `Value`: the value presented to a row closure.
- `Constant`: metadata extracted once for a batch constant.

The trait decodes one array before execution and reads one logical row from that decoded form. It
also declares whether a dense loop is safe for values stored behind nulls.

Start with primitive and Boolean elements. Do not add a hidden `scalar_at` call as a general
fallback. Such a call performs runtime dispatch in the hot loop.

### 2. Compose elements into tuples

Create an `ElementTuple` implementation for the arities that RowFn supports. Its decoded form must
distinguish two input shapes:

```text
Varying(buffer with row_count values)
Constant(buffer with one value)
```

The tuple must provide:

- Decoding for every input.
- Row lookup for mixed constant and varying inputs.
- Constant metadata for preparation.
- A validity mask for planning.

Keep the one-value constant representation. Do not expand constants to `row_count` values.

### 3. Add a typed all-varying source

Add an indexed source capability for tuples whose values can be represented by contiguous typed
slices. For a primitive pair, its varying source is equivalent to:

```rust
LaneZip<&[Left], &[Right]>
```

Validate every input length before the loop. The loop can then use unchecked indexed reads. The
single validation is both the safety proof and the condition that lets LLVM remove bounds checks.

Keep this capability separate from the general tuple trait. Stable Rust cannot express a blanket
fallback plus a more specific primitive implementation without specialization.

### 4. Define output capabilities

Support two output models:

1. An owned row value returned by the closure.
2. An output sink that lends a row handle to the closure.

The owned executor allocates final storage and writes each returned value. It requires a
compile-time proof that abandoned initialized spare capacity does not contain a type with a
destructor. Use the existing no-drop assertion. Do not expose an unnecessary `Output: Copy`
bound.

The uninitialized sink must make initialization a safe API invariant. Its row handle owns a
write-once token. Writing a value consumes the handle and returns a proof token. A successful
closure result must contain that token. This prevents safe code from reporting success without
initializing the output slot.

### 5. Separate failure evidence from errors

Represent common per-row failures with a small OR-reducible type. The loop returns failure
evidence, not a formatted `VortexError`. Convert the final evidence into an error outside the hot
loop with a cold, non-inlined helper.

Keep immediate failure for operations such as integer division when that form measures better.
Do not assume that deferred failure always vectorizes or always wins.

### 6. Add the visitor API

Define visit methods for these independent capabilities:

| Input preparation | Output | Failure |
| --- | --- | --- |
| None | Owned | None or deferred |
| None | Sink | None or immediate |
| Prepared constants | Owned | None or deferred |
| Prepared constants | Sink | None or immediate |

The RowFn implementation declares one typed row operation. The execution visitor selects the loop
and null policy. A planning visitor obtains dtype and fallibility information without running the
row closure.

### 7. Add batch planning and execution

Planning records the output dtype, validity behavior, fallibility, and optional encoded rewrite.
Execution then:

1. Decodes input arrays.
2. Computes conjoined validity.
3. Selects dense, dense-with-retry, valid-only, or filter-and-scatter execution.
4. Extracts constants and prepares batch state, when requested.
5. Runs the selected typed loop.
6. Builds the final array and validity.

The closure used by a dense policy must be total for every stored lane value, including values
behind null rows. It must not panic or perform side effects for those values.

### 8. Port primitive numeric functions first

Primitive binary arithmetic gives the smallest useful performance matrix. Port wrapping,
checked, saturating, and division operations. Keep the previous implementation available as a
benchmark control until every shape is measured.

Test at least these shapes:

- Varying plus varying.
- Varying plus constant.
- Constant plus varying.
- Dense validity.
- Mixed validity.
- Checked success.
- Checked failure behind a null row.
- Checked visible failure.

### 9. Add tensor and spatial row types

Decode tensors into typed flat buffers with width and stride. Use stride zero for a constant
tensor. Do not repeat a ptype check or buffer downcast for every output row.

Prepared tensor visits can compute a constant norm once. Prepared spatial visits can compute a
constant bounding box or relation helper once. These users prove that preparation is more than an
API placeholder.

## Historical implementation map

The branch history records useful intermediate designs. Recreate the final design from the steps
above, but use these commits to repeat an ablation or inspect why a design was rejected:

| Commit | Purpose |
| --- | --- |
| `fef191df5` | Original RowFn framework |
| `ae099e890` | Initial executor and null-policy benchmarks |
| `b324f3e26` | First numeric RowFn port |
| `aebe3caf7` | First tensor port |
| `6c13e8516` | First spatial port |
| `0a0ad0db1` | Cleaned RowFn framework based on current develop |
| `89fd28bc1` | Owned primitive numeric execution |
| `59c4578ef` | Focused executor benchmarks |
| `5c02036a2` | Refined execution contracts and initial shared length check |
| `a236e0b9d` | Self-contained kernel arguments |
| `f4617a2b5` | Merge of the research and cleaned histories |
| `69607edb6` | Pre-loop bounds proofs for owned execution |
| `892717f30` | Typed tensor and spatial row access |
| `4c936447a` | Branch-local varying proof for mixed constants |

The two histories before `f4617a2b5` are intentional. One preserves the original experiments. The
other preserves the cleaned implementation that was based on the latest develop revision.

## Benchmark procedure

### Choose the measurement before testing

CodSpeed CPU simulation and local wall time answer different questions. Do not use one as a proxy
for the other.

- Use the exact CodSpeed simulation workflow to reproduce a CodSpeed regression. Compare the
  simulated instructions, cache costs, memory costs, and differential flame graph.
- Use a pinned local wall-time run to check native performance on that host.
- Treat agreement between the two as additional evidence. Do not require it.

The repository workflow builds with AVX2 and runs `cargo codspeed run` in simulation mode. A
normal `cargo bench` invocation uses the wall-time compatibility runner and does not reproduce the
simulated metric.

### Use isolated worktrees and target directories

Build the baseline and candidate in separate worktrees. Give each build its own target directory.
This prevents one revision from reusing incompatible artifacts from another revision.

```bash
git worktree add --detach /tmp/vortex-rowfn-base 66d096b5d
git worktree add --detach /tmp/vortex-rowfn-candidate 4c936447a

RUSTFLAGS='-C target-feature=+avx2' \
  CARGO_TARGET_DIR=/tmp/rowfn-target-base \
  cargo bench -j 8 -p vortex-array --bench row_fn_executor --no-run

RUSTFLAGS='-C target-feature=+avx2' \
  CARGO_TARGET_DIR=/tmp/rowfn-target-candidate \
  cargo bench -j 8 -p vortex-array --bench row_fn_executor --no-run
```

Build independent experiments in parallel. Run their benchmark binaries serially on the same
hardware thread. Parallel benchmark runs compete for caches and memory bandwidth.

### Match the native host

Use the host CPU when native wall time is the acceptance signal:

```bash
RUSTFLAGS='-C target-cpu=native' \
  CARGO_TARGET_DIR=/tmp/rowfn-native-base \
  cargo bench -j 8 -p vortex-array --bench binary_ops --no-run
```

Build the candidate into a different target directory with the same flags. Copy or retain both
executables, pin them to the same logical CPU, and alternate their run order. Record the compiler,
CPU model, flags, timer, sample count, minimum time, and every run median.

This build answers how the code runs on that host. It does not match CodSpeed's AVX2 compilation.
For example, `target-cpu=native` enables AVX-512 on the Ryzen 9 7950X and reduces the measured
`mul_u16_nonnull` RowFn gap from 26.0% to about 9.7%.

### Match CodSpeed compilation

The repository bench profile uses the CodSpeed-relevant defaults:

```text
codegen-units = 16
lto = false
```

Set AVX2 explicitly for the local comparison:

```bash
RUSTFLAGS='-C target-feature=+avx2' cargo bench -p vortex-array --bench take_filter --no-run
```

Test one codegen unit as a compiler ablation:

```bash
RUSTFLAGS='-C target-feature=+avx2' \
  CARGO_PROFILE_BENCH_CODEGEN_UNITS=1 \
  cargo bench -p vortex-array --bench take_filter --no-run
```

The one-unit test does not emulate CodSpeed. It is only a compiler ablation.

### Run CodSpeed simulation

The CI workflow is the authoritative reproduction:

```bash
RUSTFLAGS='-C target-feature=+avx2' \
  cargo codspeed build --features _test-harness -p vortex-array --profile bench
cargo codspeed run -m simulation
```

Local simulation requires `cargo-codspeed` and CodSpeed's Valgrind fork. A standard Valgrind
installation is not equivalent. If those tools are unavailable, push the exact revision to a
branch with an open pull request. That push gives CodSpeed the comparison context it needs.

A plain `workflow_dispatch` run does not update a pull request's CodSpeed report. Do not use its
partial output as comparison evidence. Do not substitute a native timing run and label it CodSpeed.

Use the CodSpeed benchmark page to compare the candidate with the same develop baseline. Inspect
the differential flame graph and record these values for the changed stack:

- Simulated time.
- Executed instruction cost.
- Cache cost.
- Memory cost.
- Function self time and total time.

### Pin a native benchmark process

Find the generated executable under `target/release/deps`, then run it on one hardware thread:

```bash
taskset -c 4 target/release/deps/row_fn_executor-<hash> \
  --bench --sample-count 100 --max-time 1 --color never
```

Run candidate and baseline in alternating order. Repeat a surprising result. Report medians and
the full range across repetitions. Label these results as native wall time.

### Core benchmark set

Use these commands to cover the framework and its migrated users:

```bash
cargo bench -p vortex-array --bench row_fn_executor
cargo bench -p vortex-array --bench binary_ops
cargo bench -p vortex-array --bench take_filter
cargo bench -p vortex-array --bench compact
cargo bench -p vortex-tensor --bench cosine_similarity
cargo bench -p vortex-tensor --bench inner_product
cargo bench -p vortex-tensor --bench l2_norm
cargo bench -p vortex-spatial
```

Use benchmark name filters to keep each comparison focused. Record the exact filter with the
result.

## Source ablation procedure

When a small source edit causes a large result, do not infer a cause from the final diff. Use this
procedure:

1. Keep compiler flags, target CPU, benchmark input, and toolchain fixed.
2. Change one source property.
3. Build into a new target directory.
4. Run the baseline and candidate serially on one CPU.
5. Inspect LLVM IR and final assembly for the production monomorph.
6. Revert the source property and confirm that the result returns.

For the mixed-constant regression, the single property was the location of the varying-source
match and its length proof. Controls showed that all-varying execution did not move.

Do not preserve a source edit only because an unrelated benchmark report improves. First prove
that the benchmark executes the changed path or that its machine-code change is stable and
understood.

## Inspect generated code

Build a focused crate with one codegen unit when you need readable LLVM IR or assembly:

```bash
RUSTFLAGS='-C target-feature=+avx2' \
  CARGO_PROFILE_BENCH_CODEGEN_UNITS=1 \
  cargo rustc -p vortex-array --release --lib -- --emit=llvm-ir,asm
```

Search the emitted files for a concrete operation and type. Check these properties:

- Array and dtype dispatch are outside the loop.
- The loop has no per-row bounds failure edge.
- The row closure is inlined.
- Failure evidence stays as a small value.
- Rich error construction is outside the loop.
- Vector instructions exist before claiming SIMD.

For a linked benchmark binary, compare symbol sizes and disassembly:

```bash
llvm-nm --demangle --print-size --size-sort target/release/deps/<benchmark> > symbols.txt
llvm-objdump --demangle --disassemble-symbols='<symbol>' \
  target/release/deps/<benchmark> > symbol.asm
```

Normalize absolute addresses and relocation offsets before comparing instructions. Identical
instructions at different addresses still permit a layout-sensitive cache or branch result.

## Correctness checks

Run the narrow checks while iterating:

```bash
cargo nextest run -p vortex-array
cargo test --doc -p vortex-array
cargo check -p vortex-array --benches
```

Run repository Rust checks before handing off code changes:

```bash
cargo +nightly fmt --all
cargo clippy --all-targets --all-features
```

If cargo reports exactly `sccache: error: Operation not permitted`, rerun that command with
`RUSTC_WRAPPER=`.

## Known limitations of the record

- The host used a power-saving governor during some local runs. CPU pinning and repeated controls
  reduce noise, but they do not replace a fixed-frequency benchmark host.
- `perf`, Samply, and local CodSpeed simulation were not available for the final take/filter
  investigation.
- The current take/filter evidence identifies a linked-binary effect. It does not identify the
  exact cache set, branch target, or called symbol that causes the wall-time gap.
- The exact cause of the public `Copy`-bound compiler regression remains unknown.
- Several early null-strategy and bytes-length benchmarks were research scaffolding and are not
  part of the final API.
