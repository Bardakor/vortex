<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# RowFn reconstruction guide

This guide explains the RowFn design without requiring access to its source. It records the type
model, execution model, performance constraints, implementation order, and benchmark procedure.
The goal is to let a new contributor reconstruct the branch and understand each unusual choice.

The guide describes the implementation through `443aed0b9` on `ct/row-fn`. Historical CodSpeed
comparisons use develop commit `66d096b5d`.

## Reading order

1. Read [`HANDOFF.md`](HANDOFF.md) for the current branch state, corrected CodSpeed history, and
   unfinished investigation.
2. Read [`DESIGN.md`](DESIGN.md) for the API, concrete input examples, null handling, failure
   handling, and generated loop shape.
3. Read [`OPTIMIZATION.md`](OPTIMIZATION.md) for the performance history, source-placement
   constraints, rejected designs, and current CodSpeed interpretation.
4. Read [`REPRODUCE.md`](REPRODUCE.md) to rebuild the implementation and repeat the experiments.

These dated records contain the raw evidence behind this guide:

- [`rowfn-x86-2026-08-07`](../rowfn-x86-2026-08-07/README.md) records the owned-output, indexed
  source, `Copy`-bound, LLVM IR, assembly, and x86 experiments.
- [`rowfn-regressions-2026-08-08`](../rowfn-regressions-2026-08-08/README.md) records the branch
  bisection, compiler-configuration matrix, and tensor, spatial, list, and compact benchmarks.
- [`NUMERIC_ROWFN_PLAN.md`](../../NUMERIC_ROWFN_PLAN.md) records the earlier Apple Silicon work and
  the original numeric design alternatives.

## Terms

The guide uses these terms consistently:

- A _batch_ is one invocation over zero or more equally sized arrays.
- A _row closure_ computes one logical result from one element of each input.
- A _varying input_ stores one decoded value for each logical row.
- A _batch constant_ stores one decoded value that every logical row reads.
- An _owned output_ returns one independent Rust value for each row.
- An _output sink_ gives the row closure a handle into batch-owned output state.
- A _dense loop_ visits all stored rows, including payloads behind nulls.
- A _valid-only loop_ visits only rows where every input is valid.
- _Failure evidence_ is a small value that the loop OR-reduces before it creates an error.
- A _semantic dependency_ means that the benchmark executes the changed code.
- A _code-generation dependency_ means that the rebuild changes machine code or layout without a
  runtime call to the changed code.

## Main conclusions

- RowFn removes array dispatch, dtype dispatch, decoding, allocation, validity, and rich errors
  from the hot row loop.
- Rust monomorphization gives the loop concrete input, output, closure, and failure types.
- Primitive all-varying inputs use a typed indexed source with one bounds proof before the loop.
- Mixed constant inputs use one branch per argument and row. Batch constants remain one-row
  buffers and are not expanded.
- Prepared visits expose constant values once before the loop. Tensor norms and spatial bounding
  boxes use this capability.
- Owned output and sink output are separate capabilities. One abstraction did not optimize both
  use cases well.
- Deferred failure evidence keeps rich error construction outside the loop. It also lets batch
  execution suppress failures that came only from null rows.
- Integer division uses immediate failure and an uninitialized sink. Division is expensive and
  scalar, so deferred evidence does not preserve useful vectorization there.
- The mixed-constant owned loop is sensitive to one source placement with Rust 1.91.0 and LLVM
  21.1.2. The varying view and its length proof must remain in the selected branch.
- The current CodSpeed report still contains unrelated regressions. A changed result in an
  unrelated benchmark is not evidence that RowFn changed its algorithm.

## What “autovectorization” means here

RowFn does not use explicit SIMD intrinsics. It presents LLVM with ordinary counted loops over
typed slices and independent output slots. This shape lets LLVM use SIMD when the operation and
target support it.

Not every important result uses SIMD. The measured signed and unsigned 64-bit checked multiply
loops remain scalar on x86 because each lane needs a widened product. They still match the
handwritten baseline after the framework removes abstraction overhead. Tensor kernels often gain
SIMD inside each tensor row, rather than across RowFn output rows.

The exact generated code is part of the contract for performance-sensitive paths. Benchmark
parity alone does not prove vectorization, and vector-shaped LLVM IR does not prove vector machine
instructions.

## Future article structure

The material supports two independent articles:

1. The RowFn design: typed row declarations, planning through visitors, null policy, prepared
   constants, and output capabilities.
2. The performance investigation: owned output, indexed sources, failure reduction, compiler
   sensitivity, assembly inspection, and misleading unrelated benchmark movement.

The dated records contain experiment details. This guide contains the stable explanatory model
that those articles can use.
