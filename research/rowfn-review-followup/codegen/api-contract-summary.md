<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# RowFn API contract code generation

Platform: Apple Silicon `arm64`, macOS 15.7.3, Rust 1.91.0, LLVM 21.1.2.

Baseline: `833632aaa3cab59fb4a7d4f001df26975b2267a1`

Candidate: `44457b76546b40987c5328eea7759cd095a4761e`

The candidate contains only the `InputElement` and `OutputSink` contract changes. The later batch
execution fixes are not present.

## Command

Both revisions used the repository bench profile with 16 codegen units and no LTO:

```text
RUSTFLAGS='-C symbol-mangling-version=v0' \
  cargo rustc --profile bench -p vortex-array --lib -- \
  --emit=llvm-ir -C debuginfo=0
```

`symbol-mangling-version=v0` identifies each generic monomorph. It does not change optimization.
The ordinary mangling build used the same profile and produced the same structural conclusions.

## Structural comparison

| Monomorph | Vector IR | Bounds sites | Closure calls | Failure reduction |
| --- | --- | ---: | ---: | --- |
| `i64` add | `<16 x i64>` plus remainder | 2 -> 2 | 0 -> 0 | vector and scalar phi |
| `i64` sub | `<16 x i64>` plus remainder | 2 -> 2 | 0 -> 0 | vector and scalar phi |
| `i64` mul | scalar unroll plus `<2 x i64>` support | 2 -> 2 | 0 -> 0 | scalar phi |
| `i32` mul | `<4 x i32>` and `<16 x i32>` | 2 -> 2 | 0 -> 0 | vector and scalar phi |
| `u64` mul | scalar unroll plus `<2 x i64>` support | 2 -> 2 | 0 -> 0 | scalar phi |
| `u16` mul | `<8 x i16>` plus wider unroll groups | 2 -> 2 | 0 -> 0 | vector and scalar phi |
| `i64` div, dense sink | scalar | 2 -> 2 | 0 -> 0 | immediate error |
| `i64` div, valid-row sink | scalar mask walk | 1 -> 1 | 0 -> 0 | immediate error |

The bounds sites are outside the arithmetic vector bodies. No `panic_bounds_check` edge appears in
a vector body. The candidate preserves every branch, vector factor, broadcast, load, arithmetic
operation, store, and loop-carried failure phi in the six owned arithmetic monomorphs. The raw diff
contains only metadata numbering, attribute numbering, and allocation-location symbol hashes.

Signed add and subtract use the existing XOR/AND sign test instead of LLVM overflow intrinsics.
Multiply uses the existing widening or high-half checks. Rich `VortexError` construction remains
outside every row loop.

## Sink publication

`OutputSink::finish` is now unsafe. The generic sink executor contains one documented unsafe call
after successful dense traversal or successful skipped-row initialization and traversal. LLVM
removes the unsafe boundary. Both `i64` division sink monomorphs retain their prior loop structure.
