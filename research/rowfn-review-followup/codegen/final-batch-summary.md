<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# RowFn batch correctness code generation

Platform: Apple Silicon `arm64`, macOS 15.7.3, Rust 1.91.0, LLVM 21.1.2.

Baseline: `833632aaa3cab59fb4a7d4f001df26975b2267a1`

Candidate: `5204fb2be5f069fb33eee1f14f3fca2b9760f894`

The candidate includes the API contract commit and the batch length, single-probe, and retry fixes.
It used the command from `api-contract-summary.md`.

## Gate result

The batch correctness commit does not pass the arithmetic IR gate. The mixed-input branch of each
owned arithmetic monomorph gains a `panic_bounds_check` edge for
`row/execute/owned.rs:98`:

```rust
output[index].write(value);
```

The source line itself did not change. Moving the `reduce_encoded` probe out of the generic row
executor changed codegen-unit placement and inlining decisions. The all-varying and mixed-constant
vector loops remain present, the row closure remains inlined, failure remains loop-carried, and
error construction remains outside the loop. The new output bounds edge is still a structural
regression, so this commit stays on `ct/row-fn` and is not upstreamed.

## Mixed constants on `arm64`

The mixed-constant `i64` add and subtract branches contain `insertelement` and `shufflevector`
broadcasts, `<16 x i64>` arithmetic, `<16 x i1>` loop-carried failure phis, and vector stores. The
mixed-constant `i32` multiply branch also contains fixed-width vector arithmetic and broadcasts.

Apple Silicon exposes 128-bit NEON. The LLVM vector factors span groups of NEON registers, but the
optimized IR is not a scalar fallback. The claim that mixed constants fall off the vectorized lane
kernel is false on this platform.
