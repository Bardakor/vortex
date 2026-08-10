<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# RowFn target-branch code generation, round 2

Platform: Apple Silicon `arm64`, macOS 15.7.3, Rust 1.91.0, LLVM 21.1.2.

API baseline: `84837ad36f9c8e7c2cec76920ec07f303d60be11` on
`origin/ct/row-fn-api`.

Numeric baseline: `32ad0bf3b7` on `origin/ct/row-fn-numeric`, whose parent is the API baseline.

The API crate does not instantiate the production numeric `execute_owned` and `execute_sink`
monomorphs. The arithmetic gate therefore uses the numeric branch twice: first at the exact numeric
baseline, then with its two commits rebased onto the candidate API tree. This measures the code that
will actually receive the framework changes.

The final tested source trees were consolidated into API candidate `f759998dce` and downstream
numeric candidate `b4900072c`. Consolidation did not change either tree.

## Command

All captures use the repository bench profile with 16 codegen units and no LTO:

```text
RUSTFLAGS='-C symbol-mangling-version=v0' \
  cargo rustc --profile bench -p vortex-array --lib -- \
  --emit=llvm-ir -C debuginfo=0
```

## Framework candidate

The candidate includes the round 1 contract and batch commits, the round 2 early-decline and batch
corrections, `RowExecution` from `reduce_encoded`, and the added coverage.

| Monomorph | Vector IR at candidate | Bounds sites, baseline -> candidate | Closure calls | Failure reduction |
| --- | --- | ---: | ---: | --- |
| `i64` add | `<16 x i64>` plus remainder | 3 -> 2 | 0 -> 0 | vector and scalar phi |
| `i64` sub | `<16 x i64>` plus remainder | 3 -> 2 | 0 -> 0 | vector and scalar phi |
| `i64` mul | scalar unroll plus `<2 x i64>` support | 3 -> 2 | 0 -> 0 | scalar phi |
| `i32` mul | `<4 x i32>` and `<16 x i32>` | 3 -> 2 | 0 -> 0 | vector and scalar phi |
| `u64` mul | scalar unroll plus `<2 x i64>` support | 3 -> 2 | 0 -> 0 | scalar phi |
| `u16` mul | `<8 x i16>` plus wider unroll groups | 3 -> 2 | 0 -> 0 | vector and scalar phi |
| `i64` div, dense sink | scalar | 2 -> 2 | 0 -> 0 | immediate error |
| `i64` div, valid-row sink | scalar mask walk | 1 -> 1 | 0 -> 0 | immediate error |

The extra `output[index]` bounds edge observed for `5204fb2` on `ct/row-fn` does not reproduce on
the target lineage. The target candidate instead removes that edge from every owned arithmetic
monomorph. The remaining bounds sites are outside the arithmetic vector bodies.

The nullable `i64` add path reuses the same owned monomorph after validity resolution or filtered
retry. Nullable division uses the valid-row sink monomorph shown separately.

Signed add and subtract retain their XOR/AND overflow test. Multiply retains its widening or
high-half checks. Failure values remain loop-carried phis rather than stack loads and stores, and
rich error construction remains outside every row loop.

## Staged attribution

The production changes were also compiled one stage at a time on the downstream numeric lineage.
Bounds counts below are per owned arithmetic monomorph; the dense and valid-row division sinks stay
at two and one throughout.

| Stage | Owned bounds sites | Result |
| --- | ---: | --- |
| Round 1 API/batch candidate | 3 | Starting point for round 2 |
| Restore early skipped-row decline | 3 | No call, vector, failure-phi, or bounds regression |
| Fix length check, all-invalid branch, and reducer precedence | 2 | Removes the `output[index]` edge; other loop properties remain |
| Change `reduce_encoded` to `RowExecution`, alone | 3 | Fails the gate as a standalone change |
| Add the required strategy/prepared coverage and final safety comment | 2 | Final combined API tree passes |
| Delete unreachable deferred sink results | 2 | Final deletion also passes |

The reducer-signature change must not be cherry-picked alone. Test-only source and comments should
not normally affect optimized library IR, but in this framework they change codegen-unit placement
and restore the desired bounds proof. Upstreaming is therefore gated on the combined API commit and
its exact final tree, not on an intermediate commit or on `ct/row-fn`.

## Deferred sink deletion

Findings 11 and 14 were tested as a separate commit after the framework candidate. The commit
removes `OutputSink::ERRORS_ARE_DEFERRED`, the five unreachable deferred `SinkResult` word
implementations, the unused deferred evidence passed to `OutputSink::finish`, and the unreachable
retry classification in `finish_sink`.

The deletion moves the numeric monomorphs from codegen unit 10 to codegen unit 9, so the raw IR is
not byte-identical. Structurally, all eight monomorphs retain the candidate's vector factors,
broadcasts, two/one bounds sites, inlined closures, loop-carried failure phis, overflow checks, and
out-of-loop error construction. It passes the stated IR gate on the target lineage.

## Mixed constants

Mixed `i64` add and subtract retain `insertelement`/`shufflevector` broadcasts and `<16 x i64>`
vector arithmetic. Mixed `i32` multiply also retains fixed-width broadcast vector loops. LLVM's
vector factors span groups of 128-bit NEON registers; these paths are not scalar fallbacks.
