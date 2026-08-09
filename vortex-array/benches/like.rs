// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use divan::Bencher;
use rand::RngExt;
use rand::SeedableRng;
use rand::distr::Uniform;
use rand::prelude::StdRng;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::scalar_fn::fns::like::Like;
use vortex_array::scalar_fn::fns::like::LikeOptions;

fn main() {
    divan::main();
}

const ARRAY_SIZE: usize = 2_048;

/// Random lowercase strings of 4..=24 bytes, some with a `hello` infix.
fn strings() -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(0);
    let len_dist = Uniform::new_inclusive(4usize, 24).unwrap();
    VarBinViewArray::from_iter_str((0..ARRAY_SIZE).map(|i| {
        let len = rng.sample(len_dist);
        let mut s: String = (0..len)
            .map(|_| char::from(rng.random_range(b'a'..=b'z')))
            .collect();
        if i % 7 == 0 {
            s.insert_str(len / 2, "hello");
        }
        s
    }))
    .into_array()
}

fn bench_like(bencher: Bencher, pattern: &str, options: LikeOptions) {
    let session = vortex_array::array_session();
    let array = strings();
    bencher
        .with_inputs(|| {
            (
                Like.try_new_array(
                    ARRAY_SIZE,
                    options,
                    [
                        array.clone(),
                        ConstantArray::new(pattern, ARRAY_SIZE).into_array(),
                    ],
                )
                .unwrap(),
                session.create_execution_ctx(),
            )
        })
        .bench_values(|(array, mut ctx)| array.execute::<BoolArray>(&mut ctx).unwrap());
}

#[divan::bench]
fn like_exact(bencher: Bencher) {
    bench_like(bencher, "hello", LikeOptions::default());
}

#[divan::bench]
fn like_prefix(bencher: Bencher) {
    bench_like(bencher, "hello%", LikeOptions::default());
}

#[divan::bench]
fn like_suffix(bencher: Bencher) {
    bench_like(bencher, "%hello", LikeOptions::default());
}

#[divan::bench]
fn like_contains(bencher: Bencher) {
    bench_like(bencher, "%hello%", LikeOptions::default());
}

#[divan::bench]
fn like_regex(bencher: Bencher) {
    bench_like(bencher, "h_llo%w%d", LikeOptions::default());
}

fn bench_per_row_patterns(bencher: Bencher, patterns: ArrayRef) {
    let session = vortex_array::array_session();
    let array = strings();
    bencher
        .with_inputs(|| {
            (
                Like.try_new_array(
                    ARRAY_SIZE,
                    LikeOptions::default(),
                    [array.clone(), patterns.clone()],
                )
                .unwrap(),
                session.create_execution_ctx(),
            )
        })
        .bench_values(|(array, mut ctx)| array.execute::<BoolArray>(&mut ctx).unwrap());
}

#[divan::bench]
fn like_per_row_patterns(bencher: Bencher) {
    // A non-constant pattern child takes the per-row path; repeated patterns hit the
    // compile cache.
    let patterns = VarBinViewArray::from_iter_str((0..ARRAY_SIZE).map(|_| "hello%")).into_array();
    bench_per_row_patterns(bencher, patterns);
}

/// The per-row path with the compile cache hit on every row, carrying the infix pattern that
/// [`like_per_row_distinct_patterns`] varies. Both compile the same shape and match the same way,
/// so the only difference between them is how often a pattern is compiled.
#[divan::bench]
fn like_per_row_repeated_patterns(bencher: Bencher) {
    let patterns = VarBinViewArray::from_iter_str((0..ARRAY_SIZE).map(|_| "%aaa%")).into_array();
    bench_per_row_patterns(bencher, patterns);
}

/// The per-row path with the compile cache defeated: every row carries a distinct pattern of the
/// same shape, so each row pays one [`LikePattern`] compilation.
///
/// Paired with [`like_per_row_repeated_patterns`] this isolates the cost of compiling a pattern from
/// the cost of matching against it, which is what any kernel that cannot cache across rows pays.
#[divan::bench]
fn like_per_row_distinct_patterns(bencher: Bencher) {
    let patterns = VarBinViewArray::from_iter_str(
        (0..ARRAY_SIZE).map(|i| format!("%{}%", distinct_trigram(i))),
    )
    .into_array();
    bench_per_row_patterns(bencher, patterns);
}

/// A distinct three-letter lowercase infix for each index below 26³.
fn distinct_trigram(i: usize) -> String {
    let letter = |place: usize| char::from(b'a' + u8::try_from((i / place) % 26).unwrap());
    [letter(1), letter(26), letter(26 * 26)].iter().collect()
}

#[divan::bench]
fn ilike_contains(bencher: Bencher) {
    bench_like(
        bencher,
        "%HELLO%",
        LikeOptions {
            negated: false,
            case_insensitive: true,
        },
    );
}
