#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

set -Eeu -o pipefail

usage() {
    cat >&2 <<'EOF'
Usage: benchmark-rowfn.sh [OPTIONS] <baseline-worktree> <candidate-worktree> <new-output-directory>

Options:
  --suite NAME           Select a preset or benchmark label. Repeatable; defaults to full.
  --filter PATTERN       Pass a Divan benchmark filter. Repeatable.
  --config NAME          primary (1 CGU/fat LTO) or repository (16 CGUs/no LTO).
  --target-root PATH     New directory for separate baseline and candidate Cargo targets.
  --codegen-units N      Override the selected configuration.
  --lto VALUE            Override LTO with false, thin, or fat.
  --rustflags FLAGS      Override RUSTFLAGS; defaults to -C target-cpu=native.
  --build-jobs N         Jobs per concurrent revision build; defaults to 8 and cannot exceed 8.
  --bench-cpu N          Logical CPU used for every timed process; defaults to 4.
  --warm-runs N          Warm runs per revision; defaults to 2.
  --measured-pairs N     Alternating measured pairs; defaults to 7.
  --sample-count N       Divan sample count; defaults to 100.
  --min-time SECONDS     Divan minimum time; defaults to 0.25.
  --max-time SECONDS     Divan maximum time; defaults to 0.5.
  --lock-file PATH       Global timed-run lock; defaults to /tmp/vortex-rowfn-benchmark.lock.
  --list-suites          Print presets and benchmark labels, then exit.
EOF
}

suite_catalog=(
    "array-binary_ops|vortex-array|binary_ops|array,numeric,design-a-matrix,full"
    "array-compare|vortex-array|compare|array,compare,full"
    "array-row_fn_executor|vortex-array|row_fn_executor|array,framework,full"
    "array-strict_validity|vortex-array|strict_validity|array,framework,full"
    "array-like|vortex-array|like|array,full"
    "array-take_filter|vortex-array|take_filter|array,full"
    "array-varbinview_compact|vortex-array|varbinview_compact|array,full"
    "tensor-l2_norm|vortex-tensor|l2_norm|tensor,full"
    "tensor-inner_product|vortex-tensor|inner_product|tensor,full"
    "tensor-cosine_similarity|vortex-tensor|cosine_similarity|tensor,full"
    "tensor-normalized|vortex-tensor|normalized|tensor,full"
    "spatial-binary_predicates|vortex-spatial|binary_predicates|spatial,full"
    "spatial-distance|vortex-spatial|distance|spatial,full"
    "spatial-envelope|vortex-spatial|envelope|spatial,full"
    "spatial-predicate_bbox|vortex-spatial|predicate_bbox|spatial,full"
)

requested_suites=()
filters=()
configuration=primary
target_root=
codegen_units_override=
lto_override=
rustflags_override=
build_jobs=8
bench_cpu=4
warm_runs=2
measured_pairs=7
sample_count=100
min_time=0.25
max_time=0.5
lock_file=/tmp/vortex-rowfn-benchmark.lock

while [[ $# -gt 0 ]]; do
    case $1 in
        --suite) requested_suites+=("$2"); shift 2 ;;
        --filter) filters+=("$2"); shift 2 ;;
        --config) configuration=$2; shift 2 ;;
        --target-root) target_root=$2; shift 2 ;;
        --codegen-units) codegen_units_override=$2; shift 2 ;;
        --lto) lto_override=$2; shift 2 ;;
        --rustflags) rustflags_override=$2; shift 2 ;;
        --build-jobs) build_jobs=$2; shift 2 ;;
        --bench-cpu) bench_cpu=$2; shift 2 ;;
        --warm-runs) warm_runs=$2; shift 2 ;;
        --measured-pairs) measured_pairs=$2; shift 2 ;;
        --sample-count) sample_count=$2; shift 2 ;;
        --min-time) min_time=$2; shift 2 ;;
        --max-time) max_time=$2; shift 2 ;;
        --lock-file) lock_file=$2; shift 2 ;;
        --list-suites)
            echo "Presets: full array framework numeric design-a-matrix compare tensor spatial"
            printf '%s\n' "${suite_catalog[@]}" | cut -d '|' -f 1
            exit 0
            ;;
        -h|--help) usage; exit 0 ;;
        --*) echo "Unknown option: $1" >&2; usage; exit 1 ;;
        *) break ;;
    esac
done

if [[ $# -ne 3 ]]; then
    usage
    exit 1
fi
if [[ $(uname -m) != x86_64 ]]; then
    echo "RowFn native performance decisions require an x86_64 host." >&2
    exit 1
fi
if ((build_jobs < 1 || build_jobs > 8)); then
    echo "--build-jobs must be between 1 and 8 so two builds cannot exceed 16 jobs." >&2
    exit 1
fi
command -v flock >/dev/null || { echo "benchmark-rowfn.sh requires flock." >&2; exit 1; }

baseline=$(realpath "$1")
candidate=$(realpath "$2")
output=$(realpath -m "$3")
if [[ -e $output ]]; then
    echo "Output path already exists: $output" >&2
    exit 1
fi

case $configuration in
    primary) codegen_units=1; lto=fat ;;
    repository) codegen_units=16; lto=false ;;
    *) echo "Unknown configuration: $configuration" >&2; exit 1 ;;
esac
codegen_units=${codegen_units_override:-$codegen_units}
lto=${lto_override:-$lto}
rustflags=${rustflags_override:--C target-cpu=native}

if ((${#requested_suites[@]} == 0)); then
    requested_suites=(full)
fi
selected_suites=()
declare -A selected_labels=()
for request in "${requested_suites[@]}"; do
    matched=false
    for entry in "${suite_catalog[@]}"; do
        IFS='|' read -r label _ _ groups <<<"$entry"
        if [[ $request == "$label" || ,$groups, == *,$request,* ]]; then
            matched=true
            if [[ -z ${selected_labels[$label]:-} ]]; then
                selected_suites+=("$entry")
                selected_labels[$label]=1
            fi
        fi
    done
    if [[ $matched == false ]]; then
        echo "Unknown suite or benchmark label: $request" >&2
        exit 1
    fi
done

common_git_dir=$(git -C "$candidate" rev-parse --path-format=absolute --git-common-dir)
repository_root=$(dirname "$common_git_dir")
if [[ -z $target_root ]]; then
    target_root="$repository_root/target/rowfn-benchmark/$(basename "$output")"
fi
target_root=$(realpath -m "$target_root")
if [[ -e $target_root ]]; then
    echo "Target root already exists: $target_root" >&2
    exit 1
fi

mkdir -p "$output/build" "$output/warm" "$output/measured" "$target_root"
baseline_target="$target_root/baseline"
candidate_target="$target_root/candidate"
parser="$candidate/scripts/rowfn_benchmark.py"

{
    echo "RowFn benchmark machine record"
    echo "Date: $(date --iso-8601=seconds)"
    echo "Host: $(hostname)"
    echo "Kernel: $(uname -srvmo)"
    echo "Benchmark CPU: $bench_cpu"
    echo "Configuration: $configuration"
    echo "Cargo profile: bench, $codegen_units codegen units, LTO $lto"
    echo "RUSTFLAGS: $rustflags"
    echo "Warm runs: $warm_runs"
    echo "Measured pairs: $measured_pairs"
    echo "Divan: TSC timer, $sample_count samples, min $min_time s, max $max_time s"
    echo
    rustc -vV
    cargo -V
    echo
    lscpu
    echo
    rg -m1 '^microcode' /proc/cpuinfo || true
    for path in \
        /sys/devices/system/cpu/cpu"$bench_cpu"/cpufreq/scaling_governor \
        /sys/devices/system/cpu/cpu"$bench_cpu"/cpufreq/energy_performance_preference \
        /sys/devices/system/cpu/cpufreq/boost; do
        [[ -r $path ]] && echo "$path: $(<"$path")"
    done
} >"$output/machine.txt"

build_revision() {
    local worktree=$1
    local target=$2
    local log=$3

    (
        cd "$worktree"
        export CARGO_TARGET_DIR=$target
        export CARGO_PROFILE_BENCH_CODEGEN_UNITS=$codegen_units
        export CARGO_PROFILE_BENCH_LTO=$lto
        export RUSTFLAGS=$rustflags
        for entry in "${selected_suites[@]}"; do
            IFS='|' read -r _ package bench _ <<<"$entry"
            cargo bench --no-run -j "$build_jobs" -p "$package" --bench "$bench"
        done
    ) >"$log" 2>&1
}

echo "Building baseline and candidate with $build_jobs jobs each."
build_revision "$baseline" "$baseline_target" "$output/build/baseline.txt" &
baseline_pid=$!
build_revision "$candidate" "$candidate_target" "$output/build/candidate.txt" &
candidate_pid=$!
baseline_status=0
candidate_status=0
wait "$baseline_pid" || baseline_status=$?
wait "$candidate_pid" || candidate_status=$?
if [[ $baseline_status -ne 0 || $candidate_status -ne 0 ]]; then
    echo "Benchmark build failed; see $output/build/." >&2
    exit 1
fi

find_benchmark() {
    local target=$1
    local name=$2
    local binary

    binary=$(find "$target/release/deps" -maxdepth 1 -type f -executable -name "$name-*" \
        -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d ' ' -f 2-)
    [[ -n $binary ]] || { echo "Cannot find benchmark $name under $target." >&2; exit 1; }
    echo "$binary"
}

declare -A baseline_binaries=()
declare -A candidate_binaries=()
manifest_args=(
    manifest
    --output "$output/manifest.json"
    --machine-record "$output/machine.txt"
    --baseline-worktree "$baseline"
    --candidate-worktree "$candidate"
    --baseline-target "$baseline_target"
    --candidate-target "$candidate_target"
    --setting "configuration=$configuration"
    --setting "codegen_units=$codegen_units"
    --setting "lto=$lto"
    --setting "rustflags=$rustflags"
    --setting "bench_cpu=$bench_cpu"
    --setting "warm_runs=$warm_runs"
    --setting "measured_pairs=$measured_pairs"
    --setting "sample_count=$sample_count"
    --setting "min_time=$min_time"
    --setting "max_time=$max_time"
)
for filter in "${filters[@]}"; do
    manifest_args+=(--filter "$filter")
done
for entry in "${selected_suites[@]}"; do
    IFS='|' read -r label _ bench _ <<<"$entry"
    baseline_binaries[$label]=$(find_benchmark "$baseline_target" "$bench")
    candidate_binaries[$label]=$(find_benchmark "$candidate_target" "$bench")
    manifest_args+=(
        --suite "$label"
        --baseline-binary "$label=${baseline_binaries[$label]}"
        --candidate-binary "$label=${candidate_binaries[$label]}"
    )
done
python3 "$parser" "${manifest_args[@]}"

run_suite() {
    local revision=$1
    local label=$2
    local destination=$3
    local binary
    local command

    if [[ $revision == baseline ]]; then
        binary=${baseline_binaries[$label]}
    else
        binary=${candidate_binaries[$label]}
    fi
    command=(
        taskset -c "$bench_cpu" "$binary"
        --bench --timer tsc --sample-count "$sample_count"
        --min-time "$min_time" --max-time "$max_time" --color never
        "${filters[@]}"
    )
    echo "Running $label ($revision) -> $destination"
    "${command[@]}" >"$destination" 2>&1
}

echo "Waiting for the global timed benchmark lock: $lock_file"
exec {benchmark_lock}>"$lock_file"
flock "$benchmark_lock"
if pgrep -x cargo >/dev/null || pgrep -x rustc >/dev/null; then
    echo "Cargo or rustc is active after acquiring the benchmark lock; refusing to measure." >&2
    exit 1
fi

for ((round = 1; round <= warm_runs; round++)); do
    for entry in "${selected_suites[@]}"; do
        IFS='|' read -r label _ _ _ <<<"$entry"
        if ((round % 2 == 1)); then
            run_suite baseline "$label" "$output/warm/$label-baseline-$round.txt"
            run_suite candidate "$label" "$output/warm/$label-candidate-$round.txt"
        else
            run_suite candidate "$label" "$output/warm/$label-candidate-$round.txt"
            run_suite baseline "$label" "$output/warm/$label-baseline-$round.txt"
        fi
    done
done

for ((pair = 1; pair <= measured_pairs; pair++)); do
    for entry in "${selected_suites[@]}"; do
        IFS='|' read -r label _ _ _ <<<"$entry"
        if ((pair % 2 == 1)); then
            run_suite baseline "$label" "$output/measured/$label-baseline-$pair.txt"
            run_suite candidate "$label" "$output/measured/$label-candidate-$pair.txt"
        else
            run_suite candidate "$label" "$output/measured/$label-candidate-$pair.txt"
            run_suite baseline "$label" "$output/measured/$label-baseline-$pair.txt"
        fi
    done
done

python3 "$parser" summarize "$output"
echo "Raw results: $output"
echo "Summary: $output/summary.md"
