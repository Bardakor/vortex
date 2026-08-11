#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Capture and summarize evidence from ``benchmark-rowfn.sh`` runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

RESULT_FILE = re.compile(r"^(?P<suite>.+)-(?P<revision>baseline|candidate)-(?P<pair>\d+)\.txt$")
TREE_ROW = re.compile(r"^(?P<prefix>(?:│  |   )*)(?:├─ |╰─ )(?P<body>.*)$")
TIMING = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>ps|ns|µs|us|ms|s)\s*$")
UNIT_TO_NS = {
    "ps": 0.001,
    "ns": 1.0,
    "µs": 1_000.0,
    "us": 1_000.0,
    "ms": 1_000_000.0,
    "s": 1_000_000_000.0,
}


@dataclass(frozen=True)
class BenchmarkSummary:
    suite: str
    benchmark: str
    pairs: int
    baseline_median_ns: float
    candidate_median_ns: float
    median_ratio: float
    minimum_ratio: float
    maximum_ratio: float
    ratio_mad: float


def run_git(worktree: Path, *args: str, binary: bool = False) -> str | bytes:
    """Run one read-only Git command in ``worktree``."""

    result = subprocess.run(
        ["git", "-C", str(worktree), *args],
        check=True,
        capture_output=True,
        text=not binary,
    )
    return result.stdout if binary else result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def revision_record(worktree: Path, target: Path, binaries: Iterable[str]) -> dict[str, object]:
    """Describe the exact revision, dirty patch, targets, and benchmark executables."""

    status = str(run_git(worktree, "status", "--short")).splitlines()
    diff = run_git(worktree, "diff", "--binary", "HEAD", binary=True)
    assert isinstance(diff, bytes)

    untracked = run_git(worktree, "ls-files", "--others", "--exclude-standard", "-z", binary=True)
    assert isinstance(untracked, bytes)
    dirty_digest = hashlib.sha256(diff)
    dirty_digest.update(untracked)
    for relative_path in filter(None, untracked.decode().split("\0")):
        path = worktree / relative_path
        if path.is_file():
            dirty_digest.update(relative_path.encode())
            dirty_digest.update(bytes.fromhex(sha256_file(path)))

    executable_records: dict[str, object] = {}
    for entry in binaries:
        label, separator, raw_path = entry.partition("=")
        if not separator:
            raise ValueError(f"expected LABEL=PATH for benchmark binary, got {entry!r}")
        path = Path(raw_path).resolve()
        executable_records[label] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }

    return {
        "worktree": str(worktree.resolve()),
        "head": run_git(worktree, "rev-parse", "HEAD"),
        "changed_paths": status,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "dirty_state_sha256": dirty_digest.hexdigest(),
        "target": str(target.resolve()),
        "binaries": executable_records,
    }


def write_manifest(args: argparse.Namespace) -> None:
    settings = dict(setting.split("=", 1) for setting in args.setting)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "settings": settings,
        "suites": args.suite,
        "filters": args.filter,
        "machine_record": str(Path(args.machine_record).resolve()),
        "baseline": revision_record(
            Path(args.baseline_worktree),
            Path(args.baseline_target),
            args.baseline_binary,
        ),
        "candidate": revision_record(
            Path(args.candidate_worktree),
            Path(args.candidate_target),
            args.candidate_binary,
        ),
    }
    output = Path(args.output)
    output.write_text(f"{json.dumps(manifest, indent=2, sort_keys=True)}\n", encoding="utf-8")


def timing_ns(field: str) -> float:
    match = TIMING.search(field.strip())
    if match is None:
        raise ValueError(f"cannot parse Divan timing from {field!r}")

    return float(match.group("value")) * UNIT_TO_NS[match.group("unit")]


def parse_divan(path: Path) -> dict[str, float]:
    """Return benchmark paths and median nanoseconds from one Divan table."""

    parents: dict[int, str] = {}
    timings: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = re.split(r"\s+│\s+", line)
        tree_match = TREE_ROW.match(fields[0])
        if tree_match is None:
            continue

        depth = len(tree_match.group("prefix")) // 3
        body = tree_match.group("body").rstrip()
        timing_match = TIMING.search(body)
        name = body[: timing_match.start()].rstrip() if timing_match else body.strip()
        parents = {level: parent for level, parent in parents.items() if level < depth}

        if timing_match is None:
            parents[depth] = name
            continue
        if len(fields) < 3:
            raise ValueError(f"timed Divan row has no median column in {path}: {line}")

        components = [parents[level] for level in sorted(parents) if level < depth]
        benchmark = "/".join([*components, name])
        if benchmark in timings:
            raise ValueError(f"duplicate benchmark {benchmark!r} in {path}")
        timings[benchmark] = timing_ns(fields[2])

    if not timings:
        raise ValueError(f"no Divan benchmark timings found in {path}")

    return timings


def read_measurements(directory: Path) -> dict[tuple[str, str, int, str], float]:
    measurements: dict[tuple[str, str, int, str], float] = {}
    for path in sorted(directory.glob("*.txt")):
        match = RESULT_FILE.match(path.name)
        if match is None:
            continue
        suite = match.group("suite")
        revision = match.group("revision")
        pair = int(match.group("pair"))
        for benchmark, median_ns in parse_divan(path).items():
            measurements[suite, revision, pair, benchmark] = median_ns

    if not measurements:
        raise ValueError(f"no measured result files found in {directory}")

    return measurements


def summarize(measurements: dict[tuple[str, str, int, str], float]) -> list[BenchmarkSummary]:
    groups = {(suite, pair, benchmark) for suite, _, pair, benchmark in measurements}
    incomplete = [
        group
        for group in groups
        if (group[0], "baseline", group[1], group[2]) not in measurements
        or (group[0], "candidate", group[1], group[2]) not in measurements
    ]
    if incomplete:
        raise ValueError(f"unpaired benchmark measurements: {sorted(incomplete)!r}")

    by_benchmark: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for suite, pair, benchmark in sorted(groups):
        baseline = measurements[suite, "baseline", pair, benchmark]
        candidate = measurements[suite, "candidate", pair, benchmark]
        by_benchmark.setdefault((suite, benchmark), []).append((baseline, candidate))

    summaries = []
    for (suite, benchmark), pairs in sorted(by_benchmark.items()):
        baseline_values = [baseline for baseline, _ in pairs]
        candidate_values = [candidate for _, candidate in pairs]
        ratios = [candidate / baseline for baseline, candidate in pairs]
        median_ratio = statistics.median(ratios)
        summaries.append(
            BenchmarkSummary(
                suite=suite,
                benchmark=benchmark,
                pairs=len(pairs),
                baseline_median_ns=statistics.median(baseline_values),
                candidate_median_ns=statistics.median(candidate_values),
                median_ratio=median_ratio,
                minimum_ratio=min(ratios),
                maximum_ratio=max(ratios),
                ratio_mad=statistics.median(abs(ratio - median_ratio) for ratio in ratios),
            )
        )

    return summaries


def format_ns(value: float) -> str:
    for divisor, unit in ((1_000_000_000, "s"), (1_000_000, "ms"), (1_000, "µs")):
        if value >= divisor:
            return f"{value / divisor:.3f} {unit}"

    return f"{value:.3f} ns"


def write_summary(output_directory: Path, summaries: list[BenchmarkSummary]) -> None:
    csv_path = output_directory / "ratios.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(summaries[0])))
        writer.writeheader()
        writer.writerows(asdict(summary) for summary in summaries)

    markdown = [
        "# RowFn benchmark comparison",
        "",
        "Ratios are paired candidate/baseline medians. Lower is faster.",
        "",
        "| Suite | Benchmark | Pairs | Baseline | Candidate | Ratio | Change | MAD |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in sorted(summaries, key=lambda result: result.median_ratio, reverse=True):
        change = (summary.median_ratio - 1.0) * 100.0
        markdown.append(
            f"| {summary.suite} | `{summary.benchmark}` | {summary.pairs} "
            f"| {format_ns(summary.baseline_median_ns)} "
            f"| {format_ns(summary.candidate_median_ns)} "
            f"| {summary.median_ratio:.6f} | {change:+.2f}% | {summary.ratio_mad:.6f} |"
        )
    markdown.append("")
    (output_directory / "summary.md").write_text("\n".join(markdown), encoding="utf-8")


def summarize_directory(args: argparse.Namespace) -> None:
    output_directory = Path(args.output_directory)
    summaries = summarize(read_measurements(output_directory / "measured"))
    write_summary(output_directory, summaries)


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(required=True)

    manifest = subparsers.add_parser("manifest", help="capture revisions and executable hashes")
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--machine-record", required=True)
    manifest.add_argument("--baseline-worktree", required=True)
    manifest.add_argument("--candidate-worktree", required=True)
    manifest.add_argument("--baseline-target", required=True)
    manifest.add_argument("--candidate-target", required=True)
    manifest.add_argument("--setting", action="append", default=[])
    manifest.add_argument("--suite", action="append", default=[])
    manifest.add_argument("--filter", action="append", default=[])
    manifest.add_argument("--baseline-binary", action="append", default=[])
    manifest.add_argument("--candidate-binary", action="append", default=[])
    manifest.set_defaults(function=write_manifest)

    summary = subparsers.add_parser("summarize", help="write ratios.csv and summary.md")
    summary.add_argument("output_directory")
    summary.set_defaults(function=summarize_directory)

    return parser


def main() -> None:
    args = argument_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
