#!/usr/bin/env python3
"""
gget virus Direct Benchmark

Runs gget virus queries directly (without an agent) for each benchmark
configuration and records the retrieved sequence counts.

Columns are read dynamically from the CSV header.  Column names map directly
to ``gget.virus()`` keyword arguments, except for the following benchmark-only
columns which are not passed to gget:

    query_id, expected_count, pathogen

The ``tax_id`` column is used as the positional ``virus`` argument.

Usage:
    python benchmark_gget_virus.py                      # run all queries
    python benchmark_gget_virus.py --test                # run only the first query
    python benchmark_gget_virus.py -o results_custom     # custom output directory
"""

import csv
import glob
import os
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import gget
import pandas as pd

from utils import load_completed_runs_from_csv

NUM_RUNS = 3

# Columns that are benchmark metadata and should NOT be forwarded to gget.virus()
_BENCHMARK_COLUMNS = {"query_id", "expected_count", "pathogen"}

# The column whose value is passed as the positional `virus` argument
_VIRUS_COLUMN = "tax_id"

# Columns whose non-empty values should be interpreted as integers
_INT_COLUMNS = {
    "min_seq_length", "max_seq_length", "max_ambiguous_chars",
}

# Flag columns: boolean flags where False is equivalent to not passing them
_FLAG_COLUMNS = {
    "is_sars_cov2", "is_alphainfluenza",
    "is_accession",
}

# Tri-state boolean columns: True/False/None are all meaningful
# (empty cell = None = no filter, "true" = filter for, "false" = filter against)
_TRISTATE_BOOL_COLUMNS = {
    "lab_passaged", "vaccine_strain",
}

_BOOL_COLUMNS = _FLAG_COLUMNS | _TRISTATE_BOOL_COLUMNS


def _parse_bool(raw: str) -> bool:
    return raw.strip().lower() in ("true", "1", "yes")


def _coerce_value(column: str, raw: str) -> Any:
    """Convert a raw CSV string to the appropriate Python type for gget.virus()."""
    if column in _INT_COLUMNS:
        return int(raw)
    if column in _BOOL_COLUMNS:
        return _parse_bool(raw)
    return raw.strip()


def parse_csv(csv_path: str) -> list[dict]:
    """Parse the benchmark CSV and return a list of query dicts.

    Each dict contains:
      - ``query_id``: int
      - ``expected_count``: int
      - ``pathogen``: str
      - ``tax_id``: str  (used as the positional virus arg)
      - ``gget_kwargs``: dict of keyword arguments to pass to gget.virus()
    """
    configs: list[dict] = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                query_id = int(row["query_id"])
                expected_count = int(row["expected_count"])
            except (ValueError, KeyError):
                continue

            tax_id = row.get(_VIRUS_COLUMN, "").strip()
            if not tax_id:
                continue

            pathogen = row.get("pathogen", "").strip()

            # Build gget kwargs from all remaining non-empty columns
            gget_kwargs: dict[str, Any] = {}
            for col, val in row.items():
                if col in _BENCHMARK_COLUMNS or col == _VIRUS_COLUMN:
                    continue
                val = val.strip() if val else ""
                if not val:
                    continue
                try:
                    gget_kwargs[col] = _coerce_value(col, val)
                except (ValueError, TypeError) as e:
                    print(
                        f"Warning: query {query_id}: skipping column '{col}' "
                        f"(could not convert {val!r}): {e}"
                    )

            # Drop flag kwargs that resolved to False (same as default)
            gget_kwargs = {
                k: v
                for k, v in gget_kwargs.items()
                if not (k in _FLAG_COLUMNS and v is False)
            }

            configs.append({
                "query_id": query_id,
                "expected_count": expected_count,
                "pathogen": pathogen,
                "tax_id": tax_id,
                "gget_kwargs": gget_kwargs,
            })

    return configs


def count_sequences(outfolder: str) -> Optional[int]:
    """Count sequences from the gget virus output metadata CSV."""
    csv_files = [
        f
        for f in glob.glob(os.path.join(outfolder, "*_metadata.csv"))
        if "genbank_metadata" not in os.path.basename(f)
    ]
    if not csv_files:
        return None
    df = pd.read_csv(csv_files[0])
    return len(df)


_SUMMARY_FAILURE_MARKERS = [
    "Command failed",
    "FAILED OPERATIONS",
]


def check_summary_for_errors(outfolder: str) -> Optional[str]:
    """Check command_summary.txt for error or failure indicators.

    Returns an error description string if failures are found, None otherwise.
    """
    summary_path = os.path.join(outfolder, "command_summary.txt")
    if not os.path.exists(summary_path):
        return None
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError as e:
        return f"Could not read command_summary.txt: {e}"

    for marker in _SUMMARY_FAILURE_MARKERS:
        if marker in content:
            return f"command_summary.txt contains '{marker}'"
    return None


def run_gget_virus(tax_id: str, gget_kwargs: dict, outfolder: str) -> tuple[int, Optional[str]]:
    """Run gget.virus() and return (sequence_count, error_or_none)."""
    gget.virus(tax_id, outfolder=outfolder, **gget_kwargs)

    summary_error = check_summary_for_errors(outfolder)

    count = count_sequences(outfolder)
    # If gget completed without error but produced no metadata CSV, assume 0 sequences
    count = count if count is not None else 0

    return count, summary_error


def run_benchmark(
    csv_path: str,
    output_dir: str = "results/gget_virus",
    test_run: int = 0,
    resume_from: Optional[str] = None,
):
    """Run the full benchmark suite."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    configs = parse_csv(csv_path)
    print(f"Found {len(configs)} query configurations")

    if test_run:
        configs = configs[:test_run]
        print(f"Test mode: Running first {test_run} query(ies).")

    completed_runs: set[tuple[int, int]] = set()
    total_correct = 0
    total_runs = 0
    total_errors = 0

    if resume_from:
        csv_report_path = Path(resume_from)
        completed_runs = load_completed_runs_from_csv(str(csv_report_path))
        print(f"Resuming: found {len(completed_runs)} completed runs in {csv_report_path}")
        # Reconstruct tallies from existing rows
        try:
            with open(csv_report_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    total_runs += 1
                    if row.get("error") and row["error"].strip():
                        total_errors += 1
                    elif row.get("is_correct", "").strip().lower() == "true":
                        total_correct += 1
        except Exception as e:
            print(f"Warning: could not reconstruct tallies from {csv_report_path}: {e}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_report_path = output_path / f"gget_direct_benchmark_summary_{timestamp}.csv"

        with open(csv_report_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "query_id", "run_number", "expected_count", "retrieved_count",
                "is_correct", "error", "duration_seconds",
            ])

    for i, config in enumerate(configs):
        print(
            f"\nQuery {config['query_id']} ({i + 1}/{len(configs)}): "
            f"{config['pathogen']}"
        )
        print(f"  Expected: {config['expected_count']}")

        for run_num in range(1, NUM_RUNS + 1):
            if (config["query_id"], run_num) in completed_runs:
                continue

            tmpdir = tempfile.mkdtemp(
                prefix=f"gget_q{config['query_id']}_r{run_num}_"
            )

            retrieved_count = None
            error = None
            start = time.time()

            try:
                retrieved_count, summary_error = run_gget_virus(
                    config["tax_id"], config["gget_kwargs"], tmpdir,
                )
                if summary_error:
                    error = summary_error
            except Exception as e:
                error = str(e)
            finally:
                duration = time.time() - start
                shutil.rmtree(tmpdir, ignore_errors=True)

            is_correct = (
                not error
                and retrieved_count is not None
                and retrieved_count == config["expected_count"]
            )
            total_runs += 1
            if error:
                total_errors += 1
            elif is_correct:
                total_correct += 1

            status = "PASS" if is_correct else ("ERROR" if error else "FAIL")
            print(
                f"  Run {run_num}: Retrieved={retrieved_count}, "
                f"Expected={config['expected_count']} [{status}] ({duration:.1f}s)"
            )
            if error:
                print(f"    Error: {error}")

            with open(csv_report_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    config["query_id"], run_num, config["expected_count"],
                    retrieved_count, is_correct, error, duration,
                ])

    print(f"\n{'=' * 60}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total queries: {len(configs)}")
    print(f"Total runs: {total_runs}")
    print(f"Correct: {total_correct}")
    print(f"Incorrect: {total_runs - total_correct - total_errors}")
    print(f"Errors: {total_errors}")
    if total_runs:
        print(f"Accuracy: {total_correct / total_runs:.2%}")
    print(f"\nCSV saved to: {csv_report_path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark gget virus directly (no agent)"
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=str(Path(__file__).parent.parent / "docs" / "virseq_benchmark.csv"),
        help="Path to virseq_benchmark.csv",
    )
    parser.add_argument("--output-dir", "-o", default=str(Path(__file__).parent.parent / "results" / "gget_virus"))
    parser.add_argument(
        "--test", "-t",
        nargs="?",
        const=1,
        default=0,
        type=int,
        help="Run only the first N queries for testing (default N=1)",
    )
    parser.add_argument(
        "--resume",
        metavar="REPORT_CSV",
        default=None,
        help="Resume from an existing CSV report, skipping completed runs",
    )

    args = parser.parse_args()
    run_benchmark(args.csv_path, output_dir=args.output_dir, test_run=args.test, resume_from=args.resume)


if __name__ == "__main__":
    main()
