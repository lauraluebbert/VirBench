#!/usr/bin/env python3
"""
Edison Finch Analysis Agent Benchmark

Benchmarks the Edison Finch analysis agent's ability to retrieve viral
sequences with specific filters. Each query configuration is tested
NUM_RUNS times and results are compared against expected counts.

Usage:
    python benchmark_edison_analysis.py                # run all queries
    python benchmark_edison_analysis.py --test         # first query only
    python benchmark_edison_analysis.py --gget-virus   # include gget docs
"""

import os
import json
import csv
import asyncio
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import Optional

from dotenv import load_dotenv
from edison_client import EdisonClient, JobNames

from utils import (
    parse_csv,
    build_query,
    extract_count_from_response,
    BenchmarkResult,
    QueryConfig,
    NUM_RUNS,
    GGET_VIRUS_DOC_MD_PATH,
    load_completed_runs_from_json,
)

# Load environment variables
load_dotenv(override=True)

EDISON_API_KEY = os.getenv("EDISON_API_KEY")
if not EDISON_API_KEY or EDISON_API_KEY == "your_api_key_here":
    raise ValueError("Please set your EDISON_API_KEY in the .env file")


async def run_single_benchmark(
    client: EdisonClient,
    config: QueryConfig,
    run_number: int,
    use_gget_virus: bool = False,
    gget_data_storage_uris: Optional[list] = None,  # NEW: pass in URIs if available
) -> BenchmarkResult:
    """Run a single benchmark query."""
    query = build_query(config, use_gget_virus=use_gget_virus)
    tool_status = " [with gget virus]" if use_gget_virus else ""
    print(f"  Run {run_number}: Querying for {config.pathogen} ({config.filters.get('segment', '')}){tool_status}...")

    start_time = datetime.now()
    result = BenchmarkResult(
        query_id=config.query_id,
        run_number=run_number,
        expected_count=config.expected_count,
    )

    try:
        data_storage_uris = None

        if use_gget_virus:
            if gget_data_storage_uris is not None:
                data_storage_uris = gget_data_storage_uris
            else:
                # This branch should not be reached if upload is managed in run_benchmark
                raise RuntimeError("gget virus documentation URIs not provided but gget virus workflow enabled.")

        task_data = {
            "name": JobNames.ANALYSIS,
            "query": query,
        }

        if use_gget_virus:
            task_data["runtime_config"] = {
                "environment_config": {
                    "data_storage_uris": data_storage_uris,
                },
            }

        response = await client.arun_tasks_until_done(task_data)

        result.duration_seconds = (datetime.now() - start_time).total_seconds()
        result.raw_response = response.answer if hasattr(response, 'answer') else str(response)

        # Extract count from response using LLM
        retrieved_count = await extract_count_from_response(result.raw_response)
        result.retrieved_count = retrieved_count

        if retrieved_count is not None:
            result.is_correct = (retrieved_count == config.expected_count)

    except Exception as e:
        result.error = str(e)
        result.duration_seconds = (datetime.now() - start_time).total_seconds()

    return result


async def run_benchmark(
    csv_path: str,
    output_dir: str = "results/edison_analysis",
    test_run: int = 0,
    use_gget_virus: bool = False,
    resume_from: Optional[str] = None,
):
    """Run the full benchmark suite.

    Args:
        csv_path: Path to the benchmark CSV.
        output_dir: Directory to store results.
        test_run: If > 0, only run the first N queries.
        use_gget_virus: If True, tell agent to use the gget virus module.
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Parse configurations
    print("Parsing CSV configuration...")
    configs = parse_csv(csv_path)
    print(f"Found {len(configs)} query configurations")
    print(f"gget virus workflow: {'ENABLED' if use_gget_virus else 'DISABLED'}")

    if test_run:
        print(f"Test mode: Running first {test_run} query(ies).")
        configs = configs[:test_run]
        if not configs:
            print("No valid queries found for test run.")
            return

    # Initialize client
    client = EdisonClient(api_key=EDISON_API_KEY)

    # --- gget virus: upload docs ONCE if needed ---
    gget_data_storage_uris = None
    if use_gget_virus:
        if not os.path.isfile(GGET_VIRUS_DOC_MD_PATH):
            raise FileNotFoundError(
                f"gget_virus_docs.md file not found at {GGET_VIRUS_DOC_MD_PATH}. Please place the gget_virus_docs.md documentation there."
            )
        print("Uploading gget_virus documentation to Edison data storage (one-time upload)...")
        upload_response = await client.astore_file_content(
            name="gget_virus documentation",
            file_path=GGET_VIRUS_DOC_MD_PATH,
            description="Documentation for the gget virus Python and cli module.",
        )
        data_storage_id = upload_response.data_storage.id
        gget_data_storage_uris = [f"data_entry:{data_storage_id}"]

    # Prepare for appending results immediately after each run
    all_results: list[BenchmarkResult] = []
    completed_runs: set[tuple[int, int]] = set()

    if resume_from:
        report_path = Path(resume_from)
        csv_report_path = Path(
            str(report_path).replace("_report", "_summary").replace(".json", ".csv")
        )
        completed_runs = load_completed_runs_from_json(str(report_path))
        print(f"Resuming: found {len(completed_runs)} completed runs in {report_path}")
        if not csv_report_path.exists():
            with open(csv_report_path, "w", newline="") as cf:
                writer = csv.writer(cf)
                writer.writerow([
                    "query_id", "run_number", "expected_count", "retrieved_count",
                    "is_correct", "error", "duration_seconds",
                ])
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_middle = "_gv" if use_gget_virus else ""
        report_path = output_path / f"benchmark_report{file_middle}_{timestamp}.json"
        csv_report_path = output_path / f"benchmark_summary{file_middle}_{timestamp}.csv"

        json_report_dict = {
            "timestamp": timestamp,
            "total_queries": len(configs),
            "total_runs": 0,
            "results": [],
            "summary": {
                "correct_runs": 0,
                "incorrect_runs": 0,
                "error_runs": 0,
                "accuracy": 0
            }
        }
        with open(report_path, 'w') as jf:
            json.dump(json_report_dict, jf, indent=2)

        with open(csv_report_path, 'w', newline='') as cf:
            writer = csv.writer(cf)
            writer.writerow([
                "query_id", "run_number", "expected_count", "retrieved_count",
                "is_correct", "error", "duration_seconds"
            ])

    for i, config in enumerate(configs):
        print(f"\nQuery {config.query_id} ({i+1}/{len(configs)}): {config.pathogen} - {config.filters.get('segment', '')}")
        print(f"  Expected count: {config.expected_count}")

        for run_num in range(1, NUM_RUNS + 1):
            if (config.query_id, run_num) in completed_runs:
                continue

            result = await run_single_benchmark(
                client,
                config,
                run_num,
                use_gget_virus=use_gget_virus,
                gget_data_storage_uris=gget_data_storage_uris,
            )
            all_results.append(result)

            status = "PASS" if result.is_correct else "FAIL"
            if result.error:
                status = "ERROR"

            print(f"  Run {run_num}: Retrieved={result.retrieved_count}, Expected={result.expected_count} [{status}]")
            if result.error:
                print(f"    Error: {result.error}")

            # --- APPEND TO CSV after each run ---
            with open(csv_report_path, 'a', newline='') as cf:
                writer = csv.writer(cf)
                writer.writerow([
                    result.query_id, result.run_number, result.expected_count, result.retrieved_count,
                    result.is_correct, result.error, result.duration_seconds
                ])

            # --- APPEND TO JSON after each run ---
            # Load the current json report
            try:
                with open(report_path, 'r') as jf:
                    json_report_dict = json.load(jf)
            except Exception:
                json_report_dict = {
                    "timestamp": timestamp,
                    "total_queries": len(configs),
                    "total_runs": 0,
                    "results": [],
                    "summary": {
                        "correct_runs": 0,
                        "incorrect_runs": 0,
                        "error_runs": 0,
                        "accuracy": 0
                    }
                }

            json_report_dict.setdefault("results", []).append(asdict(result))
            json_report_dict["total_runs"] = len(json_report_dict["results"])

            # Update running summary/accuracy stats for currently written runs
            correct_runs = sum(1 for r in json_report_dict["results"] if r.get("is_correct", False))
            error_runs = sum(1 for r in json_report_dict["results"] if r.get("error"))
            total_runs = json_report_dict["total_runs"]
            incorrect_runs = total_runs - correct_runs - error_runs
            accuracy = correct_runs / total_runs if total_runs else 0

            json_report_dict["summary"] = {
                "correct_runs": correct_runs,
                "incorrect_runs": incorrect_runs,
                "error_runs": error_runs,
                "accuracy": accuracy
            }

            with open(report_path, 'w') as jf:
                json.dump(json_report_dict, jf, indent=2)

    # After all runs are done, print summary as before (using what we just finished writing)
    with open(report_path, 'r') as jf:
        report = json.load(jf)

    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"Total queries: {report['total_queries']}")
    print(f"Total runs: {report['total_runs']}")
    print(f"Correct: {report['summary']['correct_runs']}")
    print(f"Incorrect: {report['summary']['incorrect_runs']}")
    print(f"Errors: {report['summary']['error_runs']}")
    print(f"Accuracy: {report['summary']['accuracy']:.2%}")
    print(f"\nReport saved to: {report_path}")
    print(f"CSV summary saved to: {csv_report_path}")

    return report


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Edison Finch Analysis Agent")
    parser.add_argument(
        "csv_path",
        help="Path to the benchmark CSV file",
        nargs="?",
        default=str(Path(__file__).parent.parent / "docs" / "virseq_benchmark.csv")
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="Output directory for results",
        default=str(Path(__file__).parent.parent / "results" / "edison_analysis")
    )
    parser.add_argument(
        "--test", "-t",
        nargs="?",
        const=1,
        default=0,
        type=int,
        help="Run only the first N queries for testing (default N=1)",
    )
    parser.add_argument(
        "--gget-virus", "-gv",
        action="store_true",
        help="Enable the gget virus workflow for the analysis agent.",
    )
    parser.add_argument(
        "--resume",
        metavar="REPORT_JSON",
        default=None,
        help="Resume from an existing JSON report, skipping completed runs",
    )

    args = parser.parse_args()

    asyncio.run(run_benchmark(
        args.csv_path,
        output_dir=args.output_dir,
        test_run=args.test,
        use_gget_virus=args.gget_virus,
        resume_from=args.resume,
    ))

if __name__ == "__main__":
    main()
