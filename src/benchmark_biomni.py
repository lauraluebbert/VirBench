#!/usr/bin/env python3
"""
Biomni Agent Benchmark

Benchmarks the Biomni data-lake agent's ability to retrieve viral sequences
with specific filters. Each query configuration is tested NUM_RUNS times
and results are compared against expected counts.

Usage:
    python benchmark_biomni.py                         # run all queries
    python benchmark_biomni.py --test                  # first query only
    python benchmark_biomni.py --gget-virus            # include gget docs
    python benchmark_biomni.py --llm claude-sonnet-4-20250514
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


async def run_single_benchmark(
    agent,
    config: QueryConfig,
    run_number: int,
    use_gget_virus: bool = False,
) -> BenchmarkResult:
    """Run a single benchmark query using the Biomni agent."""
    # Use the same query string as in benchmark.py: can pass use_gget_virus flag
    query = build_query(config, use_gget_virus=use_gget_virus)
    print(f"  Run {run_number}: Querying for {config.pathogen} ({config.filters.get('segment', '')})...")

    start_time = datetime.now()
    result = BenchmarkResult(
        query_id=config.query_id,
        run_number=run_number,
        expected_count=config.expected_count,
    )

    try:
        # Biomni's agent.go() is synchronous, so run in a thread to avoid blocking
        response = await asyncio.to_thread(agent.go, query)

        result.duration_seconds = (datetime.now() - start_time).total_seconds()

        # Extract the response text - agent.go() returns the response string
        response_text = str(response) if response else ""
        result.raw_response = response_text

        # Extract count from response using LLM
        retrieved_count = await extract_count_from_response(response_text)
        result.retrieved_count = retrieved_count

        if retrieved_count is not None:
            result.is_correct = (retrieved_count == config.expected_count)

    except Exception as e:
        result.error = str(e)
        result.duration_seconds = (datetime.now() - start_time).total_seconds()

    return result

async def run_benchmark(
    csv_path: str,
    output_dir: str = "results/biomni",
    test_run: int = 0,
    llm: str = "claude-sonnet-4-20250514",
    biomni_data_path: str = str(Path(__file__).parent.parent / "data"),
    use_gget_virus: bool = False,
    resume_from: Optional[str] = None,
):
    """Run the full Biomni benchmark suite.

    Args:
        csv_path: Path to the benchmark CSV.
        output_dir: Directory to store results.
        test_run: If > 0, only run the first N queries.
        llm: LLM model name for the Biomni agent.
        biomni_data_path: Path for Biomni's data lake.
        use_gget_virus: Whether to instruct the agent to use gget_virus (query as in benchmark.py).
    """
    from biomni.agent import A1

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Parse configurations
    print("Parsing CSV configuration...")
    configs = parse_csv(csv_path)
    print(f"Found {len(configs)} query configurations")

    if test_run:
        print(f"Test mode: Running first {test_run} query(ies).")
        configs = configs[:test_run]
        if not configs:
            print("No valid queries found for test run.")
            return

    # Initialize Biomni agent
    print(f"Initializing Biomni agent (llm={llm})...")
    agent = A1(path=biomni_data_path, llm=llm)

    if use_gget_virus:
        # Attach gget_virus documentation
        doc_path = Path(GGET_VIRUS_DOC_MD_PATH)
        if not doc_path.exists():
            raise FileNotFoundError(f"gget virus doc not found: {doc_path}")

        agent.add_data({
            str(doc_path): (
                "Documentation for the gget virus Python and cli module."
            )
        })

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
        report_path = output_path / f"biomni_benchmark_report{file_middle}_{timestamp}.json"
        csv_report_path = output_path / f"biomni_benchmark_summary{file_middle}_{timestamp}.csv"

        json_report_dict = {
            "timestamp": timestamp,
            "agent": "biomni",
            "llm": llm,
            "gget_virus": use_gget_virus,
            "total_queries": len(configs),
            "total_runs": 0,
            "results": [],
            "summary": {
                "correct_runs": 0,
                "incorrect_runs": 0,
                "error_runs": 0,
                "accuracy": 0,
            },
        }
        with open(report_path, "w") as jf:
            json.dump(json_report_dict, jf, indent=2)

        with open(csv_report_path, "w", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow([
                "query_id", "run_number", "expected_count", "retrieved_count",
                "is_correct", "error", "duration_seconds",
            ])

    for i, config in enumerate(configs):
        print(f"\nQuery {config.query_id} ({i+1}/{len(configs)}): {config.pathogen} - {config.filters.get('segment', '')}")
        print(f"  Expected count: {config.expected_count}")

        for run_num in range(1, NUM_RUNS + 1):
            if (config.query_id, run_num) in completed_runs:
                continue

            result = await run_single_benchmark(agent, config, run_num, use_gget_virus=use_gget_virus)
            all_results.append(result)

            status = "PASS" if result.is_correct else "FAIL"
            if result.error:
                status = "ERROR"

            print(f"  Run {run_num}: Retrieved={result.retrieved_count}, Expected={result.expected_count} [{status}]")
            if result.error:
                print(f"    Error: {result.error}")

            # Append to CSV after each run
            with open(csv_report_path, "a", newline="") as cf:
                writer = csv.writer(cf)
                writer.writerow([
                    result.query_id, result.run_number, result.expected_count,
                    result.retrieved_count, result.is_correct, result.error,
                    result.duration_seconds,
                ])

            # Append to JSON after each run
            try:
                with open(report_path, "r") as jf:
                    json_report_dict = json.load(jf)
            except Exception:
                json_report_dict = {
                    "timestamp": timestamp,
                    "agent": "biomni",
                    "llm": llm,
                    "gget_virus": use_gget_virus,
                    "total_queries": len(configs),
                    "total_runs": 0,
                    "results": [],
                    "summary": {
                        "correct_runs": 0,
                        "incorrect_runs": 0,
                        "error_runs": 0,
                        "accuracy": 0,
                    },
                }

            json_report_dict.setdefault("results", []).append(asdict(result))
            json_report_dict["total_runs"] = len(json_report_dict["results"])

            # Update running summary/accuracy stats
            correct_runs = sum(1 for r in json_report_dict["results"] if r.get("is_correct", False))
            error_runs = sum(1 for r in json_report_dict["results"] if r.get("error"))
            total_runs = json_report_dict["total_runs"]
            incorrect_runs = total_runs - correct_runs - error_runs
            accuracy = correct_runs / total_runs if total_runs else 0

            json_report_dict["summary"] = {
                "correct_runs": correct_runs,
                "incorrect_runs": incorrect_runs,
                "error_runs": error_runs,
                "accuracy": accuracy,
            }

            with open(report_path, "w") as jf:
                json.dump(json_report_dict, jf, indent=2)

    # Print final summary
    with open(report_path, "r") as jf:
        report = json.load(jf)

    print(f"\n{'='*60}")
    print("BIOMNI BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"Agent: Biomni (llm={llm})")
    print(f"gget_virus mode: {'enabled' if use_gget_virus else 'disabled'}")
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

    parser = argparse.ArgumentParser(description="Benchmark Biomni Agent")
    parser.add_argument(
        "csv_path",
        help="Path to the benchmark CSV file",
        nargs="?",
        default=str(Path(__file__).parent.parent / "docs" / "virseq_benchmark.csv"),
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="Output directory for results",
        default=str(Path(__file__).parent.parent / "results" / "biomni"),
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
        "--llm",
        help="LLM model for the Biomni agent (default: claude-sonnet-4-20250514)",
        default="claude-sonnet-4-20250514",
    )
    parser.add_argument(
        "--biomni-data-path",
        help="Path for Biomni data lake",
        default=str(Path(__file__).parent.parent / "data"),
    )
    parser.add_argument(
        "--gget-virus", "-gv",
        action="store_true",
        help="Enable the gget virus workflow for the analysis agent (identical queries to --gget-virus in benchmark.py)",
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
        llm=args.llm,
        biomni_data_path=args.biomni_data_path,
        use_gget_virus=args.gget_virus,
        resume_from=args.resume,
    ))


if __name__ == "__main__":
    main()
