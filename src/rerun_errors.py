"""Rerun errored benchmark queries.

Identifies runs that errored out in previous benchmark reports,
reruns only those queries, and produces new .csv and .json files
where the errored runs are replaced with the new results.

A consolidated rerun report is saved to results/rerun_errors_report_<timestamp>.json.

Usage:
    python src/rerun_errors.py
    python src/rerun_errors.py --report path/to/specific_report.json
"""

import argparse
import asyncio
import copy
import csv
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime
from glob import glob
from pathlib import Path

from dotenv import load_dotenv

from utils import (
    parse_csv,
    build_query,
    extract_count_from_response,
    BenchmarkResult,
    QueryConfig,
    NUM_RUNS,
    GGET_VIRUS_DOC_MD_PATH,
)
from benchmark_edison_analysis import run_single_benchmark as edison_run_single_benchmark
from benchmark_biomni import run_single_benchmark as biomni_run_single_benchmark
from benchmark_claude import run_single_benchmark as claude_run_single_benchmark
from benchmark_gpt import run_single_benchmark as gpt_run_single_benchmark
from benchmark_gget_virus import (
    parse_csv as gget_parse_csv,
    run_gget_virus,
)

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = str(ROOT_DIR / "docs" / "virseq_benchmark.csv")

SOURCES = {
    "Edison Analysis": ("results/edison_analysis/benchmark_report_*.json", "_gv_"),
    "Edison Analysis + gget": ("results/edison_analysis/benchmark_report_gv_*.json", None),
    "Biomni": ("results/biomni/biomni_benchmark_report_*.json", "_gv_"),
    "Biomni + gget": ("results/biomni/biomni_benchmark_report_gv_*.json", None),
    "gget virus": ("results/gget_virus/gget_direct_benchmark_summary_*.csv", None),
    "Claude Sonnet 4": ("results/claude/claude_benchmark_report_*.json", "_gv_"),
    "Claude Sonnet 4 + gget": ("results/claude/claude_benchmark_report_gv_*.json", None),
    "GPT-5.2-pro": ("results/gpt/gpt_benchmark_report_*.json", "_gv_"),
    "GPT-5.2-pro + gget": ("results/gpt/gpt_benchmark_report_gv_*.json", None),
}

# Map from technology label to filename markers
TECH_MARKERS = {
    "claude": "claude",
    "gpt": "gpt",
    "biomni": "biomni",
    "gget_virus": "gget_direct",
    "edison": None,  # default / fallback
}


def get_report_paths():
    """Discover all report paths from SOURCES."""
    report_paths = []
    for pattern, exclude in SOURCES.values():
        files = sorted(glob(str(ROOT_DIR / pattern)))
        for f in files:
            if exclude and exclude in f:
                continue
            report_paths.append(os.path.abspath(f))
    return report_paths


def detect_technology(report_path: str) -> str:
    """Detect which technology a report belongs to based on its filename."""
    name = Path(report_path).name.lower()
    for tech, marker in TECH_MARKERS.items():
        if marker and marker in name:
            return tech
    return "edison"


def is_error_result(r):
    """A run 'errored' if the error field is set, retrieved_count is missing/non-integer, or result is empty."""
    if not r or r == {} or r == "" or r is None:
        return True
    error_val = r.get("error")
    if error_val not in (None, False, ""):
        return True
    rc = r.get("retrieved_count")
    if rc is None:
        return True
    if isinstance(rc, int):
        return False
    if isinstance(rc, str):
        if rc.isdigit() or (rc.startswith("-") and rc[1:].isdigit()):
            return False
        return True
    return True


def load_report(report_path: str) -> tuple:
    """Load results from a JSON report or gget_virus CSV. Returns (results_list, metadata_dict)."""
    path = Path(report_path)
    if path.suffix == ".json":
        with open(path, "r") as f:
            report = json.load(f)
        return report["results"], report
    elif path.suffix == ".csv":
        results = []
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(
                    {
                        "query_id": int(row["query_id"]),
                        "run_number": int(row["run_number"]),
                        "expected_count": int(row["expected_count"]),
                        "retrieved_count": (
                            int(row["retrieved_count"])
                            if row.get("retrieved_count") not in (None, "")
                            else None
                        ),
                        "is_correct": row.get("is_correct", "").strip().lower()
                        == "true",
                        "error": (
                            row.get("error")
                            if row.get("error") not in (None, "")
                            else None
                        ),
                        "duration_seconds": (
                            float(row["duration_seconds"])
                            if row.get("duration_seconds")
                            else None
                        ),
                    }
                )
        return results, {"results": results}
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


def inspect_reports(report_paths):
    """Print error summary for each report and return structured data."""
    summaries = []
    for report_path in report_paths:
        tech = detect_technology(report_path)
        results, _ = load_report(report_path)
        use_gget = "_gv_" in Path(report_path).name
        error_results = [r for r in results if is_error_result(r)]
        correct = sum(1 for r in results if r.get("is_correct", False))
        total = len(results)

        summary = {
            "report_path": report_path,
            "technology": tech,
            "gget_virus": use_gget if tech != "gget_virus" else None,
            "total_runs": total,
            "correct_runs": correct,
            "error_runs": len(error_results),
            "incorrect_runs": total - correct - len(error_results),
            "accuracy": correct / total if total else 0,
            "errored_queries": [
                {
                    "query_id": r.get("query_id"),
                    "run_number": r.get("run_number"),
                    "retrieved_count": r.get("retrieved_count"),
                    "error": str(r.get("error", ""))[:200],
                }
                for r in error_results
            ],
        }
        summaries.append(summary)

        print(f"\n{'='*60}")
        print(f"Report : {report_path}")
        print(f"Tech   : {tech}")
        if tech != "gget_virus":
            print(f"gget   : {use_gget}")
        print(f"Total  : {total} runs")
        print(f"Correct: {correct} runs")
        print(f"Errors : {len(error_results)} runs")

        if error_results:
            print("\nErrored runs:")
            for r in error_results:
                err_display = r.get("error", "<empty/None>") or "<empty/None>"
                rc_display = r.get("retrieved_count", "<missing>")
                print(
                    f"  query_id={r.get('query_id', '?')}, run={r.get('run_number', '?')}, "
                    f"retrieved_count={rc_display}, error={str(err_display)[:80]}"
                )
        else:
            print("  No errors found -- nothing to rerun.")

    return summaries


async def rerun_errors_for_report(
    report_path: str, config_by_id: dict, gget_config_by_id: dict
) -> dict:
    """Rerun errored queries in a report for any supported technology.

    Returns a dict with after-rerun statistics.
    """
    tech = detect_technology(report_path)
    results, full_report = load_report(report_path)

    error_indices = [i for i, r in enumerate(results) if is_error_result(r)]

    if not error_indices:
        print(f"No errors in {report_path} -- skipping.")
        return {
            "report_path": report_path,
            "technology": tech,
            "errors_before": 0,
            "errors_after": 0,
            "rerun_results": [],
            "skipped": True,
        }

    use_gget = full_report.get("gget_virus", "_gv_" in Path(report_path).name)

    print(
        f"\nRerunning {len(error_indices)} errored run(s) "
        f"for {tech} (gget_virus={use_gget})...\n"
    )

    # --- Initialise agent / client per technology ---
    system_prompt = None
    model = None

    if tech == "claude":
        from benchmark_claude import SYSTEM_PROMPT as CLAUDE_SYSTEM_PROMPT

        model = full_report.get("model", "claude-sonnet-4-20250514")
        system_prompt = CLAUDE_SYSTEM_PROMPT
        if use_gget:
            with open(GGET_VIRUS_DOC_MD_PATH, "r") as f:
                system_prompt += "\n\n--- gget virus documentation ---\n" + f.read()

    elif tech == "gpt":
        from benchmark_gpt import SYSTEM_PROMPT as GPT_SYSTEM_PROMPT

        model = full_report.get("model", "gpt-5.2-pro")
        system_prompt = GPT_SYSTEM_PROMPT
        if use_gget:
            with open(GGET_VIRUS_DOC_MD_PATH, "r") as f:
                system_prompt += "\n\n--- gget virus documentation ---\n" + f.read()

    elif tech == "biomni":
        from biomni.agent import A1

        llm = full_report.get("llm", "claude-sonnet-4-20250514")
        agent = A1(path="./data", llm=llm)

    elif tech == "edison":
        from edison_client import EdisonClient, JobNames

        edison_api_key = os.getenv("EDISON_API_KEY")
        client = EdisonClient(api_key=edison_api_key)
        gget_data_storage_uris = None
        if use_gget:
            upload_response = await client.astore_file_content(
                name="gget_virus documentation",
                file_path=GGET_VIRUS_DOC_MD_PATH,
                description="Documentation for the gget virus Python and cli module.",
            )
            data_storage_id = upload_response.data_storage.id
            gget_data_storage_uris = [f"data_entry:{data_storage_id}"]

    elif tech == "gget_virus":
        pass  # no initialisation needed

    # --- Rerun each errored result ---
    updated_results = list(results)
    rerun_details = []

    for idx in error_indices:
        old = results[idx]
        qid = old.get("query_id")
        run_num = old.get("run_number")

        if tech == "gget_virus":
            gget_cfg = gget_config_by_id.get(qid)
            if gget_cfg is None:
                print(
                    f"  WARNING: query_id {qid} not found in virseq_benchmark.csv -- skipping."
                )
                continue

            print(
                f"  Rerunning query_id={qid}, run={run_num} "
                f"({gget_cfg['pathogen']})..."
            )

            tmpdir = tempfile.mkdtemp(prefix=f"gget_rerun_q{qid}_r{run_num}_")
            retrieved_count = None
            error = None
            start = time.time()
            try:
                retrieved_count, summary_error = run_gget_virus(
                    gget_cfg["tax_id"],
                    gget_cfg["gget_kwargs"],
                    tmpdir,
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
                and retrieved_count == gget_cfg["expected_count"]
            )

            new_result_dict = {
                "query_id": qid,
                "run_number": run_num,
                "expected_count": gget_cfg["expected_count"],
                "retrieved_count": retrieved_count,
                "is_correct": is_correct,
                "error": error,
                "duration_seconds": duration,
            }

            status = (
                "PASS"
                if is_correct
                else ("ERROR (again)" if error else "FAIL")
            )
            print(
                f"    -> Retrieved={retrieved_count}, "
                f"Expected={gget_cfg['expected_count']} [{status}]"
            )

            updated_results[idx] = new_result_dict
            rerun_details.append({
                "query_id": qid,
                "run_number": run_num,
                "status": status,
                "still_error": is_error_result(new_result_dict),
                "retrieved_count": retrieved_count,
                "expected_count": gget_cfg["expected_count"],
                "error": str(error)[:200] if error else None,
                "duration_seconds": round(duration, 2),
            })

        else:
            config = config_by_id.get(qid)
            if config is None:
                print(
                    f"  WARNING: query_id {qid} not found in virseq_benchmark.csv -- skipping."
                )
                continue

            print(
                f"  Rerunning query_id={qid}, run={run_num} "
                f"({config.pathogen})..."
            )

            if tech == "claude":
                new_result = await claude_run_single_benchmark(
                    config,
                    run_num,
                    model=model,
                    system_prompt=system_prompt,
                    use_gget_virus=use_gget,
                )
            elif tech == "gpt":
                new_result = await gpt_run_single_benchmark(
                    config,
                    run_num,
                    model=model,
                    system_prompt=system_prompt,
                    use_gget_virus=use_gget,
                )
            elif tech == "biomni":
                new_result = await biomni_run_single_benchmark(
                    agent,
                    config,
                    run_num,
                    use_gget_virus=use_gget,
                )
            elif tech == "edison":
                new_result = await edison_run_single_benchmark(
                    client,
                    config,
                    run_num,
                    use_gget_virus=use_gget,
                    gget_data_storage_uris=gget_data_storage_uris,
                )

            status = "PASS" if new_result.is_correct else "FAIL"
            if new_result.error:
                status = "ERROR (again)"
            print(
                f"    -> Retrieved={new_result.retrieved_count}, "
                f"Expected={new_result.expected_count} [{status}]"
            )

            new_result_dict = asdict(new_result)
            updated_results[idx] = new_result_dict
            rerun_details.append({
                "query_id": qid,
                "run_number": run_num,
                "status": status,
                "still_error": is_error_result(new_result_dict),
                "retrieved_count": new_result.retrieved_count,
                "expected_count": new_result.expected_count,
                "error": str(new_result.error)[:200] if new_result.error else None,
                "duration_seconds": round(new_result.duration_seconds, 2) if new_result.duration_seconds else None,
            })

    # --- Recompute summary ---
    correct = sum(1 for r in updated_results if r.get("is_correct", False))
    errors_after = sum(1 for r in updated_results if is_error_result(r))
    total = len(updated_results)
    incorrect = total - correct - errors_after
    accuracy = correct / total if total else 0

    # --- Write new output files ---
    orig = Path(report_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_csv_path = None

    if orig.suffix == ".json":
        updated_report = copy.deepcopy(full_report)
        updated_report["results"] = updated_results
        updated_report["summary"] = {
            "correct_runs": correct,
            "incorrect_runs": incorrect,
            "error_runs": errors_after,
            "accuracy": accuracy,
        }

        new_json_name = orig.stem + f"_rerun_{timestamp}.json"
        new_csv_name = (
            orig.stem.replace("report", "summary") + f"_rerun_{timestamp}.csv"
        )
        new_json_path = orig.parent / new_json_name
        new_csv_path = orig.parent / new_csv_name

        with open(new_json_path, "w") as f:
            json.dump(updated_report, f, indent=2)
        print(f"\n  New JSON: {new_json_path}")

    else:
        # CSV-only output (gget_virus)
        new_csv_name = orig.stem + f"_rerun_{timestamp}.csv"
        new_csv_path = orig.parent / new_csv_name

    with open(new_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "query_id",
                "run_number",
                "expected_count",
                "retrieved_count",
                "is_correct",
                "error",
                "duration_seconds",
            ]
        )
        for r in updated_results:
            writer.writerow(
                [
                    r.get("query_id"),
                    r.get("run_number"),
                    r.get("expected_count"),
                    r.get("retrieved_count"),
                    r.get("is_correct"),
                    r.get("error"),
                    r.get("duration_seconds"),
                ]
            )

    print(f"  New CSV : {new_csv_path}")
    print(
        f"  Summary : {correct}/{total} correct ({accuracy:.2%}), "
        f"{errors_after} remaining errors"
    )

    return {
        "report_path": report_path,
        "technology": tech,
        "gget_virus": use_gget,
        "skipped": False,
        "errors_before": len(error_indices),
        "errors_after": errors_after,
        "total_runs": total,
        "correct_after": correct,
        "incorrect_after": incorrect,
        "accuracy_after": round(accuracy, 4),
        "output_csv": str(new_csv_path),
        "rerun_details": rerun_details,
    }


def write_rerun_report(before_summaries, after_summaries, timestamp):
    """Write a consolidated rerun report to results/rerun_errors_report_<timestamp>.json."""
    report_dir = ROOT_DIR / "results"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"rerun_errors_report_{timestamp}.json"

    # Build per-technology summary by matching before/after on report_path
    after_by_path = {s["report_path"]: s for s in after_summaries}
    technologies = []
    for before in before_summaries:
        path = before["report_path"]
        after = after_by_path.get(path, {})
        technologies.append({
            "report_path": path,
            "technology": before["technology"],
            "gget_virus": before["gget_virus"],
            "total_runs": before["total_runs"],
            "before": {
                "correct": before["correct_runs"],
                "incorrect": before["incorrect_runs"],
                "errors": before["error_runs"],
                "accuracy": round(before["accuracy"], 4),
            },
            "after": {
                "correct": after.get("correct_after"),
                "incorrect": after.get("incorrect_after"),
                "errors": after.get("errors_after"),
                "accuracy": after.get("accuracy_after"),
            } if not after.get("skipped", True) else "no errors — skipped",
            "rerun_details": after.get("rerun_details", []),
            "output_csv": after.get("output_csv"),
        })

    total_errors_before = sum(b["error_runs"] for b in before_summaries)
    total_errors_after = sum(
        a.get("errors_after", 0) for a in after_summaries if not a.get("skipped")
    )

    report = {
        "timestamp": timestamp,
        "summary": {
            "total_reports_processed": len(before_summaries),
            "total_errors_before": total_errors_before,
            "total_errors_after": total_errors_after,
            "total_errors_resolved": total_errors_before - total_errors_after,
        },
        "technologies": technologies,
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Rerun report saved to: {report_path}")
    print(f"  Total errors before: {total_errors_before}")
    print(f"  Total errors after:  {total_errors_after}")
    print(f"  Errors resolved:     {total_errors_before - total_errors_after}")

    return report_path


async def main(report_paths: list[str] | None = None):
    # Parse query configs
    all_configs = parse_csv(CSV_PATH)
    config_by_id = {c.query_id: c for c in all_configs}

    # gget_virus configs
    gget_configs = gget_parse_csv(CSV_PATH)
    gget_config_by_id = {c["query_id"]: c for c in gget_configs}

    if report_paths is None:
        report_paths = get_report_paths()

    print(f"Found {len(report_paths)} report(s).")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Show error summary (before rerun)
    before_summaries = inspect_reports(report_paths)

    # Rerun errors
    after_summaries = []
    for rp in report_paths:
        result = await rerun_errors_for_report(rp, config_by_id, gget_config_by_id)
        after_summaries.append(result)

    # Write consolidated rerun report
    write_rerun_report(before_summaries, after_summaries, timestamp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rerun errored benchmark queries."
    )
    parser.add_argument(
        "--report",
        type=str,
        nargs="*",
        default=None,
        help="Path(s) to specific report file(s) to rerun. "
        "If not provided, all reports are discovered automatically.",
    )
    args = parser.parse_args()

    asyncio.run(main(report_paths=args.report))
