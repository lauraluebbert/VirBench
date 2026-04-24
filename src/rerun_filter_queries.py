"""One-time script: rerun queries that use segment, vaccine_strain, or source_database filters.

Reruns all 3 runs for the matching query IDs across the 4 LLM+gget technologies,
then overwrites the existing _rerun JSON and CSV files in-place.

Usage:
    cd src/
    python rerun_filter_queries.py          # full run
    python rerun_filter_queries.py --dry    # print what would be rerun without executing
"""

import asyncio
import copy
import csv
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from utils import (
    parse_csv,
    BenchmarkResult,
    GGET_VIRUS_DOC_MD_PATH,
    NUM_RUNS,
)
from benchmark_claude import (
    run_single_benchmark as claude_run,
    SYSTEM_PROMPT as CLAUDE_SYSTEM_PROMPT,
)
from benchmark_gpt import (
    run_single_benchmark as gpt_run,
    SYSTEM_PROMPT as GPT_SYSTEM_PROMPT,
)
from benchmark_edison_analysis import run_single_benchmark as edison_run
from benchmark_biomni import run_single_benchmark as biomni_run

ROOT = Path(__file__).resolve().parent.parent

# ── Identify target query IDs ──────────────────────────────────────────────

BENCHMARK_CSV = ROOT / "docs" / "virseq_benchmark.csv"


def get_target_query_ids() -> set[int]:
    """Return query IDs where segment, vaccine_strain, or source_database is set."""
    ids = set()
    with open(BENCHMARK_CSV) as f:
        for row in csv.DictReader(f):
            if (
                row.get("segment", "").strip()
                or row.get("vaccine_strain", "").strip()
                or row.get("source_database", "").strip()
            ):
                ids.add(int(row["query_id"]))
    return ids


# ── File paths for the 4 LLM+gget rerun files ─────────────────────────────

RERUN_FILES = {
    # "claude": {
    #     "json": ROOT / "results/claude/claude_benchmark_report_gv_20260227_112923_rerun_20260302_123424.json",
    #     "csv":  ROOT / "results/claude/claude_benchmark_summary_gv_20260227_112923_rerun_20260302_123424.csv",
    # },
    # "gpt": {
    #     "json": ROOT / "results/gpt/gpt_benchmark_report_gv_20260227_114331_rerun_20260302_214334.json",
    #     "csv":  ROOT / "results/gpt/gpt_benchmark_summary_gv_20260227_114331_rerun_20260302_214334.csv",
    # },
    "edison": {
        "json": ROOT / "results/edison_analysis/benchmark_report_gv_20260227_113028_rerun_20260302_094332.json",
        "csv":  ROOT / "results/edison_analysis/benchmark_summary_gv_20260227_113028_rerun_20260302_094332.csv",
    },
    "biomni": {
        "json": ROOT / "results/biomni/biomni_benchmark_report_gv_20260227_113135_rerun_20260302_105027.json",
        "csv":  ROOT / "results/biomni/biomni_benchmark_summary_gv_20260227_113135_rerun_20260302_105027.csv",
    },
}

# ── Helpers ────────────────────────────────────────────────────────────────


def load_json_report(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_json_report(path: Path, report: dict):
    with open(path, "w") as f:
        json.dump(report, f, indent=2)


def save_csv_from_results(path: Path, results: list[dict]):
    cols = [
        "query_id", "run_number", "expected_count", "retrieved_count",
        "is_correct", "error", "duration_seconds",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for r in results:
            writer.writerow([r.get(c) for c in cols])


def build_index(results: list[dict]) -> dict[tuple[int, int], int]:
    """Map (query_id, run_number) -> list index."""
    return {
        (r["query_id"], r["run_number"]): i
        for i, r in enumerate(results)
    }


# ── Per-technology runner ──────────────────────────────────────────────────


async def rerun_tech(tech: str, target_ids: set[int], configs_by_id: dict, dry: bool):
    paths = RERUN_FILES[tech]
    json_path = paths["json"]
    csv_path = paths["csv"]

    if not json_path.exists():
        print(f"  [SKIP] JSON not found: {json_path}")
        return

    report = load_json_report(json_path)
    results = report["results"]
    idx_map = build_index(results)

    # Collect (query_id, run_number, list_index) to rerun
    to_rerun = []
    for qid in sorted(target_ids):
        for run_num in range(1, NUM_RUNS + 1):
            key = (qid, run_num)
            if key in idx_map:
                to_rerun.append((qid, run_num, idx_map[key]))
            else:
                print(f"  [WARN] ({qid}, run {run_num}) not found in {json_path.name}")

    print(f"\n{'='*60}")
    print(f"  {tech.upper()}: {len(to_rerun)} runs to rerun "
          f"({len(target_ids)} queries x {NUM_RUNS} runs)")
    print(f"  JSON: {json_path.name}")
    print(f"  CSV:  {csv_path.name}")
    print(f"{'='*60}")

    if dry:
        for qid, run_num, _ in to_rerun:
            cfg = configs_by_id.get(qid)
            print(f"  [DRY] query {qid:3d} run {run_num} — {cfg.pathogen if cfg else '?'}")
        return

    # ── Initialise clients / prompts ──

    with open(GGET_VIRUS_DOC_MD_PATH) as f:
        gget_docs = f.read()

    client = None
    agent = None
    system_prompt = None
    model = None
    gget_data_storage_uris = None

    if tech == "claude":
        model = report.get("model", "claude-sonnet-4-20250514")
        system_prompt = CLAUDE_SYSTEM_PROMPT + "\n\n--- gget virus documentation ---\n" + gget_docs

    elif tech == "gpt":
        model = report.get("model", "gpt-5.2-pro")
        system_prompt = GPT_SYSTEM_PROMPT + "\n\n--- gget virus documentation ---\n" + gget_docs

    elif tech == "edison":
        from edison_client import EdisonClient

        edison_api_key = os.getenv("EDISON_API_KEY")
        client = EdisonClient(api_key=edison_api_key)
        upload_response = await client.astore_file_content(
            name="gget_virus documentation",
            file_path=GGET_VIRUS_DOC_MD_PATH,
            description="Documentation for the gget virus Python and cli module.",
        )
        gget_data_storage_uris = [f"data_entry:{upload_response.data_storage.id}"]

    elif tech == "biomni":
        from biomni.agent import A1

        llm = report.get("llm", "claude-sonnet-4-20250514")
        agent = A1(path=str(Path(__file__).parent.parent / "data"), llm=llm)

    # ── Run each query ──

    updated_results = list(results)

    for qid, run_num, idx in to_rerun:
        config = configs_by_id.get(qid)
        if config is None:
            print(f"  [SKIP] query {qid} not in benchmark CSV")
            continue

        print(f"  Rerunning query {qid} run {run_num} ({config.pathogen})...", flush=True)

        if tech == "claude":
            result = await claude_run(
                config, run_num, model=model,
                system_prompt=system_prompt, use_gget_virus=True,
            )
        elif tech == "gpt":
            result = await gpt_run(
                config, run_num, model=model,
                system_prompt=system_prompt, use_gget_virus=True,
            )
        elif tech == "biomni":
            result = await biomni_run(
                agent, config, run_num, use_gget_virus=True,
            )
        elif tech == "edison":
            result = await edison_run(
                client, config, run_num, use_gget_virus=True,
                gget_data_storage_uris=gget_data_storage_uris,
            )

        status = "PASS" if result.is_correct else ("ERROR" if result.error else "FAIL")
        print(f"    -> Retrieved={result.retrieved_count}, "
              f"Expected={result.expected_count} [{status}]")

        updated_results[idx] = asdict(result)

    # ── Recompute summary and save ──

    correct = sum(1 for r in updated_results if r.get("is_correct", False))
    total = len(updated_results)

    updated_report = copy.deepcopy(report)
    updated_report["results"] = updated_results
    updated_report["summary"] = {
        "correct_runs": correct,
        "incorrect_runs": total - correct,
        "accuracy": correct / total if total else 0,
    }

    save_json_report(json_path, updated_report)
    save_csv_from_results(csv_path, updated_results)
    print(f"  Saved {json_path.name} and {csv_path.name} "
          f"(accuracy: {correct}/{total} = {correct/total:.2%})")


# ── Main ───────────────────────────────────────────────────────────────────


async def main():
    dry = "--dry" in sys.argv

    target_ids = get_target_query_ids()
    print(f"Target query IDs ({len(target_ids)}): {sorted(target_ids)}")

    configs = parse_csv(str(BENCHMARK_CSV))
    configs_by_id = {c.query_id: c for c in configs}

    # for tech in ["claude", "gpt", "biomni", "edison"]:
    for tech in ["biomni", "edison"]:
        await rerun_tech(tech, target_ids, configs_by_id, dry)

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
