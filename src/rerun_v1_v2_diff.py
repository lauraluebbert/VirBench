"""One-time script: rerun GPT 5.5 queries that differ between benchmark v1 and v2.

Only reruns queries with substantive changes (expected_count or pathogen name),
not cosmetic date format changes. Reruns both the with-gget and without-gget
reports, updating only the affected runs and saving new rerun files.

Substantive changes (v1 -> v2):
    Q3:   expected_count 23 -> 22
    Q28:  pathogen "O'nyong-nyong virus (ONN)" -> "Alphavirus"
    Q36:  expected_count 126 -> 125
    Q37:  expected_count 195 -> 194
    Q38:  expected_count 267 -> 266
    Q39:  expected_count 362 -> 361
    Q54:  pathogen "O'nyong-nyong virus (ONN)" -> "Onyong-nyong virus (ONN)"
    Q56:  expected_count 738 -> 663
    Q69:  expected_count 2694 -> 2907
    Q96:  expected_count 12 -> 11
    Q114: expected_count 3226 -> 3200

Usage:
    cd src/
    python rerun_v1_v2_diff.py          # full run
    python rerun_v1_v2_diff.py --dry    # print what would be rerun without executing
"""

import asyncio
import copy
import csv
import json
import re
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from utils import (
    parse_csv,
    GGET_VIRUS_DOC_MD_PATH,
    BENCHMARK_CSV_PATH,
    NUM_RUNS,
)
from benchmark_gpt import (
    run_single_benchmark as gpt_run,
    SYSTEM_PROMPT as GPT_SYSTEM_PROMPT,
)

ROOT = Path(__file__).resolve().parent.parent

# ── Benchmark CSV paths ──────────────────────────────────────────────────

V1_CSV = ROOT / "docs" / "virbench_v1_DO_NOT_SHARE.csv"
V2_CSV = BENCHMARK_CSV_PATH

# ── GPT 5.5 report files (latest reruns) ─────────────────────────────────

REPORT_FILES = {
    "without_gget": {
        "json": ROOT / "results/gpt/gpt-5.5/gpt_benchmark_report_20260512_230151_rerun_20260518_203644_rerun_20260519_234317.json",
        "csv":  ROOT / "results/gpt/gpt-5.5/gpt_benchmark_summary_20260512_230151_rerun_20260518_203644_rerun_20260519_234317.csv",
    },
    "with_gget": {
        "json": ROOT / "results/gpt/gpt-5.5/gpt_benchmark_report_gv_20260512_230230_rerun_20260518_222018_rerun_20260520_022023.json",
        "csv":  ROOT / "results/gpt/gpt-5.5/gpt_benchmark_summary_gv_20260512_230230_rerun_20260518_222018_rerun_20260520_022023.csv",
    },
}

# ── Detect substantive v1→v2 differences ─────────────────────────────────

DATE_COLS = {
    "min_collection_date", "max_collection_date",
    "min_release_date", "max_release_date",
}


def _normalize_date(d: str) -> str:
    """Normalize M/D/YY and YYYY-MM-DD to a common form for comparison."""
    d = d.strip()
    if not d:
        return ""
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2})$", d)
    if m:
        month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        year = year + 2000 if year < 50 else year + 1900
        return f"{year:04d}-{month:02d}-{day:02d}"
    return d


def get_target_query_ids() -> set[int]:
    """Return query IDs with substantive differences between v1 and v2.

    Ignores date FORMAT changes (e.g. 2020-01-01 vs 1/1/20) that represent
    the same date. Only flags queries where expected_count, pathogen, or a
    non-date filter value actually changed.
    """
    v1 = {}
    with open(V1_CSV) as f:
        for row in csv.DictReader(f):
            v1[int(row["query_id"])] = row

    v2 = {}
    with open(V2_CSV) as f:
        for row in csv.DictReader(f):
            v2[int(row["query_id"])] = row

    ids = set()
    for qid in v1:
        if qid not in v2:
            ids.add(qid)
            continue
        for col in v1[qid]:
            val1 = v1[qid][col].strip()
            val2 = v2[qid].get(col, "").strip()
            if val1 == val2:
                continue
            if col in DATE_COLS and _normalize_date(val1) == _normalize_date(val2):
                continue
            ids.add(qid)
            break

    return ids


# ── Helpers ──────────────────────────────────────────────────────────────


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


# ── Per-variant runner ───────────────────────────────────────────────────


async def rerun_variant(
    label: str,
    target_ids: set[int],
    configs_by_id: dict,
    dry: bool,
):
    paths = REPORT_FILES[label]
    json_path = paths["json"]
    csv_path = paths["csv"]

    use_gget = label == "with_gget"

    if not json_path.exists():
        print(f"  [SKIP] JSON not found: {json_path}")
        return

    report = load_json_report(json_path)
    results = report["results"]
    idx_map = build_index(results)

    to_rerun = []
    for qid in sorted(target_ids):
        for run_num in range(1, NUM_RUNS + 1):
            key = (qid, run_num)
            if key in idx_map:
                to_rerun.append((qid, run_num, idx_map[key]))
            else:
                print(f"  [WARN] ({qid}, run {run_num}) not found in {json_path.name}")

    print(f"\n{'='*60}")
    print(f"  GPT 5.5 {'+ gget' if use_gget else '(no gget)'}: "
          f"{len(to_rerun)} runs to rerun "
          f"({len(target_ids)} queries x {NUM_RUNS} runs)")
    print(f"  JSON: {json_path.name}")
    print(f"  CSV:  {csv_path.name}")
    print(f"{'='*60}")

    if dry:
        for qid, run_num, _ in to_rerun:
            cfg = configs_by_id.get(qid)
            old = results[idx_map[(qid, run_num)]]
            print(f"  [DRY] query {qid:3d} run {run_num} — {cfg.pathogen if cfg else '?'} "
                  f"(old expected={old['expected_count']}, new expected={cfg.expected_count})")
        return

    # ── Initialise ──

    model = report.get("model", "gpt-5.5")
    system_prompt = GPT_SYSTEM_PROMPT
    if use_gget:
        with open(GGET_VIRUS_DOC_MD_PATH) as f:
            system_prompt += "\n\n--- gget virus documentation ---\n" + f.read()

    # ── Run each query ──

    updated_results = list(results)

    for qid, run_num, idx in to_rerun:
        config = configs_by_id.get(qid)
        if config is None:
            print(f"  [SKIP] query {qid} not in benchmark CSV")
            continue

        print(f"  Rerunning query {qid} run {run_num} ({config.pathogen})...", flush=True)

        result = await gpt_run(
            config, run_num, model=model,
            system_prompt=system_prompt, use_gget_virus=use_gget,
        )

        status = "PASS" if result.is_correct else ("ERROR" if result.error else "FAIL")
        print(f"    -> Retrieved={result.retrieved_count}, "
              f"Expected={result.expected_count} [{status}]")

        updated_results[idx] = asdict(result)

    # ── Recompute summary and save ──

    correct = sum(1 for r in updated_results if r.get("is_correct", False))
    total = len(updated_results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_json = json_path.parent / (json_path.stem + f"_rerun_{timestamp}.json")
    new_csv = csv_path.parent / (csv_path.stem + f"_rerun_{timestamp}.csv")

    updated_report = copy.deepcopy(report)
    updated_report["results"] = updated_results
    updated_report["summary"] = {
        "correct_runs": correct,
        "incorrect_runs": total - correct,
        "accuracy": correct / total if total else 0,
    }

    save_json_report(new_json, updated_report)
    save_csv_from_results(new_csv, updated_results)
    print(f"\n  Saved {new_json.name}")
    print(f"  Saved {new_csv.name}")
    print(f"  Accuracy: {correct}/{total} = {correct / total:.2%}")


# ── Main ─────────────────────────────────────────────────────────────────


async def main():
    dry = "--dry" in sys.argv

    target_ids = get_target_query_ids()
    print(f"Target query IDs ({len(target_ids)}): {sorted(target_ids)}")

    configs = parse_csv(str(V2_CSV))
    configs_by_id = {c.query_id: c for c in configs}

    for label in ["without_gget", "with_gget"]:
        await rerun_variant(label, target_ids, configs_by_id, dry)

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
