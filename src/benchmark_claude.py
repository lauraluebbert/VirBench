#!/usr/bin/env python3
"""
Claude Sonnet 4 Benchmark

Benchmarks Claude Sonnet 4 (via Anthropic Messages API with tool use) on
viral sequence count retrieval from NCBI.

The model is given:
- web_search (server-side tool) so it can look up the best API/library.
- execute_python (client-side custom tool) so it can write and run Python locally.

SECURITY NOTE:
Local Python execution has NETWORK ACCESS, but this script enforces a DOMAIN
ALLOWLIST for all outbound connections from the execute_python subprocess to
reduce risk.

Usage:
    python benchmark_claude.py                          # run all queries
    python benchmark_claude.py --test                   # first query only
    python benchmark_claude.py --use-gget-virus         # include gget docs
    python benchmark_claude.py --model claude-sonnet-4-6

K-Dense scientific skills (optional):
    The --kdense flag loads curated scientific skill documentation from the
    K-Dense claude-scientific-skills repository into the system prompt,
    giving the model detailed API references and code examples for tools
    like BioPython and gget.

    --kdense DIR            Path to the scientific-skills directory.
    --kdense-skills s1,s2   Comma-separated skill names to load (default: all).
                            Recommended for this benchmark: biopython,gget
    --kdense-refs           Also include each skill's references/*.md and
                            scripts/*.py files (detailed API docs and examples).
"""

import asyncio
import csv
import glob as glob_module
import json
import os
import subprocess
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv

import re

from utils import (
    parse_csv,
    build_query,
    BenchmarkResult,
    QueryConfig,
    NUM_RUNS,
    GGET_VIRUS_DOC_MD_PATH,
    DOMAIN_ALLOWLIST,
    load_completed_runs_from_json,
)

load_dotenv(override=True)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("Please set your ANTHROPIC_API_KEY in the .env file")

DEFAULT_MODEL = "claude-sonnet-4-20250514"


# -----------------------------------------------------------------------------
# System prompt + tools
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a bioinformatics agent.\n"
    "You have access to:\n"
    "- web_search: search the internet for up-to-date information.\n"
    "- execute_python: run Python LOCALLY (WITH internet) to call APIs, install packages, "
    "and compute results. Note: outbound network access is restricted to an allowlist of domains.\n\n"
    "Workflow:\n"
    "1) If you need to choose the best API/library, use web_search first.\n"
    "2) Then use execute_python to implement the call and compute the final answer.\n\n"
    "Output rules:\n"
    "- When you have the final count, respond with exactly one integer on its own line.\n"
    "- After you output the final integer, also output a JSON block with keys: "
    "methods (eg APIs used) and reasoning (max 3 bullets). Keep it short.\n"
)

# When web_search is disabled (e.g. --no-web-search for claude-sonnet-4-6 compatibility)
SYSTEM_PROMPT_NO_WEB_SEARCH = (
    "You are a bioinformatics agent.\n"
    "You have access to execute_python: run Python LOCALLY (WITH internet) to call APIs, "
    "install packages, and compute results. Outbound network access is restricted to an allowlist of domains.\n\n"
    "Use execute_python to implement the API calls and compute the final answer.\n\n"
    "Output rules:\n"
    "- When you have the final count, respond with exactly one integer on its own line.\n"
    "- After you output the final integer, also output a JSON block with keys: "
    "methods (eg APIs used) and reasoning (max 3 bullets). Keep it short.\n"
)

WEB_SEARCH_TOOL = {
    "type": "web_search_20250305", # Using the old web_search tool type to avoid auto-adding an unsupported code_execution version when web_search is present.
    "name": "web_search",
    "max_uses": 5,
}

# Our custom tool for LOCAL code execution (has internet access, but restricted by allowlist).
EXECUTE_PYTHON_TOOL = {
    "type": "custom",
    "name": "execute_python",
    "description": (
        "Execute Python code LOCALLY and return stdout/stderr. Use this to run scripts that "
        "query APIs, install packages, or process data. The code runs in a local subprocess "
        "with a 120-second timeout. Outbound network access is restricted to an allowlist of domains."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute.",
            }
        },
        "required": ["code"],
    },
}


def _load_kdense_skills(
    skills_dir: str,
    skill_names: Optional[list[str]] = None,
    include_refs: bool = False,
) -> str:
    """Load K-Dense scientific skill files from the scientific-skills directory."""
    if skill_names:
        skill_dirs = []
        for name in skill_names:
            skill_path = os.path.join(skills_dir, name)
            if not os.path.isdir(skill_path):
                raise FileNotFoundError(f"Skill '{name}' not found in {skills_dir}")
            skill_dirs.append(skill_path)
    else:
        skill_files = sorted(
            glob_module.glob(os.path.join(skills_dir, "**", "SKILL.md"), recursive=True)
        )
        if not skill_files:
            raise FileNotFoundError(
                f"No SKILL.md files found in {skills_dir}. "
                "Expected the path to the K-Dense scientific-skills directory."
            )
        skill_dirs = [os.path.dirname(f) for f in skill_files]

    parts = []
    total_chars = 0
    for skill_dir in sorted(skill_dirs):
        skill_name = os.path.basename(skill_dir)
        skill_md = os.path.join(skill_dir, "SKILL.md")
        if not os.path.isfile(skill_md):
            print(f"  Warning: {skill_name}/ has no SKILL.md, skipping")
            continue

        with open(skill_md, "r") as f:
            content = f"### Skill: {skill_name}\n{f.read()}"

        if include_refs:
            refs_dir = os.path.join(skill_dir, "references")
            if os.path.isdir(refs_dir):
                for ref_file in sorted(os.listdir(refs_dir)):
                    if not ref_file.endswith(".md"):
                        continue
                    ref_path = os.path.join(refs_dir, ref_file)
                    if os.path.isfile(ref_path):
                        with open(ref_path, "r") as f:
                            content += f"\n\n#### Reference: {ref_file}\n{f.read()}"

            scripts_dir = os.path.join(skill_dir, "scripts")
            if os.path.isdir(scripts_dir):
                for script_file in sorted(os.listdir(scripts_dir)):
                    if not script_file.endswith(".py"):
                        continue
                    script_path = os.path.join(scripts_dir, script_file)
                    if os.path.isfile(script_path):
                        with open(script_path, "r") as f:
                            content += (
                                f"\n\n#### Script: {script_file}\n"
                                f"```python\n{f.read()}\n```"
                            )

        parts.append(content)
        total_chars += len(content)

    if not parts:
        raise FileNotFoundError(f"No valid skills found in {skills_dir}.")

    refs_note = " (including references and scripts)" if include_refs else ""
    print(f"Loaded {len(parts)} K-Dense skills ({total_chars:,} chars){refs_note}")
    return "\n\n".join(parts)


def _parse_integer_response(text: str) -> Optional[int]:
    """Parse an integer from the model's final response."""
    if not text:
        return None
    stripped = text.strip()
    try:
        return int(stripped)
    except ValueError:
        pass
    matches = re.findall(r"\b(\d+)\b", stripped)
    if matches:
        return int(matches[-1])
    return None


def _extract_json_block(text: str) -> tuple[str, Optional[dict]]:
    """Extract a trailing JSON metadata block from the model's response.

    Returns (text_without_block, parsed_dict_or_None).
    """
    if not text:
        return text or "", None
    # Try fenced ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            return text[: match.start()].strip(), data
        except json.JSONDecodeError:
            pass
    # Try bare JSON object containing "methods" key
    match = re.search(r'(\{[^{}]*"methods".*?\})\s*$', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            return text[: match.start()].strip(), data
        except json.JSONDecodeError:
            pass
    return text, None


def _domain_guard_prelude(allowlist: list[str]) -> str:
    """
    Python prelude that blocks outbound network connections except to allowlisted domains.

    This is a pragmatic safety measure:
    - Blocks direct IP connections
    - Allows exact domain matches and subdomains of allowlisted entries
    - Hooks common socket-level functions so it affects requests/urllib/httpx/etc.
    """
    allowlist_json = json.dumps([d.lower() for d in allowlist])
    return f"""
# --- DOMAIN GUARD (auto-injected) ---
import socket, ipaddress
from urllib.parse import urlparse

__ALLOWLIST = set({allowlist_json})

def __is_allowed_host(host: str) -> bool:
    if not host:
        return False
    host = host.strip().lower().rstrip(".")
    # Block raw IP connections by default
    try:
        ipaddress.ip_address(host)
        return False
    except Exception:
        pass
    if host in __ALLOWLIST:
        return True
    # Allow subdomains of allowlisted domains
    for d in __ALLOWLIST:
        if host.endswith("." + d):
            return True
    return False

__orig_getaddrinfo = socket.getaddrinfo
def __guarded_getaddrinfo(host, port, *args, **kwargs):
    if host and isinstance(host, str) and not __is_allowed_host(host):
        raise PermissionError(f"Blocked network access to host: {{host}} (not in allowlist)")
    return __orig_getaddrinfo(host, port, *args, **kwargs)

socket.getaddrinfo = __guarded_getaddrinfo

__orig_create_connection = socket.create_connection
def __guarded_create_connection(address, *args, **kwargs):
    host, port = address[0], address[1]
    if host and isinstance(host, str) and not __is_allowed_host(host):
        raise PermissionError(f"Blocked network access to host: {{host}} (not in allowlist)")
    return __orig_create_connection(address, *args, **kwargs)

socket.create_connection = __guarded_create_connection

# --- END DOMAIN GUARD ---
"""


def _execute_python(code: str) -> str:
    """Run Python code in a subprocess and return stdout + stderr (network restricted)."""
    guarded_code = _domain_guard_prelude(DOMAIN_ALLOWLIST) + "\n\n" + code

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write(guarded_code)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["python", tmp_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n" if output else "") + result.stderr
        if not output:
            output = "(no output)"
        return output[:10_000]
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out after 120 seconds."
    finally:
        os.unlink(tmp_path)


def _tools_list(use_web_search: bool) -> list:
    """Build tools list: always include execute_python; optionally web_search."""
    if use_web_search:
        return [EXECUTE_PYTHON_TOOL, WEB_SEARCH_TOOL]
    return [EXECUTE_PYTHON_TOOL]


async def run_claude_agent(
    query: str,
    model: str,
    system_prompt: str,
    max_turns: int = 15,
    use_web_search: bool = True,
) -> tuple[str, int]:
    """
    Run the Claude agentic loop with tool use.

    Handles:
    - Server-side tools (web_search): may require 'pause_turn' continuation.
    - Client-side tool (execute_python): we must execute and return tool_result.

    Returns (final_text, tool_call_count).
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    messages = [{"role": "user", "content": query}]
    tool_call_count = 0
    tools = _tools_list(use_web_search)

    for _ in range(max_turns):
        response = await asyncio.to_thread(
            client.messages.create,
            model=model,
            max_tokens=4096,
            system=system_prompt,
            tools=tools,
            messages=messages,
        )

        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        def block_type(b):
            return getattr(b, "type", None)

        def block_text(b):
            return getattr(b, "text", "") or ""

        # Count only client tool_use blocks (server tools use server_tool_use on newer models)
        tool_call_count += sum(
            1 for block in assistant_content if block_type(block) == "tool_use"
        )

        # If Claude is done, return final text
        if response.stop_reason == "end_turn":
            text_parts = [block_text(b) for b in assistant_content if block_type(b) == "text"]
            return "\n".join(t for t in text_parts if t).strip() or "\n".join(text_parts), tool_call_count

        # Server tools may require continuation turns (Anthropic may return pause_turn)
        if response.stop_reason == "pause_turn":
            continue

        # Unexpected stop_reason — return text if any; else continue (e.g. server tool turn with no text yet)
        if response.stop_reason != "tool_use":
            text_parts = [block_text(b) for b in assistant_content if block_type(b) == "text"]
            if text_parts:
                return ("\n".join(t for t in text_parts if t).strip(), tool_call_count)
            # No text (e.g. server_tool_use only): continue to next turn like pause_turn
            continue

        # Dispatch ONLY client tool_use (execute_python). Ignore server_tool_use (e.g. web_search).
        tool_results = []
        saw_execute_python = False

        for block in assistant_content:
            if block_type(block) != "tool_use":
                continue
            if getattr(block, "name", None) != "execute_python":
                continue

            saw_execute_python = True
            inp = getattr(block, "input", None) or {}
            code = inp.get("code", "") if isinstance(inp, dict) else ""
            print(f"    [tool] execute_python ({len(code)} chars)...")
            output = _execute_python(code)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": getattr(block, "id", ""),
                    "content": output,
                }
            )

        if saw_execute_python:
            messages.append({"role": "user", "content": tool_results})
        continue

    # Exhausted turns — return last text if present
    last = messages[-1]["content"]
    text_parts = []
    for block in last:
        if getattr(block, "type", None) == "text":
            text_parts.append(getattr(block, "text", "") or "")
    return ("\n".join(text_parts).strip() if text_parts else "Max turns reached without final answer."), tool_call_count


async def run_single_benchmark(
    config: QueryConfig,
    run_number: int,
    model: str,
    system_prompt: str,
    use_gget_virus: bool = False,
    use_web_search: bool = True,
) -> BenchmarkResult:
    """Run a single benchmark query."""
    query = build_query(
        config, use_gget_virus=use_gget_virus, return_integer_only=True
    )
    tool_status = " [with gget virus]" if use_gget_virus else ""
    print(
        f"  Run {run_number}: Querying for {config.pathogen} "
        f"({config.filters.get('segment', '')}){tool_status}..."
    )

    start_time = datetime.now()
    result = BenchmarkResult(
        query_id=config.query_id,
        run_number=run_number,
        expected_count=config.expected_count,
    )

    try:
        raw_response, tool_call_count = await run_claude_agent(
            query, model=model, system_prompt=system_prompt, use_web_search=use_web_search
        )
        result.duration_seconds = (datetime.now() - start_time).total_seconds()
        result.raw_response = raw_response
        result.tool_call_count = tool_call_count

        # Extract JSON metadata block before parsing the integer
        text_for_int, metadata = _extract_json_block(raw_response)
        if metadata:
            result.methods = metadata.get("methods")
            result.reasoning = metadata.get("reasoning")

        retrieved_count = _parse_integer_response(text_for_int)
        result.retrieved_count = retrieved_count

        if retrieved_count is not None:
            result.is_correct = retrieved_count == config.expected_count

    except Exception as e:
        result.error = str(e)
        result.duration_seconds = (datetime.now() - start_time).total_seconds()

    return result


async def run_benchmark(
    csv_path: str,
    output_dir: str = "results/claude",
    test_run: int = 0,
    use_gget_virus: bool = False,
    use_web_search: bool = True,
    model: str = DEFAULT_MODEL,
    kdense_dir: Optional[str] = None,
    kdense_skills: Optional[list[str]] = None,
    kdense_refs: bool = False,
    resume_from: Optional[str] = None,
):
    """Run the full benchmark suite."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    print("Parsing CSV configuration...")
    configs = parse_csv(csv_path)
    print(f"Found {len(configs)} query configurations")
    print(f"Model: {model}")
    print(f"gget virus workflow: {'ENABLED' if use_gget_virus else 'DISABLED'}")
    print(f"web_search: {'ENABLED' if use_web_search else 'DISABLED'}")
    print(f"execute_python domain allowlist: {', '.join(DOMAIN_ALLOWLIST)}")
    if kdense_dir:
        if kdense_skills:
            print(f"K-Dense skills: {', '.join(kdense_skills)}")
        else:
            print("K-Dense skills: ALL")
        if kdense_refs:
            print("K-Dense references: INCLUDED")
    else:
        print("K-Dense skills: DISABLED")

    if test_run:
        print(f"Test mode: Running first {test_run} query(ies).")
        configs = configs[:test_run]
        if not configs:
            print("No valid queries found for test run.")
            return

    # Build system prompt (with or without web_search mention), then gget docs and K-Dense skills
    system_prompt = SYSTEM_PROMPT if use_web_search else SYSTEM_PROMPT_NO_WEB_SEARCH
    if use_gget_virus:
        if not os.path.isfile(GGET_VIRUS_DOC_MD_PATH):
            raise FileNotFoundError(
                f"gget_virus_docs.md not found at {GGET_VIRUS_DOC_MD_PATH}"
            )
        with open(GGET_VIRUS_DOC_MD_PATH, "r") as f:
            gget_docs = f.read()
        system_prompt += "\n\n--- gget virus documentation ---\n" + gget_docs

    if kdense_dir:
        skills_content = _load_kdense_skills(
            kdense_dir,
            skill_names=kdense_skills,
            include_refs=kdense_refs,
        )
        system_prompt += (
            "\n\n--- K-Dense scientific skills ---\n"
            "The following scientific skills are available. Use them as "
            "reference for writing Python code to solve the task.\n\n"
            + skills_content
        )

    # Prepare incremental result files
    completed_runs: set[tuple[int, int]] = set()

    if resume_from:
        report_path = Path(resume_from)
        # Derive CSV path: _report -> _summary, .json -> .csv
        csv_report_path = Path(
            str(report_path).replace("_report", "_summary").replace(".json", ".csv")
        )
        completed_runs = load_completed_runs_from_json(str(report_path))
        print(f"Resuming: found {len(completed_runs)} completed runs in {report_path}")
        # Ensure CSV companion exists (recreate header if missing)
        if not csv_report_path.exists():
            with open(csv_report_path, "w", newline="") as cf:
                writer = csv.writer(cf)
                writer.writerow([
                    "query_id", "run_number", "expected_count", "retrieved_count",
                    "is_correct", "error", "duration_seconds", "tool_call_count",
                ])
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffixes = []
        if use_gget_virus:
            suffixes.append("gv")
        if not use_web_search:
            suffixes.append("nws")
        if kdense_dir:
            suffixes.append("kd")
        file_middle = ("_" + "_".join(suffixes)) if suffixes else ""
        report_path = output_path / f"claude_benchmark_report{file_middle}_{timestamp}.json"
        csv_report_path = (
            output_path / f"claude_benchmark_summary{file_middle}_{timestamp}.csv"
        )

        json_report_dict = {
            "timestamp": timestamp,
            "model": model,
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
            writer.writerow(
                [
                    "query_id",
                    "run_number",
                    "expected_count",
                    "retrieved_count",
                    "is_correct",
                    "error",
                    "duration_seconds",
                    "tool_call_count",
                ]
            )

    for i, config in enumerate(configs):
        print(
            f"\nQuery {config.query_id} ({i + 1}/{len(configs)}): "
            f"{config.pathogen} - {config.filters.get('segment', '')}"
        )
        print(f"  Expected count: {config.expected_count}")

        for run_num in range(1, NUM_RUNS + 1):
            if (config.query_id, run_num) in completed_runs:
                continue

            result = await run_single_benchmark(
                config,
                run_num,
                model=model,
                system_prompt=system_prompt,
                use_gget_virus=use_gget_virus,
                use_web_search=use_web_search,
            )

            status = "PASS" if result.is_correct else "FAIL"
            if result.error:
                status = "ERROR"
            print(
                f"  Run {run_num}: Retrieved={result.retrieved_count}, "
                f"Expected={result.expected_count} [{status}] "
                f"(tool_calls={result.tool_call_count})"
            )
            if result.error:
                print(f"    Error: {result.error}")

            # Append to CSV
            with open(csv_report_path, "a", newline="") as cf:
                writer = csv.writer(cf)
                writer.writerow(
                    [
                        result.query_id,
                        result.run_number,
                        result.expected_count,
                        result.retrieved_count,
                        result.is_correct,
                        result.error,
                        result.duration_seconds,
                        result.tool_call_count,
                    ]
                )

            # Append to JSON
            try:
                with open(report_path, "r") as jf:
                    json_report_dict = json.load(jf)
            except Exception:
                json_report_dict = {
                    "timestamp": timestamp,
                    "model": model,
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

            correct_runs = sum(
                1 for r in json_report_dict["results"] if r.get("is_correct", False)
            )
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

    # Final summary
    with open(report_path, "r") as jf:
        report = json.load(jf)

    print(f"\n{'=' * 60}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    print(f"Model: {model}")
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

    parser = argparse.ArgumentParser(
        description="Benchmark Claude Sonnet 4 on viral sequence retrieval"
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=str(Path(__file__).parent.parent / "docs" / "virseq_benchmark.csv"),
        help="Path to the benchmark CSV file",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=str(Path(__file__).parent.parent / "results" / "claude"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--test",
        "-t",
        nargs="?",
        const=1,
        default=0,
        type=int,
        help="Run only the first N queries for testing (default N=1)",
    )
    parser.add_argument(
        "--use-gget-virus",
        "-gv",
        action="store_true",
        help="Include gget virus documentation in the system prompt",
    )
    parser.add_argument(
        "--no-web-search",
        action="store_true",
        help="Disable web_search tool; use only execute_python.",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help=f"Anthropic model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--kdense",
        metavar="DIR",
        default=None,
        help="Path to K-Dense scientific-skills directory. "
        "Loads SKILL.md files into the system prompt.",
    )
    parser.add_argument(
        "--kdense-skills",
        default=None,
        help="Comma-separated skill names to load (e.g. biopython,gget). "
        "Requires --kdense. Default: all skills in the directory.",
    )
    parser.add_argument(
        "--kdense-refs",
        action="store_true",
        help="Also include references/*.md and scripts/*.py from each "
        "loaded skill. Requires --kdense.",
    )
    parser.add_argument(
        "--resume",
        metavar="REPORT_JSON",
        default=None,
        help="Resume from an existing JSON report, skipping completed runs",
    )

    args = parser.parse_args()

    kdense_skill_list = None
    if args.kdense_skills:
        if not args.kdense:
            parser.error("--kdense-skills requires --kdense DIR")
        kdense_skill_list = [s.strip() for s in args.kdense_skills.split(",") if s.strip()]
    if args.kdense_refs and not args.kdense:
        parser.error("--kdense-refs requires --kdense DIR")

    asyncio.run(
        run_benchmark(
            args.csv_path,
            output_dir=args.output_dir,
            test_run=args.test,
            use_gget_virus=args.use_gget_virus,
            use_web_search=not args.no_web_search,
            model=args.model,
            kdense_dir=args.kdense,
            kdense_skills=kdense_skill_list,
            kdense_refs=args.kdense_refs,
            resume_from=args.resume,
        )
    )


if __name__ == "__main__":
    main()