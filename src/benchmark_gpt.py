#!/usr/bin/env python3
"""
GPT-5.2-pro Benchmark

Benchmarks GPT-5.2-pro (via OpenAI Responses API) on viral sequence count
retrieval from NCBI. The model is given a code execution function tool so it
can write and run Python (e.g. gget, BioPython) to query NCBI.

Enforces a DOMAIN ALLOWLIST for all outbound network calls from the
execute_python subprocess (same guard pattern as for the Claude benchmark).

Usage:
    python benchmark_gpt.py                         # run all queries
    python benchmark_gpt.py --test                  # first query only
    python benchmark_gpt.py --use-gget-virus        # include gget docs
    python benchmark_gpt.py --model gpt-5.2-pro
"""

import asyncio
import csv
import json
import os
import subprocess
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import re

from dotenv import load_dotenv
from openai import OpenAI

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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set your OPENAI_API_KEY in the .env file")

DEFAULT_MODEL = "gpt-5.2-pro"

SYSTEM_PROMPT = (
    "You are a bioinformatics agent.\n"
    "You have access to:\n"
    "- web_search_preview: search the internet for up-to-date information.\n"
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

# Function tool definition for the OpenAI Responses API
EXECUTE_PYTHON_TOOL = {
    "type": "function",
    "name": "execute_python",
    "description": (
        "Execute Python code and return stdout/stderr. Use this to run "
        "scripts that query NCBI, install packages, or process data. "
        "The code runs in a fresh subprocess with a 120-second timeout. "
        "Outbound network access is restricted to an allowlist of domains."
    ),
    "parameters": {
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


def _parse_integer_response(text: str) -> Optional[int]:
    """Parse an integer from the model's final response.

    The model is instructed to respond with only the integer count.
    Falls back to extracting the last integer found in the text.
    """
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

    - Blocks direct IP connections
    - Allows exact matches and subdomains
    - Hooks socket-level resolution/connection so it affects requests/urllib/httpx/etc.
    """
    allowlist_json = json.dumps([d.lower() for d in allowlist])
    return f"""
# --- DOMAIN GUARD (auto-injected) ---
import socket, ipaddress

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
    if not isinstance(DOMAIN_ALLOWLIST, (list, tuple)) or not DOMAIN_ALLOWLIST:
        raise ValueError(
            "DOMAIN_ALLOWLIST must be a non-empty list/tuple imported from utils.py"
        )

    guarded_code = _domain_guard_prelude(list(DOMAIN_ALLOWLIST)) + "\n\n" + code

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


async def run_gpt_agent(
    query: str,
    model: str,
    system_prompt: str,
    max_turns: int = 15,
) -> tuple[str, int]:
    """Run the GPT agent loop with tool execution via the Responses API.

    Returns (final_text, tool_call_count).
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    tool_call_count = 0

    tools = [EXECUTE_PYTHON_TOOL, {"type": "web_search_preview"}]

    # 1) Initial request
    response = await asyncio.to_thread(
        client.responses.create,
        model=model,
        instructions=system_prompt,
        input=query,
        tools=tools,
    )

    # 2) Tool loop
    for _ in range(max_turns):
        # Count tool calls in this response (function_call + web_search_call)
        tool_call_count += sum(
            1 for item in response.output
            if item.type in ("function_call", "web_search_call")
        )

        tool_calls = [item for item in response.output if item.type == "function_call"]

        if tool_calls:
            tool_outputs = []
            for call in tool_calls:
                if getattr(call, "name", None) != "execute_python":
                    tool_outputs.append(
                        {
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": f"Error: Unsupported function tool '{getattr(call, 'name', None)}'",
                        }
                    )
                    continue

                args = json.loads(call.arguments) if isinstance(call.arguments, str) else call.arguments
                code = (args or {}).get("code", "")
                print(f"    [tool] executing code ({len(code)} chars)...")
                output = _execute_python(code)
                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": output,
                    }
                )

            # Continue from prior response state
            response = await asyncio.to_thread(
                client.responses.create,
                model=model,
                instructions=system_prompt,
                previous_response_id=response.id,
                input=tool_outputs,
                tools=tools,
            )
            continue

        # No tool calls: return final text
        if getattr(response, "output_text", None):
            return response.output_text, tool_call_count

        # Fallback extraction if output_text absent
        text_parts = []
        for item in getattr(response, "output", []):
            if item.type == "message":
                for content in getattr(item, "content", []):
                    if hasattr(content, "text") and content.text:
                        text_parts.append(content.text)
        if text_parts:
            return "\n".join(text_parts), tool_call_count

        return "No final text output.", tool_call_count

    return "Max turns reached without final answer.", tool_call_count


async def run_single_benchmark(
    config: QueryConfig,
    run_number: int,
    model: str,
    system_prompt: str,
    use_gget_virus: bool = False,
) -> BenchmarkResult:
    """Run a single benchmark query."""
    query = build_query(config, use_gget_virus=use_gget_virus, return_integer_only=True)
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
        raw_response, tool_call_count = await run_gpt_agent(
            query, model=model, system_prompt=system_prompt
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
    output_dir: str = "results/gpt",
    test_run: int = 0,
    use_gget_virus: bool = False,
    model: str = DEFAULT_MODEL,
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
    print(f"execute_python domain allowlist: {', '.join(DOMAIN_ALLOWLIST)}")

    if test_run:
        print(f"Test mode: Running first {test_run} query(ies).")
        configs = configs[:test_run]
        if not configs:
            print("No valid queries found for test run.")
            return

    # Build system prompt, optionally including gget docs
    system_prompt = SYSTEM_PROMPT
    if use_gget_virus:
        if not os.path.isfile(GGET_VIRUS_DOC_MD_PATH):
            raise FileNotFoundError(f"gget_virus_docs.md not found at {GGET_VIRUS_DOC_MD_PATH}")
        with open(GGET_VIRUS_DOC_MD_PATH, "r") as f:
            gget_docs = f.read()
        system_prompt += "\n\n--- gget virus documentation ---\n" + gget_docs

    # Prepare incremental result files
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
                    "is_correct", "error", "duration_seconds", "tool_call_count",
                ])
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_middle = "_gv" if use_gget_virus else ""
        report_path = output_path / f"gpt_benchmark_report{file_middle}_{timestamp}.json"
        csv_report_path = output_path / f"gpt_benchmark_summary{file_middle}_{timestamp}.csv"

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

    parser = argparse.ArgumentParser(description="Benchmark GPT-5.2-pro on viral sequence retrieval")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=str(Path(__file__).parent.parent / "docs" / "virseq_benchmark.csv"),
        help="Path to the benchmark CSV file",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=str(Path(__file__).parent.parent / "results" / "gpt"),
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
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--resume",
        metavar="REPORT_JSON",
        default=None,
        help="Resume from an existing JSON report, skipping completed runs",
    )

    args = parser.parse_args()

    asyncio.run(
        run_benchmark(
            args.csv_path,
            output_dir=args.output_dir,
            test_run=args.test,
            use_gget_virus=args.use_gget_virus,
            model=args.model,
            resume_from=args.resume,
        )
    )


if __name__ == "__main__":
    main()