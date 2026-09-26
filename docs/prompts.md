# VirBench prompts

This page gives the exact prompts the VirBench harness sends to each agent. Every prompt string below was taken from the literals in `src/` and evaluated, not retyped. The source code is the authoritative reference. If this page and `src/` ever disagree, `src/` is correct.

No benchmark query appears on this page. The worked example uses a pathogen and filter set that are not part of VirBench.

## 1. Who receives what

| Arm | System prompt set by the harness | User prompt | gget virus condition |
|---|---|---|---|
| Claude (`src/benchmark_claude.py`) | `SYSTEM_PROMPT` (§2.1) | `build_query(config, use_gget_virus=..., return_integer_only=True)` | the prefix in §4 is prepended to the user prompt, and `docs/gget_virus_docs.md` is appended to the system prompt (§5) |
| GPT (`src/benchmark_gpt.py`) | `SYSTEM_PROMPT` (§2.2), passed as the Responses API `instructions` | `build_query(config, use_gget_virus=..., return_integer_only=True)` | same as Claude |
| Edison Analysis (`src/benchmark_edison_analysis.py`) | none; the platform uses its own internal prompt | `build_query(config, use_gget_virus=...)` (default closing sentence) | prefix prepended, and the documentation file is uploaded to Edison data storage and attached to each task (§5) |
| Biomni OSS (`src/benchmark_biomni.py`) | none; the framework uses its own internal prompt | `build_query(config, use_gget_virus=...)` (default closing sentence) | prefix prepended, and the documentation file is registered with `agent.add_data` (§5) |
| gget virus standalone (`src/benchmark_gget_virus.py`) | n/a (no language model) | n/a | n/a |

## 2. System prompts (Claude and GPT)

### 2.1 Claude: `SYSTEM_PROMPT` (`src/benchmark_claude.py`, line 77)

Rendered text:

```
You are a bioinformatics agent.
You have access to:
- web_search: search the internet for up-to-date information.
- execute_python: run Python LOCALLY (WITH internet) to call APIs, install packages, and compute results. Note: outbound network access is restricted to an allowlist of domains.

Workflow:
1) If you need to choose the best API/library, use web_search first.
2) Then use execute_python to implement the call and compute the final answer.

Output rules:
- When you have the final count, respond with exactly one integer on its own line.
- After you output the final integer, also output a JSON block with keys: methods (eg APIs used) and reasoning (max 3 bullets). Keep it short.

```

Source literal:

```python
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
```

When the script is run with `--no-web-search`, `SYSTEM_PROMPT_NO_WEB_SEARCH` (line 93) is used instead. The manuscript reports Claude runs with web search enabled.

```
You are a bioinformatics agent.
You have access to execute_python: run Python LOCALLY (WITH internet) to call APIs, install packages, and compute results. Outbound network access is restricted to an allowlist of domains.

Use execute_python to implement the API calls and compute the final answer.

Output rules:
- When you have the final count, respond with exactly one integer on its own line.
- After you output the final integer, also output a JSON block with keys: methods (eg APIs used) and reasoning (max 3 bullets). Keep it short.

```

**Optional K-Dense skills (Claude only).** If `--kdense DIR` is given, the script appends a header followed by the loaded `SKILL.md` files to the system prompt. The header is `"\n\n--- K-Dense scientific skills ---\nThe following scientific skills are available. Use them as reference for writing Python code to solve the task.\n\n"`. The result files do not record whether this flag was used. The manuscript does not describe it.

### 2.2 GPT: `SYSTEM_PROMPT` (`src/benchmark_gpt.py`, line 55)

This is the same as the Claude prompt except that the web search tool is named `web_search_preview` in the first bullet.

Rendered text:

```
You are a bioinformatics agent.
You have access to:
- web_search_preview: search the internet for up-to-date information.
- execute_python: run Python LOCALLY (WITH internet) to call APIs, install packages, and compute results. Note: outbound network access is restricted to an allowlist of domains.

Workflow:
1) If you need to choose the best API/library, use web_search first.
2) Then use execute_python to implement the call and compute the final answer.

Output rules:
- When you have the final count, respond with exactly one integer on its own line.
- After you output the final integer, also output a JSON block with keys: methods (eg APIs used) and reasoning (max 3 bullets). Keep it short.

```

Source literal:

```python
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
```

## 3. User-prompt template: `build_query()` (`src/utils.py`, `src/utils.py` lines 195–327)

`build_query()` turns one benchmark row into the user prompt. It has four parts, joined with single spaces:

1. **Opening sentence**, which depends on the query type:
   - taxonomy query: `Retrieve viral sequences from NCBI for TaxID {tax_id} ({pathogen})`
   - accession query: `Retrieve the sequence(s) that belong to NCBI accession ID {tax_id}`
   - all-virus query: `Retrieve all viral sequences (for any virus) from NCBI`
2. **Filter clause:** `that adhere to the following criteria: ` followed by the filter phrases, separated by commas. Each filter has a fixed phrasing.
3. **Closing sentence:** `. Return the final count as a single integer on its own line.` for Claude and GPT, or `. Return only the count of sequences that match these criteria.` for Edison and Biomni.
4. **gget prefix (§4):** inserted in front of everything else in tool-guided runs.

Because the parts are joined with spaces, the prompt contains a space before the period of the closing sentence (`... bp . Return ...`). This spacing is in every prompt that was sent; the prompts echoed back in the Edison run records show it too. The shorter examples in `README.md` leave the space out.

Full source:

```python
def build_query(
    config: QueryConfig,
    use_gget_virus: bool = False,
    return_integer_only: bool = False,
) -> str:
    """Build a natural language query from the VirBench row.

    Converts every filter in ``config.filters`` (except those in
    ``_QUERY_EXCLUDE``) into a human-readable sentence fragment.

    When *return_integer_only* is True the closing instruction asks the
    model to respond with nothing but the integer count (no prose).
    This lets callers parse the response directly instead of using
    ``extract_count_from_response``.
    """
    is_accession = config.filters.get("is_accession", False)

    if is_accession and config.tax_id:
        opening = f"Retrieve the sequence(s) that belong to NCBI accession ID {config.tax_id}"
    elif config.tax_id and config.pathogen:
        opening = f"Retrieve viral sequences from NCBI for TaxID {config.tax_id} ({config.pathogen})"
    elif config.tax_id:
        opening = f"Retrieve viral sequences from NCBI for TaxID {config.tax_id}"
    else:
        opening = "Retrieve all viral sequences (for any virus) from NCBI"

    query_parts = [opening]

    f = config.filters
    filter_phrases = []

    # -- String / value filters ------------------------------------------------

    if f.get("host"):
        filter_phrases.append(f"host organism: {f['host']}")

    if f.get("nuc_completeness"):
        filter_phrases.append(f"nucleotide completeness: {f['nuc_completeness']}")

    if f.get("geographic_location"):
        filter_phrases.append(f"geographic location of sample collection: {f['geographic_location']}")

    if f.get("submitter_country"):
        filter_phrases.append(f"sample submitter country: {f['submitter_country']}")

    if f.get("lineage"):
        filter_phrases.append(f"SARS-CoV-2 lineage: {f['lineage']}")

    if f.get("segment"):
        filter_phrases.append(f"contains the genome segment: {f['segment']}")

    if f.get("source_database"):
        filter_phrases.append(f"source database: {f['source_database']}")

    # -- Date filters ----------------------------------------------------------

    if f.get("min_collection_date"):
        filter_phrases.append(f"collected on or after {f['min_collection_date']}")
    if f.get("max_collection_date"):
        filter_phrases.append(f"collected on or before {f['max_collection_date']}")

    if f.get("min_release_date"):
        filter_phrases.append(f"released on or after {f['min_release_date']}")
    if f.get("max_release_date"):
        filter_phrases.append(f"released on or before {f['max_release_date']}")

    # -- Numeric range filters -------------------------------------------------

    if f.get("min_seq_length") is not None:
        filter_phrases.append(f"minimum sequence length: {f['min_seq_length']} bp")
    if f.get("max_seq_length") is not None:
        filter_phrases.append(f"maximum sequence length: {f['max_seq_length']} bp")

    if f.get("max_ambiguous_chars") is not None:
        filter_phrases.append(
            f"maximum {f['max_ambiguous_chars']} ambiguous characters (N's)"
        )

    # -- Boolean / flag filters ------------------------------------------------

    if "lab_passaged" in f:
        if f["lab_passaged"]:
            filter_phrases.append("only lab-passaged samples")
        else:
            filter_phrases.append("exclude lab-passaged samples")

    if "vaccine_strain" in f:
        if f["vaccine_strain"]:
            filter_phrases.append("vaccine strains only")
        else:
            filter_phrases.append("exclude vaccine strains")

    # -- Catch-all for any future columns not explicitly handled above ----------

    _HANDLED = (
        _QUERY_EXCLUDE
        | _META_COLUMNS
        | {
            "host", "nuc_completeness", "geographic_location", "submitter_country",
            "min_collection_date", "max_collection_date",
            "min_release_date", "max_release_date",
            "min_seq_length", "max_seq_length", "max_ambiguous_chars",
            "lineage", "lab_passaged", "vaccine_strain",
            "segment", "source_database",
        }
    )
    for col, val in f.items():
        if col not in _HANDLED and val not in (None, "", False):
            label = col.replace("_", " ")
            filter_phrases.append(f"{label}: {val}")

    # -- Assemble the final query ----------------------------------------------

    if filter_phrases:
        query_parts.append(
            "that adhere to the following criteria: " + ", ".join(filter_phrases)
        )

    if return_integer_only:
        query_parts.append(
            ". Return the final count as a single integer on its own line."
        )
    else:
        query_parts.append(". Return only the count of sequences that match these criteria.")

    if use_gget_virus:
        query_parts.insert(
            0,
            "Use the gget virus module installable with "
            "'pip install gget==0.30.3'. The documentation is attached.",
        )

    return " ".join(query_parts)
```

## 4. gget virus instruction (tool-guided runs)

When `use_gget_virus=True`, `build_query()` inserts this sentence at the start of the user prompt:

```
Use the gget virus module installable with 'pip install gget==0.30.3'. The documentation is attached.
```

## 5. gget virus documentation delivery

The documentation text is `docs/gget_virus_docs.md` in this repository.

- **Claude and GPT:** appended to the system prompt, after the separator shown here (the separator's first two characters are newlines):
```
'\n\n--- gget virus documentation ---\n'
```
- **Edison Analysis:** uploaded once per benchmark run with `client.astore_file_content(name="gget_virus documentation", description="Documentation for the gget virus Python and cli module.")`, then passed to every task as `data_storage_uris=["data_entry:<id>"]`.
- **Biomni OSS:** `agent.add_data({<path to gget_virus_docs.md>: "Documentation for the gget virus Python and cli module."})`.

## 6. Worked example (synthetic query, not in VirBench)

These are the exact `build_query()` outputs for a synthetic configuration. The pathogen is Nipah virus (TaxID 121791), and the filters are host *Homo sapiens*, nucleotide completeness complete, collected 2001-01-01 to 2018-12-31, and minimum length 18000 bp. Neither this pathogen nor this filter set is in the benchmark.

**Claude/GPT, baseline:**
```
Retrieve viral sequences from NCBI for TaxID 121791 (Nipah virus) that adhere to the following criteria: host organism: Homo sapiens, nucleotide completeness: complete, collected on or after 2001-01-01, collected on or before 2018-12-31, minimum sequence length: 18000 bp . Return the final count as a single integer on its own line.
```

**Claude/GPT, with gget virus:**
```
Use the gget virus module installable with 'pip install gget==0.30.3'. The documentation is attached. Retrieve viral sequences from NCBI for TaxID 121791 (Nipah virus) that adhere to the following criteria: host organism: Homo sapiens, nucleotide completeness: complete, collected on or after 2001-01-01, collected on or before 2018-12-31, minimum sequence length: 18000 bp . Return the final count as a single integer on its own line.
```

**Edison/Biomni, baseline:**
```
Retrieve viral sequences from NCBI for TaxID 121791 (Nipah virus) that adhere to the following criteria: host organism: Homo sapiens, nucleotide completeness: complete, collected on or after 2001-01-01, collected on or before 2018-12-31, minimum sequence length: 18000 bp . Return only the count of sequences that match these criteria.
```

**Edison/Biomni, with gget virus:**
```
Use the gget virus module installable with 'pip install gget==0.30.3'. The documentation is attached. Retrieve viral sequences from NCBI for TaxID 121791 (Nipah virus) that adhere to the following criteria: host organism: Homo sapiens, nucleotide completeness: complete, collected on or after 2001-01-01, collected on or before 2018-12-31, minimum sequence length: 18000 bp . Return only the count of sequences that match these criteria.
```

**The same template with placeholder values**, showing the phrasing of other common filters (Claude/GPT closing):
```
Retrieve viral sequences from NCBI for TaxID <TAXID> (<PATHOGEN>) that adhere to the following criteria: host organism: <HOST>, geographic location of sample collection: <LOCATION>, released on or after <YYYY-MM-DD>, released on or before <YYYY-MM-DD>, minimum sequence length: <N> bp, maximum <K> ambiguous characters (N's) . Return the final count as a single integer on its own line.
```

## Appendix A: agent-facing tool definitions (Claude and GPT)

Claude `EXECUTE_PYTHON_TOOL` (`src/benchmark_claude.py`):
```python
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
```

Claude `WEB_SEARCH_TOOL` (server-side; the description comes from the provider):
```python
WEB_SEARCH_TOOL = {
    "type": "web_search_20250305", # Using the old web_search tool type to avoid auto-adding an unsupported code_execution version when web_search is present.
    "name": "web_search",
    "max_uses": 5,
}
```

GPT `EXECUTE_PYTHON_TOOL` (`src/benchmark_gpt.py`). GPT also gets the provider's built-in `web_search_preview` tool.
```python
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
```

Both scripts allow at most 25 model turns per run (`max_turns=25`).

## Appendix B: count-extraction prompt (Edison and Biomni scoring)

The agents never see this prompt. Edison and Biomni return free text, so `extract_count_from_response()` in `src/utils.py` sends the last 4,000 characters of the response to `claude-sonnet-4-20250514` with this template:

```python
prompt = f"""Extract the final sequence count from this analysis response.

            The response is from an agent that was asked to count viral sequences matching certain criteria.
            Return ONLY the integer count, nothing else. If no clear count is found, return -1.

            Response:
            {truncated_response}

            Final count (integer only):"""
```
