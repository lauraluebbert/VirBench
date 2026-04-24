<p align="center">
  <img src="figures/virbench_logo.png" alt="VirBench Logo" width="200">
</p>

**VirBench** is a benchmark for evaluating how well different AI agents and tools retrieve viral sequence data. Each agent is given a natural-language query describing a pathogen and a set of filters (host, date range, sequence length, geographic location, etc.) and asked to return the number of matching sequences. Results are compared against a manually curated set of expected counts based on the NCBI Virus web UI.

**NOTE:** The full VirBench benchmark dataset is not publicly released to prevent potential data leakage into large language model training corpora, which could compromise the validity of evaluation results by enabling models to retrieve or memorize answers. Researchers interested in accessing the benchmark are encouraged to [contact the corresponding authors](https://www.lauraluebbert.com/contact/) to request access.


## File structure

```
VirBench/
├── requirements.txt              # Python dependencies
│
├── docs/                         # Benchmark inputs and reference data
│   ├── virseq_benchmark.csv          # Benchmark queries and expected counts - CONTACT US FOR ACCESS
│   └── gget_virus_docs.md            # gget virus module documentation (provided to agents)
│
├── src/                          # Benchmark scripts and shared utilities
│   ├── utils.py                      # Shared data classes, CSV parsing, query building, LLM extraction
│   ├── benchmark_edison_analysis.py  # Edison Analysis agent benchmark
│   ├── benchmark_biomni.py           # Biomni data-lake agent benchmark
│   ├── benchmark_gget_virus.py       # gget virus benchmark (no agent)
│   ├── benchmark_claude.py           # Claude Sonnet 4 benchmark (Anthropic API + code execution)
│   └── benchmark_gpt.py              # GPT-5.2-pro benchmark (OpenAI Responses API + code execution)
│   └── rerun_errors.py               # Rerun failed runs for all technologies
│
├── results/                      # Benchmark output (CSV summaries + JSON reports)
│   ├── edison_analysis/          # Edison agent results
│   ├── biomni/                   # Biomni agent results
│   ├── gget_virus/               # gget virus results
│   ├── claude/                   # Claude Sonnet 4 results
│   └── gpt/                      # GPT-5.2-pro results
│
├── notebooks/                    # Analysis and visualization notebooks
│
└── figures/                      # Plots generated in notebooks
```

## Usage

```bash
# Set up environment
cp .env.example .env   # add your API keys
pip install -r requirements.txt

# Run a quick test (first query only, 3 runs)
python src/benchmark_claude.py --test
python src/benchmark_gpt.py --test
python src/benchmark_edison_analysis.py --test
python src/benchmark_biomni.py --test
python src/benchmark_gget_virus.py --test

# Full benchmark run
python src/benchmark_claude.py

# With gget virus documentation
python src/benchmark_claude.py --use-gget-virus

# With K-Dense scientific skills (Claude only)
python src/benchmark_claude.py --test \
    --kdense data/claude-scientific-skills/scientific-skills \
    --kdense-skills biopython,gget --kdense-refs

# Custom model / output directory
python src/benchmark_claude.py --model claude-sonnet-4-20250514 -o results/claude
python src/benchmark_gpt.py --model gpt-5.2-pro -o results/gpt
```

## Benchmarking process

### Query definitions

All queries are defined in `docs/virseq_benchmark.csv`. Each row specifies:

- **query_id** and **pathogen** — identifies the query
- **tax_id** — NCBI taxonomy ID for the virus
- **expected_count** — the ground-truth sequence count based on manual queries through the NCBI Virus web UI
- **Filter columns** — host, geographic_location, date ranges, sequence length bounds, refseq_only, annotated, segment, lineage, and many others

All queries include a maximum release date to increase the stability of expected counts over time.  

The shared `build_query()` function in `src/utils.py` converts each row into a natural-language prompt (e.g. *"Retrieve viral sequences from NCBI for TaxID 3052518 (CCHFV) with the following filters: geographic location: Africa, collected on or after 2020-01-01, minimum sequence length: 11000 bp..."*).

### Prompts

The shared `build_query()` function in `src/utils.py` converts each CSV row into a natural-language prompt. The prompt structure varies depending on the query type:

- **Standard queries** (most rows) — the prompt opens with *"Retrieve viral sequences from NCBI for TaxID {tax_id} ({pathogen})"* followed by any filter criteria (host, dates, sequence length, etc.).
- **Accession queries** (`is_accession=TRUE`) — instead of searching by taxonomy, the prompt asks *"Retrieve the sequence(s) that belong to NCBI accession ID {tax_id}"*. In these rows, the `tax_id` column holds an accession ID (e.g. `NC_105106.1`) rather than a numeric taxonomy ID.
- **All-virus queries** (no `tax_id` or `pathogen`) — the prompt opens with *"Retrieve all viral sequences (for any virus) from NCBI"* and relies entirely on the filter columns to narrow results.

Filter columns (host, date ranges, sequence length, geographic location, lineage, etc.) are converted into human-readable phrases and appended after the opening sentence. Boolean flags like `vaccine_strain` and `lab_passaged` produce phrases such as *"exclude vaccine strains"* or *"only lab-passaged samples"*. Columns in the exclude set (`is_sars_cov2`, `is_alphainfluenza`, `is_accession`) are used for routing logic but are not included in the prompt text.

When gget virus mode is enabled, the instruction *"Use the gget virus module installable with 'pip install gget==0.30.3'. The documentation is attached."* is prepended to the prompt.

**Example — standard query** (query 1, CCHFV from Africa, without gget virus):
> Retrieve viral sequences from NCBI for TaxID 3052518 (Crimean-Congo Hemorrahagic fever virus (CCHFV)) that adhere to the following criteria: geographic location of sample collection: Africa, collected on or after 2020-01-01, collected on or before 2025-12-31, released on or after 2020-01-01, released on or before 2025-12-31, minimum sequence length: 11000 bp, maximum 10 ambiguous characters (N's). Return only the count of sequences that match these criteria.

**Example — same query with gget virus**:
> Use the gget virus module installable with 'pip install gget==0.30.3'. The documentation is attached. Retrieve viral sequences from NCBI for TaxID 3052518 (Crimean-Congo Hemorrahagic fever virus (CCHFV)) that adhere to the following criteria: geographic location of sample collection: Africa, collected on or after 2020-01-01, collected on or before 2025-12-31, released on or after 2020-01-01, released on or before 2025-12-31, minimum sequence length: 11000 bp, maximum 10 ambiguous characters (N's). Return only the count of sequences that match these criteria.

**Example — accession query** (query 89):
> Retrieve the sequence(s) that belong to NCBI accession ID NC_105106.1. Return only the count of sequences that match these criteria.

### Agents under test

| Agent | Script | How it works |
|-------|--------|-------------|
| **Edison Analysis** | `benchmark_edison_analysis.py` | Proprietary agent via Edison client SDK. Queries are sent as analysis jobs. |
| **Biomni** | `benchmark_biomni.py` | Data-lake agent accessed via HTTP API. Uses `claude-sonnet-4-20250514` as its LLM (configurable via `--llm`). |
| **gget virus** | `benchmark_gget_virus.py` | Direct API call to `gget.virus()` — no LLM involved. Serves as a deterministic baseline. |
| **Claude Sonnet 4** | `benchmark_claude.py` | Claude Sonnet 4 (`claude-sonnet-4-20250514`) via the Anthropic Messages API with tool use. Tools: `execute_python` (local code execution) and `web_search` (server-side web search). Optionally enhanced with K-Dense scientific skills (see below). Note: newer Claude models (e.g. `claude-sonnet-4-6`) refuse viral sequence queries due to biosecurity safety filters, so we use `claude-sonnet-4-20250514`. |
| **GPT-5.2-pro** | `benchmark_gpt.py` | GPT-5.2-pro via the OpenAI Responses API with tool use. Tools: `execute_python` (local code execution) and `web_search_preview` (server-side web search). |

### Claude and GPT tool use

Both the Claude and GPT benchmarks call their respective APIs with two tools:

- **Web search** (server-side) — a built-in tool provided by each API (`web_search_20260209` for Claude, `web_search_preview` for GPT). The model can search the web for API documentation, examples, or any other information. Search execution and result retrieval are handled entirely by the provider; the benchmark script does not need to process these tool calls.
- **`execute_python`** (client-side) — a custom function tool that runs model-generated Python code locally in a subprocess with a 120-second timeout. The script executes the code and returns stdout/stderr (truncated to 10K chars) as the tool result.

The agentic loop sends the query, lets the model call tools (up to 15 turns), and collects the final integer response. This gives the model full autonomy to search for documentation, install packages (e.g. `pip install gget`), write multi-step scripts, and retry on errors.

#### Domain allowlist

For safety reasons, the Claude and GPT benchmarks restrict outbound network access from `execute_python` to a configurable domain allowlist. Before each code execution, a socket-level guard is injected that monkey-patches `socket.getaddrinfo` and `socket.create_connection` to block connections to any host not on the allowlist. This catches all common Python HTTP libraries (requests, urllib, httpx, etc.) at the socket layer. Raw IP connections are also blocked.

The default allowlist permits only NCBI and PyPI domains:

| Domain | Purpose |
|--------|---------|
| `ncbi.nlm.nih.gov` | NCBI web and FTP |
| `eutils.ncbi.nlm.nih.gov` | NCBI E-utilities API |
| `api.ncbi.nlm.nih.gov` | NCBI Datasets API |
| `ftp.ncbi.nlm.nih.gov` | NCBI FTP |
| `pypi.org` | pip package index |
| `files.pythonhosted.org` | pip package downloads |

To override the default allowlist, set the `EXECUTE_PYTHON_DOMAIN_ALLOWLIST` environment variable to a comma-separated list of domains:

```bash
export EXECUTE_PYTHON_DOMAIN_ALLOWLIST="ncbi.nlm.nih.gov,eutils.ncbi.nlm.nih.gov,pypi.org"
```

Subdomain matching is supported: allowing `ncbi.nlm.nih.gov` also permits `eutils.ncbi.nlm.nih.gov`.

### Execution flow

Each benchmark script follows the same pattern:

1. **Parse** `docs/virseq_benchmark.csv` into query configurations
2. **For each query**, run `NUM_RUNS` (default 3) independent trials:
   - Build the natural-language query from the config
   - Send it to the agent/tool and collect the raw response
   - For Edison and Biomni, extract the integer count from the response using `claude-sonnet-4-20250514` (`extract_count_from_response`)
   - Compare against the expected count
3. **Write results incrementally** — after every single run, append to both the CSV summary and JSON report so partial results are preserved if the process is interrupted

### gget virus mode

All agent benchmarks support a `--use-gget-virus` / `-gv` flag. When enabled, the agent is instructed to use the `gget virus` Python module and the documentation from `docs/gget_virus_docs.md` is provided (either uploaded to the agent's environment or prepended to the system prompt). This tests whether giving agents access to a purpose-built tool improves retrieval accuracy.

### K-Dense scientific skills mode (Claude only)

The Claude benchmark supports loading [K-Dense scientific skills](https://github.com/kdense/claude-scientific-skills) into the system prompt via the `--kdense` flag. Each skill is a curated documentation bundle (a `SKILL.md` file plus optional `references/` and `scripts/` directories) that teaches the model how to use a specific bioinformatics tool or database API.

The skills relevant to this benchmark are **biopython** (NCBI Entrez access, sequence I/O) and **gget** (unified queries to 20+ genomic databases). Loading these gives the model detailed API references and working code examples it can adapt when writing its NCBI queries.

**Flags:**

| Flag | Description |
|------|-------------|
| `--kdense DIR` | Path to the `scientific-skills` directory. Loads `SKILL.md` from every skill folder. |
| `--kdense-skills s1,s2` | Comma-separated list of skill folder names to load instead of all ~140 skills. |
| `--kdense-refs` | Also load `references/*.md` and `scripts/*.py` from each selected skill (detailed API docs and example code). |

**Setup:**

```bash
# Clone the skills repo into data/
git clone https://github.com/kdense/claude-scientific-skills.git data/claude-scientific-skills
```

**Example runs:**

```bash
# Load biopython + gget skills (~40K chars added to system prompt)
python src/benchmark_claude.py --test \
    --kdense data/claude-scientific-skills/scientific-skills \
    --kdense-skills biopython,gget

# Include full reference docs and example scripts (~195K chars)
python src/benchmark_claude.py --test \
    --kdense data/claude-scientific-skills/scientific-skills \
    --kdense-skills biopython,gget --kdense-refs
```

Output files include a `_kd` suffix when K-Dense skills are enabled (e.g. `claude_benchmark_summary_kd_20260220_120000.csv`). When combined with gget virus mode the suffix is `_gv_kd`.

### Output format

Each benchmark run produces two files in its results subdirectory:

- **CSV summary** (`*_benchmark_summary_YYYYMMDD_HHMMSS.csv`) — one row per run with query_id, run_number, expected_count, retrieved_count, is_correct, error, and duration_seconds
- **JSON report** (`*_benchmark_report_YYYYMMDD_HHMMSS.json`) — full report including raw responses and running accuracy statistics

Files with `_gv` in the name were run with gget virus mode enabled.

### Handling of errors and None results

When computing metrics (accuracy, stability, MAE, duration), there are two categories of incomplete runs:

1. **Explicit errors** — runs where the `error` column in the CSV contains a non-empty value (e.g. a timeout, API error, or exception traceback). These are always excluded from all metric computations (treated as `NaN`).

2. **None results** — runs where `retrieved_count` is missing/empty but the `error` column is also empty. These represent cases where the agent completed without raising an exception but did not return a parseable integer count.

The analysis notebook (`notebooks/create_figures.ipynb`) provides a toggle to control how None results are handled:

```python
NONE_TREATED_AS_ZERO = True   # None results count as 0 (included in metrics)
NONE_TREATED_AS_ZERO = False  # None results are treated as errors (excluded from metrics)
```

When `NONE_TREATED_AS_ZERO = True`, a missing `retrieved_count` with no error is interpreted as "the agent found no matching sequences" (i.e. 0). When `False`, it is treated the same as an explicit error and excluded from all metric computations.

### Rerunning failed queries

The `src/rerun_errors.py` script identifies errored runs across all benchmark reports and reruns only those queries. It produces updated result files (with a `_rerun_` suffix) and a consolidated report at `results/rerun_errors_report_<timestamp>.json` summarizing errors before and after the rerun.

```bash
# Rerun all errored queries across all technologies
python src/rerun_errors.py

# Rerun errors for a specific report
python src/rerun_errors.py --report results/gpt/gpt_benchmark_report_20260226_211117.json
```

### Shared utilities (`src/utils.py`)

| Export | Description |
|--------|------------|
| `QueryConfig` | Dataclass holding query_id, pathogen, expected_count, tax_id, and a filters dict |
| `BenchmarkResult` | Dataclass for a single run's outcome (counts, correctness, timing, errors) |
| `parse_csv(path)` | Parses `docs/virseq_benchmark.csv` into a list of `QueryConfig` objects |
| `build_query(config, use_gget_virus)` | Converts a `QueryConfig` into a natural-language prompt |
| `extract_count_from_response(text)` | Uses Claude to extract an integer count from free-text agent responses |
| `NUM_RUNS` | Number of independent trials per query (default: 3) |
| `GGET_VIRUS_DOC_MD_PATH` | Path to the gget virus documentation file |


