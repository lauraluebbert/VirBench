"""
Shared utilities for the virseq benchmark suite.

Contains data classes, CSV parsing, query building, and LLM-based
response extraction used by the Edison, Biomni, and rerun notebooks.
"""

import asyncio
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_RUNS = 3

GGET_VIRUS_DOC_MD_PATH = str(Path(__file__).parent.parent / "docs" / "gget_virus_docs.md")

# ---------------------------------------------------------------------------
# Anthropic client (used by extract_count_from_response)
# ---------------------------------------------------------------------------

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("Please set your ANTHROPIC_API_KEY in the .env file")

anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# -----------------------------------------------------------------------------
# Network/domain allowlist for LOCAL python execution
# -----------------------------------------------------------------------------
DEFAULT_DOMAIN_ALLOWLIST = [
    # NCBI
    "ncbi.nlm.nih.gov",
    "eutils.ncbi.nlm.nih.gov",
    "api.ncbi.nlm.nih.gov",
    "ftp.ncbi.nlm.nih.gov",
    # Common package install hosts
    "pypi.org",
    "files.pythonhosted.org",
]

DOMAIN_ALLOWLIST = [
    d.strip().lower()
    for d in os.getenv("EXECUTE_PYTHON_DOMAIN_ALLOWLIST", "").split(",")
    if d.strip()
] or DEFAULT_DOMAIN_ALLOWLIST

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class QueryConfig:
    """Configuration for a viral sequence query.

    ``filters`` holds all filter columns from the CSV as a dict
    (column name -> typed value).  ``build_query`` converts these into a
    natural-language prompt.
    """
    query_id: int
    pathogen: str
    expected_count: int
    tax_id: Optional[str] = None
    filters: dict = None

    def __post_init__(self):
        if self.filters is None:
            self.filters = {}


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    query_id: int
    run_number: int
    expected_count: int
    retrieved_count: Optional[int] = None
    is_correct: bool = False
    error: Optional[str] = None
    raw_response: Optional[str] = None
    duration_seconds: float = 0.0
    tool_call_count: int = 0
    methods: Optional[list] = None
    reasoning: Optional[list] = None


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------


# Columns that are benchmark metadata, not filters to pass downstream
_META_COLUMNS = {"query_id", "expected_count", "pathogen", "tax_id"}

# Columns whose non-empty values should be parsed as integers
_INT_FILTER_COLUMNS = {
    "min_seq_length", "max_seq_length", "max_ambiguous_chars",
}

# Columns whose non-empty values should be parsed as booleans
_BOOL_FILTER_COLUMNS = {
    "is_sars_cov2", "is_alphainfluenza", "is_accession",
    "vaccine_strain", "lab_passaged",
}

# Columns that should never appear in the natural-language query
_QUERY_EXCLUDE = {
    "is_sars_cov2", "is_alphainfluenza", "is_accession",
}

def _parse_filter_value(column: str, raw: str):
    """Convert a raw CSV string to the appropriate Python type."""
    if column in _INT_FILTER_COLUMNS:
        return int(raw)
    if column in _BOOL_FILTER_COLUMNS:
        return raw.strip().lower() in ("true", "1", "yes")
    return raw.strip()


def parse_csv(csv_path: str) -> list[QueryConfig]:
    """Parse the benchmark CSV file and return query configurations.

    All columns beyond the metadata columns (query_id, expected_count,
    pathogen, tax_id) are stored in ``QueryConfig.filters`` as a dict
    so that ``build_query`` can convert them to natural language.
    """
    configs = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                query_id = int(row["query_id"]) if row.get("query_id") else None
                expected_count = int(row["expected_count"]) if row.get("expected_count") else None
                tax_id = row.get("tax_id", "").strip() or None

                if expected_count is None or query_id is None:
                    continue

                # Build filters dict from all remaining non-empty columns
                filters = {}
                for col, val in row.items():
                    if col in _META_COLUMNS:
                        continue
                    val = val.strip() if val else ""
                    if not val:
                        continue
                    try:
                        filters[col] = _parse_filter_value(col, val)
                    except (ValueError, TypeError):
                        filters[col] = val.strip()

                config = QueryConfig(
                    query_id=query_id,
                    pathogen=row.get("pathogen", "").strip(),
                    expected_count=expected_count,
                    tax_id=tax_id,
                    filters=filters,
                )
                configs.append(config)
            except (ValueError, KeyError) as e:
                print(f"Warning: Skipping row due to parsing error: {e}")
                continue

    return configs


# ---------------------------------------------------------------------------
# Query building
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# LLM-based response extraction
# ---------------------------------------------------------------------------


async def extract_count_from_response(response_text: str) -> Optional[int]:
    """Extract the sequence count from the agent's response using an LLM.

    Uses Claude to reliably extract the final count from varied response formats.
    """
    if not response_text:
        return None

    # Truncate very long responses to avoid excessive token usage
    max_chars = 4000
    truncated_response = response_text[-max_chars:] if len(response_text) > max_chars else response_text

    prompt = f"""Extract the final sequence count from this analysis response.

            The response is from an agent that was asked to count viral sequences matching certain criteria.
            Return ONLY the integer count, nothing else. If no clear count is found, return -1.

            Response:
            {truncated_response}

            Final count (integer only):"""

    try:
        message = await asyncio.to_thread(
            anthropic_client.messages.create,
            model="claude-sonnet-4-20250514",
            max_tokens=32,
            messages=[{"role": "user", "content": prompt}]
        )
        count_str = message.content[0].text.strip()
        count = int(count_str)
        return count if count >= 0 else None
    except (ValueError, IndexError, anthropic.APIError) as e:
        print(f"    Warning: LLM extraction failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------


def load_completed_runs_from_json(path: str) -> set[tuple[int, int]]:
    """Read a JSON report file and return the set of (query_id, run_number) pairs."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return {
            (int(r["query_id"]), int(r["run_number"]))
            for r in data.get("results", [])
        }
    except Exception as e:
        print(f"Warning: could not parse resume file {path}: {e}")
        return set()


def load_completed_runs_from_csv(path: str) -> set[tuple[int, int]]:
    """Read a CSV summary file and return the set of (query_id, run_number) pairs."""
    try:
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            return {
                (int(row["query_id"]), int(row["run_number"]))
                for row in reader
            }
    except Exception as e:
        print(f"Warning: could not parse resume file {path}: {e}")
        return set()
