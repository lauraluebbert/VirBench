# GPT 5.5 Failure Mode Analysis

Analysis of GPT 5.5 benchmark results from the latest rerun reports (May 2026).
Max tool call turns: 25. All None results are "Max turns reached" timeouts.

## Overall Performance

| Condition | Correct | Wrong Count | None (timeout) | Total | Accuracy |
|-----------|---------|-------------|----------------|-------|----------|
| Without gget | 305 | 40 | 15 | 360 | 84.7% |
| With gget | 338 | 14 | 8 | 360 | 93.9% |

## Failure Categories

### 1. Consistent Off-by-One Errors (all 3 runs return the same wrong value)

These queries return a deterministic but slightly wrong count in every run, both with and without gget. The model's API query logic consistently misses or over-counts by 1 sequence.

| Query | Pathogen | Expected | Retrieved | Delta | Key Filters |
|-------|----------|----------|-----------|-------|-------------|
| Q3 | CCHFV | 23 | 22 | -1 | Africa, length ≥11000, max 10 N's |
| Q36 | CCHFV | 126 | 125 | -1 | length ≥11000, max 10 N's |
| Q96 | Rabies lyssavirus | 12 | 11 | -1 | host=10090 (mouse) |

**Root cause:** Likely edge-case differences in how the model applies filters (e.g., boundary inclusion for ambiguous characters or sequence length) compared to the NCBI Virus web UI.

### 2. Consistent Wrong Count (larger discrepancy)

| Query | Pathogen | Expected | Retrieved | Key Filters |
|-------|----------|----------|-----------|-------------|
| Q69 | Dengue virus 2 | 2694 | 2907 | completeness=partial, length 10416–13416 bp |

**Root cause:** The model consistently over-counts by 213 sequences. The query specifies `nuc_completeness: partial`, which maps to a specific NCBI Virus UI filter. The model uses NCBI Datasets API with `completeness == PARTIAL`, which may classify completeness differently than the NCBI Virus web interface.

### 3. Taxonomy/API Confusion (sporadic 0 results)

| Query | Pathogen | Expected | Without gget | With gget |
|-------|----------|----------|-------------|-----------|
| Q28 | O'nyong-nyong virus | 231 | [231, 231, 0] | [231, 0, 231] |
| Q54 | O'nyong-nyong virus | 1475 | [1475, 1475, 0] | [1475, 1475, 1475] |
| Q66 | Influenza A (Korea, RefSeq) | 3 | [0, 0, 0] | [3, 3, 3] |
| Q92 | Ebola virus (Italy) | 4 | [2, 2, 2] | [4, 4, 4] |

**Q28/Q54 root cause:** O'nyong-nyong virus (TaxID 11019) has undergone NCBI taxonomy reclassification. The model sometimes discovers that TaxID 11019 now maps to "Alphavirus" rather than ONNV specifically, and fails to find the right descendant taxon. When it navigates this correctly, it gets the right count.

**Q66 root cause (without gget):** The model searches for RefSeq Influenza A complete genomes from Korea with host=human, but its E-utilities/Datasets queries return 0. The NCBI Virus web UI uses different internal filtering logic for RefSeq + complete genome combinations. gget fixes this completely (3/3 correct).

**Q92 root cause:** Ebola virus from Italy — the model consistently gets 2 instead of 4 without gget. Its reasoning mentions "excluded records marked by filovirus" suggesting it's applying an extra lab-passage or vaccine-strain exclusion filter that the benchmark doesn't require. gget fixes this completely.

### 4. Influenza Timeouts (None results from max turns)

Without gget, Influenza A queries are the dominant source of timeouts:

| Query | Geo | Segment | Expected | Without gget | With gget |
|-------|-----|---------|----------|-------------|-----------|
| Q59 | North America | — | 304 | [None, None, None] | [304, 304, 304] |
| Q61 | — | — | 217 | [217, None, 217] | [217, 217, 217] |
| Q62 | Hong Kong | — | 1 | [None, None, 1] | [1, 1, 1] |
| Q64 | Cambodia | — | 98 | [98, None, 98] | [98, 98, 98] |
| Q65 | France | — | 216 | [216, 216, None] | [216, 216, 216] |
| Q118 | Morocco | HA | 5 | [5, None, None] | [5, None, None] |

**Root cause:** Without gget, the model uses E-utilities to query Influenza A (TaxID 11320), which returns massive result sets. The model then attempts to download and filter sequences locally, exhausting the 25-turn tool call limit before converging. Influenza A is one of the most heavily sequenced pathogens, making brute-force E-utilities approaches impractical.

**gget impact:** gget virus dramatically fixes Influenza queries (Q59 goes from 0/3 to 3/3 correct). Q118 remains problematic even with gget, likely because segment-level filtering (HA) adds complexity.

### 5. Large Dataset Timeouts (non-Influenza)

| Query | Pathogen | Expected | Without gget | With gget |
|-------|----------|----------|-------------|-----------|
| Q18 | HIV-1 | 16 | [16, 16, None] | [None, 16, 16] |
| Q43 | HIV-1 | 928 | [928, 928, 928] | [None, None, 928] |
| Q57 | mpox virus | 1782 | [1782, 1782, None] | [None, 1782, 1782] |
| Q78 | Retroviridae | 3 | [None, 3, 3] | [3, 3, 3] |
| Q85 | HBV (New Caledonia) | 1 | [1, 1, 1] | [None, 1, 1] |
| Q103 | Retroviridae (Taiwan) | 4 | [4, None, 4] | [4, None, 4] |

**Root cause:** These are sporadic timeouts on queries involving large datasets (HIV-1, mpox) or complex taxonomy (Retroviridae). The model sometimes takes an inefficient code path that burns through the turn limit.

**Q43 regression with gget:** This is the most notable gget regression. Without gget, Q43 (HIV-1, 928 expected) is 3/3 correct. With gget, it drops to 1/3 — the gget virus module likely struggles with the large HIV-1 dataset, causing timeouts.

### 6. Inconsistent Filter Application (without gget only)

These queries return wrong counts without gget but are fully fixed by gget:

| Query | Pathogen | Expected | Without gget counts | With gget counts |
|-------|----------|----------|-------------------|-----------------|
| Q32 | HBV (Africa) | 162 | [156, 156, 162] | [162, 162, 162] |
| Q37 | Hantavirus | 195 | [199, 195, 199] | [195, 195, 195] |
| Q38 | Hantavirus | 267 | [272, 272, 267] | [267, 267, 267] |
| Q39 | Hantavirus | 362 | [367, 367, 367] | [362, 362, 362] |
| Q45 | Dengue type 1 | 1535 | [1535, 1390, 1535] | [1535, 1535, 1535] |
| Q51 | West Nile virus | 764 | [750, 717, 764] | [764, 764, 764] |
| Q58 | HBV | 1367 | [1361, 1361, 1367] | [1367, 1367, 1367] |
| Q67 | Zika virus | 43 | [42, 43, 42] | [43, 43, 43] |
| Q82 | Yellow fever | 7 | [8, 6, 7] | [7, 7, 7] |
| Q87 | HBV (France) | 35 | [35, 35, 33] | [35, 35, 35] |
| Q115 | West Nile virus | 43 | [43, 57, 43] | [43, 43, 43] |

**Root cause:** Without gget, the model writes its own E-utilities/Datasets API code, which sometimes applies date filters, geographic filters, or ambiguous character counting slightly differently across runs. The non-determinism comes from the model choosing different API strategies or filter implementations on each run.

**gget impact:** gget virus provides a consistent, tested API that applies filters reliably, eliminating this class of errors entirely.

## Impact of gget virus

### Queries fixed by gget (21 queries)
gget virus fixes or improves results on 21 queries spanning all failure categories: off-by-one counts, taxonomy confusion, Influenza timeouts, and inconsistent filter application.

### Queries broken by gget (3 queries)
- **Q43** (HIV-1): 3/3 → 1/3 correct — large dataset causes gget timeouts
- **Q85** (HBV, New Caledonia): 3/3 → 2/3 correct — sporadic timeout
- **Q94** (mpox): 3/3 → 2/3 correct — one run returns 2 instead of 11

### Queries unchanged by gget (7 queries)
Q3, Q36, Q96 (off-by-one), Q69 (completeness mismatch), Q18/Q103/Q118 (sporadic timeouts) — these fail at similar rates with and without gget.

## Summary of Failure Modes

| Failure Mode | Queries Affected | Without gget | With gget | Fixable? |
|-------------|-----------------|-------------|-----------|----------|
| Off-by-one filter edge cases | Q3, Q36, Q96 | 9 wrong | 9 wrong | Unlikely — systematic API/UI discrepancy |
| Completeness definition mismatch | Q69 | 3 wrong | 3 wrong | Unlikely — API vs UI semantics |
| Taxonomy reclassification | Q28, Q54 | 2 wrong | 1 wrong | Partially — depends on model's taxonomy navigation |
| RefSeq/filter logic mismatch | Q66, Q92 | 6 wrong | 0 wrong | Yes — gget handles correctly |
| Influenza timeouts | Q59, Q61–65, Q118 | 8 None | 1 None | Mostly — gget fixes 5/6 queries |
| Large dataset timeouts | Q18, Q43, Q57, Q78, Q85, Q103 | 4 None | 5 None | Mixed — gget causes regressions on HIV-1 |
| Inconsistent filter application | Q32, Q37–39, Q45, Q51, Q58, Q67, Q82, Q87, Q115 | 20 wrong | 0 wrong | Yes — gget eliminates entirely |
