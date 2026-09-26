# VirBench curation protocol

This protocol explains how VirBench reference values are set, versioned and adjudicated. It is written from the Methods section "VirBench benchmark curation" of the manuscript *Deterministic access to global viral sequence data enables robust agentic scientific discovery*, together with Supplementary Tables S4 and S7. Where this document adds operational detail, the source is named.

This document contains no query definitions and no reference values. The complete benchmark is only available through the gated deposit (§8).

---

## 1. Terms

| Term | Meaning |
|---|---|
| **Query** | One VirBench row: a pathogen (an NCBI TaxID or accession ID, or no taxon for the two all-virus queries) plus between 0 and 9 filters. |
| **Reference value** | The number of NCBI Virus records that match a query, set by manual curation. |
| **Reference accession set** | The accession IDs behind a reference value, exported from the NCBI Virus web interface. |
| **Version** | One frozen set of all 120 reference values, taken during one curation round. |
| **Arm** | One evaluated system under one condition, for example an agent with or without gget virus, or standalone gget virus. |

---

## 2. Establishing a reference value: manual execution in NCBI Virus

Every reference value comes from running the query by hand in the NCBI Virus web interface (https://www.ncbi.nlm.nih.gov/labs/virus/vssi/#/). Do the following for every query:

1. **Enter the taxon.** Type the virus name or TaxID into the search field. For an accession query, enter the accession instead.
2. **Apply every filter** the query specifies, using the filter categories on the left-hand side of the results page (for example Dates, Geographic Region, Host, Sequence Length, Nucleotide Completeness, Genome Organization). Apply only the filters the query specifies.
3. **Set the date bounds carefully.** Operational note from Supplementary Table S7:
   - The date fields take MM/DD/YYYY.
   - After you press Submit, the interface moves the start date by one day.
   - The interface does not include the maximum date.
   - Enter the bounds so that the range actually applied matches the range the query intends, then check the applied bounds after submitting.
4. **Record the applied filter state.** Record the filters exactly as the interface shows them after submission, not as you meant to set them.
5. **Record the returned count.** This is the reference value for this query in this curation round.

Record both the applied filter state and the count for every query. The recorded filter state is what lets you find and fix a mis-set filter later.

---

## 3. The maximum-release-date bound

**Rule:** every query except the three accession lookups includes a maximum release date. An accession lookup names one fixed record, so it doesn't need a date bound.

**Why this closes the record set:**
- New records are deposited in NCBI all the time, so an open-ended query returns more records every day.
- Capping the release date limits the query to records that had been released by that date.
- The set of matching records is therefore closed. The reference value stays fixed instead of growing with the database, which is what makes it possible to re-run and score the query later.

**Limits of the bound:** the bound is strong but not absolute. The matching set can still change after curation when:
- a record is deposited into INSDC retrospectively, with a release date inside the bound, or
- a record is withdrawn or reclassified after deposition.

Because of this residual drift, reference values are versioned (§4). They are not treated as permanent.

---

## 4. Versioning of reference values

### 4.1 Rules

1. Each curation round produces a new, frozen **version** of all reference values.
2. A reference value is never edited in place. Any change, whether from database drift or from adjudication (§6), goes into a new version.
3. **Each arm is scored against the version current when it was run.** An arm keeps that version even after later versions exist. The only exception is an arm that is deliberately re-executed against a later round (§5).

### 4.2 Versions and arm assignments (as reported in the manuscript)

| Curation round | When | Arms scored against it |
|---|---|---|
| **Version 1** | February 2026 | Claude Sonnet 4, Biomni OSS, Edison Analysis and GPT-5.2-pro (each with and without gget virus), and the original standalone gget virus arm. All were executed in February 2026. |
| **Version 2** | May 2026 | Claude Opus 4.7 and GPT-5.5 (each with and without gget virus). Both were executed in May 2026. |
| **Accession-export round** | after version 2 | The standalone gget virus arm, re-executed for the set-level comparison in Supplementary Table S4. |

Size of the changes between rounds:
- 9 of the 120 reference values differ between version 1 and version 2.
- 5 values in the accession-export round differ from version 1, and 7 differ from version 2.

### 4.3 Correspondence to the files in this repository

- The expected values in `results/` fall into exactly two groups, and they differ on 9 queries. That matches versions 1 and 2.
- The Claude Opus 4.7 files and the GPT-5.5 `_rerun_` files use version 2.
- All other summary files use version 1, including the initial GPT-5.5 files from 2026-05-12, which were later rerun (see the previous point).
- The standalone gget virus files in `results/gget_virus/` are the version 1 runs. **The re-execution against the accession-export round is not included in this repository.**

---

## 5. The accession-export round

**Purpose:** in versions 1 and 2, each reference value is only a count. A count alone can't tell whether a system that returned the right number returned the right records. This round attaches a reference accession set to each reference value, so results can be compared by composition as well as by count.

**Procedure:**
1. For every query, repeat the manual execution in §2.
2. From the NCBI Virus web interface, export the full record table the query returns. Supplementary Table S7 describes the export path: "Download All Results", then "CSV format" under "Results Table", then "Download All Records", with all columns selected, including accession version.
3. Store each exported accession set with its count. The count, the set and the recorded filter state together make up that query's reference entry for this round.
4. Re-execute the standalone gget virus arm against this round, so its accession sets can be compared with the exported reference sets.
5. Leave every other arm on the version it was originally scored against (§4.1, rule 3).

**Rules for comparing sets** (from Supplementary Table S4):
- Compare accession identifiers **without version suffixes**.
- A reported list counts as **complete** when the number of distinct accession IDs equals the count the run returned. Only complete lists are compared.
- Leave out the three accession lookups when scoring agent arms, because the accession is given in the prompt and reporting it back is not evidence of retrieval. They are included for the standalone gget virus arm, which returns a full set for every query.
- An **offsetting error** is a run that returns the reference count with a set that is not the reference set. Report these separately. A count-based score marks them correct.

---

## 6. Adjudication: web interface vs gget virus

**Why agreement matters:** gget virus implements the geographic-location filter and several other filters so that they reproduce the field coverage and matching behavior of the NCBI Virus web interface.
- When gget virus agrees with a manually curated value, that rules out execution errors in the portal (a mis-set filter, a mis-read count, a transcription slip). Any such error would have to be repeated independently by code that uses a different set of endpoints.
- Agreement does **not** rule out a shared misreading of what a filter is meant to mean.

**Rule:** when the web interface and gget virus disagree on the same criterion, **the primary GenBank record is authoritative.**

**Procedure:**
1. Identify the disputed accessions: the records returned by one implementation and not the other.
2. Check each disputed accession's annotation against its GenBank entry.
3. Record the filter semantics that caused the difference. For example: exact versus case-insensitive string matching, or which record field a filter reads.
4. If the GenBank records show that the reference value must change, issue a new version (§4.1, rule 2). Do not edit the existing value.

**Known case:** Supplementary Table S7 documents one case outside the benchmark. The web interface's protein filter matches protein names as exact strings, so it misses records whose annotation differs only in capitalization. gget virus matches case-insensitively and returns them. No VirBench query uses a filter of this kind, so no reference value depends on this adjudication. The manuscript reports that such cases are rare and that none occur in VirBench.

---

## 7. Release checklist for a curation round

- [ ] Every query was run manually in NCBI Virus, and the applied filter state and count were recorded (§2).
- [ ] Date bounds were checked against the interface's actual applied range (§2, step 3).
- [ ] Every query except accession lookups has a maximum release date (§3).
- [ ] The round is frozen as a new version, and no earlier version was changed (§4.1).
- [ ] The arm-to-version assignments are recorded (§4.2).
- [ ] For an accession-export round, every query has an exported accession set (§5).
- [ ] Every disagreement between the web interface and gget virus was adjudicated against GenBank, and the filter semantics were recorded (§6).
- [ ] The version was deposited in the gated dataset (§8). No reference values or query definitions were added to any public file.

---

## 8. Deposit and access

- All versions are deposited in a gated dataset at huggingface.co/datasets/ferbsx/VirBench.
- The full benchmark is not publicly released. This prevents the query-reference pairs from entering language-model training data, which would let a model give the correct count without doing any retrieval.
- Access is granted on the condition that recipients do not republish the reference values or post them anywhere they can be crawled.
