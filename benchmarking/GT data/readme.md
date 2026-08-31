# This folder contains the Ground Truth CSV data files

The GT files are created by human-annotating the workload news stories, for comparison with LLM generated data.

Steps:

1. Students are given training using the same definitions LLMs are given, and some example files
2. Principal investigator reviews all the student completed GT work and finalizes them.
3. Convert into CSV with the standard column definitions and order used for this project and push into this folder.

## File organization

Each file corresponds to one article, with a single header row: `No, Sourced Statements, Type of Source, Name of Source, Title of Source, Source Justification`.

For a story with a file in both this folder and `rev2025/`, the `rev2025/` version supersedes the original and should be treated as canonical — it reflects a later revision pass that fixed errors and brought the annotations closer to the current definitions. Only use the base-folder version for a story if no `rev2025/` file exists for it.

## Data fidelity review (2026-07-24)

A review corrected a small number of `Source Justification` values that stated only the medium of contact (e.g. "said by email") rather than substantive justification, which is no longer valid per current definitions — see commit `ae27e70`. `Title of Source` was also reviewed across all 43 canonical stories for a different error class (relational descriptions like "Victim's granddaughter" recorded as a title instead of added context); none were found here, though the same issue did turn up in `inter_coder_reliability/`'s annotator practice files for one story. See `inter_coder_reliability/readme.md`'s data-quality-fix section for the full reasoning behind both checks.

## What `20250904/` is (and isn't)

`20250904/` is a **frozen point-in-time snapshot** of this folder plus `rev2025/` as they stood on 2025-09-04, added in commit `612209d` alongside LLM-precision-metric changes to `eval_multi_llm_article.py`. It exists so one specific historical benchmark run — `llm_results/all_llm_results_20250905102307/`, scored to `benchmarking/metrics/09_05` — stays reproducible against the exact GT data it was originally scored against, decoupled from ongoing edits to `rev2025/` and this folder. `eval_multi_llm_article.py`'s `main()` currently hardcodes `human_gt_dir` to this folder, reflecting that pin.

**It is not a canonical or rationalized GT baseline, and it is stale.** Annotation work has continued since the snapshot was taken — Anonymous Source Type reformatting (`92885a0`), Source Justification fixes (`ff8c31e`), the medium-of-contact removal above (`ae27e70`), and other substantive corrections (e.g. story 9's joint "Stephen and Ayesha Curry" row later split into two separate named-person rows). As of 2026-08, 29 of the 30 stories with a `rev2025` revision have diverged from their `20250904` counterpart; only story 14 still matches exactly. The 13 non-revised stories match their `20250904` counterpart, since nothing in the base folder changed after the snapshot except story 22 (see commit `53b384e`, which normalized story 22's `ae27e70` fix into a proper `rev2025/22-Judge-rules-breona-taylor-rev2.csv` file — `20250904/22-Judge-rules-breona-taylor.csv` still reflects the pre-fix text, as expected for a pin taken before that fix existed).

For current/canonical GT data, use `rev2025/` + this folder as described above, not `20250904/`.
