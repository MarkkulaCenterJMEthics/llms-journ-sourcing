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
