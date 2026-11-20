# GT Phase 4 tools

Scripts used for Phase 4 of the GT migration (Sourced Statement row-set audit) on GT-I
(`benchmarking/GT data/GT-2026/`) and GT-II (`benchmarking/GT data/GT-II/`). Background and every
decision made with them: `development-of-v59.md` items 76 (GT-I) and 77 (original GT-II batch);
per-annotator change log: `gt-ii-student-housekeeping.md`.

Snapshot committed 2026-11-19, mid-way through Phase 4 on the original GT-II batch. Until Phase 4 on
that batch and SZ-9 is finished, the working copy is `~/Documents/gt_phase4_tools/`; this folder is
refreshed from it when the phase is done.

## Workflow

1. **Step 1 -- text fidelity only.** `step1.py` proposes mechanical fixes that make each row match the
   article text (whitespace, quote marks, Title capitals, fragments -> full sentences, paragraph
   splits, wording differences). It never decides whether a row is a Sourced Statement.
   `python3 step1.py --set orig|sz9|N,N,..` writes `step1_proposals.json` / `step1_report.md`;
   `python3 step1.py --apply ID,ID,..` applies approved proposals. `step1_tables.py` and
   `step1_partB.py` print the review tables (full before-rows).
2. **Step 2 -- judgment.** `step2.py` lists candidates only (never edits): 4a missed statements
   (recall), then 4b invalid rows (precision; includes `speakercheck.py`), SJ issues and Title
   wording against the text (`titlecheck.py`). Each candidate is read against the text and the
   written rules (v61 Minor list and patches, v62 items in `prompt-revision-list.md`) before it is
   shown with a lean. Approved edits go through `step2_apply.py ops.json` (content-matched,
   rows renumbered, text order kept).
3. **Step 3 -- merges.** Same-source, same-paragraph rows are merged only after 4b, re-detected
   with `step1.py` on the surviving and added rows.

Shared helpers: `gtlib.py` (paths, CSV I/O preserving BOM/newlines, sentence splitting, text
location, annotator mapping per file). Calibration: `calib_step2.py` (GT-I before/after Step 2),
`calib_title.py` (Title check against the 2026-11-13 SZ Pass 1, commit 5647714).
`partB_housekeeping.py` drafts housekeeping entries for a Step 1 Part B apply.

Requires Python 3.10+ and `rapidfuzz`. Repo root is found automatically (or set `GT_REPO`).
