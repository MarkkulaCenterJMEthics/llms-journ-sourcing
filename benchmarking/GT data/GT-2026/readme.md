# GT-2026/ — the 2026 canonical GT baseline

This folder is the single canonical ground-truth source for all 43 stories (1–43), consolidated from the base folder and `rev2025/` one level up, as the base to receive the 100–200 incoming student-annotated stories for the 2026 dataset expansion. Every story has exactly one file here, named `{story}-{slug}.csv`.

## How it was built (commit `479e00a`)

- 31 stories copied from `rev2025/*-rev2.csv` (the superseding revision at the time).
- 12 stories with no `rev2025` revision at the time, copied from the base folder as-is.
- Filenames normalized to drop the `-rev2` suffix carried over from `rev2025/` — every file follows one consistent pattern instead of a mix of suffixed and unsuffixed names.

Since then, story 17 was promoted into `rev2025/` too (see below), so the current split feeding this folder is 32 revised + 11 base-only, though that no longer matters day to day — this folder is itself the thing to read from.

## Type of Source taxonomy cleanup (2026-09-01, commits `fd6d33b`, `554cee4`, `d9e1711`, `0731656`)

Full review of `Type of Source` across all 43 stories, fixing data-quality variance at the GT source rather than accounting for it in eval-code normalization — this was an explicit decision: the 2024–2025 benchmark code handled variance on the eval side (`eval_multi_llm_article.py`'s `SourceTypeMapping`), but going forward the GT side gets fixed instead. Mirrors `inter_coder_reliability/v13all-icrclaude.py`'s `TYPE_OF_SOURCE_VARIANTS` approach, but fixed at the source this time.

All 645 `Type of Source` values across the 43 stories now fall into exactly one of the 6 canonical categories, with zero blank or unrecognized values:

| Category | Rows | What was fixed |
|---|---|---|
| Named Person | 383 | Legacy `named individual` synonym (3 casing variants, 205 cells) normalized to `Named Person`; embedded newlines stripped from 16 cells (stories 6, 24) |
| Named Organization | 109 | Lowercase `named organization` (5 cells, story 17) normalized; embedded newlines stripped from 10 cells (story 6) |
| Unnamed Group of People | 58 | 8 casing/typo variants (`Unamed group of people`, etc., 68 cells) normalized; embedded newlines stripped from 5 cells |
| Document | 71 | Trailing whitespace stripped from 3 cells (stories 19, 43) — found via a blanket sweep of the whole `Type of Source` column across every category, after the per-category passes above kept surfacing the same artifact in the same couple of files |
| Anonymous Source | 17 | Already clean |
| Unnamed Person | 7 | Already clean |

Every structural edit was verified with a CSV round-trip test (parse → rewrite unchanged → byte-identical to the original) before being applied, so each diff touches only the intended cells.

Per-category review CSVs from this cleanup are kept in the sibling `GT-inspection-2026/` folder.

## Other fixes from the same pass

- **Legacy anonymity column removed.** A `Y/N` anonymity column (five header spellings across files: `Anomynity? (Y/N)`, `Anonymity? (Y/N)`, `Anonymity Y/N`, `Anomynity (Y/N)`, `If Unnamed, Anomynity? (Y/N)`) predated the `Anonymous Source` / `Unnamed Person` category split and had become fully redundant — verified before removal: only 9 of 496 affected rows had any value at all, and every one was 100% derivable from `Type of Source` (`Y` only on `Anonymous Source` rows, `N` only on a non-anonymous row). Dropped from 31 files here.
- **Story 17 row-numbering bug fixed.** `17-homelessness-santa-clara.csv` had 2 rows with a completely blank `No` — real statements that were never assigned a sequence number. Renumbered all 20 rows sequentially. This is also why story 17 now has a `rev2025/17-homelessness-santa-clara-rev2.csv` file it didn't have before (see that folder's readme).
- **Story 35 header fixed.** `35-Trump-Tariffs-Rattling-Meta.csv` had `Source Justification (includes addition source characterizations beyond the title)` instead of the plain canonical header, which broke `eval_multi_llm_article.py`'s strict `pd.read_csv(usecols=[...])` matching.

All 43 files were confirmed to load cleanly under that same `usecols` check, and `eval_multi_llm_article.py` was run end-to-end against this folder (5-article subset) with no errors after each round of fixes.

## File organization

Single header row: `No, Sourced Statements, Type of Source, Name of Source, Title of Source, Source Justification`. No anonymity column (see above).

## Not yet formalized

`eval_multi_llm_article.py`'s `main()` still hardcodes `human_gt_dir = "benchmarking/GT data/20250904/"` (a stale pin for one historical benchmark run — see the parent folder's readme) rather than pointing at this folder. That's deliberate: this folder was set up and sanity-tested first, formalizing it as the actual eval default is a separate step still to come.

## Base folder and `rev2025/`

The base folder (`benchmarking/GT data/*.csv`) and `rev2025/` were intentionally left untouched by all of the above except where a fix was promoted into a proper `rev2025/*-rev2.csv` file (stories 17, 22) — both are treated as frozen originals now that this folder is the canonical baseline going forward. Any future GT revision work should happen here, not there.
