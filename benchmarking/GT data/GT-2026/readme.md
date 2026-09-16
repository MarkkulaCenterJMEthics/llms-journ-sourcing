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

## Source Descriptors / Source Justification re-audit (v59/60 schema migration, 2026-09-15 ongoing)

Third and final leg of a 3-part re-audit (new-material GT-II batch, then the original 48-story GT-II batch, then this folder — see `development-of-v59.md`'s Migration punchlist item 29) checking already-populated `Source Descriptors` values against the actual v59/60 schema rules, after a structural gap was found where the schema-application rules weren't rigorously re-applied row-by-row during the original population pass. Full scan: 648 rows across all 43 files, 174 with a populated `Source Descriptors` value, 0 with a single-word `Source Justification`. Fixes applied so far, reviewed and approved one at a time:

- **Data hygiene.** `14-many-police-calls.csv` row 20 (Daniel Quipp): stripped a literal trailing `\n` from `Title of Source` ("Chair of the selectboard"). `15-Newsom-declares-emergency.csv` rows 4, 12, 16 (Southern California Edison): stripped a trailing space from `Name of Source`.
- **`25-ballot-access-trans.csv` row 2 — Type of Source reclassified from Document to Named Organization.** The row's `Source Descriptors` ("Letter") and `Sourced Statement` described "lawyers for the American Civil Liberties Union" drafting a letter to election workers, but nothing in the article shows the reporter directly accessed the letter's contents (no URL, no direct quote, no "the letter states") — the word "letter" appears exactly once in the whole piece, in a sentence structurally parallel to two other organization-roundup sentences right after it (VoteRiders, Equality Florida) that are background descriptions of org activity, not document citations. Per the Document definition ("the document is the source only if the author of the story has accessed the document itself and used its contents directly for the story"), this doesn't qualify as a Document source. The lawyers are never named individually — only "lawyers for the ACLU," acting on the org's behalf — and reporters referencing a letter this way (naming the org) typically mean an org-authorized letter, not lawyers acting in a personal capacity, even where the reporter has seen it. That maps to Named Organization's own carve-out for "an unnamed spokesperson/official of a named org." Fixed: Type of Source → Named Organization, Name of Source → "American Civil Liberties Union", Title of Source and Source Descriptors blanked (no bare category word for ACLU stated in the text), Source Justification left untouched — the existing text ("Staff for the organization have also held clinics elsewhere...") now reads correctly as justification for the org as source, rather than appearing mismatched against a Document row.

- **`39-84pct-coral-reefs.csv` rows 2 & 3 — Name typo fixed, Source Descriptors blanked.** `Name of Source` was "International Coral Relief Intiative" — the article's own Sourced Statement text spells it correctly ("International Coral Reef Initiative"), so "Relief" should have been "Reef" and "Intiative" was missing an "i." Fixed both rows to the correct name. Separately, `Source Descriptors` held "a mix of more than 100 governments, non-governmental organizations and others" — a composition description (what the org is made of), duplicated verbatim from `Source Justification`, not a bare category word (nonprofit/coalition/partnership/etc.) as the Named Organization Source Descriptors rule requires. No such category word is stated anywhere in the article for this org, so per the rule ("if no bare category word is present in the text, leave Source Descriptors null"), blanked it on both rows — the composition text stays where it already correctly sits, in Source Justification. This corrects an earlier sign-off in `development-of-v59.md` (item 15) that had conflated composition with category and cleared this value as fine; see that item's correction note for the full reasoning.

- **`6-OUSD-basic-job.csv` row 13 — reviewed, no change.** `Source Descriptors` "safe drinking water policy" (Document, SS: "The district's safe drinking water policy states that..."). Confirmed correct as-is: "policy" is the genre word, and "safe drinking water" functions as the policy's actual subject/name rather than pure elaboration.
- **`34-human-ca_guanranteed_income.csv` row 23 — reviewed, no change.** `Title of Source` "Researchers" and `Source Descriptors` "advocates" both attach to one Unnamed Group of People row ("Researchers and advocates stress how..."), each word describing a different one of the two groups jointly cited in that sentence — not a mismatch.
- **`2-Best-DDR-player.csv` (Roger Clark, 7 rows) and `6-OUSD-basic-job.csv` row 3 (OUSD) — reviewed, no change.** Both are comma-separated multi-value Source Descriptors ("friend, programmer, DDR enthusiast"; "Parents, students, school staff") — confirmed each value is manifest verbatim in the article text. The v60 schema explicitly sanctions this (multiple descriptors captured to distinguish richer, multi-faceted source portrayal from flattening/reductive labeling — see `system_prompt_v60.txt`'s Source Descriptors definition) — no gap to log, this was already correctly drafted into the prompt (punchlist item 11).

**Batch complete (2026-09-15).** All 174 populated-`Source Descriptors` rows across the 43 files were reviewed; 0 single-word `Source Justification` values were found. This closes step 3 of `development-of-v59.md`'s punchlist item 29 (the 3-part SD re-audit plan) — step 4, formally adding an SD audit sub-pass to `gt-migration-methodology.md`, is still open.

## Not yet formalized

`eval_multi_llm_article.py`'s `main()` still hardcodes `human_gt_dir = "benchmarking/GT data/20250904/"` (a stale pin for one historical benchmark run — see the parent folder's readme) rather than pointing at this folder. That's deliberate: this folder was set up and sanity-tested first, formalizing it as the actual eval default is a separate step still to come.

## Base folder and `rev2025/`

The base folder (`benchmarking/GT data/*.csv`) and `rev2025/` were intentionally left untouched by all of the above except where a fix was promoted into a proper `rev2025/*-rev2.csv` file (stories 17, 22) — both are treated as frozen originals now that this folder is the canonical baseline going forward. Any future GT revision work should happen here, not there.
