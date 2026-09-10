# SJ Analysis Notes

Exploratory Source Justification typology work (Task 1) — **not** a benchmark-project deliverable, not logged in `development-of-v59.md`. For Mizzou/Penn State talks + eventual dataset validity cleanup. Revive this note after GT-II migration, when both corpora share one schema.

## Original asks (re-run these post-migration, same order)

1. Pull all non-empty Source Justification text from GT-I (GT-2026, migrated) + GT-II (raw XLSx, unmigrated) combined. First pass: examine what *types* of SJ content exist, using the system prompt's SJ definition/Notes/Example 10 as a starting hint, but stay inductive.
2. Intersect the resulting categories with Type of Source — look for preponderance/clustering by type.
3. Within Named Person only, intersect further by Title of Source present vs. absent — test the hunch that non-titled sources are justified more via lived experience than titled ones.

## Method used (2026-09-09/10 first pass)

- Distinct-text basis (deduped SJ text, not raw rows) — avoids inflating counts from legitimate SJ carry-forward (Note 22/24). Keep this convention for comparability.
- 488 distinct values, single-coder, rule-assisted classification. **First pass only — not double-coded, no IRR run on the categories themselves.**
- 61 literal `"null"` strings (GT-II) excluded as a data-quality artifact, not content.
- 12 categories used: EXP, LIV, STK, ORG, ANN (from the prompt's own SJ definition/Example 10) + DOC, FAM, LGL, BAR (inductive content categories) + ATV, BKG (inductive validity-flag categories) + OTH (catch-all, ended at 0 after manual review).
- Deliverable: `~/Documents/SJ_Typology_by_TypeOfSource.docx` (two sections — Type-of-Source cross-tab, then Titled-vs-Non-titled Named Person follow-up).

## Key numeric findings to compare against the re-run

- Named Person = 65% of corpus; Lived experience 40%, Expertise 21%.
- Named Organization: Org function/mission 41%, Document/provenance 20%.
- Document: Document/study provenance 66%.
- Anonymous Source: Anonymity/access disclosure 79%.
- Unnamed Person: Anonymity-adjacent language 45% despite no formal disclosure — watch this one specifically against item 1's migration.
- Unnamed Group of People: flattest distribution; highest BKG rate (20%) — hardest type to write source-specific SJ for.
- Named Person, Titled vs Non-titled: LIV 32% vs 51%; EXP 28% vs 10%. Hunch confirmed.
- ATV (Note 20 violation) appeared only in Titled Named Person rows (9 of 191, 0 of 134 Non-titled).

## Caveats to carry forward into the post-migration re-run

- GT-II's Type of Source values are **pre-migration** — the Anonymous Source/Unnamed Person split (and the 45% finding above) may shift once item 1's narrow disclosure boundary is applied. Re-check specifically.
- ~7% of all SJ entries flagged as validity concerns by the schema's own definitions: 2.3% ATV (Note 20, mere attribution pathway) + 4.5% BKG (Note 21, general story background not about that specific source). User intends to use these for final dataset cleanup before benchmark scoring, not just as typology commentary — don't just re-describe them next time, action them.
- Incidental GT-II schema-quality issues spotted while pulling this data (name sitting in the Type-of-Source column, multiple names crammed into one Type field, non-canonical Type spellings) — not SJ-typology findings themselves, but feed the migration's schema-violation cleanup step.
- Extraction/classification scripts and intermediate JSON from this pass live only in a session scratchpad, not committed anywhere — regenerate from source files, don't expect to find them.

## Open design question for the re-run

- 12 categories is likely more than needed. User plans to simplify toward fewer, larger principles **without losing signal** before the post-migration run. When redesigning, note that EXP/LIV and STK/LIV were the boundaries needing the most judgment calls during manual classification — the likeliest places a simplified scheme could quietly lose signal if collapsed carelessly.
