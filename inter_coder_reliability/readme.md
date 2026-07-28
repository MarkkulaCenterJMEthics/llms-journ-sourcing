# Intercode Reliability Scoring (ICR) for human annotations that were used to create the ground truth data for the LLM benchmarking project. 

## Setup

```bash
conda create --name llm-news python=3.10 -y
conda activate llm-news
cd inter_coder_reliability
pip install -r requirements.txt
```

(This directory's `requirements.txt` is separate from the one in the repo root — it pulls in `simpledorff`, `sentence-transformers`, `scikit-learn`, and `irrCAC`, which the ICR script needs but the rest of the repo does not. `numpy` is pinned to `1.26.4`, not a newer release, because `irrCAC`'s packaged metadata hard-pins `scipy==1.12.0`, which caps `numpy<1.29.0` — verified this doesn't change any of the script's Krippendorff's Alpha results versus the newer numpy it replaced.)

**If you already have the `llm-news` env set up from before:** `git pull` updates the code but does *not* update your installed packages. If a later `git pull` adds a new dependency to `requirements.txt` (as happened when `irrCAC` was added for Gwet's AC1), your code will be current but your environment won't be — showing up as a `ModuleNotFoundError` for whatever was just added. Fix: re-run `pip install -r requirements.txt` (with `llm-news` activated, from this directory) after every `git pull`. It's safe to re-run any time — it only installs what's missing or out of date.

The first run downloads the `all-MiniLM-L6-v2` sentence-embedding model, so an internet connection is required at least once.

**Windows:** run the commands above from **Anaconda Prompt** (installed alongside Miniconda/Anaconda). If you use PowerShell or VS Code's terminal instead, you'll need to run `conda init powershell` once and reopen the terminal before `conda activate` works.

**Expected harmless warning:** every run prints `Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.` This is safe to ignore — it fires on every load (not just the first), including when the model is already cached, because `sentence-transformers` does a quick unauthenticated metadata check against the Hub each time. It doesn't affect correctness or require any action. (Investigated forcing offline mode to silence it — `HF_HUB_OFFLINE` gets locked in at import time with no safe same-process fallback if the model isn't cached yet, so that route was dropped as too fragile for a cosmetic fix. The clean fix, if wanted later, is setting a real `HF_TOKEN` — a free Hugging Face account's read-only token — which removes the warning via the officially supported mechanism.)

## Usage

usage: v13all-icrclaude.py [-h] csv1 csv2

Calculate Inter-Coder Reliability using semantic similarity and fuzzy matching

positional arguments:
  csv1        Path to first annotator CSV file
  csv2        Path to second annotator CSV file

options:
  -h, --help  show this help message and exit

## How the Krippendorff's Alpha input is built, column by column

This section documents exactly what data each column's alpha score is computed from — the functions involved, in call order:

1. **`load_and_prepare_data(csv1, csv2)`** — reads both CSVs, checks the 5 required columns exist (`Sourced Statements`, `Type of Source`, `Name of Source`, `Title of Source`, `Source Justification` — case-sensitive), replaces every blank cell in those columns with the literal string `MISSING_SENTINEL = "MISSING_VALUE"`, and assigns `item_id` 1..N by row position (row *N* of csv1 is always compared to row *N* of csv2 — there's no content-based matching).
2. **`compute_ss_gate_mask(df1, df2)`** — builds one boolean array, one entry per row: `True` if *both* annotators have a non-missing `Sourced Statements` value for that row, `False` otherwise. Used only by the four "dependent" columns below.
3. **`prepare_data_for_column(df1, df2, column_name, gate_mask=None)`** — builds the actual table handed to the alpha calculation: three columns, `item_id` / `annotator` / `annotation_text`, two rows per item (one per annotator). If a gate mask is given, rows where it's `False` are dropped from this table entirely before it ever reaches `simpledorff` — not scored as agreement or disagreement, just absent.
4. **`calculate_icr_for_column(...)`** — passes that table plus a *distance function* to `simpledorff.calculate_krippendorffs_alpha_for_df()`. The distance function is the entire definition of "how different are these two values" for that column; `simpledorff` calls it internally, both within-item (for observed disagreement, `Do`) and between every pair of distinct values that appear anywhere in the column (for expected disagreement by chance, `De`).

There are three distance functions now, and which one applies depends on what kind of data the column actually holds:

- **`fuzzy_distance_metric(a, b)`** — Levenshtein-ratio string similarity (via `fuzzywuzzy`/`python-Levenshtein`), converted to a `0.0`–`1.0` distance. Used for **`Name of Source`** only: names are arbitrary proper nouns with no fixed vocabulary, so forgiving a minor spelling variant of the *same* name is the right behavior, and there's no finite category list to normalize against.
- **`semantic_distance_metric` / `build_cached_semantic_distance_fn(model, texts)`** — cosine distance between `all-MiniLM-L6-v2` sentence embeddings, `0.0`–`1.0`. Used for **`Sourced Statements`**, **`Source Justification`**, and **`Title of Source`**: free text where two annotators rarely type the identical sentence, but may mean the same thing (paraphrase, partial quote, punctuation differences). The cached variant precomputes one embedding per distinct value up front (see the "silent-drop" history below for why — `simpledorff` calls the distance function once per *pair* of distinct values, which is quadratic and was taking ~25 minutes per column on a 100-row file before caching).
- **`canonicalize_type_of_source(value, source_file, row_number)` + `nominal_distance_metric(a, b)`** — used for **`Type of Source`** only. See the "Why `Type of Source` isn't fuzzy or semantic" note just below — it's a closed 6-category taxonomy, not free text, so it gets normalized to a canonical category first, then compared with a strict same-category-or-not (`0.0`/`1.0`, no partial credit) metric instead.

The free-text and name functions share the same missing-value rule via `_is_missing(value)` (true for real `NaN` or the `MISSING_VALUE` sentinel): **both missing → distance `0.0`** (agreement — neither annotator found anything there); **one missing → distance `1.0`** (disagreement); **both present → the real fuzzy/semantic comparison**. `nominal_distance_metric` follows the same missing-value rule, but compares already-canonicalized categories exactly once both are present.

Column by column, this is what actually feeds the alpha calculation:

| Column | Distance function | Gated on `Sourced Statements`? | A "value" looks like |
|---|---|---|---|
| `Sourced Statements` | semantic (cached) | **No** — never gated | A full sentence/quote, or `MISSING_VALUE` if an annotator missed the statement entirely |
| `Type of Source` | canonicalize + nominal | Yes | One of the 6 canonical categories (`Named Person`, `Named Organization`, `Document`, `Anonymous Source`, `Unnamed Person`, `Unnamed Group of People`), or `MISSING_VALUE` |
| `Name of Source` | fuzzy | Yes | A name string, or `MISSING_VALUE` |
| `Title of Source` | semantic (cached) | Yes | A title phrase, or `MISSING_VALUE` |
| `Source Justification` | semantic (cached) | Yes | A quoted/paraphrased justification sentence, or `MISSING_VALUE` |

"Gated" means: rows where either annotator's `Sourced Statements` is missing are excluded from that column's alpha entirely (see `compute_ss_gate_mask` above) — because there was no annotation work on `Type of Source`/etc. to compare if one annotator never registered a sourced statement at that row in the first place. `Sourced Statements` itself is exempt from gating by design: a missing statement there *is* the disagreement being measured.

### Why `Type of Source` isn't fuzzy or semantic (added 2026-07-24)

`Type of Source` originally used the same fuzzy string metric as `Name of Source`. That's wrong for a closed taxonomy: fuzzy (and semantic) distance measures how similar two strings *look* or *mean* — for genuinely different categories that happen to share words, that gives false partial credit instead of the full disagreement a category mismatch should score. Verified with real category-label pairs:

| Pair | Fuzzy distance | Semantic distance |
|---|---|---|
| `Named Person` vs `Named Organization` | `0.4000` | `0.5975` |
| `Anonymous Source` vs `Named Organization` | `0.8200` | `0.7563` |
| `Document` vs `Named Person` | `0.7000` | `0.8159` |
| `Anonymous Source` vs `Unnamed Person` | `0.6700` | `0.5146` |

Neither metric is uniformly better: semantic distance pushes `Named Person`/`Named Organization` closer to the correct `1.0` (`0.60` vs `0.40`), but pushes `Anonymous Source`/`Unnamed Person` *further away* from it (`0.51` vs `0.67`) — embeddings recognize these as conceptually related ("both about unidentified sourcing") even though they're meant to be mutually exclusive categories. Switching metrics just relocates the problem; it doesn't fix it, because both metrics answer "how similar is this text/meaning," not "is this the same category."

The actual fix: `Type of Source` values are normalized to one of the 6 canonical categories first (`canonicalize_type_of_source`, using the `TYPE_OF_SOURCE_VARIANTS` mapping — built fresh for this project's own CSVs, covering every real typo/casing/whitespace variant found across all the ICR files and all 30 `benchmarking/GT data/rev2025/` ground-truth files, including a legacy synonym `"Named Individual"` → `Named Person` from an older annotation batch), *then* compared with `nominal_distance_metric`: exactly `0.0` if the same canonical category, exactly `1.0` if not. Typos/casing/whitespace collapse correctly (no longer penalized); genuinely different categories score full disagreement (no longer given partial credit). A value that doesn't match any known category or variant raises immediately, naming the exact CSV file and row — `Type of Source` is a defined, closed set, so an unrecognized value is treated as an annotator error to fix in the source data, not something to guess at or silently score.

## Worked example: from CSV rows to a Krippendorff's Alpha score

Using a real pair, `Spr25-AppleAI-KG-ann1.csv` / `Spr25-AppleAI-SV-ann2.csv` (19 rows), run as `python3 v13all-icrclaude.py Spr25-AppleAI-KG-ann1.csv Spr25-AppleAI-SV-ann2.csv`.

### Example 1: `Type of Source` (canonicalize + nominal metric) — how `1.0000` is actually computed

Item 6 has no `Sourced Statements` from annotator2, so it's gated out of this column, leaving 18 items / 36 total ratings. Here's what the two annotators actually wrote for the rest (condensed):

```
Item 1:  'Unnamed Group of People '  vs  'Unamed Group of People'     (trailing space / typo)
Item 2:  'Document'                  vs  'Document'
Item 3:  'Named Organization'        vs  'Named Organization'
Item 5:  'Anonymous Source'          vs  'Anonymous Source'
Item 7:  'Named Person'              vs  'Named Person'
Item 9-19 (mixed): eight more 'Unnamed Group of People' vs 'Unamed Group of People' typo pairs
                   (items 9, 10, 11, 12, 14, 16, 18, 19), plus exact matches on Named Person
                   (items 13, 17) and Document (item 15)
```

**Step 1 — canonicalization**, via `canonicalize_type_of_source`, runs *before* any distance is computed (in `load_and_prepare_data`, on load). Both the trailing-space variant and the typo resolve to the same canonical string:

| Raw value | Canonical category |
|---|---|
| `Unnamed Group of People ` | `Unnamed Group of People` |
| `Unamed Group of People` | `Unnamed Group of People` |
| `Document`, `Named Organization`, `Anonymous Source`, `Named Person` | (already canonical, unchanged) |

**Step 2 — per-item distances**, via `nominal_distance_metric`, now operating on canonical values only:

| a | b | distance |
|---|---|---|
| `Named Person` | `Named Person` | `0.0000` |
| `Unnamed Group of People` | `Unnamed Group of People` | `0.0000` (both sides already canonicalized to the same category) |
| `Document` | `Named Person` | `1.0000` (hypothetically — doesn't occur in this file, but this is what a real mismatch scores now, vs. `0.7000` under the old fuzzy metric) |

**Step 3 — `simpledorff` builds the marginal frequency table.** Because the typo variants no longer exist as distinct strings, the class count drops from 7 (under fuzzy) to 5:

```
class_freqs = {
  'Anonymous Source': 2, 'Document': 6, 'Named Organization': 4,
  'Named Person': 6, 'Unnamed Group of People': 18
}
total ratings = 36
```

**Step 4 — observed and expected disagreement:**

```
Do = 0.0000    (every item, canonicalized, is an exact match -- zero observed disagreement)
De = 880.0000
N  = 36 (total ratings)

alpha = 1 - (Do / De) × (N - 1)
      = 1 - (0.0000 / 880.0000) × 35
      = 1.0000
```

That matches the script's printed `✅ Krippendorff's Alpha for 'Type of Source': 1.0000` exactly. This is the clean result of the fix: under the old fuzzy metric, this file scored `0.9784` — not because of any real disagreement, but because the repeated `Unamed`/`Unnamed` typo was treated as *slight* disagreement instead of none. With canonicalization, that phantom disagreement is gone and the score correctly reflects what actually happened: perfect agreement on every comparable item.

### Example 1b: `Type of Source` at real scale — 100-row file, same formula, no shortcuts

`Type of Source` is a fixed taxonomy, not free text, so its vocabulary doesn't grow with sample size — even at 100 rows (`ICRcombined/100R-ICRcombined-ann1.csv` / `-ann2.csv`) there are only 5 canonical classes among the 95 gated-in items (190 total ratings), so this is exactly as hand-tractable as the 19-row example, just at real scale:

```
class_freqs = {
  'Named Person': 108, 'Named Organization': 42, 'Unnamed Group of People': 24,
  'Document': 14, 'Anonymous Source': 2
}
total ratings = 190   (95 items x 2 annotators; 5 items gated out for a missing Sourced Statement)
```

(One more `Unnamed Group of People` than under the old fuzzy count, since the whitespace-variant item that used to sit in its own `'Unnamed Group of People '` bucket now correctly merges into the real category.)

Only 2 of the 95 comparable items have *any* disagreement left — every other item is an exact match after canonicalization (`Do` contribution `0.0000`):

| Item | ann1 | ann2 | distance |
|---|---|---|---|
| 40 | `Named Organization` | `Document` | `1.0000` (genuinely different categories — up from `0.6900` under the old fuzzy metric) |
| 54 | `Named Organization` | `Document` | `1.0000` (same pair again) |

```
Do = 4.0000
De = 21896.0000   (sum over all 5 x 5 = 25 class pairs, e.g. 'Named Person' x 'Document' pulls in 108 x 14 x 1.0)
N  = 190

alpha = 1 - (Do / De) x (N - 1)
      = 1 - (4.0000 / 21896.0000) x 189
      = 0.9655
```

Matches the script's printed `Type of Source: 0.9655` on this file exactly. Compare this to the earlier fuzzy-metric result of `0.9558`: `Do` actually roughly doubled (`2.76` → `4.00`, since the two genuine mismatches now score full `1.0` instead of partial `0.69`), but `De` grew even more (every one of the 25 class pairs is now valued at its true nominal distance rather than a mix of partial fuzzy values), so the net effect is alpha moving slightly *up*. This is the corrected, methodologically sound number — not just a different number.

### Example 2: `Sourced Statements` (semantic metric) — same mechanism, not hand-tractable at scale

The formula above (`Do`, `De`, `alpha = 1 - (Do/De)(N-1)`) is identical for the semantic columns — only the distance function changes. The difference is scale: free-text columns are almost entirely *unique* values, so `De`'s pairwise sum runs over every distinct sentence in the column, not a handful of fixed categories. On this same 19-row file, `Sourced Statements` reaches 31 distinct texts out of 37 total ratings (31² = 961 pairwise terms); on the 100-row file it's 155 distinct texts (~24,000 terms) — infeasible to hand-verify, which is exactly why `build_cached_semantic_distance_fn` exists (see the pipeline description above). A few real per-item distances from this same run illustrate what feeds that sum:

| Item | ann1 | ann2 | distance |
|---|---|---|---|
| 2 | *"While Vision Pro sales have been disappointing..."* | *(identical)* | `0.0000` |
| 9 | *"The A.I. stumble was set in motion in early 2023. Mr. Giannandrea..."* | *"Mr. Giannandrea..."* (same sentence, missing the leading clause) | `0.2907` |
| 6 | *"Apple's software chief, told employees that..."* | `MISSING_VALUE` (annotator2 never recorded a statement here) | `1.0000` |

For the full, verified real-world numbers on a larger sample, see the saved run `ICRcombined/July24-100R-CanonicalTypeOfSource-ICR.txt` (100 rows; `Source Justification` computed at `0.7008` there) and the troubleshooting case below for how these numbers were validated against a same-data control run of the pre-fix code. (`ICRcombined/July26-100R-revisedICR.txt` is an earlier snapshot from right after the silent-drop fix, before the `Type of Source` canonicalization and a `Title of Source`/`Source Justification` data-quality fix described further below — kept for history, not current.)

### Summary: all five columns, 100-row file

The same formula (`Do`, `De`, `alpha = 1 - (Do/De)(N-1)`) applied to every column on the full `100R-ICRcombined-ann1.csv` / `-ann2.csv` pair. `Type of Source` is fully derived above; the free-text columns' pairwise sums aren't reproduced here (tens of thousands of terms), but every number below is the actual verified output of the real calculation, not an estimate or approximation:

| Column | Items (post-gate) | Total ratings | Distinct values | `Do` | `De` | `alpha` |
|---|---|---|---|---|---|---|
| `Sourced Statements` | 100 | 200 | 155 | 18.6287 | 35209.9727 | `0.8947` |
| `Source Justification` | 95 | 190 | 64 | 39.0708 | 24683.3926 | `0.7008` |
| `Type of Source` | 95 | 190 | 5 | 4.0000 | 21896.0000 | `0.9655` |
| `Name of Source` | 95 | 190 | 41 | 8.1200 | 27828.6800 | `0.9449` |
| `Title of Source` | 95 | 190 | 41 | 8.1204 | 25318.0215 | `0.9394` |

`Sourced Statements` is never gated, so it keeps all 100 items; the other four columns lose the same 5 items where one annotator recorded no `Sourced Statements` at all. These `alpha` values match `ICRcombined/July24-100R-CanonicalTypeOfSource-ICR.txt` exactly, and reflect both the `Type of Source` canonicalization fix and the `Title of Source`/`Source Justification` data-quality fix described below.

*This documentation was written as a clear methodology baseline before evaluating Gwet's AC1/AC2 as an alternative reliability measure, so any difference in results could be attributed to the formula, not to ambiguity about what data each one actually consumed — see the "Gwet's AC1" section below for what came of that.*

## Data-quality fix: relational descriptions don't belong in `Title of Source` (found 2026-07-24)

While reviewing the `Type of Source` canonicalization work above, a real annotation error turned up: one story (Nebraska carjacking/murder case — the source of the "Melanie Roberts, victim's granddaughter" quotes; also informally nicknamed `DogWalker` in this dataset after a detail of the crime) had `Victim's granddaughter` recorded in `Title of Source` for its `Melanie Roberts` rows, with `Source Justification` left as a `none` placeholder.

That's a miscategorization. `Title of Source` is meant for a formal position/rank/authority/expertise (`SFO spokesperson`, `King County Superior Court Judge`) — not a relational description of how the source is connected to the story (`victim's granddaughter`, `victim's neighbor`). "Victim's granddaughter" is exactly the kind of *added context about the source* that belongs in `Source Justification` instead.

Both annotators made the identical classification choice independently, so this wasn't a disagreement the ICR score would have caught — it required a human reviewing the actual content. Fixed by moving the text from `Title of Source` to `Source Justification` (replacing the `none` placeholder) and blanking `Title of Source`, applied consistently across every copy of this story in the repo: the standalone `Spr25-DogWalker-ann1/2/3.csv` files, their `ICRcombined/` duplicates, and all three combined snapshots (`ICRcombined-ann1/2.csv`, `79R-ICRcombined-ann1/2.csv`, `100R-ICRcombined-ann1/2.csv`) — 11 files, 3 rows each (33 field edits total). Applied as precise text substitution rather than a full CSV rewrite, to avoid disturbing these files' original CRLF line endings or other formatting.

This is a narrower, one-story data fix — a separate, broader pass to bring `benchmarking/GT data/` (the ground-truth CSVs used by the LLM-benchmarking pipeline, not this ICR script) to the same standard is tracked as follow-up work, not done here.

## A note on "Undefined" alpha scores

A column can report `Undefined` instead of a number between -1 and 1. This isn't a bug — it means every comparable item in that column got the exact same value from both annotators (e.g. every source classified as `Named Person`, with no other category present at all). Krippendorff's Alpha needs some variability across categories to establish a chance-agreement baseline; with zero variability there's nothing to divide by, so the result is mathematically indeterminate (0/0), not 0.0 or 1.0.

This is most likely on **small samples**, where it's entirely possible for every annotator to agree by chance on a single category simply because the sample never contained a different one to disagree about — it says more about sample size than about annotator reliability. It gets rarer as the sample grows and more categories naturally appear (an 11-row test file hit this on `Type of Source`; the same column computed fine on a 100-row file). When you see `Undefined`, the message names the value and how many items/ratings tied, so you don't need to dig through the per-item log to explain it.

## Gwet's AC1, printed alongside Krippendorff's Alpha for `Type of Source` (added 2026-07-24)

`Type of Source`'s summary line and per-column log now also print **Gwet's AC1**, computed via the [`irrCAC`](https://pypi.org/project/irrCAC/) package, purely for comparison — it does not replace Krippendorff's Alpha, and no other column is affected.

**Why this column, and why AC1 specifically:**

- Gwet's AC1/AC2 was designed as a "paradox-resistant" alternative to Cohen's/Fleiss' Kappa — its formula divides by `(1 - pe)` rather than by `pe` itself, so it doesn't hit the same `0/0` singularity Krippendorff's Alpha does at zero variability. Verified directly: on the `94-SZ-Met_Gala.csv`/`94-AV-Met_Gala.csv` file, where Krippendorff's Alpha reports `Undefined` for `Type of Source` (see above), **AC1 evaluates cleanly to `1.0000`**.
- AC1 uses strictly nominal (identity) weights: same category = full credit, different category = none. **AC2** is the same formula with a *weighted* matrix, appropriate when categories have a meaningful degree of closeness (ordinal/Likert-style data). `Type of Source`'s 6 categories don't — there's no defined "closeness" between `Named Person` and `Named Organization` — which is the same reasoning that led to `nominal_distance_metric` for this column's Krippendorff's Alpha (see "Why `Type of Source` isn't fuzzy or semantic" above). **AC2 is deliberately not implemented** — it would require a real design decision about which category pairs should get partial credit (e.g. `Anonymous Source`/`Unnamed Person`, both about unidentified sourcing) and is left as an open, separate question, not a default.
- Gwet's AC-family is a fundamentally *categorical* agreement statistic, unlike Krippendorff's Alpha's ability to accept an arbitrary custom distance function. That flexibility is what makes the semantic columns (`Sourced Statements`, `Source Justification`, `Title of Source`) work at all with Krippendorff's Alpha — there's no established "continuous embedding distance" variant of Gwet's method, so it isn't a natural substitute there. `Name of Source` is a secondary, unexplored possibility (open-ended proper nouns, not a fixed taxonomy) but wasn't implemented here.

**Missing-value handling matches Krippendorff's Alpha exactly, not by accident.** `compute_gwet_ac1_for_type_of_source` passes `MISSING_SENTINEL` through as one of the categories fed to `irrCAC`, rather than converting it to real `NaN`. This matters: `irrCAC` only drops a row if *both* raters are missing; a row with one real value and one `NaN` would otherwise be silently excluded from `pa` the same way `simpledorff` once silently dropped one-sided-missing rows (see the troubleshooting case below) — exactly the bug this project already spent real effort fixing for Krippendorff's Alpha. Passing the sentinel as a real category instead means the identity weight matrix naturally reproduces `_is_missing()`'s convention for free: same category (including both-missing) = agreement, one side missing = disagreement.

**On the 100-row file**, where Krippendorff's Alpha has real data to work with (`Type of Source: 0.9655`), **AC1 comes out to `0.9766`** — close but not identical, which is expected: different formula, not a discrepancy to chase down.

## Troubleshooting case: Krippendorff's Alpha was silently dropping missing-data rows (found 2026-07-22)

**The problem, in plain English.** When one annotator left a cell blank and the other didn't, the script's own distance functions were written to score that as a real disagreement. But that scoring code never actually ran for those rows. The Krippendorff's Alpha library (`simpledorff`) drops any real blank (`NaN`) *before* our distance function ever sees it, and then throws out the whole row if fewer than 2 ratings are left. So every row with a blank cell — agreements and disagreements alike — was silently excluded from the score, with no warning, no error, nothing to indicate it happened.

**How it happened.** `simpledorff` builds its internal per-item rating table with `row.dropna()`, then masks out any item left with only 1 rating (Krippendorff's convention: you need 2+ ratings to compare). A blank cell reduces a 2-rating item to 1, so it gets excluded entirely — not scored as agreement, not scored as disagreement, just gone.

**Why we didn't catch it right away.** On a 100-row test file, this defect never crashed anything. It just quietly computed alpha from whatever subset of rows happened to have no blanks — for `Source Justification`, that turned out to be only 37 of 100 rows — and printed a perfectly normal-looking score (0.8810). Nothing about the output looked wrong.

**How we did catch it.** On a smaller 11-row file, one column (`Type of Source`) happened to have *zero variability* among the rows that survived the silent drop (every remaining row was `Named Person`). That's a degenerate case for Krippendorff's Alpha — the "expected disagreement by chance" denominator becomes exactly 0 — so it crashed with `ZeroDivisionError` instead of quietly producing a wrong number. The crash is what triggered the investigation. If that column had had any variability left, the same silent-drop bug would have kept hiding in plain sight.

**How we fixed it.** Traced `simpledorff`'s source directly to confirm the `dropna()`/masking mechanism (rather than guessing). Then, as a methodology decision (not just a code fix): a blank `Sourced Statements` cell should always count as real disagreement (one annotator missed a statement the other found), but the four columns that depend on a sourced statement existing (`Type of Source`, `Name of Source`, `Title of Source`, `Source Justification`) should only be scored on rows where *both* annotators found the statement in the first place — otherwise there was no annotation work to compare. Implemented by substituting a non-null sentinel for blanks (so `simpledorff` can't silently drop them) and adding an explicit gate that excludes (not scores) the dependent columns on rows where `Sourced Statements` was missing on either side.

**How we verified the fix was actually correct, not just different.** Ran the untouched original code against the *same current data* as a control (isolating the code change from any possible data drift). Then, for each affected column, programmatically confirmed that every row the old code ever actually scored produces an identical distance value under the new code (diffs at floating-point noise level, ~1e-7) — proving the score changes come entirely from previously-invisible rows now being counted, not from any change to the underlying comparison logic.

**Takeaways for next time:**
- A crash is a lucky break. The same defect can just as easily hide behind a normal-looking score — always sanity-check how many rows actually fed the calculation, not just the final alpha.
- If you want missing data to count as disagreement in Krippendorff's Alpha, confirm it actually reaches your distance function — many implementations (including `simpledorff`) drop real `NaN`s internally before your metric ever runs.
- When validating a fix to a stats pipeline, diff the intermediate per-row values against a same-data control run of the old code, not just the final aggregate number.

**To reproduce the original (pre-fix) behavior**, e.g. to demo the silent-drop bug or the `ZeroDivisionError` on `94-SZ-Met_Gala.csv`/`94-AV-Met_Gala.csv`: the buggy version is preserved under the git tag `icr-silent-drop-bug` (last commit before this fix). Pull just that file out without touching your current working copy:

```bash
git show icr-silent-drop-bug:inter_coder_reliability/v13all-icrclaude.py > v13_buggy_reference.py
python3 v13_buggy_reference.py 94-SZ-Met_Gala.csv 94-AV-Met_Gala.csv
```
