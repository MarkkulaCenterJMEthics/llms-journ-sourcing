# Intercode Reliability Scoring (ICR) for human annotations that were used to create the ground truth data for the LLM benchmarking project. 

## Setup

```bash
conda create --name llm-news python=3.10 -y
conda activate llm-news
cd inter_coder_reliability
pip install -r requirements.txt
```

(This directory's `requirements.txt` is separate from the one in the repo root — it pulls in `simpledorff`, `sentence-transformers`, and `scikit-learn`, which the ICR script needs but the rest of the repo does not.)

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

There are two distance functions, and which one applies depends on whether a column holds free text or a short fixed category:

- **`fuzzy_distance_metric(a, b)`** — Levenshtein-ratio string similarity (via `fuzzywuzzy`/`python-Levenshtein`), converted to a `0.0`–`1.0` distance. Used for **`Type of Source`** and **`Name of Source`**: short, low-vocabulary strings where a near-identical typo (`"Unamed Group of People"` vs `"Unnamed Group of People"`) should score as *nearly* agreeing, not as a coin-flip-different category.
- **`semantic_distance_metric` / `build_cached_semantic_distance_fn(model, texts)`** — cosine distance between `all-MiniLM-L6-v2` sentence embeddings, `0.0`–`1.0`. Used for **`Sourced Statements`**, **`Source Justification`**, and **`Title of Source`**: free text where two annotators rarely type the identical sentence, but may mean the same thing (paraphrase, partial quote, punctuation differences). The cached variant precomputes one embedding per distinct value up front (see the "silent-drop" history below for why — `simpledorff` calls the distance function once per *pair* of distinct values, which is quadratic and was taking ~25 minutes per column on a 100-row file before caching).

Both functions share the same missing-value rule via `_is_missing(value)` (true for real `NaN` or the `MISSING_VALUE` sentinel): **both missing → distance `0.0`** (agreement — neither annotator found anything there); **one missing → distance `1.0`** (disagreement); **both present → the real fuzzy/semantic comparison**.

Column by column, this is what actually feeds the alpha calculation:

| Column | Distance function | Gated on `Sourced Statements`? | A "value" looks like |
|---|---|---|---|
| `Sourced Statements` | semantic (cached) | **No** — never gated | A full sentence/quote, or `MISSING_VALUE` if an annotator missed the statement entirely |
| `Type of Source` | fuzzy | Yes | A short category string (`Named Person`, `Document`, ...), or `MISSING_VALUE` |
| `Name of Source` | fuzzy | Yes | A name string, or `MISSING_VALUE` |
| `Title of Source` | semantic (cached) | Yes | A title phrase, or `MISSING_VALUE` |
| `Source Justification` | semantic (cached) | Yes | A quoted/paraphrased justification sentence, or `MISSING_VALUE` |

"Gated" means: rows where either annotator's `Sourced Statements` is missing are excluded from that column's alpha entirely (see `compute_ss_gate_mask` above) — because there was no annotation work on `Type of Source`/etc. to compare if one annotator never registered a sourced statement at that row in the first place. `Sourced Statements` itself is exempt from gating by design: a missing statement there *is* the disagreement being measured.

## Worked example: from CSV rows to a Krippendorff's Alpha score

Using a real pair, `Spr25-AppleAI-KG-ann1.csv` / `Spr25-AppleAI-SV-ann2.csv` (19 rows), run as `python3 v13all-icrclaude.py Spr25-AppleAI-KG-ann1.csv Spr25-AppleAI-SV-ann2.csv`.

### Example 1: `Type of Source` (fuzzy metric) — how `0.9784` is actually computed

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

**Step 1 — per-item distances**, via `fuzzy_distance_metric`. A few representative values it actually returns:

| a | b | distance |
|---|---|---|
| `Named Person` | `Named Person` | `0.0000` |
| `Unamed Group of People` | `Unnamed Group of People` | `0.0200` (one missing letter — treated as *near*-agreement, not disagreement) |
| `Unnamed Group of People ` | `Unnamed Group of People` | `0.0000` (trailing whitespace is stripped before comparing) |
| `Document` | `Named Person` | `0.7000` |
| `Anonymous Source` | `Named Organization` | `0.8200` |
| `Named Organization` | `Named Person` | `0.4000` |

**Step 2 — `simpledorff` builds a per-item table** (`Counter` of values per item) and a marginal frequency table across *all* 36 ratings (the typo variants count as separate literal strings — Krippendorff's Alpha treats distinct strings as distinct classes for this frequency table, even though the *distance function* between two of those classes can be small):

```
class_freqs = {
  'Anonymous Source': 2, 'Document': 6, 'Named Organization': 4, 'Named Person': 6,
  'Unamed Group of People': 9, 'Unnamed Group of People': 8, 'Unnamed Group of People ': 1
}
total ratings = 36
```

**Step 3 — observed and expected disagreement.** `Do` sums the within-item pairwise distance for every item with 2+ ratings (weighted `1/(ratings-1)`); `De` sums `freq[c] × freq[k] × distance(c,k)` over every pair of classes in the table above (that's `7 × 7 = 49` terms — mostly large, since `Document`/`Named Person`/etc. are genuinely different categories):

```
Do = 0.3600
De = 583.7200
N  = 36 (total ratings)

alpha = 1 - (Do / De) × (N - 1)
      = 1 - (0.3600 / 583.7200) × 35
      = 0.9784
```

That matches the script's printed `✅ Krippendorff's Alpha for 'Type of Source': 0.9784` exactly. The near-1.0 score reflects that almost all "disagreement" here was really the same typo repeating, scored as nearly-identical by the fuzzy metric rather than as a coin-flip category mismatch.

### Example 1b: `Type of Source` at real scale — 100-row file, same formula, no shortcuts

`Type of Source` is a fixed taxonomy, not free text, so its vocabulary doesn't grow with sample size — even at 100 rows (`ICRcombined/100R-ICRcombined-ann1.csv` / `-ann2.csv`) there are only 6 distinct classes among the 95 gated-in items (190 total ratings), so this is exactly as hand-tractable as the 19-row example, just at real scale:

```
class_freqs = {
  'Named Person': 108, 'Named Organization': 42, 'Unnamed Group of People': 23,
  'Document': 14, 'Anonymous Source': 2, 'Unnamed Group of People ': 1
}
total ratings = 190   (95 items x 2 annotators; 5 items gated out for a missing Sourced Statement)
```

Only 3 of the 95 comparable items have *any* within-item disagreement at all — every other item is an exact match (`Do` contribution `0.0000`):

| Item | ann1 | ann2 | distance |
|---|---|---|---|
| 12 | `Unnamed Group of People ` | `Unnamed Group of People` | `0.0000` (trailing whitespace only) |
| 40 | `Named Organization` | `Document` | `0.6900` (genuinely different categories) |
| 54 | `Named Organization` | `Document` | `0.6900` (same pair again) |

```
Do = 2.7600
De = 11808.9600   (sum over all 6 x 6 = 36 class pairs, e.g. 'Named Person' x 'Document' pulls in 108 x 14 x distance(...))
N  = 190

alpha = 1 - (Do / De) x (N - 1)
      = 1 - (2.7600 / 11808.9600) x 189
      = 0.9558
```

Matches the script's printed `Type of Source: 0.9558` on this file exactly. Just two genuine category-level mismatches (`Named Organization` vs `Document`, items 40 and 54) out of 95 items are enough to pull alpha down from a hypothetical `1.0` to `0.9558` — a concrete illustration of how sensitive alpha is to even a couple of real disagreements once weighted by the `(N-1)` term.

### Example 2: `Sourced Statements` (semantic metric) — same mechanism, not hand-tractable at scale

The formula above (`Do`, `De`, `alpha = 1 - (Do/De)(N-1)`) is identical for the semantic columns — only the distance function changes. The difference is scale: free-text columns are almost entirely *unique* values, so `De`'s pairwise sum runs over every distinct sentence in the column, not 7 categories. On this same 19-row file, `Sourced Statements` reaches 31 distinct texts out of 37 total ratings (31² = 961 pairwise terms); on the 100-row file it's 155 distinct texts (~24,000 terms) — infeasible to hand-verify, which is exactly why `build_cached_semantic_distance_fn` exists (see the pipeline description above). A few real per-item distances from this same run illustrate what feeds that sum:

| Item | ann1 | ann2 | distance |
|---|---|---|---|
| 2 | *"While Vision Pro sales have been disappointing..."* | *(identical)* | `0.0000` |
| 9 | *"The A.I. stumble was set in motion in early 2023. Mr. Giannandrea..."* | *"Mr. Giannandrea..."* (same sentence, missing the leading clause) | `0.2907` |
| 6 | *"Apple's software chief, told employees that..."* | `MISSING_VALUE` (annotator2 never recorded a statement here) | `1.0000` |

For the full, verified real-world numbers on a larger sample, see the saved run `ICRcombined/July26-100R-revisedICR.txt` (100 rows; `Source Justification` computed at `0.6997` there) and the troubleshooting case below for how that number was validated against a same-data control run of the pre-fix code.

### Summary: all five columns, 100-row file

The same formula (`Do`, `De`, `alpha = 1 - (Do/De)(N-1)`) applied to every column on the full `100R-ICRcombined-ann1.csv` / `-ann2.csv` pair. `Type of Source` is fully derived above; the free-text columns' pairwise sums aren't reproduced here (tens of thousands of terms), but every number below is the actual verified output of the real calculation, not an estimate or approximation:

| Column | Items (post-gate) | Total ratings | Distinct values | `Do` | `De` | `alpha` |
|---|---|---|---|---|---|---|
| `Sourced Statements` | 100 | 200 | 155 | 18.6287 | 35209.9727 | `0.8947` |
| `Source Justification` | 95 | 190 | 62 | 39.0708 | 24585.9746 | `0.6997` |
| `Type of Source` | 95 | 190 | 6 | 2.7600 | 11808.9600 | `0.9558` |
| `Name of Source` | 95 | 190 | 41 | 8.1200 | 27828.6800 | `0.9449` |
| `Title of Source` | 95 | 190 | 43 | 8.1204 | 26041.4297 | `0.9411` |

`Sourced Statements` is never gated, so it keeps all 100 items; the other four columns lose the same 5 items where one annotator recorded no `Sourced Statements` at all. These `alpha` values match `ICRcombined/July26-100R-revisedICR.txt` exactly.

*Documenting this now, before trying Gwet's AC2 as an alternative reliability measure, so any difference in results between the two can be attributed to the formula, not to ambiguity about what data each one actually consumed.*

## A note on "Undefined" alpha scores

A column can report `Undefined` instead of a number between -1 and 1. This isn't a bug — it means every comparable item in that column got the exact same value from both annotators (e.g. every source classified as `Named Person`, with no other category present at all). Krippendorff's Alpha needs some variability across categories to establish a chance-agreement baseline; with zero variability there's nothing to divide by, so the result is mathematically indeterminate (0/0), not 0.0 or 1.0.

This is most likely on **small samples**, where it's entirely possible for every annotator to agree by chance on a single category simply because the sample never contained a different one to disagree about — it says more about sample size than about annotator reliability. It gets rarer as the sample grows and more categories naturally appear (an 11-row test file hit this on `Type of Source`; the same column computed fine on a 100-row file). When you see `Undefined`, the message names the value and how many items/ratings tied, so you don't need to dig through the per-item log to explain it.

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
