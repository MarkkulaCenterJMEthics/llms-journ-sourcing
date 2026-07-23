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

## Usage

usage: v13all-icrclaude.py [-h] csv1 csv2

Calculate Inter-Coder Reliability using semantic similarity and fuzzy matching

positional arguments:
  csv1        Path to first annotator CSV file
  csv2        Path to second annotator CSV file

options:
  -h, --help  show this help message and exit

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
