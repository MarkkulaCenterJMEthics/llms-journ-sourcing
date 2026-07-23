# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Directory Does

Calculates inter-coder reliability (ICR) between two human annotators who labeled the same article's sourced statements. This validates the human ground-truth data used elsewhere in `llms-journ-sourcing` for LLM benchmarking — it does not involve LLMs itself.

## Running It

```bash
# From repo root or this directory (adjust path accordingly)
python v13all-icrclaude.py <csv_annotator1> <csv_annotator2>
```

Example:
```bash
cd ICRcombined
python3 ../v13all-icrclaude.py ICRcombined-ann1.csv ICRcombined-ann2.csv
```

Output is printed to stdout: per-item distance scores for every row, followed by Krippendorff's Alpha per column and a summary table. The `.txt` files alongside the CSVs (e.g. `57rows-ICRcombined5colmetric.txt`) are saved output from past runs (shell-redirected), not something the script generates itself.

## Environment Setup

The repo-root `requirements.txt` (fuzzywuzzy, pandas, etc.) is **not sufficient** here — use the one in this directory instead, which additionally pins `simpledorff`, `sentence-transformers`, and `scikit-learn`:

```bash
conda create --name llm-news python=3.10 -y
conda activate llm-news
cd inter_coder_reliability
pip install -r requirements.txt
```

`sentence-transformers` downloads the `all-MiniLM-L6-v2` model on first run, so an internet connection is required at least once.

**Windows:** use Anaconda Prompt (or run `conda init powershell`/`conda init cmd.exe` once and reopen the terminal before `conda activate` works elsewhere). The script forces UTF-8 stdout/stderr at startup (`sys.stdout.reconfigure(encoding="utf-8")`) so the `✅`/`α`/`≥`/`≤` symbols it prints don't raise `UnicodeEncodeError` under Windows' default codepage — this matters especially when redirecting output to a file (`> results.txt`).

## Architecture

`v13all-icrclaude.py` computes Krippendorff's Alpha (via `simpledorff`) separately for five columns, each using its own distance metric:

| Column | Metric |
|---|---|
| `Sourced Statements` | semantic (cosine distance of sentence embeddings) |
| `Source Justification` | semantic |
| `Title of Source` | semantic |
| `Type of Source` | fuzzy string (`fuzzywuzzy.fuzz.ratio`) |
| `Name of Source` | fuzzy string |

Both distance functions treat "both values missing" as perfect agreement (distance 0.0) and "only one missing" as total disagreement (distance 1.0) — this reflects that a blank cell means the annotator found no such source, not that they skipped it.

**Undefined alpha (`N/A`) on a column:** Krippendorff's Alpha is mathematically undefined when the paired (both-annotators-rated) items for a column show zero variability — e.g. every `Type of Source` happens to be `Named Person`. The denominator ("expected disagreement by chance," computed in `simpledorff`'s `calculate_de`) is exactly zero in that case, which is a `ZeroDivisionError` inside `simpledorff`, not a bug in this script or the data. `calculate_icr_for_column` catches this and reports `N/A (undefined - no variability in the data)` for that column while still computing the rest — this is expected on small or homogeneous test files, not something to "fix" in the CSV.

**Row correspondence is positional, not key-matched**: the two CSVs are assumed to have the same number of rows in the same order (`item_id` is just `range(1, len(df)+1)` assigned independently to each file). There is no join on the `No` column or on statement text. If one annotator added/removed/reordered a row, the comparison will silently misalign that row and every row after it.

## Data Files (Gotchas)

CSVs in this directory and `ICRcombined/` are pairs of annotator files named `<article>-ann1.csv`, `<article>-ann2.csv` (and occasionally `-ann3.csv`, e.g. `Spr25-DogWalker`, for a third annotator — run the script pairwise for each combination you need). `ICRcombined/` is a separate, overlapping collection of these same per-article files plus multi-article "combined" runs (e.g. `57rows-ICRcombined-ann1.csv`, `100R-ICRcombined-ann1.csv`) that concatenate several articles' rows for a larger-N alpha calculation.

The script hard-requires columns named exactly `Sourced Statements`, `Source Justification`, `Type of Source`, `Name of Source`, `Title of Source` (case-sensitive) and raises `ValueError` if any is missing. **Several existing CSVs use `Type of source` (lowercase "source")** — e.g. `Harris-Poll-1.csv`, `Harris-Poll-2.csv`, `Wtr24-C98-Harris-Poll-ann1.csv`, `4-SFO-Labor-Day-for-CSV-ann1.csv`/`-ann2.csv` — and will fail against this script as-is; fix the header casing before running. Extra columns (e.g. `Anonymity Y/N?`, trailing blank columns from Excel exports) are harmless and ignored.
