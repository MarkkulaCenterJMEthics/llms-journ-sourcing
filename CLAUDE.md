# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

Benchmarks multiple LLMs on journalistic source annotation: given a news article, models identify sourced statements and classify each source by type (Named Person, Named Organization, Anonymous Source, Documents, etc.), along with name, title, and justification. LLM output is compared to human-annotated ground truth using fuzzy/semantic matching to produce precision/recall/F1 metrics.

## Environment Setup

```bash
conda create --name llm-news python=3.10 -y
conda activate llm-news
pip install -r requirements.txt
pip install -U sentence-transformers  # only needed for semantic matching
python -c "import nltk; nltk.download('punkt_tab')"
```

**API key:** Place OpenRouter API key in `openrouter_key.txt`, or set `OPENROUTER_API_KEY` env var. All LLM calls go through OpenRouter.

## Key Commands

```bash
# Run extraction with all models, prompt version v55, 3 repetitions
python v10-extract-multiple-LLMs.py -m all -pv v55 -t 3

# Run a single provider
python v10-extract-multiple-LLMs.py -m anthropic -pv v55

# Run with custom output folder prefix
python v10-extract-multiple-LLMs.py -m openai -s myexperiment -pv v55

# Evaluate LLM output against ground truth
python eval_multi_llm_article.py

# Inter-coder reliability between two human annotators
# (needs inter_coder_reliability/requirements.txt installed too -- see inter_coder_reliability/CLAUDE.md
# for architecture and inter_coder_reliability/readme.md for full methodology, worked examples, and
# Gwet's AC1 as an alternative measure for Type of Source)
python inter_coder_reliability/v13all-icrclaude.py <csv1> <csv2>
```

## Pipeline Architecture

**Step 1 — Extraction** (`v10-extract-multiple-LLMs.py`):
- Reads `.txt` articles from `2025_input_stories/`
- Loads `new_prompts/system_prompt_{version}.txt` and `new_prompts/user_prompt_{version}.txt`
- Sends each article to each model via OpenRouter (temperature=0.0)
- Parses JSON from response; saves `.json` + `.csv` per article/model
- Output tree: `llm_results/{prefix}_llm_results_{timestamp}/{model}/{article}/`
- Prompts v50+ use `save_json_and_csv()` (flat `Sourcing Table` array); pre-v50 use `orig_save_json_and_csv()` (nested by source type)

**Step 2 — Evaluation** (`eval_multi_llm_article.py`):
- Loads human ground truth CSVs from `benchmarking/GT data/`. Single header row; data columns are: `Sourced Statements`, `Type of Source`, `Name of Source`, `Title of Source`, `Source Justification`. For stories with both a base-folder file and a `rev2025/` revision, the `rev2025/` version supersedes the original and should be treated as canonical.
- Matches LLM sourced statements to GT statements via fuzzy/semantic scoring
- Computes per-article and aggregate precision/recall/F1 for type, name, title
- Saves metrics CSVs and comparison plots to `benchmarking/metrics/`
- **Requires manual edits** to `main()` to set `human_gt_dir`, `llm_base_dirs`, `output_dir`, `model_names`, and `valid_article_ids` before running
- A 2026-07-24 data fidelity review corrected a small number of `Source Justification` values that stated only the medium of contact (e.g. "said by email") rather than substantive justification — see commit `ae27e70` and `inter_coder_reliability/readme.md`'s data-quality-fix section for the same class of issue as found in the ICR CSVs

## Configuration

`configure.py` — controls matching in the eval script:
```python
match_method = "fuzz_split"   # Options: "fuzz", "fuzz_split", "semantic", "semantic_split"
match_threshold = { "fuzz_split": 70, "semantic_split": 0.7, ... }
```

`fuzz_split` (default) splits statements at sentence/comma boundaries before fuzzy-matching — more forgiving than whole-string comparison.

## Source Type Taxonomy

The six source types defined in the system prompt (v55):
- **Named Person** — a named individual directly quoted or paraphrased
- **Named Organization** — a named org, including when an unnamed spokesperson/official of a named org is cited
- **Document** — an authentic, publicly accessible document used directly as a source
- **Anonymous Source** — a person known to the reporter but granted anonymity
- **Unnamed Person** — an individual referenced without a name and no anonymity granted (e.g. "a police officer said…")
- **Unnamed Group of People** — a group the reporter witnessed or accessed (e.g. "protestors said…", "teachers chanted…")

`SourceTypeMapping` in `eval_multi_llm_article.py` normalizes the many variant strings LLMs may return (e.g. "named person sources", "anonymous_groups") to internal canonical keys used during evaluation.

`inter_coder_reliability/v13all-icrclaude.py` independently canonicalizes this same taxonomy for its own purposes (`TYPE_OF_SOURCE_VARIANTS`), built from a scan of human-annotated typos/casing/legacy naming across the ICR CSVs and `benchmarking/GT data/rev2025/`. This mapping is deliberately separate from `SourceTypeMapping` above — different data sources (human annotation vs. LLM output) tend to produce different variant strings, so fixing one mapping doesn't fix the other. See `inter_coder_reliability/readme.md`'s "Why `Type of Source` isn't fuzzy or semantic" section for why.

## Directory Layout

```
2025_input_stories/      # Input articles (.txt) for v10 extraction
extracted_articles/      # Older article set (v1–v30 era, no boilerplate)
extracted_articles_boilerplate/  # Same articles with boilerplate text
benchmarking/GT data/    # Human-annotated ground truth CSVs
benchmarking/metrics/    # Evaluation output (CSVs + plots)
new_prompts/             # Versioned prompts (system_prompt_vXX.txt, user_prompt_vXX.txt)
llm_results/             # All extraction run outputs
inter_coder_reliability/ # ICR analysis between human annotators
```

## Prompt Versioning

Prompts live in `new_prompts/` as `system_prompt_v{N}.txt` / `user_prompt_v{N}.txt`. Current active versions are v50–v55. The extraction script behavior branches at v50: v50+ expects a flat `Sourcing Table` array in the JSON response; pre-v50 expects nested source-type keys.

## Supported Models (via OpenRouter)

| Provider | Models |
|---|---|
| anthropic | claude-3.5-sonnet, claude-3.7-sonnet, claude-3.7-sonnet:thinking, claude-sonnet-4 |
| google | gemini-2.5-pro, gemini-pro-1.5 |
| openai | gpt-4.1-mini, chatgpt-4o-latest, gpt-4.1 |
| meta | llama-4-maverick, llama-3.1-405b-instruct |
| nvidia | llama-3.1-nemotron-70b-instruct |
| deepseek | deepseek-r1-0528, deepseek-chat-v3-0324 |

To add a model, update the `MODELS` dict in `v10-extract-multiple-LLMs.py` using OpenRouter model IDs.
