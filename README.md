# Multi-LLM Article Analysis

This project benchmarks multiple Large Language Models (LLMs) on their performance for journalistic sourcing annotations. Given a news article, models identify sourced statements and classify each source by type, name, title, and justification. LLM output is compared to human-annotated ground truth to produce accuracy metrics using straightforward, fuzzy or semantic comparison scoring.

## Environment Setup

```bash
conda create --name llm-news python=3.10 -y
conda activate llm-news
pip install -r requirements.txt
pip install -U sentence-transformers  # skip if using fuzz matching only
python -c "import nltk; nltk.download('punkt_tab')"
```

**API key:** Place your OpenRouter API key in `openrouter_key.txt` in the project root, or set the `OPENROUTER_API_KEY` environment variable. All LLM calls go through OpenRouter.

## Flow of Commands

1. Place input articles (`.txt` files) in `2025_input_stories/`
2. Run extraction across models:
   ```bash
   python v10-extract-multiple-LLMs.py -m all -pv v55
   ```
3. Edit the `main()` function in `eval_multi_llm_article.py` to set `llm_base_dirs`, `model_names`, and `valid_article_ids`
4. Run evaluation:
   ```bash
   python eval_multi_llm_article.py
   ```

## Extraction Script: `v10-extract-multiple-LLMs.py`

```
usage: v10-extract-multiple-LLMs.py [-h] [-m MODELS] [-s STRING] [-t TIMES] [-pv PROMPT_VERSION]

options:
  -m, --models          'all', a provider name, or comma-separated model names
  -s, --string          Custom prefix for the output directory name
  -t, --times           Number of times to repeat the experiment loop (default: 1)
  -pv, --prompt-version Prompt version to use (e.g., v55)
```

Examples:
```bash
# Run all models with prompt v55, three repetitions
python v10-extract-multiple-LLMs.py -m all -pv v55 -t 3

# Run only Anthropic models
python v10-extract-multiple-LLMs.py -m anthropic -pv v55

# Run with a custom output folder prefix
python v10-extract-multiple-LLMs.py -m openai -s myexperiment -pv v55
```

Output is saved to: `llm_results/{prefix}_llm_results_{timestamp}/{model}/{article}/`

Each article folder contains a `.json` (raw LLM response) and a `.csv` with columns: `SourcedStatement`, `SourceType`, `Name`, `Title`, `Justification`.

### Supported Models

| Provider | Models |
|---|---|
| anthropic | claude-3.5-sonnet, claude-3.7-sonnet, claude-3.7-sonnet:thinking, claude-sonnet-4 |
| google | gemini-2.5-pro, gemini-pro-1.5 |
| openai | gpt-4.1-mini, chatgpt-4o-latest, gpt-4.1 |
| meta | llama-4-maverick, llama-3.1-405b-instruct |
| nvidia | llama-3.1-nemotron-70b-instruct |
| deepseek | deepseek-r1-0528, deepseek-chat-v3-0324 |

To add a model, update the `MODELS` dict in `v10-extract-multiple-LLMs.py` using OpenRouter model IDs.

## Evaluation Script: `eval_multi_llm_article.py`

Compares LLM-generated CSVs against human-annotated ground truth CSVs in `benchmarking/GT data/`.

Before running, edit the `main()` function to configure:
- `human_gt_dir` — path to the GT CSV folder
- `llm_base_dirs` — list of extraction output directories to evaluate
- `model_names` — list of model folder names to include
- `valid_article_ids` — set of article IDs to evaluate
- `output_dir` — where to write metrics CSVs and plots

Matching method and threshold are controlled in `configure.py` (default: `fuzz_split` at threshold 70).

Output per model: `{model}_all_metrics.csv`, `{model}_avg_metrics.csv`, comparison plots.

## Prompts

Versioned prompts live in `new_prompts/` as `system_prompt_v{N}.txt` / `user_prompt_v{N}.txt`. Current active versions are v50–v55. Pass the version with `-pv` at runtime.

Prompts v50 and later use a flat `Sourcing Table` array in the JSON response. Earlier versions used a nested structure grouped by source type.

## Inter-Coder Reliability

To measure agreement between two human annotators:

```bash
python inter_coder_reliability/v13all-icrclaude.py <annotator1.csv> <annotator2.csv>
```

Annotator CSV pairs are stored in `inter_coder_reliability/`. Combined sets are in `inter_coder_reliability/ICRcombined/`.

## Ground Truth Data

Human-annotated ground truth CSVs are in `benchmarking/GT data/`. Each file corresponds to one article and uses the same column structure as the LLM output CSVs. GT files have a 5-row header before the data rows.
