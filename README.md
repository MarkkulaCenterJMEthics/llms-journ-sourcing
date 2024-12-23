# Multi-LLM Article Analysis

This project aims to analyze articles using multiple Large Language Models (LLMs) and evaluate their performance.

## Environment Setup

#### Create new environment 
```bash
conda create --name llm-news python=3.10 -y
conda activate llm-news
pip install -r requirements.txt
pip install -U sentence-transformers # skip if using fuzz matching
```

#### Setup nltk packages 
```python
>>> import nltk
>>> nltk.download('punkt_tab')
```

## Flow of Commands

1. Extract information from articles using multiple LLMs:
   - Run `python3 extract-multiple-llm-at-once.py`

2. Evaluate the extracted information:
   - Run `python3 eval-multi-llm-article.py`

## Code Files Description

### 1. extract-multiple-llm-at-once.py

This script is responsible for extracting information from input articles using both GPT-4 and Claude 3 models.

Key features:
- Reads API keys for OpenAI and Anthropic
- Loads the latest system and user prompts
- Processes input articles and generates analysis using both models
- Saves the results in JSON and CSV formats


### 2. eval-multi-llm-article.py

This script evaluates the performance of the LLMs by comparing their extracted information to ground truth data.

Key features:
- Preprocesses the extracted data and ground truth
- Compares attributions and calculates performance metrics
- Generates visualizations for performance comparison
- Outputs detailed results and summary statistics

## API Access Information

### OpenAI (GPT-4)
- API key location: `openai_key.txt`
- Model used: gpt-4-turbo

### Anthropic (Claude 3)
- API key location: `claude_api_key.txt`
- Model used: claude-3-opus-20240229

## Prompt File Structure

The project uses a specific folder structure for managing prompts:
prompts/
├── system_prompt_v.txt
└── user_prompt_v.txt

Note:The script automatically selects the latest version of each prompt type based on the file creation time.

## Usage

1. Ensure you have the necessary API keys in the correct files.
2. Place your input articles in the `input_articles` directory.
3. Run `extract-multiple-llm-at-once.py` to generate LLM analyses.
4. Prepare your ground truth data in the `eval_data` directory.
5. Run `eval-multi-llm-article.py` to evaluate the LLM performance.

## Output

- The extraction script will create a timestamped experiment directory in the `output` folder, containing JSON and CSV files for each analyzed article.
- The evaluation script will generate performance metrics, comparison tables, and visualization plots in the experiment's evaluation directory.

## Open Issues
The eval-multi-llm-article.py script uses string comparison methods:
Fuzzy string matching using the fuzzywuzzy library:

```python
def fuzzy_compare(row, gt_col, llm_col):
    return fuzz.ratio(clean_text(row[gt_col]), clean_text(row[llm_col]))
```
This method calculates the similarity ratio between two strings using the **Levenshtein distance algorithm**.

Here's a brief explanation of how it works:
1. Levenshtein Distance: This is the minimum number of single-character edits (insertions, deletions, or substitutions) required to change one string into another.
2. Ratio Calculation: The ratio is computed using the following formula:

```
ratio = (len(str1) + len(str2) - levenshtein_distance) / len(str2) * 100
```

3. Scaling: The result is then multiplied by 100 to get a percentage-like score between 0 and 100.

**Limitations:**
While effective for aligning sentences, this method falls short when comparing nuanced elements like "association to the story". The Levenshtein ratio fails to capture semantic similarities, potentially missing meaning-equivalent but lexically different phrases. This limitation affects the accuracy of our association matching scores and suggests the need for a dual approach in future research: maintaining lexical comparison for direct extractions while incorporating semantic similarity measures for contextual information.

~~~
5. Sahas to document OpenRouter here



   
