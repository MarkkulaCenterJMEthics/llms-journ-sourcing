# Multi-LLM Article Analysis

This project is a research proposal to benchmark multiple Large Language Models (LLMs) on their performance for journalistic sourcing annotations.

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

### 1. Updated extraction script, using the v10-extract-multiple-LLMs.py with --help 

The v10 script will produce annotations using Anthropic, Google, Openai, Meta, Nvidia (llama-3.1 70B), and Deepseek models, using Openrouter.ai, for different prompt versions, with output in custome-prefixed folder names. 

python v10-extract-multiple-LLMs.py --help                             
usage: v10-extract-multiple-LLMs.py [-h] [-m MODELS] [-s STRING] [-t TIMES] [-pv PROMPT_VERSION]

Process articles with multiple LLM models

options:
  -h, --help            show this help message and exit
  
  -m, --models MODELS   Models to use: 'all', provider names (anthropic, google, openai, meta, nvidia, deepseek), or specific
                        model names (comma-separated)

  -s, --string STRING   Custom prefix string for output directory
  
  -t, --times TIMES     Number of times to run the experiment loop (default: 1)
  
  -pv, --prompt-version PROMPT_VERSION
                        Prompt version to use (e.g., v41, v50). If not specified, uses global setting


The following models are supported. You can add  new models by updating the models array using OpenRouter.ai's model names into the python script. 

```# Define all available models organized by provider
MODELS = {
    "anthropic": [
        ("anthropic/claude-3.5-sonnet", "claude-3.5-sonnet"),
        ("anthropic/claude-3.7-sonnet", "claude-3.7-sonnet"),
        ("anthropic/claude-3.7-sonnet:thinking", "claude-3.7-sonnet-thinking"),
        ("anthropic/claude-sonnet-4", "claude-sonnet-4")
    ],
    "google": [
        ("google/gemini-2.5-pro", "gemini-2.5-pro"),
        ("google/gemini-pro-1.5", "gemini-pro-1.5")
    ],
    "openai": [
        ("openai/gpt-4.1-mini", "gpt-4.1-mini"),
        ("openai/chatgpt-4o-latest", "chatgpt-4o-latest"),
        ("openai/gpt-4.1", "gpt-4.1")
    ],
    "meta": [
        ("meta-llama/llama-4-maverick", "llama-4-maverick"),
        ("meta-llama/llama-3.1-405b-instruct", "llama-3.1-405b-instruct")
    ],
    "nvidia": [
        ("nvidia/llama-3.1-nemotron-70b-instruct", "llama-3.1-70b-instruct")
    ],
    "deepseek": [
        ("deepseek/deepseek-r1-0528", "deepseek-r1-0528"),
        ("deepseek/deepseek-chat-v3-0324", "deepseek-chat-v3-0324")
    ]
}
```

### 2. Original (2024) -- extract-multiple-llm-at-once.py

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
