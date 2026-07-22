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
