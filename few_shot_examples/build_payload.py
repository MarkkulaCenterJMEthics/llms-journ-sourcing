"""
Reference implementation: assemble an OpenRouter chat-completions payload
from examples_bank.json plus the v60 system/user prompts.

This is a reference for how the pieces fit together, not a production
pipeline -- adapt error handling, model selection, etc. as needed.

Usage:
    python3 build_payload.py path/to/target_article.txt > payload.json
"""
import csv
import io
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BANK_PATH = Path(__file__).resolve().parent / "examples_bank.json"
SYSTEM_PROMPT_PATH = REPO_ROOT / "new_prompts" / "system_prompt_v60.txt"
USER_PROMPT_PATH = REPO_ROOT / "new_prompts" / "user_prompt_v60_csv.txt"


def rows_to_csv(field_order, rows):
    """Render correct_annotation rows into the exact CSV shape the user
    prompt's output rules require: exact header, every field quoted,
    internal quotes doubled, no blank lines."""
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_ALL, lineterminator="\n")
    writer.writerow(field_order)
    for r in rows:
        writer.writerow([("" if r[f] is None else r[f]) for f in field_order])
    return buf.getvalue()


def build_messages(target_article_text, model_name=None):
    with open(BANK_PATH, encoding="utf-8") as f:
        bank = json.load(f)
    field_order = bank["csv_field_order"]
    examples = bank["examples"]

    with open(SYSTEM_PROMPT_PATH, encoding="utf-8") as f:
        system_prompt = f.read()
    with open(USER_PROMPT_PATH, encoding="utf-8") as f:
        user_prompt = f.read()

    messages = [{"role": "system", "content": system_prompt}]

    for i, ex in enumerate(examples):
        csv_text = rows_to_csv(field_order, ex["correct_annotation"])
        is_last = i == len(examples) - 1
        messages.append({"role": "user", "content": ex["article_excerpt"]})
        if is_last:
            # Single cache_control breakpoint at the end of the whole
            # stable prefix (system prompt + all few-shot examples).
            # Required for Anthropic/Qwen to get the cache discount;
            # harmless no-op for providers that cache automatically.
            messages.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": csv_text,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            })
        else:
            messages.append({"role": "assistant", "content": csv_text})

    # user_prompt_v60_csv.txt has a "%s" placeholder for the article
    real_user_turn = user_prompt.replace("%s", "") + "\n\nArticle:\n" + target_article_text
    messages.append({"role": "user", "content": real_user_turn})

    payload = {"messages": messages}
    if model_name:
        payload["model"] = model_name
    return payload


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python3 build_payload.py path/to/target_article.txt [model_name]", file=sys.stderr)
        sys.exit(1)
    article_path = Path(sys.argv[1])
    model_name = sys.argv[2] if len(sys.argv) > 2 else None
    article_text = article_path.read_text(encoding="utf-8")
    payload = build_messages(article_text, model_name)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
