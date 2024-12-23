import csv
import datetime
import glob
import json
import os
import re

import requests

version = "v30"


def get_latest_prompt(prompt_type, version):
    prompt_dir = "new_prompts"
    prompt_file = os.path.join(prompt_dir, f"{prompt_type}_prompt_{version}.txt")
    if not prompt_file:
        raise FileNotFoundError(
            f"No {prompt_type} prompt files found in the 'prompts' directory."
        )
    # latest_prompt_file = max(prompt_files, key=os.path.getctime)
    with open(prompt_file, "r") as file:
        prompt = file.read().strip()
    return prompt, os.path.basename(prompt_file).split(".")[
        0
    ]  # Return prompt and filename without extension


# Load prompts
system_prompt_content, SYSTEM_PROMPT_VERSION = get_latest_prompt("system", version)
user_prompt_content, USER_PROMPT_VERSION = get_latest_prompt("user", version)


def analyze_with_openrouter(article_text, model_name):
    chat_prompt = [
        {
            "role": "system",
            "content": system_prompt_content,
        },
        {
            "role": "user",
            "content": user_prompt_content + "%" + article_text,
        },
    ]

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"bearer {os.environ.get('OpenRouter_API_Key')}",
            "Content-Type": "application/json",
        },
        data=json.dumps(
            {
                "model": model_name,
                "messages": chat_prompt,
                "temperature": 0.0,
                "max_tokens": 4096,
            }
        ),
    )
    if response.status_code == 200:
        data = response.json()
        try:
            act_res = data["choices"][0]["message"]["content"]
        except:
            print(data)
            raise
        return act_res
    else:
        print(response)
        raise Exception("Call not successful")


# new json convertion function to match the v5 prompts json schema.
def save_json_and_csv(response, model_name, input_file, experiment_dir, json_dir):
    # Extract JSON from the LLM response
    json_match = re.search(r"```json\s*([\s\S]*?)\s*```", response)
    if json_match:
        cleaned_response = json_match.group(1).strip()
    else:
        json_match = re.search(r"(\{[\s\S]*\})", response)
        if json_match:
            cleaned_response = json_match.group(1).strip()
        else:
            cleaned_response = response.strip()

    base_name = os.path.splitext(os.path.basename(input_file))[0]

    output_json = os.path.join(json_dir, f"{model_name}-{base_name}.json")
    output_csv = os.path.join(experiment_dir, f"{model_name}-{base_name}.csv")

    with open(output_json, "w") as json_file:
        json_file.write(cleaned_response)

    json_data = json.loads(cleaned_response)
    data = []

    for source_type in [
        "Named Organization Sources",
        "Named Person Sources",
        "Document Sources",
        "Unnamed Group of People",
        "Anonymous Sources",
    ]:
        for source in json_data.get(source_type, []):
            name = source.get("Name of Source", "")
            title = source.get("Title of Source", "")
            association = source.get("Source Justification", "")
            sourced_statements = source.get("Sourced Statement", [])

            if isinstance(sourced_statements, list):
                for sourced_statement in sourced_statements:
                    data.append(
                        [source_type, name, title, association, sourced_statement]
                    )
            else:
                data.append([source_type, name, title, association, sourced_statements])

    with open(output_csv, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "Source Type",
                "Name",
                "Title",
                "Source Justification",
                "Sourced Statement",
            ]
        )
        writer.writerows(data)

    print(f"CSV file generated: {output_csv}")

    print(
        f"Analysis completed with {model_name}. JSON saved to {output_json}, CSV saved to {output_csv}"
    )


def process_files(input_dir, output_dir):
    for i in range(5):  # Loop to run the experiment 5 times
        # Create a new directory for this experiment run
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        experiment_dir = os.path.join(output_dir, f"experiment_{i+1}_{timestamp}")
        os.makedirs(experiment_dir)

        # Create a subdirectory for JSON files
        json_dir = os.path.join(experiment_dir, "json")
        os.makedirs(json_dir)

        # Save the prompts used for this experiment
        with open(os.path.join(experiment_dir, "system_prompt.txt"), "w") as f:
            f.write(system_prompt_content)
        with open(os.path.join(experiment_dir, "user_prompt.txt"), "w") as f:
            f.write(user_prompt_content)

        # Process each file in the input directory
        for input_file in glob.glob(os.path.join(input_dir, "*.txt")):
            with open(input_file, "r", encoding="utf-8") as file:
                article_text = file.read()

            print(f"Processing file: {input_file}")
            print(article_text)

            # Analyze with GPT-4
            gpt4o_response = analyze_with_openrouter(
                article_text, "openai/chatgpt-4o-latest"
            )
            save_json_and_csv(
                gpt4o_response,
                "chatgpt-4o-latest",
                input_file,
                experiment_dir,
                json_dir,
            )

            claude35_response = analyze_with_openrouter(
                article_text, "anthropic/claude-3.5-sonnet"
            )
            save_json_and_csv(
                claude35_response,
                "claude-3.5-sonnet",
                input_file,
                experiment_dir,
                json_dir,
            )

            llama70_response = analyze_with_openrouter(
                article_text, "meta-llama/llama-3.1-70b-instruct:free"
            )
            save_json_and_csv(
                llama70_response,
                "llama-3.1-70b-instruct:free",
                input_file,
                experiment_dir,
                json_dir,
            )

            llama405_response = analyze_with_openrouter(
                article_text, "meta-llama/llama-3.1-405b-instruct:free"
            )
            save_json_and_csv(
                llama405_response,
                "llama-3.1-405b-instruct:free",
                input_file,
                experiment_dir,
                json_dir,
            )

            claude3opus_response = analyze_with_openrouter(
                article_text, "anthropic/claude-3-opus"
            )
            save_json_and_csv(
                claude3opus_response,
                "claude-3-opus",
                input_file,
                experiment_dir,
                json_dir,
            )

        print(f"Experiment run {i+1} completed. Results saved in {experiment_dir}")


if __name__ == "__main__":
    input_dir = "article6_test"
    output_dir = "output_article6"
    process_files(input_dir, output_dir)
