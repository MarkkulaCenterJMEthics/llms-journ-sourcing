import csv
import datetime
import json
import os
import re

import requests

input_file = "input_articles/openai_board.txt"
with open(input_file, "r", encoding="utf-8") as file:
    article = file.read()


def load_prompt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            system_prompt = file.read()
            return system_prompt
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None


def extract_content_after_string(content, target_string):
    start_index = content.find(target_string)

    if start_index == -1:
        return "String not found."

    return content[start_index + len(target_string) :].strip()


system_prompt_content = load_prompt("new_prompts/system-v30-all5.txt")
user_prompt_content = load_prompt("new_prompts/user-v30-all5.txt")


chat_prompt = [
    {
        "role": "system",
        "content": system_prompt_content,
    },
    {
        "role": "user",
        "content": user_prompt_content + "%" + article,
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
            "model": "meta-llama/llama-3.1-70b-instruct:free",
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

    # print(data)
    # cleaned_response = re.search(r'\{.*\}', act_res, re.DOTALL).group()
    cleaned_response = act_res
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")

    folder_name = os.path.splitext(input_file)[0]
    file_name = os.path.split(input_file)[1]
    custom_folder = "v30_openAI_test"

    os.makedirs(custom_folder, exist_ok=True)

    output_json = os.path.join(
        custom_folder, f"v30_llama3.1_70Bfree__openai_test{timestamp}.json"
    )
    output_csv = os.path.join(custom_folder, f"combined_test{timestamp}.csv")

    with open(output_json, "w") as json_file:
        json_file.write(cleaned_response)

    json_data = json.loads(cleaned_response)

    data = []
    for source_type in [
        "Anonymous Sources",
        "Named Sources",
        "Document Sources",
        "Organization Sources",
        "Person Sources",
        "Sourcing",
    ]:
        # for source_type in ["Anonymous Sources", "Named Sources","Document Sources", "Organization Sources", "Person Sources", "Sourcing"]:

        for source in json_data.get(source_type, []):
            name = source.get("Name", "")
            title = source.get("Title", "")
            association = source.get("Association", "")
            sourced_statements = source.get("SourcedStatement", [])

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
            ["Source Type", "Name", "Title", "Association", "Sourced Statement"]
        )
        writer.writerows(data)

    print(f"CSV file generated: {output_csv}")

else:
    print(f"Failed to fetch data: {response.status_code}")
