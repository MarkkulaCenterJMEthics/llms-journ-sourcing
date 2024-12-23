import json
import os
import csv
from openai import OpenAI
import anthropic
import datetime
import glob
import re

# Read API keys
with open("openai_key.txt", "r") as key_file:
    openai_api_key = key_file.read().strip()

with open("claude_api_key.txt", "r") as key_file:
    anthropic_api_key = key_file.read().strip()

# Create API clients
openai_client = OpenAI(api_key=openai_api_key)
anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)

# Define models
GPT_MODEL = "gpt-4-turbo"
CLAUDE_MODEL = "claude-3-opus-20240229"

def get_latest_prompt(prompt_type):
    prompt_dir = "prompts"
    prompt_files = glob.glob(os.path.join(prompt_dir, f"{prompt_type}_prompt_v*.txt"))
    if not prompt_files:
        raise FileNotFoundError(f"No {prompt_type} prompt files found in the 'prompts' directory.")
    latest_prompt_file = max(prompt_files, key=os.path.getctime)
    with open(latest_prompt_file, 'r') as file:
        prompt = file.read().strip()
    return prompt, os.path.basename(latest_prompt_file).split('.')[0]  # Return prompt and filename without extension

# Load prompts
SYSTEM_PROMPT, SYSTEM_PROMPT_VERSION = get_latest_prompt("system")
USER_PROMPT, USER_PROMPT_VERSION = get_latest_prompt("user")

def analyze_with_gpt4(article_text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.replace("<<ARTICLE_TEXT>>", article_text)}
    ]
    
    response = openai_client.chat.completions.create(
        model=GPT_MODEL,
        messages=messages,
        temperature=0,
        max_tokens=4096
    )
    
    return response.choices[0].message.content, response.usage

def analyze_with_claude3(article_text):
    response = anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": USER_PROMPT.replace("<<ARTICLE_TEXT>>", article_text)
            }
        ]
    )
    
    return response.content[0].text, response.usage

# def save_json_and_csv(response, model_name, input_file, experiment_dir, json_dir):
#     # Extract JSON from the LLM response
#     json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
#     if json_match:
#         cleaned_response = json_match.group(1).strip()
#     else:
#         json_match = re.search(r'(\{[\s\S]*\})', response)
#         if json_match:
#             cleaned_response = json_match.group(1).strip()
#         else:
#             cleaned_response = response.strip()
    
#     base_name = os.path.splitext(os.path.basename(input_file))[0]
    
#     output_json = os.path.join(json_dir, f"{model_name}-{base_name}.json")
#     output_csv = os.path.join(experiment_dir, f"{model_name}-{base_name}.csv")
    
#     with open(output_json, "w") as json_file:
#         json_file.write(cleaned_response)
    
#     response_data = json.loads(cleaned_response)
#     data = []
    
#     def process_source(source_type, source):
#         name = source.get("Name") or source.get("Identifier", "")
#         title = source.get("Title", "")
#         association = source.get("Associations to the story") or source.get("Association", "")
#         statements = source.get("Attributed Statements") or source.get("SourcedStatement", [])
        
#         if not isinstance(statements, list):
#             statements = [statements]
        
#         for statement in statements:
#             data.append([source_type, name, title, association, statement])
    
#     for source_type, sources in response_data.items():
#         if isinstance(sources, list):
#             for source in sources:
#                 process_source(source_type, source)
#         elif isinstance(sources, dict):
#             process_source(source_type, sources)
    
#     with open(output_csv, "w", newline="", encoding='utf-8') as csv_file:
#         writer = csv.writer(csv_file)
#         writer.writerow(["Source Type", "Name", "Title", "Association", "SourcedStatement"])
#         writer.writerows(data)
    
#     print(f"Analysis completed with {model_name}. JSON saved to {output_json}, CSV saved to {output_csv}")

#new json convertion function to match the v5 prompts json schema.
def save_json_and_csv(response, model_name, input_file, experiment_dir, json_dir):
    # Extract JSON from the LLM response
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
    if json_match:
        cleaned_response = json_match.group(1).strip()
    else:
        json_match = re.search(r'(\{[\s\S]*\})', response)
        if json_match:
            cleaned_response = json_match.group(1).strip()
        else:
            cleaned_response = response.strip()
    
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    output_json = os.path.join(json_dir, f"{model_name}-{base_name}.json")
    output_csv = os.path.join(experiment_dir, f"{model_name}-{base_name}.csv")
    
    with open(output_json, "w") as json_file:
        json_file.write(cleaned_response)
    
    response_data = json.loads(cleaned_response)
    data = []
    
    def process_source(source_type, source):
        name = source.get("Name") or source.get("Identifier", "")
        title = source.get("Title", "")
        association = source.get("Credibility_statement") or source.get("Association", "")
        statements = source.get("Statements") or source.get("SourcedStatement", [])
        
        if not isinstance(statements, list):
            statements = [statements]
        
        for statement in statements:
            data.append([source_type, name, title, association, statement])
    
    for source_type, sources in response_data.items():
        if isinstance(sources, list):
            for source in sources:
                process_source(source_type, source)
        elif isinstance(sources, dict):
            process_source(source_type, sources)
    
    with open(output_csv, "w", newline="", encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Source Type", "Name", "Title", "Association", "SourcedStatement"])
        writer.writerows(data)
    
    print(f"Analysis completed with {model_name}. JSON saved to {output_json}, CSV saved to {output_csv}")

def process_files(input_dir, output_dir):
    # Create a new directory for this experiment run
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    experiment_dir = os.path.join(output_dir, f"experiment_{timestamp}")
    os.makedirs(experiment_dir)

    # Create a subdirectory for JSON files
    json_dir = os.path.join(experiment_dir, "json")
    os.makedirs(json_dir)

    # Save the prompts used for this experiment
    with open(os.path.join(experiment_dir, "system_prompt.txt"), "w") as f:
        f.write(SYSTEM_PROMPT)
    with open(os.path.join(experiment_dir, "user_prompt.txt"), "w") as f:
        f.write(USER_PROMPT)

    for input_file in glob.glob(os.path.join(input_dir, "*.txt")):
        with open(input_file, "r") as file:
            article_text = file.read()
        
        print(f"Processing file: {input_file}")
        
        # Analyze with GPT-4
        gpt4_response, gpt4_usage = analyze_with_gpt4(article_text)
        save_json_and_csv(gpt4_response, "gpt4", input_file, experiment_dir, json_dir)
        print(f"GPT-4 usage: {gpt4_usage}")
        
        # Analyze with Claude 3
        claude3_response, claude3_usage = analyze_with_claude3(article_text)
        save_json_and_csv(claude3_response, "claude3", input_file, experiment_dir, json_dir)
        print(f"Claude 3 usage: {claude3_usage}")

    print(f"Experiment completed. Results saved in {experiment_dir}")

if __name__ == "__main__":
    input_dir = "input_articles"
    output_dir = "output"
    process_files(input_dir, output_dir)
