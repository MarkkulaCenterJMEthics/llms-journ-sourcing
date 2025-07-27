import json
import os
import csv
import datetime
import glob
import re
import requests
import time
import logging
import argparse

prompt_version = "v50"
or_api_key = ""

# Define all available models organized by provider
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

# Load api key from config 
def load_api_key():
    # Same function as in the main script
    api_key = os.getenv('OPENROUTER_API_KEY')
    if api_key:
        return api_key
    
    config_files = ['config.json', '.env', 'api_key.txt']
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                if config_file.endswith('.json'):
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                        api_key = config.get('openrouter_api_key') or config.get('OPENROUTER_API_KEY')
                        if api_key:
                            return api_key
                elif config_file.endswith('.env'):
                    with open(config_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith('OPENROUTER_API_KEY='):
                                return line.split('=', 1)[1].strip()
                elif config_file.endswith('.txt'):
                    with open(config_file, 'r') as f:
                        api_key = f.read().strip()
                        if api_key:
                            return api_key
            except:
                continue
    return None


#load prompts 
def get_latest_prompt(prompt_type, prompt_version):
    prompt_dir = "new_prompts"
    prompt_file = os.path.join(prompt_dir, f"{prompt_type}_prompt_{prompt_version}.txt")
    print(prompt_file)
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(
            f"No {prompt_type} prompt files found in the 'prompts' directory."
        )
    with open(prompt_file, "r") as file:
        prompt = file.read().strip()
    return prompt, os.path.basename(prompt_file).split(".")[0]

def compare_versions(version1, version2):
    """Compare two version strings (e.g., 'v41', 'v50')
    Returns: -1 if version1 < version2, 0 if equal, 1 if version1 > version2
    """
    # Extract numeric part from version strings
    num1 = int(version1.lstrip('v'))
    num2 = int(version2.lstrip('v'))
    
    if num1 < num2:
        return -1
    elif num1 > num2:
        return 1
    else:
        return 0

def should_use_orig_function(prompt_version):
    """Determine whether to use orig_save_json_and_csv based on version"""
    return compare_versions(prompt_version, "v50") < 0

def parse_model_argument(model_arg):
    """Parse the model argument and return list of (api_name, file_name) tuples"""
    if model_arg.lower() == "all":
        # Return all models
        all_models = []
        for provider_models in MODELS.values():
            all_models.extend(provider_models)
        return all_models
    
    # Parse comma-separated list of providers or specific models
    requested = [item.strip().lower() for item in model_arg.split(",")]
    selected_models = []
    
    for item in requested:
        # Check if it's a provider name
        if item in MODELS:
            selected_models.extend(MODELS[item])
        else:
            # Check if it's a specific model name (partial match)
            found = False
            for provider_models in MODELS.values():
                for api_name, file_name in provider_models:
                    if item in api_name.lower() or item in file_name.lower():
                        selected_models.append((api_name, file_name))
                        found = True
            if not found:
                print(f"Warning: Model '{item}' not found in available models")
    
    return selected_models

def analyze_with_openrouter(article_text, model_name, retry_limit=3, size_threshold=50):
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

    retries = 0
    failure_counter = 0

    while retries < retry_limit:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"bearer {or_api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(
                {
                    "model": model_name,
                    "messages": chat_prompt,
                    "temperature": 0.0,
                    "max_tokens": 9048,
                    "reasoning": {
                        "exclude": True
                    }
                }
            ),
        )

        if response.status_code == 200:
            try:
                data = response.json()
                act_res = data["choices"][0]["message"]["content"]
                json_size = len(json.dumps(data).encode("utf-8"))

                if json_size < size_threshold:
                    failure_counter += 1
                    print(
                        f"Response size too small ({json_size} bytes). Retrying... (Attempt {retries + 1}/{retry_limit})"
                    )
                    retries += 1
                    continue

                return act_res

            except requests.exceptions.JSONDecodeError:
                print(f"JSON decoding failed. Response text: {response.text}")
                failure_counter += 1
                retries += 1
            except KeyError as e:
                print(
                    f"Missing expected key in response: {e}. Response text: {response.text}"
                )
                failure_counter += 1
                retries += 1
        else:
            print(
                f"Request failed with status code {response.status_code}. Retrying... (Attempt {retries + 1}/{retry_limit})"
            )
            print(f"Response text: {response.text}")
            retries += 1

    print(f"Failed to get a valid response after {retry_limit} retries.")
    raise Exception(f"Exceeded retry limit. Total failures: {failure_counter}")


# This is the original json and csv conversion function to match the v5 prompts up until v49 
def orig_save_json_and_csv(response, model_name, input_file, base_output_dir, run_num):
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
    current_date = datetime.datetime.now().strftime("%b%d")

    # Save JSON to the dedicated JSON folder

    # Create a model folder and article folder for CSVs
    model_dir = os.path.join(base_output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    article_dir = os.path.join(model_dir, base_name)
    os.makedirs(article_dir, exist_ok=True)

    # Save CSV in the respective article folder
    output_csv = os.path.join(
        article_dir, f"{base_name}-{model_name}-v{run_num + 2}-{current_date}.csv"
    )
    output_json = os.path.join(
        article_dir, f"{base_name}-{model_name}-v{run_num + 2}-{current_date}.json"
    )
    with open(output_json, "w") as json_file:
        json_file.write(cleaned_response)

    try:
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
                if not name:  # This handles None, empty string, and other falsy values
                    name = source.get("Name_of_Source", "") or ""
#                if len(name) == 0:
#                    name = source.get("Name_of_Source", "")

                title = source.get("Title of Source", "")
                if not title:  # This handles None, empty string, and other falsy values
                    title = source.get("Title of Source", "") or ""

                association = source.get("Source Justification", "")
                if not association:  # This handles None, empty string, and other falsy values
                    association = source.get("Source Justification", "") or ""

                sourced_statements = source.get("Sourced Statement", [])

                if isinstance(sourced_statements, list):
                    for sourced_statement in sourced_statements:
                        data.append(
                            [sourced_statement, source_type, name, title, association]
                        )
                else:
                    data.append(
                        [sourced_statements, source_type, name, title, association]
                    )

        with open(output_csv, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                ["SourcedStatement", "SourceType", "Name", "Title", "Justification"]
            )
            writer.writerows(data)

        print(f"CSV file generated: {output_csv}")
    except Exception as e:
        print(f"CSV not generated for {base_name}: {e}")


# This is the revised save_json_and_csv function for v50 prompts and later; 
#v50 user prompt asks model to do identify sourced statements line by line first and then the rest of the data
def save_json_and_csv(response, model_name, input_file, base_output_dir, run_num):
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
    current_date = datetime.datetime.now().strftime("%b%d")

    model_dir = os.path.join(base_output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    article_dir = os.path.join(model_dir, base_name)
    os.makedirs(article_dir, exist_ok=True)

    output_csv = os.path.join(
        article_dir, f"{base_name}-{model_name}-v{run_num + 2}-{current_date}.csv"
    )
    output_json = os.path.join(
        article_dir, f"{base_name}-{model_name}-v{run_num + 2}-{current_date}.json"
    )
    
    with open(output_json, "w") as json_file:
        json_file.write(cleaned_response)

    try:
        json_data = json.loads(cleaned_response)
        data = []

        sourcing_table = json_data.get("Sourcing Table", [])
        
        for source in sourcing_table:
            sourced_statement = source.get("Sourced Statement", "")
            source_type = source.get("Type of Source", "")
            name = source.get("Name of Source", "")
            if not name:
                name = ""
            
            title = source.get("Title of Source", "")
            if not title:
                title = ""

            justification = source.get("Source Justification", "")
            if not justification:
                justification = ""

            data.append([sourced_statement, source_type, name, title, justification])

        with open(output_csv, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                ["SourcedStatement", "SourceType", "Name", "Title", "Justification"]
            )
            writer.writerows(data)

        print(f"CSV file generated: {output_csv}")
    except Exception as e:
        print(f"CSV not generated for {base_name}: {e}")

def extract_json_strings(content):
    json_pattern = re.compile(r"\{.*?\}", re.DOTALL)
    json_matches = json_pattern.findall(content)

    valid_json_objects = []
    for match in json_matches:
        try:
            if "[" in match:
                match = match.split("[", 1)[1]
            if "]" in match:
                match = match.split("]", 1)[0]
            json_obj = json.loads(match)
            valid_json_objects.append(json_obj)
        except json.JSONDecodeError:
            print("parse error" + match)
            continue
    return valid_json_objects

def process_files(input_dir, output_dir, selected_models, loop_times, prefix_string="", current_prompt_version=None):
    # Use global prompt_version if not provided
    if current_prompt_version is None:
        current_prompt_version = prompt_version
    
    # Create experiment directory with optional prefix
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if prefix_string:
        experiment_dir = os.path.join(output_dir, f"{prefix_string}_llm_results_{timestamp}")
    else:
        experiment_dir = os.path.join(output_dir, f"llm_results_{timestamp}")
    
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Results will be saved in: {experiment_dir}")
    print(f"Using prompt version: {current_prompt_version}")

    for i in range(loop_times):
        print(f"\n=== Starting experiment run {i + 1}/{loop_times} ===")
        
        # Save prompts for this experiment run (only once)
        if i == 0:
            with open(os.path.join(experiment_dir, "system_prompt.txt"), "w") as f:
                f.write(system_prompt_content)
            with open(os.path.join(experiment_dir, "user_prompt.txt"), "w") as f:
                f.write(user_prompt_content)

        # Process each file in the input directory
        for input_file in glob.glob(os.path.join(input_dir, "*.txt")):
            with open(input_file, "r", encoding="utf-8") as file:
                article_text = file.read()

            print(f"\nProcessing file: {input_file}")

            # Process each selected model
            for api_name, file_name in selected_models:
                print(f"  Processing with {file_name}...")
                try:
                    response = analyze_with_openrouter(article_text, api_name)
                    
                    # Choose appropriate save function based on prompt version
                    if should_use_orig_function(current_prompt_version):
                        orig_save_json_and_csv(response, file_name, input_file, experiment_dir, i)
                        print(f"  ✓ Completed {file_name} (using orig function for {current_prompt_version})")
                    else:
                        save_json_and_csv(response, file_name, input_file, experiment_dir, i)
                        print(f"  ✓ Completed {file_name} (using new function for {current_prompt_version})")
                        
                except Exception as e:
                    print(f"  ✗ Failed {file_name}: {e}")
                
                time.sleep(5)  # Rate limiting

        print(f"Experiment run {i + 1} completed.")

    print(f"\nAll experiments completed. Results saved in {experiment_dir}")


def main():
    parser = argparse.ArgumentParser(description="Process articles with multiple LLM models")
    parser.add_argument("-m", "--models", default="all", 
                       help="Models to use: 'all', provider names (anthropic, google, openai, meta, nvidia, deepseek), or specific model names (comma-separated)")
    parser.add_argument("-s", "--string", default="", 
                       help="Custom prefix string for output directory")
    parser.add_argument("-t", "--times", type=int, default=1, 
                       help="Number of times to run the experiment loop (default: 1)")
    parser.add_argument("-pv", "--prompt-version", default=None,
                       help="Prompt version to use (e.g., v41, v50). If not specified, uses global setting")
    
    args = parser.parse_args()

    # Load API key and set the global variable
    global or_api_key
    or_api_key = load_api_key()
    
    if or_api_key:
        print('✓ API key loaded successfully')
        print(f'Key starts with: {or_api_key[:10]}...')
    else:
        print('✗ No API key found')
        print('Please set OPENROUTER_API_KEY environment variable or create a config file')
        return  # Exit if no API key found
    
    # Determine which prompt version to use
    effective_prompt_version = args.prompt_version if args.prompt_version else prompt_version
    
    # Parse and validate models
    selected_models = parse_model_argument(args.models)
    if not selected_models:
        print("Error: No valid models selected")
        return
    
    print(f"Selected models: {[model[1] for model in selected_models]}")
    print(f"Loop times: {args.times}")
    print(f"Prompt version: {effective_prompt_version}")
    if args.string:
        print(f"Output prefix: {args.string}")
    
    # Load prompts
    global system_prompt_content, user_prompt_content
    system_prompt_content, SYSTEM_PROMPT_VERSION = get_latest_prompt("system", effective_prompt_version)
    user_prompt_content, USER_PROMPT_VERSION = get_latest_prompt("user", effective_prompt_version)
    
    # Process files
    input_dir = "2025_input_stories"
    output_dir = "llm_results"
    
    process_files(input_dir, output_dir, selected_models, args.times, args.string, effective_prompt_version)

if __name__ == "__main__":
    main()
