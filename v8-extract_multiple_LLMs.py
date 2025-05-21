import json
import os
import csv
import datetime
import glob
import re
import requests
import time
import logging

version = "v40"


def get_latest_prompt(prompt_type,version):
    prompt_dir = "new_prompts"
    prompt_file = os.path.join(prompt_dir, f"{prompt_type}_prompt_{version}.txt")
    print(prompt_file)
    if not prompt_file:
        raise FileNotFoundError(f"No {prompt_type} prompt files found in the 'prompts' directory.")
    #latest_prompt_file = max(prompt_files, key=os.path.getctime)
    with open(prompt_file, 'r') as file:
        prompt = file.read().strip()
    return prompt, os.path.basename(prompt_file).split('.')[0]  # Return prompt and filename without extension

# Load prompts
system_prompt_content, SYSTEM_PROMPT_VERSION = get_latest_prompt("system", version)
user_prompt_content, USER_PROMPT_VERSION = get_latest_prompt("user", version)
#print(user_prompt_content)

def analyze_with_openrouter(article_text, model_name, retry_limit=3, size_threshold=50):
    chat_prompt = [
        {
            "role": "system",
            "content": system_prompt_content,
        },
        {
            "role": "user",
            "content": user_prompt_content + '%' + article_text,
        }
    ]

    retries = 0
    failure_counter = 0  # Counter to track bad responses

    while retries < retry_limit:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"bearer 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",
                "Content-Type": "application/json"
            },
            data=json.dumps({
                "model": model_name,
                "messages": chat_prompt,
                "temperature": 0.0,
                "max_tokens": 9048,
                "include_reasoning": True
            })
        )

        if response.status_code == 200:
            try:
                data = response.json()
                act_res = data['choices'][0]['message']['content']
                json_size = len(json.dumps(data).encode('utf-8'))

                # Check JSON size
                if json_size < size_threshold:
                    failure_counter += 1
                    print(f"Response size too small ({json_size} bytes). Retrying... (Attempt {retries + 1}/{retry_limit})")
                    retries += 1
                    continue

                return act_res  # Valid response

            except requests.exceptions.JSONDecodeError:
                print(f"JSON decoding failed. Response text: {response.text}")
                failure_counter += 1
                retries += 1
            except KeyError as e:
                print(f"Missing expected key in response: {e}. Response text: {response.text}")
                failure_counter += 1
                retries += 1
        else:
            print(f"Request failed with status code {response.status_code}. Retrying... (Attempt {retries + 1}/{retry_limit})")
            print(f"Response text: {response.text}")
            retries += 1

    print(f"Failed to get a valid response after {retry_limit} retries.")
    raise Exception(f"Exceeded retry limit. Total failures: {failure_counter}")




#new json convertion function to match the v5 prompts json schema.
def save_json_and_csv(response, model_name, input_file, base_output_dir, run_num):
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
    current_date = datetime.datetime.now().strftime("%b%d")

    # Save JSON to the dedicated JSON folder
   
    # Create a model folder and article folder for CSVs
    model_dir = os.path.join(base_output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    article_dir = os.path.join(model_dir, base_name)
    os.makedirs(article_dir, exist_ok=True)

    # Save CSV in the respective article folder
    output_csv = os.path.join(article_dir, f"{base_name}-{model_name}-v{run_num+2}-{current_date}.csv")
    output_json = os.path.join(article_dir, f"{base_name}-{model_name}-v{run_num+2}-{current_date}.json")
    with open(output_json, "w") as json_file:
        json_file.write(cleaned_response)
    
    try:
        json_data = json.loads(cleaned_response)
        data = []

        for source_type in ["Named Organization Sources", "Named Person Sources", "Document Sources", "Unnamed Group of People", "Anonymous Sources"]:
            for source in json_data.get(source_type, []):
                name = source.get("Name of Source", "")
                if len(name) == 0:
                    name = source.get("Name_of_Source", "")
                title = source.get("Title of Source", "")
                association = source.get("Source Justification", "")
                sourced_statements = source.get("Sourced Statement", [])

                if isinstance(sourced_statements, list):
                    for sourced_statement in sourced_statements:
                        data.append([sourced_statement, source_type, name, title, association])
                else:
                    data.append([sourced_statements, source_type, name, title, association])

        with open(output_csv, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["SourcedStatement", "SourceType", "Name", "Title", "Justification"])
            writer.writerows(data)

        print(f"CSV file generated: {output_csv}")
    except Exception as e:
        print(f"CSV not generated for {base_name}: {e}")

def extract_json_strings(content):
   
    json_pattern = re.compile(r'\{.*?\}', re.DOTALL)
    json_matches = json_pattern.findall(content)
 
    valid_json_objects = []
    for match in json_matches:
        
        try:
            # Parse JSON and validate
            #print(match)
            if '[' in match :  
                 match = match.split('[',1)[1]
            if ']' in match:
                match = match(']',1)[0]
            json_obj = json.loads(match)
            #print(json_obj)
            valid_json_objects.append(json_obj)
        except json.JSONDecodeError:
            print("parse error" + match)
            continue
    return valid_json_objects


def process_files(input_dir, output_dir):
     # Create a new directory for this experiment run
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    experiment_dir=os.path.join(output_dir, f"llms_{timestamp}")
    os.makedirs(experiment_dir)

    for i in range(2):  # Loop to run the experiment 5 times
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
            # print(article_text)

            # Analyze with GPT-4
            
            # gpt4o_response = analyze_with_openrouter(article_text, "openai/chatgpt-4o-latest")
            # save_json_and_csv(gpt4o_response, "chatgpt-4o-latest", input_file, experiment_dir, i+1)
            # time.sleep(120)
            # claude35_response = analyze_with_openrouter(article_text, "anthropic/claude-3.5-sonnet")
            # save_json_and_csv(claude35_response, "claude-3.5-sonnet", input_file, experiment_dir, i+1)
            # time.sleep(120)
            
            # gemini_response = analyze_with_openrouter(article_text, "google/gemini-pro-1.5")
            # save_json_and_csv(gemini_response, "gemini-pro-1.5", input_file, experiment_dir, i+1)
            # time.sleep(120)
        
            # llama70_response = analyze_with_openrouter(article_text, "nvidia/llama-3.1-nemotron-70b-instruct")
            # save_json_and_csv(llama70_response, "llama-3.1-70b-instruct", input_file, experiment_dir,i+1)
            
            # time.sleep(120)
            # llama405_response = analyze_with_openrouter(article_text, "meta-llama/llama-3.1-405b-instruct")
            # save_json_and_csv(llama405_response, "llama-3.1-405b-instruct", input_file, experiment_dir, i+1)

            # time.sleep(120)
            # model_dir = os.path.join(experiment_dir, "deepseek-chat-free")      
            # base_name = os.path.splitext(os.path.basename(input_file))[0]      
            # article_dir = os.path.join(model_dir, base_name)
        
            # if os.path.isdir(article_dir):
            #     continue
            
            try:
                deepseek_response = analyze_with_openrouter(article_text, "deepseek/deepseek-r1")
                save_json_and_csv(deepseek_response, "deepseek-r1", input_file, experiment_dir, i+1)
                time.sleep(30)
            except Exception as e:
                logging.error(f"Error with message {e}")
                pass

            try:
                gemini_response = analyze_with_openrouter(article_text, "google/gemini-2.5-flash-preview")
                save_json_and_csv(gemini_response, "gemini-2.5-flash", input_file, experiment_dir, i+1)
                time.sleep(30)
            except Exception as e:
                logging.error(f"Error with message {e}")
                pass
            
            
           
        print(f"Experiment run {i+1} completed. Results saved in {experiment_dir}")


if __name__ == "__main__":
    input_dir = "extracted_articles_boilerplate"
    output_dir = "llm_results"
    process_files(input_dir, output_dir)
