import os
import json
import requests
import time
from dotenv import load_dotenv
from llms import LLM_CONFIGS
import random
import time
import requests
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not set. Check your .env file and environment variables.")
# Load model name -> label ID mapping
with open("label_encoder.json", "r") as f:
    label_encoder = json.load(f)
model_to_label = {v: k for k, v in label_encoder.items()}

# Load prompts
with open("filtered_prompts.json", "r", encoding="utf-8") as f:
    prompts = [json.loads(line)["prompt"] for line in f]

# OpenRouter API key (make sure it's set in your environment)
import os
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not set")

# Replaces {OPENROUTER_API_KEY} in config
def resolve_headers(config):
    headers = config["headers"].copy()
    headers["Authorization"] = headers["Authorization"].replace("{OPENROUTER_API_KEY}", OPENROUTER_API_KEY)
    return headers

# Make a real call to OpenRouter
def generate_response(model_name, prompt, max_retries=5, min_wait=3, max_wait=6):
    config = LLM_CONFIGS[model_name]
    url = config["url"]
    headers = resolve_headers(config)
    body = config["body_template"](prompt)

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=body)
            response.raise_for_status()
            output = response.json()
            return output["choices"][0]["message"]["content"]
        
        except requests.exceptions.HTTPError as e:
            if response.status_code in [429, 503]:
                wait_time = min_wait + attempt * 2 + random.uniform(0, 2)
                print(f"[RATE LIMIT] {model_name} attempt {attempt+1} - sleeping {wait_time:.1f}s")
                time.sleep(wait_time)
            else:
                print(f"[ERROR] {model_name} failed: {e}")
                break  # Other HTTP errors – don't retry

        except Exception as e:
            print(f"[ERROR] {model_name} unexpected error: {e}")
            break

    return "[ERROR] Failed after retries."
# Generate and write labeled samples
output_path = "labeled_prompts_groundtruth.jsonl"
with open(output_path, "w", encoding="utf-8") as out_f:
    for prompt in prompts:
        for model_name in model_to_label:
            label = model_to_label[model_name]
            response = generate_response(model_name, prompt)
            sample = {
                "prompt": prompt,
                "response": response,
                "label": label
            }
            out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            print(f"[✓] Wrote output from {model_name}")
            time.sleep(1.5)  # Be nice to OpenRouter, avoid rate limiting

