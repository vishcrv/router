import json
from llm_selector import LLMSelector
from tqdm import tqdm

# Load filtered prompts
prompts = []
with open("filtered_prompts.json", "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:  # Skip empty lines
            continue
        try:
            data = json.loads(line)
            if "prompt" in data:
                prompts.append(data["prompt"])
            else:
                print(f"Warning: Line {line_num} does not contain 'prompt' key")
        except json.JSONDecodeError as e:
            print(f"Warning: Skipping line {line_num} due to JSON parsing error: {e}")
            print(f"Line content: {repr(line)}")
            continue

print(f"Successfully loaded {len(prompts)} prompts")

# Initialize selector
selector = LLMSelector()

# Run selector on each prompt
labeled_data = []
for prompt in tqdm(prompts, desc="Labeling prompts with selector"):
    result = selector.select_best_model(prompt)
    labeled_data.append({
        "prompt": prompt,
        "selected_model": result["selected_model"],
        "confidence": result["confidence"],
        "all_scores": result["all_scores"],
        "reasoning": result["reasoning"]
    })

# Save labeled file
with open("labeled_prompts_selector.json", "w") as f:
    for item in labeled_data:
        json.dump(item, f)
        f.write("\n")

print(f"Labeled {len(labeled_data)} prompts with selector.")

