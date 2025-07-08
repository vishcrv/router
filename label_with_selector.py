import json
from llm_selector import LLMSelector
from tqdm import tqdm

# Load filtered prompts
with open("filtered_prompts.json", "r") as f:
    prompts = [json.loads(line)["prompt"] for line in f]

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

