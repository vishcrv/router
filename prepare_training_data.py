import json
import random
from sklearn.model_selection import train_test_split

# Load selector-labeled prompts
with open("labeled_prompts_selector.json", "r") as f:
    data = [json.loads(line) for line in f]

# Extract relevant fields
examples = [{"prompt": d["prompt"], "label": d["selected_model"]} for d in data]

# Shuffle and split into train/val/test
train, temp = train_test_split(examples, test_size=0.2, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

# Save splits
for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
    with open(f"dataset_{split_name}.json", "w") as f:
        for item in split_data:
            json.dump(item, f)
            f.write("\n")

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

