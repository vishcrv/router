import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load label encoder
# Load label encoder in reverse
with open("label_encoder.json", "r") as f:
    raw_map = json.load(f)

# Convert string index to int
index_to_label = {int(k): v for k, v in raw_map.items()}
label_to_index = {v: k for k, v in index_to_label.items()}

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("selector_model")
tokenizer = AutoTokenizer.from_pretrained("selector_model")

def load_dataset(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def evaluate(path):
    data = load_dataset(path)
    y_true, y_pred = [], []

    for item in data:
        prompt = item["prompt"]
        label = item["label"]

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        pred_index = torch.argmax(outputs.logits, dim=1).item()

        y_true.append(label_to_index[label])
        y_pred.append(pred_index)

    print(f"\n=== Evaluation on {path} ===")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=[index_to_label[i] for i in sorted(index_to_label)]
    ))

evaluate("dataset_val.json")
evaluate("dataset_test.json")
