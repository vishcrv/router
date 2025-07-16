# selector_classifier.py

import json
import torch
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

MODEL_DIR = "selector_model"
ENCODER_PATH = "label_encoder.json"

# Load tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# Load label encoder
with open(ENCODER_PATH, "r") as f:
    index_to_label = json.load(f)
    index_to_label = {int(k): v for k, v in index_to_label.items()}

class SelectorClassifier:
    def __init__(self):
        self.model = model
        self.tokenizer = tokenizer
        self.label_map = index_to_label

    def select_best_model(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            prediction = torch.argmax(logits, dim=-1).item()
            confidence = torch.softmax(logits, dim=-1)[0][prediction].item()

        model_name = self.label_map[prediction]
        return {
            "selected_model": model_name,
            "confidence": round(confidence, 3),
            "reasoning": f"DistilBERT classifier chose {model_name} with {confidence:.2f} confidence.",
            "all_scores": {
                self.label_map[i]: round(score, 3)
                for i, score in enumerate(torch.softmax(logits, dim=-1)[0].tolist())
            }
        }
