# selector_classifier.py

import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "./selector_model/"
ENCODER_PATH = "label_encoder.json"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# Load label encoder (expects id2label mapping inside the JSON)
with open(ENCODER_PATH, "r") as f:
    label_data = json.load(f)
    index_to_label = {int(k): v for k, v in label_data["id2label"].items()}

class SelectorClassifier:
    def __init__(self):
        self.model = model
        self.tokenizer = tokenizer
        self.label_map = index_to_label

    def select_best_model(self, prompt: str):
        # Tokenize the input prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            prediction = torch.argmax(probs).item()
            confidence = probs[prediction].item()

        # Build result
        model_name = self.label_map[prediction]
        return {
            "selected_model": model_name,
            "confidence": round(confidence, 3),
            "reasoning": f"Classifier chose {model_name} with {confidence:.2f} confidence.",
            "all_scores": {
                self.label_map[i]: round(score.item(), 3)
                for i, score in enumerate(probs)
            }
        }
