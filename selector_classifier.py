# selector_classifier.py

import os
import json
import joblib
import numpy as np
import re

# Load the trained classifier model and label encoder
MODEL_PATH = "selector_model/classifier.joblib"
ENCODER_PATH = "label_encoder.json"

if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError("Trained classifier or label encoder not found. Run train_classifier.py first.")

classifier = joblib.load(MODEL_PATH)
with open(ENCODER_PATH, "r") as f:
    label_encoder = json.load(f)

# Reverse label encoder (index to label)
index_to_label = {int(v): k for k, v in label_encoder.items()}

# Feature extractor – directly implement analyze_prompt logic
def extract_features(prompt: str):
    return np.array([
        len(prompt),  # length
        sum(c.isdigit() for c in prompt),  # has_numbers
        int(bool(re.search(r"\bcode|function|algorithm|implement\b", prompt.lower()))),  # has_code
        int(bool(re.search(r"\bwhat|how|why|when|who|explain|describe\b", prompt.lower()))),  # question_words
        int(bool(re.search(r"\banalyze|compare|discuss|elaborate\b", prompt.lower()))),  # complexity_indicators
        int(bool(re.search(r"\bstory|poem|write|creative\b", prompt.lower()))),  # creative_indicators
        int(bool(re.search(r"\bimage|picture|visualize\b", prompt.lower()))),  # visual_indicators
        int(bool(re.search(r"\bfact|capital|president|data\b", prompt.lower()))),  # factual_indicators
        int(bool(re.search(r"\bhello|hi|how are you|today\b", prompt.lower())))  # conversational_indicators
    ]).reshape(1, -1)

# Inference wrapper
def predict_model(prompt: str) -> str:
    features = extract_features(prompt)
    prediction = classifier.predict(features)[0]
    return index_to_label[prediction]


class SelectorClassifier:
    def __init__(self):
        self.classifier = classifier
        self.index_to_label = index_to_label

    def extract_features(self, prompt: str):
        return extract_features(prompt)

    def select_best_model(self, prompt: str):
        features = self.extract_features(prompt)
        prediction = self.classifier.predict(features)[0]
        model_name = self.index_to_label[prediction]

        # Optional: add dummy confidence + reasoning for now
        return {
            "selected_model": model_name,
            "confidence": 0.9,  # Dummy — you can use predict_proba if classifier supports it
            "reasoning": f"Classifier predicted {model_name} based on extracted features.",
            "all_scores": {}  # Fill if using predict_proba
        }
