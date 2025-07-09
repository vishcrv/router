import os
import json
import joblib
import numpy as np
from router.llm_selector import LLMSelector

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

# Feature extractor
selector = LLMSelector()

def extract_features(prompt: str):
    analysis = selector.analyze_prompt(prompt)
    return np.array([
        analysis["length"],
        analysis["has_numbers"],
        analysis["has_code"],
        analysis["question_words"],
        analysis["complexity_indicators"],
        analysis["creative_indicators"],
        analysis["visual_indicators"],
        analysis["factual_indicators"],
        analysis["conversational_indicators"]
    ]).reshape(1, -1)

def predict_model(prompt: str) -> str:
    features = extract_features(prompt)
    prediction = classifier.predict(features)[0]
    return index_to_label[prediction]
