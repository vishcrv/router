# LLM Router API – Smart Model Selection for LLMs

by vishnu and ivan

## 🔍 Overview

This API dynamically selects and routes a user's prompt to the best-suited LLM from a pool of 5 models:

- DeepSeek R1
- Mistral Small
- Qwen2.5
- Gemini Flash
- Llama 3.3

Unlike static rule-based routing, this system uses a fine-tuned classifier (DistilBERT) to select the most appropriate model for a given prompt.

## 🧠 Core Logic

- **Model Selection**: `llm_selector.py` uses a trained transformer classifier to predict the best LLM.
- **Model Querying**: `router.py` sends requests to OpenRouter-backed APIs.
- **Response Evaluation**: `evaluator.py` compares all model responses and scores them for benchmarking.

## 📁 File Structure

```bash
├── app.py                     # FastAPI entrypoint (formerly main.py)
├── llm_selector.py           # Classifier-based model selection
├── router.py                 # Model querying + health checks
├── evaluator.py              # Multi-dimensional response scoring
├── llms.py                   # API config for each model
├── selector_model/           # Trained DistilBERT classifier
├── label_encoder.json        # Model-to-class label mapping
├── *.json                    # Sample prompts & datasets
```

## 🚀 Running Locally

- Prerequisites:

```bash
pip install -r requirements.txt
```

## 🏋️‍♂️ Training the Classifier

The classifier learns to route prompts to the best model by training on labeled examples. Labels are initially generated using the rule-based system.

### 🔁 Step 1: Label Prompts (Rule-based Autolabeling)

Use your old rule-based logic to generate model labels for raw prompts:

```bash
python label_with_selector.py
```
- This creates labeled_prompts_selector.json with entries like:
```bash
{ "text": "Write a Python function to calculate factorial", "label": "deepseek-r1" }
```

### 🧹 Step 2: Prepare Training Data
- Convert labeled prompts into format compatible with Hugging Face's datasets library:
```bash
python prepare_training_data.py
```
- This generates dataset_train.json, dataset_val.json, and dataset_test.json.

### 🧠 Step 3: Train the Classifier
-Fine-tune a DistilBERT classifier on the prepared dataset:
```bash
python train_classifier.py
```
- The model is saved to: `selector_model/`
- Label encoding (class ↔︎ model) is saved in:
```bash
label_encoder.json
```



- Run the API:

```bash
export OPENROUTER_API_KEY=your-api-key
python app.py
```

##🔌 Endpoints

`POST /query`

- Selects the best model via classifier and routes the prompt.

- Request:

```bash
{ "prompt": "Write a Python function to calculate factorial" }
```

-Response

```bash
{
  "prompt": "...",
  "selected_model": "llama-3.3",
  "response": "...",
  "selection_confidence": 0.93,
  "reasoning": "Predicted 'llama-3.3' with confidence 0.93"
}
```

`POST /query-compare`

- Compares classifier-selected model to all models and evaluates correctness.

```bash
curl -X POST http://localhost:8000/query-compare \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain the benefits of renewable energy"}'
```

```bash
curl -X POST http://localhost:8000/query-specific \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of India?", "model": "gemini-flash"}'
```

`POST /batch-query`

- Handles multiple prompts and optionally enables comparison

`POST /test-selection`

- Returns classifier decisions across sample prompts.

## 📊 Evaluator Scoring

| Dimension                  | Max Points |
| -------------------------- | ---------- |
| Structure                  | 2          |
| Examples                   | 1.5        |
| Formatting                 | 1          |
| Code Quality               | 4          |
| Math Accuracy              | 4          |
| Creativity                 | 4          |
| Factual Content            | 3          |
| Visual Descrip.            | 3          |
| Repetition/Error Penalties | -14        |

## 🛠 Classifier Info

- Model: DistilBERT (Fine-tuned)
- Input: Natural language prompt
- Output: Predicted best model label
- Confidence: Softmax class probability

## 🧠 Future Plans

- Add few-shot prompting support
- Live feedback loop for continual learning
- Model latency & cost-based routing



## Why Update the RULE based selector to a model
### 1. Generalization beyond the rules
- Rule-based = rigid logic.
- Classifier = pattern recognition.
- After training, the model can generalize from prompt features (e.g. keywords, phrasing, structure) that aren’t explicitly covered in the rule logic.
### 2. Robustness to variation
- Rule-based selectors break if input phrasing changes.
- e.g. "Sum of 2 and 2?" vs. "Add 2 + 2"
- Classifier learns semantic patterns, not fixed keyword triggers.
### 3. Foundation for improvement
- You can fine-tune the classifier on better ground-truth labels later:
- Human-annotated routes
- Evaluated best-response LLMs
- But rule-based logic can't "learn" anything.
### 4. Faster runtime than complex rule graphs
- Once trained, inference is a single vectorized prediction.
- Rule-based selectors (especially with nested heuristics) can get slow and messy.
### 5. Easier to expand
Want to add a new LLM to the router? Just:
- Label a few hundred prompts with the new model
- Retrain the classifier
- No need to redesign rules or engineer edge cases.


### Evaluation

```bash
=== Evaluation on dataset_val.json ===
Confusion Matrix:
[[211   2   1   6   1]
 [  0 423   4   4   0]
 [  4   0 110   7   0]
 [  3   7  14 182   0]
 [  3   1   0   0  43]]

Classification Report:
               precision    recall  f1-score   support

  deepseek-r1       0.95      0.95      0.95       221
 gemini-flash       0.98      0.98      0.98       431
    llama-3.3       0.85      0.91      0.88       121
mistral-small       0.91      0.88      0.90       206
     qwen-2.5       0.98      0.91      0.95        47

     accuracy                           0.94      1026
    macro avg       0.94      0.93      0.93      1026
 weighted avg       0.94      0.94      0.94      1026


=== Evaluation on dataset_test.json ===
Confusion Matrix:
[[196   2   0  11   1]
 [  0 419   8   4   1]
 [  1   0 106  13   1]
 [  5   2  22 182   0]
 [  1   0   0   2  49]]

Classification Report:
               precision    recall  f1-score   support

  deepseek-r1       0.97      0.93      0.95       210
 gemini-flash       0.99      0.97      0.98       432
    llama-3.3       0.78      0.88      0.82       121
mistral-small       0.86      0.86      0.86       211
     qwen-2.5       0.94      0.94      0.94        52

     accuracy                           0.93      1026
    macro avg       0.91      0.92      0.91      1026
 weighted avg       0.93      0.93      0.93      1026
```
