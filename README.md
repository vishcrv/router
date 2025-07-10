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
