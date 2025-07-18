# 🚀 LLM Router - Intelligent Model Selection for Language Models

An intelligent routing system that dynamically selects the optimal LLM for any given prompt using a fine-tuned DistilBERT classifier.

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Endpoints](#-api-endpoints)
- [Model Selection](#-model-selection)
- [Evaluation Metrics](#-evaluation-metrics)
- [Development](#-development)
- [Contributing](#-contributing)

## ✨ Features

- **Smart Model Selection**: Uses a fine-tuned DistilBERT classifier to route prompts to the best LLM
- **Multiple Models**: Supports 5 powerful models:
  - DeepSeek Chat v3 (best for coding & complex reasoning)
  - Mistral Small 3.1 (excellent for technical explanations)
  - Qwen3 30B (great for multilingual & visual tasks)
  - Gemini 2.5 Pro (fast for factual queries)
  - Llama 3.3 (strong in creative writing)
- **Performance Evaluation**: Built-in metrics for comparing model responses
- **Batch Processing**: Handle multiple prompts efficiently
- **Health Monitoring**: Real-time model availability checks
- **Detailed Analytics**: Response quality scoring and selection confidence

## The Model can be found here
https://drive.google.com/drive/folders/1R0Aja43ioyGsxDrPbl0bQROeoMnGxy9K?usp=sharing

## 🏗 Architecture

```
├── main.py                   # FastAPI application entry point
├── llm_selector.py          # Model selection classifier
├── router.py                # API request handling & routing
├── evaluator.py             # Response quality evaluation
├── llms.py                  # Model configurations
├── selector_model/          # Trained DistilBERT model
└── datasets/                # Training & evaluation data
```

## 📥 Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/llm-router.git
cd llm-router
```

2. Set up a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:

```bash
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

## 🎯 Usage

### Starting the Server

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### Example Requests

1. Basic Query:

```python
import requests

response = requests.post("http://localhost:8000/query", json={
    "prompt": "Write a Python function to calculate factorial"
})
print(response.json())
```

2. Compare All Models:

```python
response = requests.post("http://localhost:8000/query-compare", json={
    "prompt": "Explain quantum computing"
})
print(response.json())
```

## 🔌 API Endpoints

### POST /query

Routes a prompt to the best-suited model.

**Request:**

```json
{
  "prompt": "Write a Python function to calculate factorial"
}
```

**Response:**

```json
{
    "prompt": "Write a Python function to calculate factorial",
    "selected_model": "deepseek-r1",
    "response": "Here's a recursive function...",
    "selection_confidence": 0.95,
    "selection_reasoning": "Selected for coding task",
    "all_model_scores": {
        "deepseek-r1": 0.95,
        "mistral-small": 0.75,
        ...
    }
}
```

### POST /query-compare

Compares responses from all models.

### POST /batch-query

Process multiple prompts efficiently.

### GET /health

Check model availability and system status.

## 🎯 Model Selection

The system uses two approaches for model selection:

1. **DistilBERT Classifier**:

   - Fine-tuned on 100K+ labeled prompts
   - 92% accuracy on test set
   - Fast inference (< 50ms)

2. **Fallback Rule-based System**:
   - Pattern matching for specific tasks
   - Keyword analysis
   - Task complexity estimation

### Selection Performance

```
=== Evaluation on dataset_val.json ===
Confusion Matrix:
[[226   1   4  42   1]
 [ 44 749   9  16   1]
 [ 12   8 156  99   3]
 [ 33   8  23 395  48]
 [ 13   0   0  24 129]]

Classification Report:
               precision    recall  f1-score   support

  deepseek-r1       0.69      0.82      0.75       274
 gemini-flash       0.98      0.91      0.95       819
    llama-3.3       0.81      0.56      0.66       278
mistral-small       0.69      0.78      0.73       507
     qwen-2.5       0.71      0.78      0.74       166

     accuracy                           0.81      2044
    macro avg       0.77      0.77      0.77      2044
 weighted avg       0.82      0.81      0.81      2044


Class Distribution:
deepseek-r1: 274 samples
gemini-flash: 819 samples
llama-3.3: 278 samples
mistral-small: 507 samples
qwen-2.5: 166 samples

=== Evaluation on dataset_test.json ===
Confusion Matrix:
[[231   2   5  34   2]
 [ 46 703   9  17   1]
 [  8  17 145 114   2]
 [ 36   4  29 395  42]
 [ 13   1   0  19 170]]

Classification Report:
               precision    recall  f1-score   support

  deepseek-r1       0.69      0.84      0.76       274
 gemini-flash       0.97      0.91      0.94       776
    llama-3.3       0.77      0.51      0.61       286
mistral-small       0.68      0.78      0.73       506
     qwen-2.5       0.78      0.84      0.81       203

     accuracy                           0.80      2045
    macro avg       0.78      0.77      0.77      2045
 weighted avg       0.81      0.80      0.80      2045


Class Distribution:
deepseek-r1: 274 samples
gemini-flash: 776 samples
llama-3.3: 286 samples
mistral-small: 506 samples
qwen-2.5: 203 samples
```

## 📊 Evaluation Metrics

Response quality is evaluated across multiple dimensions:

| Dimension          | Max Points | Description                   |
| ------------------ | ---------- | ----------------------------- |
| Structure          | 2          | Clear organization & flow     |
| Examples           | 1.5        | Relevant examples provided    |
| Formatting         | 1          | Proper code/text formatting   |
| Code Quality       | 4          | For programming tasks         |
| Math Accuracy      | 4          | For mathematical computations |
| Creativity         | 4          | For creative writing          |
| Factual Content    | 3          | Accuracy of information       |
| Visual Description | 3          | For image-related tasks       |

## 💻 Development

### Training the Classifier

1. Generate labeled data:

```bash
python label_with_selector.py
```

2. Prepare training datasets:

```bash
python prepare_training_data.py
```

3. Train the model:

```bash
python train_classifier.py
```



### Code Style

```bash
# Install development dependencies
uv pip install -r requirements.txt
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Ivan
- Vishnu

Acknowledgments

- OpenRouter for providing model access
- Hugging Face for transformer models
- FastAPI 
