# LLM Router: Model Selection Logic and Evolution

## 1. Current Logic and Technologies

### 1.1. Regex-Based Rule System (in `llm_selector.py`)

The initial approach for selecting the best LLM (Large Language Model) for a given prompt is a rule-based system that relies heavily on regular expressions (regex) and keyword matching. This system is implemented in the `LLMSelector` class in `llm_selector.py`.

#### How the Regex Logic Works
- **Selection Rules**: The system defines a list of selection rules, each associated with a specific LLM. Each rule contains:
  - A set of `keywords` (e.g., "calculate", "code", "story", "translate", etc.)
  - A set of `patterns` (regex patterns, e.g., `r'def\s+\w+'`, `r'print\s*\('`, `r'what\s+is\s+the\s+'`)
  - A target `model` (e.g., "deepseek-r1", "mistral-small", etc.)
  - A `confidence` score (how strongly the rule should influence selection)

- **Prompt Analysis**: When a prompt is received, the system:
  - Converts the prompt to lowercase for case-insensitive matching.
  - Checks for the presence of keywords and regex pattern matches in the prompt.
  - Analyzes the prompt for features like length, presence of numbers, code, print statements, question words, and indicators for complexity, creativity, visual, factual, or conversational content.

- **Scoring**: For each rule, the system:
  - Counts keyword and pattern matches, computes a score for each model based on the number of matches and the rule's confidence.
  - Applies additional bonuses based on prompt analysis (e.g., code presence boosts DeepSeek R1, creative indicators boost Llama 3.3).
  - Normalizes scores and applies fallback logic if no model scores highly.

- **Selection**: The model with the highest score is selected, and reasoning is generated based on the analysis.

#### Example Regex Patterns Used
- `r'def\s+\w+'`: Matches Python function definitions.
- `r'print\s*\('`: Matches print statements in code.
- `r'what\s+is\s+the\s+'`: Matches factual questions.
- `r'describe\s+(?:this|the)\s+image'`: Matches visual description prompts.

#### Strengths of Regex-Based Logic
- **Transparency**: Easy to understand and debug; rules are explicit.
- **Control**: Fine-grained control over which prompts trigger which models.
- **No Training Required**: Works out-of-the-box without labeled data.

#### Weaknesses of Regex-Based Logic
- **Brittleness**: Regex and keyword rules can miss edge cases or be too broad, leading to misclassification.
- **Scalability**: Adding new models or prompt types requires manual rule updates.
- **Limited Generalization**: Cannot capture nuanced or implicit prompt meanings; struggles with ambiguous or creative prompts.
- **Maintenance Overhead**: As the number of rules grows, the system becomes harder to maintain and test.

#### When Regex Works Well
- For highly structured prompts (e.g., code, math, explicit questions).
- When prompt types are well-defined and limited in variety.

#### When Regex Fails
- For prompts with subtle intent, creative language, or mixed tasks.
- When users phrase requests in unexpected ways.
- For multilingual or code-mixed prompts.

---

## 2. Transition to a Classifier Model

### 2.1. Motivation for Switching

As the project grew, the limitations of the regex-based approach became apparent:
- **Coverage**: Regex rules could not cover the full diversity of real-world prompts.
- **Accuracy**: Misclassifications increased as prompt variety grew.
- **Adaptability**: Manual rule updates were not scalable for new domains or models.
- **Performance**: The need for more robust, data-driven selection became clear.

### 2.2. The Classifier Approach

The new approach uses a machine learning classifier to predict the best LLM for a given prompt. This is implemented in `train_classifier.py` (training), `selector_classifier.py` (inference), and `evaluate_classifier.py` (evaluation).

#### How the Classifier Works
- **Data**: The classifier is trained on a large dataset of prompts labeled with the best model for each.
- **Model**: The project uses a BERT-family model, specifically DistilBERT, for sequence classification.
- **Training**: Prompts are tokenized and fed into DistilBERT, which learns to predict the correct model label. Class weights are used to handle imbalanced data.
- **Inference**: For a new prompt, the classifier outputs probabilities for each model, and the model with the highest probability is selected.
- **Evaluation**: The classifier achieves high accuracy (80-82% on test/validation sets) and provides confidence scores for each prediction.

#### Why BERT/DistilBERT?
- **Contextual Understanding**: BERT models use attention mechanisms to understand context, making them far superior to regex for nuanced language understanding.
- **Generalization**: BERT can generalize to new prompt phrasings and unseen examples, reducing brittleness.
- **Efficiency**: DistilBERT is a lightweight, fast variant of BERT, making it suitable for real-time inference.
- **Proven Performance**: BERT-based models are state-of-the-art for many NLP tasks, including classification.
- **Transfer Learning**: Pretrained on large corpora, BERT models require less labeled data to achieve good performance.

#### Implementation Details
- **Model**: `DistilBertForSequenceClassification` (with class weights for imbalance)
- **Tokenization**: `DistilBertTokenizerFast` with max length 512
- **Training**: HuggingFace Trainer API, with metrics for accuracy, precision, recall, and F1
- **Inference**: Softmax probabilities, confidence reporting, and reasoning output

---

## 3. Summary

- The project started with a regex-based rule system for LLM selection, which is transparent but limited in flexibility and coverage.
- To improve accuracy and scalability, the system transitioned to a BERT-based classifier (DistilBERT), which leverages deep contextual understanding to robustly select the best LLM for any prompt.
- This evolution enables the router to handle a much wider variety of prompts, adapt to new domains, and provide reliable, explainable model selection at scale. 