# LLM Router API Documentation

## Overview

This API dynamically routes prompts to the most appropriate Large Language Model (LLM) from a selection of 5 models:

- DeepSeek R1
- Mistral Small
- Qwen2.5
- Gemini Flash
- Llama 3.3

The system evaluates prompts based on their characteristics and selects the optimal model using a rules-based scoring system.

## System Architecture

├── evaluator.py # Response scoring and evaluation logic
├── llm_selector.py # Model selection algorithm
├── llms.py # Model configurations
├── main.py # FastAPI application and endpoints
└── router.py # Model querying and health checks

## Key Components

### 1. LLM Selector (`llm_selector.py`)

The core model selection logic that analyzes prompts and selects the best model.

#### Features:

- Rule-based scoring system with 8 predefined categories
- Dynamic model scoring based on prompt characteristics
- Fallback logic for ambiguous prompts
- Detailed selection reasoning

#### Methods:

- `analyze_prompt()`: Extracts prompt characteristics
- `calculate_model_scores()`: Computes confidence scores for each model
- `select_best_model()`: Returns the optimal model with reasoning

### 2. Evaluator (`evaluator.py`)

Evaluates response quality and selection accuracy.

#### Features:

- Multi-dimensional response scoring (structure, examples, formatting, etc.)
- Task-specific scoring (coding, math, creative, factual, visual)
- Penalties for errors, repetition, and incompleteness
- Batch evaluation capabilities

#### Methods:

- `score_response()`: Rates response quality (0-10 scale)
- `evaluate_selection_accuracy()`: Compares predicted vs actual best model
- `batch_evaluate_selections()`: Evaluates multiple prompts
- `generate_evaluation_report()`: Creates human-readable reports

### 3. Model Configurations (`llms.py`)

Contains configurations for all supported LLMs.

#### Supported Models:

- `deepseek-r1`: Best for coding and complex reasoning
- `mistral-small`: Good for language processing
- `qwen-2.5`: Specialized for visual tasks
- `gemini-flash`: Optimized for quick factual queries
- `llama-3.3`: Excellent for creative tasks

### 4. API Endpoints (`main.py`)

FastAPI application with the following endpoints:

#### Core Endpoints:

- `POST /query`: Routes prompt to best model
- `POST /query-compare`: Queries all models and evaluates selection
- `POST /query-specific`: Bypasses selection for specific model
- `GET /health`: Checks model availability
- `POST /test-selection`: Tests selection logic with sample prompts
- `POST /batch-query`: Processes multiple prompts

## API Reference

### `POST /query`

**Purpose**: Route prompt to best model  
**Request**:

```json
{
  "prompt": "Your question or instruction"
}
```

**Response**:

```json
{
  "prompt": "...",
  "selected_model": "model-name",
  "response": "...",
  "selection_confidence": 0.85,
  "selection_reasoning": "Explanation...",
  "all_model_scores": {"model1": 0.8, ...}
}
```

### `POST /query-compare`

**Purpose**: Compare all models and evaluate selection  
**Response**:

```json
{
  "prompt": "...",
  "predicted_best": "model-name",
  "actual_best": "model-name",
  "prediction_correct": true,
  "selection_quality": "excellent",
  "relative_performance": 0.95,
  "score_difference": 0.2,
  "all_responses": {"model1": "response1", ...}
}
```

### `POST /batch-query`

**Purpose**: Process multiple prompts  
**Request**:

```json
{
  "prompts": ["prompt1", "prompt2"],
  "compare_mode": false
}
```

**Response**:

```json
{
  "batch_results": [...],
  "total_prompts": 2,
  "mode": "query"
}
```

## API Reference

### The evaluator scores responses based on:

#### Quality Indicators:

- Structure (2 pts)
- Examples (1.5 pts)
- Formatting (1 pt)
- Code quality (up to 4 pts)
- Math accuracy (up to 4 pts)
- Creativity (up to 4 pts)
- Factual content (up to 3 pts)
- Visual description (up to 3 pts)

#### Penalties:

- Too short/long (-4/-2 pts)
- Repetition (-2 pts)
- Errors (-5 pts)
- Incomplete (-3 pts)

### Selection Rules

The system uses these primary rules for model selection:

- Mathematical/Coding: DeepSeek R1 (confidence: 0.9)
- Complex Reasoning: DeepSeek R1 (0.85)
- Creative Writing: Llama 3.3 (0.85)
- Language Tasks: Mistral Small (0.8)
- Visual Tasks: Qwen2.5 (0.8)
- Quick Facts: Gemini Flash (0.75)
- Conversation: Llama 3.3 (0.7)
- Technical Docs: Mistral Small (0.75)

## Setup

- Set the environment variable

```bash
export OPENROUTER_API_KEY="your-api-key"
```

- Install dependencies:

```bash
pip install fastapi uvicorn httpx python-dotenv
```

- Run the api:

```bash
python main.py
```
