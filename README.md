# Enhanced LLM Router API

A FastAPI backend that intelligently routes user prompts to the most suitable Large Language Model (LLM) among five supported models (DeepSeek R1, Mistral Small, Qwen 2.5, Gemini Flash, Llama 3.3) using prompt analysis and scoring logic. The project supports prompt evaluation, model comparison, batch queries, and health checks, and is designed for easy extension and experimentation.

---

## Features

- **Dynamic Model Selection:**  
  Analyzes each prompt and selects the best LLM based on keywords, patterns, and prompt characteristics.
- **Supports 5 LLMs:**  
  DeepSeek R1, Mistral Small, Qwen 2.5, Gemini Flash, Llama 3.3 (via OpenRouter API).
- **Prompt Evaluation:**  
  Scores and compares responses from all models for accuracy and quality.
- **Batch and Comparison Endpoints:**  
  Send multiple prompts at once, compare model outputs, and test selection logic.
- **Health Checks:**  
  Check the availability of all configured models.
- **Configurable API Key:**  
  Uses a `.env` file for the OpenRouter API key (never hardcoded).
- **REST Client Examples:**  
  Includes a `prompt.rest` file for easy API testing in VS Code.

---

## Project Structure

```
router1/
├── .env                # Your OpenRouter API key (not tracked by git)
├── .gitignore
├── evaluator.py        # Response scoring and evaluation logic
├── llm_selector.py     # Prompt analysis and model selection logic
├── llms.py             # Model configuration (endpoints, headers, etc.)
├── main.py             # FastAPI app and all endpoints
├── prompt.rest         # REST Client examples for testing endpoints
├── router.py           # Model querying and orchestration logic
├── test_selector.py    # Unit tests for selection logic
└── __pycache__/        # Python cache files (ignored)
```

---

## Setup & Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/vishcrv/backend.git
   cd backend/vishnu/router1
   ```

2. **Create and activate a virtual environment:**
   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```sh
   pip install fastapi uvicorn httpx python-dotenv
   ```

4. **Configure your API key:**
   - Create a `.env` file in this directory:
     ```
     OPENROUTER_API_KEY=your-openrouter-api-key-here
     ```

---

## Running the Server

```sh
uvicorn main:app --reload
```
The API will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000).

---

## API Endpoints

- `GET /`  
  API info and available models.

- `POST /query`  
  **Body:** `{ "prompt": "your prompt here" }`  
  Returns the best model's response and selection reasoning.

- `POST /query-compare`  
  **Body:** `{ "prompt": "your prompt here" }`  
  Returns responses from all models and evaluates selection accuracy.

- `POST /query-specific`  
  **Body:** `{ "prompt": "...", "model": "model-name" }`  
  Query a specific model directly.

- `GET /health`  
  Health check for all models.

- `POST /test-selection`  
  **Body:** `{ "prompts": [ ... ] }`  
  Test selection logic with multiple prompts.

- `POST /batch-query`  
  **Body:** `{ "prompts": [ ... ], "compare_mode": true/false }`  
  Batch process multiple prompts, optionally with comparison.

---

## Testing with REST Client

Use the included [`prompt.rest`](prompt.rest) file with the [REST Client VS Code extension](https://marketplace.visualstudio.com/items?itemName=humao.rest-client) for easy API testing.  
You can add or modify queries as needed.

---

## File-by-File Overview

- **main.py**  
  FastAPI app, all endpoint definitions, and server entrypoint.

- **llms.py**  
  Contains the configuration for all supported LLMs (API URLs, headers, body templates).  
  Uses `{OPENROUTER_API_KEY}` as a placeholder, replaced at runtime.

- **llm_selector.py**  
  Implements the `LLMSelector` class, which analyzes prompts and scores each model based on rules, keywords, and patterns.

- **router.py**  
  Handles preparing model configs, making async HTTP requests to LLM APIs, and orchestrating queries to one or all models.

- **evaluator.py**  
  Functions for scoring model responses, evaluating selection accuracy, and generating evaluation reports.

- **test_selector.py**  
  Contains test functions to validate the selection logic with a variety of prompt types and edge cases.

- **prompt.rest**  
  Example API requests for all endpoints, ready to use with the REST Client extension.

- **.gitignore**  
  Ignores `.env`, `__pycache__/`, virtual environments, logs, and other non-source files.

---

## Security

- **Never commit your `.env` file or API keys to git.**
- The `.gitignore` ensures `.env` and other sensitive files are not tracked.

---


