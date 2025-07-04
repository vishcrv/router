LLM_CONFIGS = {
    "deepseek-r1": {
        "name": "deepseek-r1",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "headers": {
            "Authorization": "Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        "body_template": lambda prompt: {
            "model": "deepseek/deepseek-r1-0528:free",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
        }
    },
    "mistral-small": {
        "name": "mistral-small",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "headers": {
            "Authorization": "Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        "body_template": lambda prompt: {
            "model": "mistralai/mistral-small-3.2-24b-instruct:free",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
        }
    },
    "qwen-2.5": {
        "name": "qwen-2.5",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "headers": {
            "Authorization": "Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        "body_template": lambda prompt: {
            "model": "qwen/qwen2.5-vl-72b-instruct:free",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
        }
    },
    "gemini-flash": {
        "name": "gemini-flash",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "headers": {
            "Authorization": "Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        "body_template": lambda prompt: {
            "model": "google/gemini-2.5-flash-lite-preview-06-17",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
        }
    },
    "llama-3.3": {
        "name": "llama-3.3",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "headers": {
            "Authorization": "Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        "body_template": lambda prompt: {
            "model": "nvidia/llama-3.3-nemotron-super-49b-v1:free",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
        }
    }
}
