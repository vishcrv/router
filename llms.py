LLM_CONFIGS = {
    "deepseek-r1": {
        "name": "deepseek-r1",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "headers": {
            "Authorization": "Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/",
            "X-Title": "LLM Router"
        },
        "body_template": lambda prompt: {
            "model": "deepseek/deepseek-chat-v3-0324:free",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0.3,
            "top_p": 0.95,
            "stream": False
        }
    },
    "mistral-small": {
        "name": "mistral-small",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "headers": {
            "Authorization": "Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/",
            "X-Title": "LLM Router"
        },
        "body_template": lambda prompt: {
            "model": "mistralai/mistral-small-3.1-24b-instruct:free",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.95,
            "stream": False
        }
    },
    "qwen-2.5": {
        "name": "qwen-2.5",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "headers": {
            "Authorization": "Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/",
            "X-Title": "LLM Router"
        },
        "body_template": lambda prompt: {
            "model": "qwen/qwen3-30b-a3b:free",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.95,
            "stream": False
        }
    },
    "gemini-flash": {
        "name": "gemini-flash",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "headers": {
            "Authorization": "Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/",
            "X-Title": "LLM Router"
        },
        "body_template": lambda prompt: {
            "model": "google/gemini-2.0-flash-exp:free",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.95,
            "stream": False
        }
    },
    "llama-3.3": {
        "name": "llama-3.3",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "headers": {
            "Authorization": "Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/",
            "X-Title": "LLM Router"
        },
        "body_template": lambda prompt: {
            "model": "meta-llama/llama-3.3-70b-instruct:free",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.95,
            "stream": False
        }
    }
}
