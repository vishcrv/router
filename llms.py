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
    }
}
