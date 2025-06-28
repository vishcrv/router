LLM_CONFIGS = {
    "deepseek-r1": {
        "name": "deepseek-r1",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "headers": {
            "Authorization": "Bearer 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",
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
            "Authorization": "Bearer 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",
            "Content-Type": "application/json"
        },
        "body_template": lambda prompt: {
            "model": "mistralai/mistral-small-3.2-24b-instruct:free",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
        }
    }
}