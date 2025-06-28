LLM_CONFIGS = {
    "deepseek-r1": {
        "name": "deepseek-r1",
        "url": "https://openrouter.ai/api/v1/chat/completions",  # ✅ Use chat endpoint
        "headers": {
            "Authorization": "Bearer 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"
        },
        "body_template": lambda prompt: {
            "model": "deepseek/deepseek-r1-0528:free",
            "messages": [{"role": "user", "content": prompt}],  # ✅ Use chat-style messages
            "max_tokens": 512,
        }
    },
    "mistral-small": {
        "name": "mistral-small",
        "url": "https://openrouter.ai/api/v1/chat/completions",  # ✅ Use chat endpoint
        "headers": {
            "Authorization": "Bearer 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"
        },
        "body_template": lambda prompt: {
            "model": "mistralai/mistral-small-3.2-24b-instruct:free",
            "messages": [{"role": "user", "content": prompt}],  # ✅ Chat format!
            "max_tokens": 512,
        }
    }
}
