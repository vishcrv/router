import httpx
from llms import LLM_CONFIGS

async def query_model(model_config, prompt):
    """Query a single model with the given configuration"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                model_config["url"],
                headers=model_config["headers"],
                json=model_config["body_template"](prompt),
                timeout=20.0
            )
            data = response.json()
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"]
            elif "error" in data:
                return f"API ERROR: {data['error']}"
            else:
                return f"Unexpected response: {data}"
        except Exception as e:
            return f"ERROR: {e}"

async def query_single_model(model_name, prompt):
    """Query a specific model by name"""
    if model_name not in LLM_CONFIGS:
        return f"ERROR: Model '{model_name}' not found in configurations"
    
    model_config = LLM_CONFIGS[model_name]
    return await query_model(model_config, prompt)

async def get_all_responses(prompt):
    """Query all configured models (kept for comparison/evaluation purposes)"""
    from asyncio import gather
    tasks = [query_model(cfg, prompt) for cfg in LLM_CONFIGS.values()]
    responses = await gather(*tasks)
    return {name: resp for name, resp in zip(LLM_CONFIGS.keys(), responses)}
