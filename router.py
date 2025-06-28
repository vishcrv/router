import httpx
from llms import LLM_CONFIGS

async def query_model(model_config, prompt):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                model_config["url"],
                headers=model_config["headers"],
                json=model_config["body_template"](prompt),
                timeout=20.0
            )
            return response.json()["choices"][0]["message"]["content"]

        except Exception as e:
            return f"ERROR: {e}"

async def get_all_responses(prompt):
    from asyncio import gather
    tasks = [query_model(cfg, prompt) for cfg in LLM_CONFIGS.values()]
    responses = await gather(*tasks)
    return {name: resp for name, resp in zip(LLM_CONFIGS.keys(), responses)}




