import httpx
import os
from llms import LLM_CONFIGS

import os

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not set. Check your .env file and environment variables.")

    
def prepare_model_config(model_config):
    """Prepare model configuration with API key"""
    config = model_config.copy()
    
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not set")

    # replace the key placeholder in headers
    config["headers"] = {
        key: value.replace("{OPENROUTER_API_KEY}", str(OPENROUTER_API_KEY)) if isinstance(value, str) else value
        for key, value in config["headers"].items()
    }
    return config





async def query_model(model_config, prompt):
    """Query a single model with the given configuration"""
    config = prepare_model_config(model_config)
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                config["url"],
                headers=config["headers"],
                json=config["body_template"](prompt),
                timeout=30.0  # Increased timeout for 5 models
            )
            response.raise_for_status()  # Raise exception for HTTP errors
            data = response.json()
            
            # Enhanced response validation and debugging
            if not data:
                return f"ERROR: Empty response from {model_config['name']}"
            
            if "choices" in data and data["choices"]:
                content = data["choices"][0].get("message", {}).get("content")
                if not content:
                    return f"ERROR: Empty content from {model_config['name']}"
                return content
            elif "error" in data:
                error_msg = data.get("error", {})
                if isinstance(error_msg, dict):
                    error_msg = error_msg.get("message", str(error_msg))
                return f"API ERROR ({model_config['name']}): {error_msg}"
            else:
                return f"Unexpected response format from {model_config['name']}: {str(data)[:200]}"
                
        except httpx.TimeoutException:
            return f"ERROR: Request timeout for {model_config['name']} (30s)"
        except httpx.HTTPStatusError as e:
            error_body = ""
            try:
                error_body = e.response.json()
            except:
                error_body = e.response.text[:200]
            return f"ERROR: HTTP {e.response.status_code} for {model_config['name']} - {error_body}"
        except Exception as e:
            return f"ERROR: {str(e)} for {model_config['name']}"

async def query_single_model(model_name, prompt):
    """Query a specific model by name"""
    if model_name not in LLM_CONFIGS:
        available_models = ", ".join(LLM_CONFIGS.keys())
        return f"ERROR: Model '{model_name}' not found. Available models: {available_models}"
    
    model_config = LLM_CONFIGS[model_name]
    return await query_model(model_config, prompt)

async def get_all_responses(prompt):
    """Query all configured models (kept for comparison/evaluation purposes)"""
    from asyncio import gather
    
    # creates tasks for all models
    tasks = []
    model_names = []
    
    for model_name, model_config in LLM_CONFIGS.items():
        tasks.append(query_model(model_config, prompt))
        model_names.append(model_name)
    
    # executes all tasks concurrently
    responses = await gather(*tasks, return_exceptions=True)
    
    # handles any exceptions that occurred
    result = {}
    for model_name, response in zip(model_names, responses):
        if isinstance(response, Exception):
            result[model_name] = f"ERROR: {str(response)}"
        else:
            result[model_name] = response
    
    return result

async def get_model_health():
    """Check health/availability of all models"""
    health_check_prompt = "Hello, respond with 'OK' if you're working."
    
    health_status = {}
    for model_name in LLM_CONFIGS.keys():
        try:
            response = await query_single_model(model_name, health_check_prompt)
            is_healthy = not response.startswith("ERROR")
            health_status[model_name] = {
                "healthy": is_healthy,
                "response": response[:100] + "..." if len(response) > 100 else response
            }
        except Exception as e:
            health_status[model_name] = {
                "healthy": False,
                "response": f"Health check failed: {str(e)}"
            }
    
    return health_status