from dotenv import load_dotenv
load_dotenv()
import os
import os

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not set. Check your .env file and environment variables.")

    
from fastapi import FastAPI, Request
from router import query_single_model, get_all_responses, get_model_health
from llm_selector import LLMSelector
from evaluator import score_response, evaluate_selection_accuracy
import uvicorn


app = FastAPI(title="Enhanced LLM Router API", version="2.0")
selector = LLMSelector()

@app.get("/")
async def root():
    return {
        "message": "Enhanced LLM Router API - 5 Models",
        "models": ["deepseek-r1", "mistral-small", "qwen-2.5", "gemini-flash", "llama-3.3"],
        "endpoints": ["/query", "/query-compare", "/health", "/test-selection"]
    }

@app.post("/query")
async def query_router(request: Request):
    """
    Main endpoint - selects the best model and returns its response
    """
    data = await request.json()
    prompt = data.get("prompt")
    
    if not prompt:
        return {"error": "No prompt provided"}
    
    # Select the best model dynamically
    selection_result = selector.select_best_model(prompt)
    selected_model = selection_result["selected_model"]
    
    # Query only the selected model
    response = await query_single_model(selected_model, prompt)
    
    return {
        "prompt": prompt,
        "selected_model": selected_model,
        "response": response,
        "selection_confidence": selection_result["confidence"],
        "selection_reasoning": selection_result["reasoning"],
        "all_model_scores": selection_result["all_scores"]
    }

@app.post("/query-compare")
async def query_compare_models(request: Request):
    """
    Comparison endpoint - queries all models and evaluates selection accuracy
    """
    data = await request.json()
    prompt = data.get("prompt")
    
    if not prompt:
        return {"error": "No prompt provided"}
    
    # Get selection prediction
    selection_result = selector.select_best_model(prompt)
    predicted_model = selection_result["selected_model"]
    
    # Get all responses for comparison
    all_responses = await get_all_responses(prompt)
    
    # Evaluate selection accuracy
    evaluation = evaluate_selection_accuracy(prompt, predicted_model, all_responses)
    
    return {
        "prompt": prompt,
        "predicted_best": predicted_model,
        "actual_best": evaluation["actual_best_model"],
        "prediction_correct": evaluation["prediction_correct"],
        "selection_quality": evaluation["selection_quality"],
        "selection_reasoning": selection_result["reasoning"],
        "selection_confidence": selection_result["confidence"],
        "relative_performance": evaluation["relative_performance"],
        "score_difference": evaluation["score_difference"],
        "all_model_scores": selection_result["all_scores"],
        "response_quality_scores": evaluation["all_scores"],
        "all_responses": all_responses
    }

@app.post("/query-specific")
async def query_specific_model(request: Request):
    """
    Query a specific model directly (bypass selection logic)
    """
    data = await request.json()
    prompt = data.get("prompt")
    model = data.get("model")
    
    if not prompt or not model:
        return {"error": "Both prompt and model must be provided"}
    
    response = await query_single_model(model, prompt)
    
    return {
        "prompt": prompt,
        "model": model,
        "response": response
    }

@app.get("/health")
async def health_check():
    """
    Check the health/availability of all models
    """
    health_status = await get_model_health()
    
    healthy_models = [model for model, status in health_status.items() if status["healthy"]]
    unhealthy_models = [model for model, status in health_status.items() if not status["healthy"]]
    
    return {
        "overall_health": "healthy" if len(healthy_models) >= 3 else "degraded" if len(healthy_models) >= 1 else "unhealthy",
        "healthy_models": healthy_models,
        "unhealthy_models": unhealthy_models,
        "total_models": len(health_status),
        "healthy_count": len(healthy_models),
        "detailed_status": health_status
    }

@app.post("/test-selection")
async def test_selection_logic(request: Request):
    """
    Test the selection logic with multiple prompts
    """
    data = await request.json()
    test_prompts = data.get("prompts", [])
    
    if not test_prompts:
        # Default test prompts for each model type
        test_prompts = [
            "Write a Python function to calculate factorial",  # deepseek-r1
            "Solve this equation: 2x + 5 = 13",  # deepseek-r1
            "Analyze the pros and cons of renewable energy step by step",  # deepseek-r1
            "Write a short story about a lonely robot",  # llama-3.3
            "Create a poem about autumn leaves",  # llama-3.3
            "Translate this to Spanish: Hello, how are you?",  # mistral-small
            "Explain how neural networks work",  # mistral-small
            "Describe this image in detail",  # qwen-2.5
            "What do you see in this picture?",  # qwen-2.5
            "What is the capital of India?",  # gemini-flash
            "Who is the current president of the United States?",  # gemini-flash
            "Hello, how are you today?",  # llama-3.3
            "I need advice on choosing a career"  # llama-3.3
        ]
    
    results = []
    model_counts = {
        "deepseek-r1": 0,
        "mistral-small": 0,
        "qwen-2.5": 0,
        "gemini-flash": 0,
        "llama-3.3": 0
    }
    
    for prompt in test_prompts:
        selection_result = selector.select_best_model(prompt)
        model_counts[selection_result["selected_model"]] += 1
        
        results.append({
            "prompt": prompt,
            "selected_model": selection_result["selected_model"],
            "confidence": selection_result["confidence"],
            "reasoning": selection_result["reasoning"],
            "all_scores": selection_result["all_scores"]
        })
    
    return {
        "test_results": results,
        "model_selection_counts": model_counts,
        "total_prompts": len(test_prompts),
        "selection_distribution": {
            model: f"{count}/{len(test_prompts)} ({count/len(test_prompts)*100:.1f}%)"
            for model, count in model_counts.items()
        }
    }

@app.post("/batch-query")
async def batch_query(request: Request):
    """
    Process multiple prompts in batch
    """
    data = await request.json()
    prompts = data.get("prompts", [])
    compare_mode = data.get("compare_mode", False)
    
    if not prompts:
        return {"error": "No prompts provided"}
    
    results = []
    
    for prompt in prompts:
        if compare_mode:
            # Use comparison mode
            selection_result = selector.select_best_model(prompt)
            all_responses = await get_all_responses(prompt)
            evaluation = evaluate_selection_accuracy(prompt, selection_result["selected_model"], all_responses)
            
            results.append({
                "prompt": prompt,
                "predicted_model": selection_result["selected_model"],
                "actual_best": evaluation["actual_best_model"],
                "prediction_correct": evaluation["prediction_correct"],
                "selection_quality": evaluation["selection_quality"],
                "confidence": selection_result["confidence"],
                "relative_performance": evaluation["relative_performance"]
            })
        else:
            # Use regular query mode
            selection_result = selector.select_best_model(prompt)
            response = await query_single_model(selection_result["selected_model"], prompt)
            
            results.append({
                "prompt": prompt,
                "selected_model": selection_result["selected_model"],
                "response": response,
                "confidence": selection_result["confidence"],
                "reasoning": selection_result["reasoning"]
            })
    
    return {
        "batch_results": results,
        "total_prompts": len(prompts),
        "mode": "comparison" if compare_mode else "query"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)