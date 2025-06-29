from fastapi import FastAPI, Request
from router import query_single_model
from llm_selector import LLMSelector

app = FastAPI()
selector = LLMSelector()

@app.post("/query")
async def query_router(request: Request):
    data = await request.json()
    prompt = data.get("prompt")
    
    # Select the best model dynamically
    selection_result = selector.select_best_model(prompt)
    selected_model = selection_result["selected_model"]
    
    # Query only the selected model
    response = await query_single_model(selected_model, prompt)
    
    return {
        "selected_model": selected_model,
        "response": response,
        "selection_confidence": selection_result["confidence"],
        "selection_reasoning": selection_result["reasoning"],
        "all_model_scores": selection_result["all_scores"]
    }

@app.post("/query-compare")
async def query_compare_models(request: Request):
    """
    Optional endpoint that still compares all models if needed for testing/evaluation
    """
    from router import get_all_responses
    from evaluator import score_response
    
    data = await request.json()
    prompt = data.get("prompt")
    
    # Get selection prediction
    selection_result = selector.select_best_model(prompt)
    
    # Get all responses for comparison
    responses = await get_all_responses(prompt)
    scored = {model: score_response(prompt, resp) for model, resp in responses.items()}
    best_by_evaluation = max(scored, key=scored.get)
    
    return {
        "predicted_best": selection_result["selected_model"],
        "actual_best": best_by_evaluation,
        "prediction_correct": selection_result["selected_model"] == best_by_evaluation,
        "selection_reasoning": selection_result["reasoning"],
        "all_scores": scored,
        "all_responses": responses,
        "selection_confidence": selection_result["confidence"]
    }
