from fastapi import FastAPI, Request
from router import get_all_responses
from evaluator import score_response

app = FastAPI()

@app.post("/query")
async def query_router(request: Request):
    data = await request.json()
    prompt = data.get("prompt")

    responses = await get_all_responses(prompt)
    scored = {model: score_response(prompt, resp) for model, resp in responses.items()}
    best_model = max(scored, key=scored.get)

    return {
        "best_model": best_model,
        "response": responses[best_model],
        "all_scores": scored,
        "all_responses": responses
    }
