import re
from typing import Dict, Any

def score_response(prompt: str, response: str) -> float:
    """
    Enhanced scoring function that considers multiple factors
    """
    if not response or response.startswith("ERROR"):
        return 0.0
    
    # Basic metrics
    word_count = len(response.strip().split())
    sentence_count = len(re.findall(r'[.!?]+', response))
    
    # Quality indicators
    has_structure = bool(re.search(r'(?:first|second|third|finally|however|therefore)', response.lower()))
    has_examples = bool(re.search(r'(?:for example|such as|like|including)', response.lower()))
    has_code = bool(re.search(r'```|`[^`]+`', response))
    
    # Calculate base score
    base_score = min(word_count / 50, 10)  # Normalize word count
    
    # Bonuses
    structure_bonus = 2 if has_structure else 0
    example_bonus = 1.5 if has_examples else 0
    code_bonus = 3 if has_code and any(kw in prompt.lower() for kw in ['code', 'program', 'function']) else 0
    
    # Penalties
    too_short_penalty = -3 if word_count < 10 else 0
    repetition_penalty = -1 if len(set(response.split())) / len(response.split()) < 0.7 else 0
    
    total_score = base_score + structure_bonus + example_bonus + code_bonus + too_short_penalty + repetition_penalty
    
    return max(0, total_score)

def evaluate_selection_accuracy(prompt: str, predicted_model: str, all_responses: Dict[str, str]) -> Dict[str, Any]:
    """
    Evaluate how well the model selection performed
    """
    # Score all responses
    scores = {model: score_response(prompt, response) for model, response in all_responses.items()}
    actual_best = max(scores, key=scores.get)
    
    # Calculate accuracy metrics
    prediction_correct = predicted_model == actual_best
    score_difference = scores[actual_best] - scores[predicted_model]
    relative_performance = scores[predicted_model] / scores[actual_best] if scores[actual_best] > 0 else 0
    
    return {
        "prediction_correct": prediction_correct,
        "predicted_model": predicted_model,
        "actual_best_model": actual_best,
        "score_difference": score_difference,
        "relative_performance": relative_performance,
        "all_scores": scores,
        "selection_quality": "excellent" if relative_performance > 0.95 else 
                           "good" if relative_performance > 0.8 else
                           "fair" if relative_performance > 0.6 else "poor"
    }
