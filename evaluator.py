import re
from typing import Dict, Any, List
import json
import os

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not set. Check your .env file and environment variables.")

    


def score_response(prompt: str, response: str) -> float:
    """
    Enhanced scoring function that evaluates response quality across multiple dimensions
    """
    if not response or response.startswith("ERROR"):
        return 0.0
    
    # clean the response
    response = response.strip()
    if not response:
        return 0.0
    
    # basic metrics
    word_count = len(response.split())
    sentence_count = len(re.findall(r'[.!?]+', response))
    char_count = len(response)
    
    # quality indicators
    has_structure = bool(re.search(r'(?:first|second|third|finally|however|therefore|moreover|furthermore)', response.lower()))
    has_examples = bool(re.search(r'(?:for example|such as|like|including|instance)', response.lower()))
    has_code = bool(re.search(r'```|`[^`]+`|def\s+\w+|class\s+\w+', response))
    has_numbers = bool(re.search(r'\d+', response))
    has_lists = bool(re.search(r'(?:^|\n)\s*[-*•]\s+|\d+\.\s+', response, re.MULTILINE))
    
    # prompt-specific scoring
    prompt_lower = prompt.lower()
    
    # base score calculation
    base_score = min(word_count / 30, 8)  # Normalize word count (max 8 points)
    
    # content quality bonuses
    structure_bonus = 2 if has_structure else 0
    example_bonus = 1.5 if has_examples else 0
    formatting_bonus = 1 if has_lists else 0
    
    # Task-specific bonuses
    code_bonus = 0
    math_bonus = 0
    creative_bonus = 0
    factual_bonus = 0
    visual_bonus = 0
    
    # Code-related tasks
    if any(kw in prompt_lower for kw in ['code', 'program', 'function', 'algorithm', 'debug']):
        if has_code:
            code_bonus = 4
        elif 'def ' in response or 'class ' in response or 'import ' in response:
            code_bonus = 3
        elif any(lang in response.lower() for lang in ['python', 'java', 'javascript', 'c++']):
            code_bonus = 2
    
    # Math-related tasks
    if any(kw in prompt_lower for kw in ['calculate', 'solve', 'equation', 'math']):
        if has_numbers:
            math_bonus = 3
        if any(op in response for op in ['=', '+', '-', '*', '/', '^']):
            math_bonus += 1
    
    # Creative tasks
    if any(kw in prompt_lower for kw in ['write', 'story', 'poem', 'creative', 'imagine']):
        creativity_indicators = len(re.findall(r'(?:beautiful|amazing|wonderful|magical|enchanting|mysterious)', response.lower()))
        dialogue_present = bool(re.search(r'["""].*["""]|said|replied|asked', response))
        narrative_elements = bool(re.search(r'(?:once upon|suddenly|meanwhile|finally)', response.lower()))
        
        creative_bonus = min(creativity_indicators + (2 if dialogue_present else 0) + (1 if narrative_elements else 0), 4)
    
    # Factual/knowledge tasks
    if any(kw in prompt_lower for kw in ['what is', 'who is', 'when', 'where', 'capital', 'definition']):
        specific_facts = len(re.findall(r'\b\d{4}\b|\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', response))
        factual_bonus = min(specific_facts * 0.5, 3)
    
    # Visual/description tasks
    if any(kw in prompt_lower for kw in ['describe', 'image', 'picture', 'visual', 'see']):
        descriptive_words = len(re.findall(r'(?:color|shape|size|texture|bright|dark|large|small)', response.lower()))
        visual_bonus = min(descriptive_words * 0.3, 3)
    
    # Penalties
    too_short_penalty = -4 if word_count < 10 else 0
    too_long_penalty = -2 if word_count > 500 else 0
    
    # Repetition penalty
    words = response.lower().split()
    unique_words = set(words)
    repetition_ratio = len(unique_words) / len(words) if words else 1
    repetition_penalty = -2 if repetition_ratio < 0.6 else 0
    
    # Error penalty
    error_penalty = -5 if any(err in response.upper() for err in ['ERROR', 'FAILED', 'TIMEOUT']) else 0
    
    # Incomplete response penalty
    incomplete_penalty = -3 if response.endswith(('...', 'etc.', 'and so on')) else 0
    
    # Calculate total score
    total_score = (
        base_score +
        structure_bonus +
        example_bonus +
        formatting_bonus +
        code_bonus +
        math_bonus +
        creative_bonus +
        factual_bonus +
        visual_bonus +
        too_short_penalty +
        too_long_penalty +
        repetition_penalty +
        error_penalty +
        incomplete_penalty
    )
    
    # Ensure score is non-negative
    return max(0, total_score)

def evaluate_selection_accuracy(prompt: str, predicted_model: str, all_responses: Dict[str, str]) -> Dict[str, Any]:
    """
    Evaluate how well the model selection performed by comparing predicted vs actual best model
    """
    # Score all responses
    scores = {}
    for model, response in all_responses.items():
        scores[model] = score_response(prompt, response)
    
    # Find the actual best model
    actual_best = max(scores, key=scores.get) if scores else predicted_model
    
    # Calculate performance metrics
    predicted_score = scores.get(predicted_model, 0)
    best_score = scores.get(actual_best, 0)
    
    prediction_correct = predicted_model == actual_best
    score_difference = best_score - predicted_score
    relative_performance = predicted_score / best_score if best_score > 0 else 0
    
    # Determine selection quality
    if relative_performance >= 0.95:
        selection_quality = "excellent"
    elif relative_performance >= 0.85:
        selection_quality = "good"
    elif relative_performance >= 0.70:
        selection_quality = "fair"
    else:
        selection_quality = "poor"
    
    # Calculate ranking
    sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    predicted_rank = next(i for i, (model, _) in enumerate(sorted_models, 1) if model == predicted_model)
    
    return {
        "prediction_correct": prediction_correct,
        "predicted_model": predicted_model,
        "actual_best_model": actual_best,
        "predicted_score": predicted_score,
        "best_score": best_score,
        "score_difference": score_difference,
        "relative_performance": relative_performance,
        "selection_quality": selection_quality,
        "predicted_rank": predicted_rank,
        "total_models": len(scores),
        "all_scores": scores,
        "model_rankings": sorted_models
    }

def batch_evaluate_selections(prompts: List[str], predictions: List[str], all_responses: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Evaluate selection accuracy across multiple prompts in batch
    """
    if len(prompts) != len(predictions) or len(prompts) != len(all_responses):
        raise ValueError("Prompts, predictions, and responses lists must have the same length")
    
    evaluations = []
    correct_predictions = 0
    total_score_difference = 0
    total_relative_performance = 0
    
    quality_counts = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
    model_accuracy = {}
    
    for prompt, predicted, responses in zip(prompts, predictions, all_responses):
        eval_result = evaluate_selection_accuracy(prompt, predicted, responses)
        evaluations.append(eval_result)
        
        if eval_result["prediction_correct"]:
            correct_predictions += 1
        
        total_score_difference += eval_result["score_difference"]
        total_relative_performance += eval_result["relative_performance"]
        quality_counts[eval_result["selection_quality"]] += 1
        
        # Track per-model accuracy
        if predicted not in model_accuracy:
            model_accuracy[predicted] = {"correct": 0, "total": 0}
        model_accuracy[predicted]["total"] += 1
        if eval_result["prediction_correct"]:
            model_accuracy[predicted]["correct"] += 1
    
    # Calculate aggregate metrics
    total_prompts = len(prompts)
    accuracy_rate = correct_predictions / total_prompts
    avg_score_difference = total_score_difference / total_prompts
    avg_relative_performance = total_relative_performance / total_prompts
    
    # Calculate per-model accuracy rates
    for model in model_accuracy:
        model_accuracy[model]["accuracy_rate"] = model_accuracy[model]["correct"] / model_accuracy[model]["total"]
    
    return {
        "total_prompts": total_prompts,
        "correct_predictions": correct_predictions,
        "accuracy_rate": accuracy_rate,
        "avg_score_difference": avg_score_difference,
        "avg_relative_performance": avg_relative_performance,
        "quality_distribution": quality_counts,
        "model_accuracy": model_accuracy,
        "individual_evaluations": evaluations,
        "summary": {
            "overall_performance": "excellent" if accuracy_rate >= 0.8 else "good" if accuracy_rate >= 0.6 else "fair" if accuracy_rate >= 0.4 else "poor",
            "avg_quality_score": avg_relative_performance,
            "selection_efficiency": f"{accuracy_rate:.1%} accurate predictions"
        }
    }

def compare_model_performance(all_responses: Dict[str, Dict[str, str]], prompts: List[str]) -> Dict[str, Any]:
    """
    Compare performance across all models for multiple prompts
    """
    model_stats = {}
    
    # Initialize model statistics
    for model in ["deepseek-r1", "mistral-small", "qwen-2.5", "gemini-flash", "llama-3.3"]:
        model_stats[model] = {
            "total_score": 0,
            "wins": 0,
            "avg_score": 0,
            "best_categories": [],
            "worst_categories": []
        }
    
    category_performance = {}
    
    for i, prompt in enumerate(prompts):
        prompt_responses = {model: all_responses[model][i] for model in all_responses.keys()}
        scores = {model: score_response(prompt, response) for model, response in prompt_responses.items()}
        
        # Find winner for this prompt
        winner = max(scores, key=scores.get)
        model_stats[winner]["wins"] += 1
        
        # Add to total scores
        for model, score in scores.items():
            if model in model_stats:
                model_stats[model]["total_score"] += score
        
        # Categorize prompt
        prompt_lower = prompt.lower()
        if any(kw in prompt_lower for kw in ['code', 'program', 'function', 'algorithm']):
            category = "coding"
        elif any(kw in prompt_lower for kw in ['analyze', 'reasoning', 'logic', 'step by step']):
            category = "reasoning"
        elif any(kw in prompt_lower for kw in ['write', 'story', 'poem', 'creative']):
            category = "creative"
        elif any(kw in prompt_lower for kw in ['what is', 'who is', 'when', 'where']):
            category = "factual"
        elif any(kw in prompt_lower for kw in ['describe', 'image', 'picture', 'visual']):
            category = "visual"
        else:
            category = "general"
        
        if category not in category_performance:
            category_performance[category] = {}
        
        for model, score in scores.items():
            if model not in category_performance[category]:
                category_performance[category][model] = []
            category_performance[category][model].append(score)
    
    # Calculate averages
    for model in model_stats:
        model_stats[model]["avg_score"] = model_stats[model]["total_score"] / len(prompts)
    
    # Find best categories for each model
    for category, models in category_performance.items():
        category_winners = sorted(models.items(), key=lambda x: sum(x[1])/len(x[1]), reverse=True)
        if category_winners:
            best_model = category_winners[0][0]
            if best_model in model_stats:
                model_stats[best_model]["best_categories"].append(category)
    
    return {
        "model_statistics": model_stats,
        "category_performance": category_performance,
        "overall_ranking": sorted(model_stats.items(), key=lambda x: x[1]["avg_score"], reverse=True),
        "win_distribution": {model: stats["wins"] for model, stats in model_stats.items()},
        "recommendation": max(model_stats.items(), key=lambda x: x[1]["avg_score"])[0]
    }

def generate_evaluation_report(evaluation_results: Dict[str, Any]) -> str:
    """
    Generate a human-readable evaluation report
    """
    report = []
    report.append("=== LLM SELECTION EVALUATION REPORT ===")
    report.append(f"Total Prompts Evaluated: {evaluation_results['total_prompts']}")
    report.append(f"Correct Predictions: {evaluation_results['correct_predictions']}")
    report.append(f"Accuracy Rate: {evaluation_results['accuracy_rate']:.1%}")
    report.append(f"Average Relative Performance: {evaluation_results['avg_relative_performance']:.2f}")
    report.append("")
    
    report.append("Quality Distribution:")
    for quality, count in evaluation_results['quality_distribution'].items():
        percentage = (count / evaluation_results['total_prompts']) * 100
        report.append(f"  {quality.capitalize()}: {count} ({percentage:.1f}%)")
    report.append("")
    
    report.append("Per-Model Accuracy:")
    for model, stats in evaluation_results['model_accuracy'].items():
        report.append(f"  {model}: {stats['correct']}/{stats['total']} ({stats['accuracy_rate']:.1%})")
    
    report.append("")
    report.append(f"Overall Performance: {evaluation_results['summary']['overall_performance'].upper()}")
    
    return "\n".join(report)