import re
from typing import Dict, Any

class LLMSelector:
    """
    Dynamically selects the best LLM based on the input prompt characteristics
    """
    
    def __init__(self):
        # Define rules for model selection based on prompt analysis
        self.selection_rules = [
            # Mathematical/Coding tasks - DeepSeek R1 is better for reasoning
            {
                "keywords": ["calculate", "solve", "equation", "math", "code", "program", 
                           "algorithm", "function", "debug", "error", "programming"],
                "patterns": [r'\d+[\+\-\*/]\d+', r'def\s+\w+', r'class\s+\w+', r'import\s+\w+'],
                "model": "deepseek-r1",
                "confidence": 0.9
            },
            
            # Reasoning/Logic tasks - DeepSeek R1 excels at step-by-step reasoning
            {
                "keywords": ["analyze", "reasoning", "logic", "step by step", "explain why",
                           "compare", "evaluate", "assess", "conclude", "infer"],
                "patterns": [r'why\s+(?:does|is|do)', r'how\s+(?:does|do|can)', r'what\s+if'],
                "model": "deepseek-r1",
                "confidence": 0.8
            },
            
            # Creative/Language tasks - Mistral is good for creative content
            {
                "keywords": ["write", "create", "story", "poem", "creative", "imagine",
                           "describe", "summarize", "translate", "rewrite"],
                "patterns": [r'write\s+a\s+', r'create\s+a\s+', r'tell\s+me\s+about'],
                "model": "mistral-small",
                "confidence": 0.8
            },
            
            # General knowledge/Factual questions - Mistral for quick factual responses
            {
                "keywords": ["what is", "who is", "when", "where", "capital", "definition",
                           "meaning", "fact", "information", "tell me"],
                "patterns": [r'what\s+is\s+the\s+', r'who\s+is\s+', r'when\s+did\s+'],
                "model": "mistral-small",
                "confidence": 0.7
            }
        ]
    
    def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Analyze the prompt to determine its characteristics
        """
        prompt_lower = prompt.lower()
        
        analysis = {
            "length": len(prompt.split()),
            "has_numbers": bool(re.search(r'\d+', prompt)),
            "has_code": bool(re.search(r'[{}();]|def\s|class\s|import\s', prompt)),
            "question_words": len(re.findall(r'\b(?:what|how|why|when|where|who)\b', prompt_lower)),
            "complexity_indicators": len(re.findall(r'\b(?:complex|difficult|analyze|compare|evaluate)\b', prompt_lower))
        }
        
        return analysis
    
    def calculate_model_scores(self, prompt: str) -> Dict[str, float]:
        """
        Calculate confidence scores for each model based on the prompt
        """
        prompt_lower = prompt.lower()
        scores = {"deepseek-r1": 0.0, "mistral-small": 0.0}
        
        for rule in self.selection_rules:
            # Check keyword matches
            keyword_matches = sum(1 for keyword in rule["keywords"] if keyword in prompt_lower)
            keyword_score = (keyword_matches / len(rule["keywords"])) * rule["confidence"]
            
            # Check pattern matches
            pattern_matches = sum(1 for pattern in rule["patterns"] if re.search(pattern, prompt_lower))
            pattern_score = (pattern_matches / max(len(rule["patterns"]), 1)) * rule["confidence"] * 0.5
            
            # Add to model score
            total_score = keyword_score + pattern_score
            scores[rule["model"]] += total_score
        
        # Add base scores and prompt analysis adjustments
        analysis = self.analyze_prompt(prompt)
        
        # DeepSeek R1 bonuses
        if analysis["has_code"] or analysis["has_numbers"]:
            scores["deepseek-r1"] += 0.3
        if analysis["complexity_indicators"] > 0:
            scores["deepseek-r1"] += 0.2
        if analysis["length"] > 20:  # Longer prompts might need more reasoning
            scores["deepseek-r1"] += 0.1
            
        # Mistral bonuses
        if analysis["question_words"] > 0 and analysis["length"] < 15:  # Short factual questions
            scores["mistral-small"] += 0.3
        
        # Normalize scores to ensure they're between 0 and 1
        max_score = max(scores.values()) if max(scores.values()) > 0 else 1
        scores = {model: score / max_score for model, score in scores.items()}
        
        # Ensure minimum scores (fallback logic)
        if all(score < 0.1 for score in scores.values()):
            # Default to mistral for simple queries, deepseek for complex ones
            if analysis["length"] < 10 and analysis["question_words"] > 0:
                scores["mistral-small"] = 0.6
                scores["deepseek-r1"] = 0.4
            else:
                scores["deepseek-r1"] = 0.6
                scores["mistral-small"] = 0.4
        
        return scores
    
    def select_best_model(self, prompt: str) -> Dict[str, Any]:
        """
        Select the best model for the given prompt
        """
        scores = self.calculate_model_scores(prompt)
        best_model = max(scores, key=scores.get)
        
        return {
            "selected_model": best_model,
            "confidence": scores[best_model],
            "all_scores": scores,
            "reasoning": self._get_selection_reasoning(prompt, best_model, scores)
        }
    
    def _get_selection_reasoning(self, prompt: str, selected_model: str, scores: Dict[str, float]) -> str:
        """
        Provide reasoning for the model selection
        """
        analysis = self.analyze_prompt(prompt)
        
        reasons = []
        
        if analysis["has_code"]:
            reasons.append("contains code")
        if analysis["has_numbers"]:
            reasons.append("involves numbers/calculations")
        if analysis["complexity_indicators"] > 0:
            reasons.append("requires complex analysis")
        if analysis["question_words"] > 0 and analysis["length"] < 15:
            reasons.append("simple factual question")
        if analysis["length"] > 20:
            reasons.append("lengthy/detailed prompt")
            
        reasoning = f"Selected {selected_model} (confidence: {scores[selected_model]:.2f})"
        if reasons:
            reasoning += f" because prompt {', '.join(reasons)}"
        
        return reasoning
