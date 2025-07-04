import re
from typing import Dict, Any
import os

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not set. Check your .env file and environment variables.")

    
class LLMSelector:
    """
    Dynamically selects the best LLM based on the input prompt characteristics
    Now supports 5 models: DeepSeek R1, Mistral Small, Qwen2.5, Gemini Flash, Llama 3.3
    """
    
    def __init__(self):
        # Define rules for model selection based on prompt analysis
        self.selection_rules = [
            # Mathematical/Coding tasks - DeepSeek R1 is best for complex reasoning
            {
                "keywords": ["calculate", "solve", "equation", "math", "mathematics", "code", "program", 
                           "algorithm", "function", "debug", "error", "programming", "implement", "coding"],
                "patterns": [r'\d+[\+\-\*/]\d+', r'def\s+\w+', r'class\s+\w+', r'import\s+\w+', r'```'],
                "model": "deepseek-r1",
                "confidence": 0.9
            },
            
            # Complex Reasoning/Logic tasks - DeepSeek R1 excels at step-by-step reasoning
            {
                "keywords": ["analyze", "reasoning", "logic", "step by step", "explain why", "complex",
                           "compare", "evaluate", "assess", "conclude", "infer", "deduce", "prove"],
                "patterns": [r'why\s+(?:does|is|do)', r'how\s+(?:does|do|can)', r'what\s+if', r'step\s+by\s+step'],
                "model": "deepseek-r1",
                "confidence": 0.85
            },
            
            # Creative Writing/Storytelling - Llama 3.3 is excellent for creative tasks
            {
                "keywords": ["write", "create", "story", "poem", "creative", "imagine", "narrative",
                           "fiction", "character", "plot", "dialogue", "scene", "chapter"],
                "patterns": [r'write\s+a\s+story', r'create\s+a\s+poem', r'tell\s+a\s+story', r'once\s+upon'],
                "model": "llama-3.3",
                "confidence": 0.85
            },
            
            # Language Tasks/Translation/Rewriting - Mistral Small is good for language processing
            {
                "keywords": ["translate", "rewrite", "paraphrase", "summarize", "grammar", "language",
                           "correct", "edit", "improve", "rephrase", "simplify"],
                "patterns": [r'translate\s+to', r'rewrite\s+this', r'summarize\s+this', r'correct\s+the'],
                "model": "mistral-small",
                "confidence": 0.8
            },
            
            # Visual/Multimodal tasks - Qwen2.5 VL is designed for vision-language tasks
            {
                "keywords": ["image", "picture", "visual", "describe", "see", "photo", "diagram",
                           "chart", "graph", "illustration", "drawing", "artwork"],
                "patterns": [r'describe\s+(?:this|the)\s+image', r'what\s+(?:do\s+you\s+)?see', r'analyze\s+(?:this|the)\s+picture'],
                "model": "qwen-2.5",
                "confidence": 0.8
            },
            
            # Quick Facts/Information - Gemini Flash is fast for factual queries
            {
                "keywords": ["what is", "who is", "when", "where", "capital", "definition", "fact",
                           "meaning", "information", "tell me", "quick", "fast"],
                "patterns": [r'what\s+is\s+the\s+', r'who\s+is\s+', r'when\s+did\s+', r'where\s+is'],
                "model": "gemini-flash",
                "confidence": 0.75
            },
            
            # General Conversation/Chat - Llama 3.3 is good for conversational tasks
            {
                "keywords": ["hello", "hi", "chat", "talk", "conversation", "discuss", "opinion",
                           "think", "feel", "advice", "help", "recommend"],
                "patterns": [r'what\s+do\s+you\s+think', r'can\s+you\s+help', r'i\s+need\s+advice'],
                "model": "llama-3.3",
                "confidence": 0.7
            },
            
            # Technical Documentation/Explanation - Mistral Small for technical explanations
            {
                "keywords": ["explain", "documentation", "technical", "how to", "tutorial", "guide",
                           "manual", "instructions", "procedure", "process", "method"],
                "patterns": [r'how\s+to\s+', r'explain\s+how', r'what\s+are\s+the\s+steps'],
                "model": "mistral-small",
                "confidence": 0.75
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
            "has_code": bool(re.search(r'[{}();]|def\s|class\s|import\s|```', prompt)),
            "question_words": len(re.findall(r'\b(?:what|how|why|when|where|who)\b', prompt_lower)),
            "complexity_indicators": len(re.findall(r'\b(?:complex|difficult|analyze|compare|evaluate|step by step)\b', prompt_lower)),
            "creative_indicators": len(re.findall(r'\b(?:creative|story|poem|imagine|write|create)\b', prompt_lower)),
            "visual_indicators": len(re.findall(r'\b(?:image|picture|visual|photo|see|describe)\b', prompt_lower)),
            "factual_indicators": len(re.findall(r'\b(?:what is|who is|when|where|capital|definition)\b', prompt_lower)),
            "conversational_indicators": len(re.findall(r'\b(?:hello|hi|chat|opinion|think|advice|help)\b', prompt_lower))
        }
        
        return analysis
    
    def calculate_model_scores(self, prompt: str) -> Dict[str, float]:
        """
        Calculate confidence scores for each model based on the prompt
        """
        prompt_lower = prompt.lower()
        scores = {
            "deepseek-r1": 0.0,
            "mistral-small": 0.0,
            "qwen-2.5": 0.0,
            "gemini-flash": 0.0,
            "llama-3.3": 0.0
        }
        
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
        
        # DeepSeek R1 bonuses - Best for complex reasoning and coding
        if analysis["has_code"] or analysis["has_numbers"]:
            scores["deepseek-r1"] += 0.3
        if analysis["complexity_indicators"] > 0:
            scores["deepseek-r1"] += 0.25
        if analysis["length"] > 25:  # Very long prompts might need deep reasoning
            scores["deepseek-r1"] += 0.15
            
        # Mistral Small bonuses - Good for language processing and explanations
        if analysis["length"] > 15 and analysis["complexity_indicators"] == 0:  # Detailed but not complex
            scores["mistral-small"] += 0.2
        if "explain" in prompt_lower or "how to" in prompt_lower:
            scores["mistral-small"] += 0.25
            
        # Qwen2.5 bonuses - Excellent for visual and multimodal tasks
        if analysis["visual_indicators"] > 0:
            scores["qwen-2.5"] += 0.4
        if "describe" in prompt_lower and analysis["visual_indicators"] > 0:
            scores["qwen-2.5"] += 0.3
            
        # Gemini Flash bonuses - Fast for quick factual queries
        if analysis["factual_indicators"] > 0 and analysis["length"] < 15:
            scores["gemini-flash"] += 0.35
        if analysis["question_words"] > 0 and analysis["length"] < 10:
            scores["gemini-flash"] += 0.25
            
        # Llama 3.3 bonuses - Great for creative and conversational tasks
        if analysis["creative_indicators"] > 0:
            scores["llama-3.3"] += 0.35
        if analysis["conversational_indicators"] > 0:
            scores["llama-3.3"] += 0.25
        if analysis["length"] > 20 and analysis["creative_indicators"] > 0:  # Long creative prompts
            scores["llama-3.3"] += 0.2
        
        # Normalize scores to ensure they're between 0 and 1
        max_score = max(scores.values()) if max(scores.values()) > 0 else 1
        scores = {model: score / max_score for model, score in scores.items()}
        
        # Enhanced fallback logic for 5 models
        if all(score < 0.1 for score in scores.values()):
            analysis = self.analyze_prompt(prompt)
            
            if analysis["length"] < 8 and analysis["question_words"] > 0:
                # Short factual questions → Gemini Flash
                scores["gemini-flash"] = 0.7
                scores["mistral-small"] = 0.3
            elif analysis["creative_indicators"] > 0 or "write" in prompt_lower:
                # Creative tasks → Llama 3.3
                scores["llama-3.3"] = 0.7
                scores["mistral-small"] = 0.3
            elif analysis["has_code"] or analysis["has_numbers"] or analysis["complexity_indicators"] > 0:
                # Technical/complex tasks → DeepSeek R1
                scores["deepseek-r1"] = 0.7
                scores["mistral-small"] = 0.3
            elif analysis["visual_indicators"] > 0:
                # Visual tasks → Qwen2.5
                scores["qwen-2.5"] = 0.7
                scores["mistral-small"] = 0.3
            else:
                # Default to Mistral Small for general tasks
                scores["mistral-small"] = 0.6
                scores["llama-3.3"] = 0.4
        
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
        
        # Model-specific reasoning
        if selected_model == "deepseek-r1":
            if analysis["has_code"]:
                reasons.append("contains code")
            if analysis["has_numbers"]:
                reasons.append("involves calculations")
            if analysis["complexity_indicators"] > 0:
                reasons.append("requires complex reasoning")
            if analysis["length"] > 25:
                reasons.append("lengthy/complex prompt")
                
        elif selected_model == "mistral-small":
            if "explain" in prompt.lower():
                reasons.append("requires explanation")
            if analysis["length"] > 15:
                reasons.append("detailed prompt")
            reasons.append("good for language processing")
            
        elif selected_model == "qwen-2.5":
            if analysis["visual_indicators"] > 0:
                reasons.append("involves visual content")
            reasons.append("specialized for vision-language tasks")
            
        elif selected_model == "gemini-flash":
            if analysis["factual_indicators"] > 0:
                reasons.append("factual query")
            if analysis["length"] < 15:
                reasons.append("quick question")
            reasons.append("optimized for fast responses")
            
        elif selected_model == "llama-3.3":
            if analysis["creative_indicators"] > 0:
                reasons.append("creative task")
            if analysis["conversational_indicators"] > 0:
                reasons.append("conversational")
            reasons.append("excellent for creative/chat tasks")
        
        reasoning = f"Selected {selected_model} (confidence: {scores[selected_model]:.2f})"
        if reasons:
            reasoning += f" because prompt {', '.join(reasons)}"
        
        return reasoning