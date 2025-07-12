import re
from typing import Dict, Any
from dotenv import load_dotenv
import os

load_dotenv()
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
            {
                "keywords": ["algorithm", "complexity", "recursion", "dynamic programming", "data structure", 
                            "optimization", "computational", "asymptotic", "big o", "time complexity", 
                            "space complexity", "leetcode", "competitive programming", "proof by induction"],
                "patterns": [r'\b(?:solve|implement|optimize)\s+(?:algorithm|problem)\b', 
                            r'\bO\([nlogmk\d\^\+\*\s]+\)', r'\brecursive\s+(?:function|solution)\b',
                            r'\bdynamic\s+programming\b', r'\bgreedy\s+algorithm\b',
                            r'\b(?:dfs|bfs|dijkstra|bellman|floyd)\b', r'\bbacktrack(?:ing)?\b'],
                "model": "deepseek-r1",
                "confidence": 0.95
            },
            {
                "keywords": ["script", "code", "program", "function", "method", "class", "module", 
                            "programming", "coding", "software", "implementation", "python", "javascript",
                            "java", "c++", "html", "css", "sql", "bash", "shell", "powershell",
                            "hello world", "print", "console", "output", "return", "variable"],
                "patterns": [r'\bwrite\s+(?:a|an|some)?\s*(?:script|code|program|function)\b',
                            r'\bcreate\s+(?:a|an|some)?\s*(?:script|code|program|function)\b',
                            r'\bimplement\s+(?:a|an|some)?\s*(?:script|code|program|function)\b',
                            r'\bhello\s+world\b', r'\bprint\s*\(', r'\bconsole\.log\b',
                            r'\breturn\s+["\'].*["\']', r'\bdef\s+\w+\b', r'\bfunction\s+\w+\b'],
                "model": "deepseek-r1",
                "confidence": 0.92
            },
            {
                "keywords": ["mathematical proof", "theorem", "lemma", "corollary", "axiom", "derive", 
                            "mathematical induction", "contradiction", "contrapositive", "formal logic",
                            "propositional logic", "predicate logic", "set theory", "topology"],
                "patterns": [r'\bprove\s+that\b', r'\bshow\s+that\b', r'\bdemonstrate\s+that\b',
                            r'\bQ\.E\.D\b', r'\btherefore\b.*\bthus\b', r'\bassume\s+(?:for\s+)?contradiction\b',
                            r'\bby\s+(?:mathematical\s+)?induction\b', r'\biff\b|\bif\s+and\s+only\s+if\b'],
                "model": "deepseek-r1",
                "confidence": 0.94
            },
            {
                "keywords": ["debug", "error", "exception", "traceback", "stack trace", "segmentation fault",
                            "memory leak", "null pointer", "undefined behavior", "compilation error",
                            "runtime error", "syntax error", "logical error"],
                "patterns": [r'(?:error|exception|fault).*(?:line|at|in)\s+\d+', 
                            r'(?:traceback|stack\s+trace)', r'\bsegfault\b',
                            r'(?:compile|compilation)\s+(?:error|failed)', r'\bnull\s+pointer\b',
                            r'(?:memory|buffer)\s+(?:leak|overflow)', r'\bundefined\s+(?:reference|behavior)\b'],
                "model": "deepseek-r1",
                "confidence": 0.93
            },
            {
                "keywords": ["logical reasoning", "deduction", "inference", "syllogism", "premise", 
                            "conclusion", "logical fallacy", "critical thinking", "cause and effect",
                            "correlation", "causation", "hypothesis", "scientific method"],
                "patterns": [r'\bif\s+.*\bthen\s+.*\btherefore\b', r'\bgiven\s+that\b.*\bwe\s+can\s+(?:conclude|infer)\b',
                            r'\bpremise\s+\d+', r'\bsyllogism\b', r'\blogical\s+fallacy\b',
                            r'\bcause\s+and\s+effect\b', r'\bcorrelation\s+(?:does\s+not\s+imply|vs)\s+causation\b'],
                "model": "deepseek-r1",
                "confidence": 0.91
            },
            {
                "keywords": ["creative writing", "fiction", "narrative", "character development", "plot", 
                            "dialogue", "screenplay", "novel", "short story", "poetry", "prose",
                            "literary device", "metaphor", "allegory", "symbolism", "world building"],
                "patterns": [r'\bwrite\s+(?:a|an)\s+(?:story|novel|screenplay|poem|book|article|essay)\b',
                            r'\bonce\s+upon\s+a\s+time\b', r'\bcharacter\s+development\b',
                            r'\bplot\s+(?:twist|device|structure)\b', r'\bworld\s+building\b',
                            r'\bliterary\s+device\b', r'\bfirst\s+person\s+narrative\b'],
                "model": "llama-3.3",
                "confidence": 0.92
            },
            {
                "keywords": ["language translation", "localization", "linguistics", "grammar", "syntax", 
                            "morphology", "phonetics", "semantics", "pragmatics", "etymology",
                            "multilingual", "bilingual", "polyglot"],
                "patterns": [r'\btranslate\s+(?:from|to)\s+\w+', r'\bgrammar\s+(?:rules|check)\b',
                            r'\bsyntax\s+(?:error|structure)\b', r'\blinguistic\s+analysis\b',
                            r'\betymology\s+of\b', r'\bphonetic\s+transcription\b'],
                "model": "mistral-small",
                "confidence": 0.89
            },
            {
                "keywords": ["technical documentation", "api documentation", "user manual", "instruction manual",
                            "system architecture", "software documentation", "code documentation",
                            "technical writing", "specification", "requirements"],
                "patterns": [r'\bdocument(?:ation)?\s+(?:how|the)\b', r'\bapi\s+(?:reference|docs)\b',
                            r'\buser\s+(?:manual|guide)\b', r'\btechnical\s+specification\b',
                            r'\bsystem\s+(?:architecture|design)\b', r'\brequirements\s+document\b'],
                "model": "mistral-small",
                "confidence": 0.87
            },
            {
                "keywords": ["computer vision", "image analysis", "object detection", "pattern recognition",
                            "machine learning", "deep learning", "neural network", "convolutional",
                            "image processing", "feature extraction", "classification"],
                "patterns": [r'\banalyze\s+(?:this|the)\s+image\b', r'\bobject\s+detection\b',
                            r'\bcomputer\s+vision\b', r'\bimage\s+(?:processing|analysis|recognition)\b',
                            r'\bfeature\s+extraction\b', r'\bpattern\s+recognition\b',
                            r'\bconvolutional\s+neural\s+network\b'],
                "model": "qwen-2.5",
                "confidence": 0.93
            },
            {
                "keywords": ["satellite imagery", "remote sensing", "geospatial", "gis", "cartography",
                            "topography", "aerial photography", "landsat", "modis", "sentinel"],
                "patterns": [r'\bsatellite\s+(?:image|imagery)\b', r'\bremote\s+sensing\b',
                            r'\bgeospatial\s+analysis\b', r'\bgis\s+(?:data|analysis)\b',
                            r'\b(?:landsat|modis|sentinel)\b', r'\baerial\s+photography\b'],
                "model": "qwen-2.5",
                "confidence": 0.90
            },
            {
                "keywords": ["encyclopedia", "factual", "definition", "statistics", "demographics",
                            "geographical", "historical", "scientific fact", "data lookup"],
                "patterns": [r'\bwhat\s+is\s+the\s+(?:capital|population|area|height)\s+of\b',
                            r'\bwho\s+(?:is|was)\s+(?:the\s+)?(?:president|king|queen|emperor)\b',
                            r'\bwhen\s+(?:did|was)\s+.*\b(?:born|died|founded|established)\b',
                            r'\bdefine\s+\w+\b', r'\bstatistics\s+(?:for|of|about)\b'],
                "model": "gemini-flash",
                "confidence": 0.85
            },
            {
                "keywords": ["conversational", "chat", "advice", "opinion", "personal", "emotional support",
                            "relationship", "life advice", "motivation", "inspiration"],
                "patterns": [r'\b(?:hello|hi|hey)\b.*\bhow\s+are\s+you\b',
                            r'\bcan\s+you\s+help\s+me\s+with\s+(?:my|a)\s+(?:problem|situation)\b',
                            r'\bi\s+(?:need|want)\s+(?:advice|help|support)\b',
                            r'\bwhat\s+do\s+you\s+think\s+about\b', r'\blet\'?s\s+(?:talk|chat)\b'],
                "model": "llama-3.3",
                "confidence": 0.78
            },
            {
                "keywords": ["quick question", "simple query", "basic help", "general assistance"],
                "patterns": [r'\bquick\s+question\b', r'\bsimple\s+(?:question|query)\b',
                            r'\bcan\s+you\s+(?:quickly|briefly)\b', r'\bjust\s+wondering\b'],
                "model": "gemini-flash",
                "confidence": 0.75
            }
        ]
    
    def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Analyze the prompt to determine its characteristics
        """
        prompt_lower = prompt.lower()
        
        # Enhanced code detection - now includes more coding-related terms
        code_patterns = [
            r'[{}();]|def\s|class\s|import\s|```',  # Original pattern
            r'\bscript\b|\bcode\b|\bprogram\b|\bfunction\b|\bmethod\b',  # Programming terms
            r'\bhello\s+world\b|\bprint\s*\(|\breturn\s+["\']',  # Common coding examples
            r'\bpython\b|\bjavascript\b|\bjava\b|\bc\+\+\b|\bhtml\b|\bcss\b|\bsql\b',  # Languages
            r'\bwrite\s+(?:a|an|some)?\s*(?:script|code|program|function)\b',  # Write code requests
            r'\bcreate\s+(?:a|an|some)?\s*(?:script|code|program|function)\b'  # Create code requests
        ]
        
        has_code = any(re.search(pattern, prompt_lower) for pattern in code_patterns)
        
        analysis = {
            "length": len(prompt.split()),
            "has_numbers": bool(re.search(r'\d+', prompt)),
            "has_code": has_code,
            "question_words": len(re.findall(r'\b(?:what|how|why|when|where|who)\b', prompt_lower)),
            "complexity_indicators": len(re.findall(r'\b(?:complex|difficult|analyze|compare|evaluate|step by step)\b', prompt_lower)),
            "creative_indicators": len(re.findall(r'\b(?:creative|story|poem|imagine|novel|essay|article)\b', prompt_lower)),  # Removed generic "write" and "create"
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
            elif analysis["creative_indicators"] > 0 or ("write" in prompt_lower and not analysis["has_code"]):
                # Creative tasks → Llama 3.3 (but not if it's code writing)
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
                reasons.append("contains code/programming request")
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