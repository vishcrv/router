from llm_selector import LLMSelector

def test_selection_logic():
    """Test the LLM selection logic with various prompts"""
    
    selector = LLMSelector()
    
    test_prompts = [
        # Mathematical/Coding (should prefer deepseek-r1)
        "Write a Python function to calculate the factorial of a number",
        "Solve this equation: 2x + 5 = 13",
        "Debug this code: def add(a, b): return a - b",
        
        # Reasoning/Logic (should prefer deepseek-r1)
        "Analyze the pros and cons of renewable energy",
        "Explain step by step why the sky is blue",
        "Compare and evaluate different sorting algorithms",
        
        # Creative/Language (should prefer mistral-small)
        "Write a short story about a lonely robot",
        "Create a poem about autumn leaves",
        "Describe a beautiful sunset in vivid detail",
        
        # General Knowledge (should prefer mistral-small)
        "What is the capital of India?",
        "Who is the current president of the United States?",
        "When did World War II end?",
        
        # Mixed/Ambiguous
        "How does machine learning work?",
        "Tell me about quantum computing",
    ]
    
    print("Testing LLM Selection Logic")
    print("=" * 50)
    
    for prompt in test_prompts:
        result = selector.select_best_model(prompt)
        print(f"\nPrompt: {prompt}")
        print(f"Selected: {result['selected_model']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Reasoning: {result['reasoning']}")
        print(f"Scores: {result['all_scores']}")

if __name__ == "__main__":
    test_selection_logic()
