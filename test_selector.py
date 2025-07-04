from llm_selector import LLMSelector
import os

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not set. Check your .env file and environment variables.")

    
def test_selection_logic():
    """Test the LLM selection logic with various prompts for all 5 models"""
    
    selector = LLMSelector()
    
    test_prompts = [
        # Mathematical/Coding (should prefer deepseek-r1)
        "Write a Python function to calculate the factorial of a number",
        "Solve this equation: 2x + 5 = 13",
        "Debug this code: def add(a, b): return a - b",
        "Implement a binary search algorithm in Python",
        
        # Complex Reasoning/Logic (should prefer deepseek-r1)
        "Analyze the pros and cons of renewable energy step by step",
        "Explain why the sky is blue using scientific reasoning",
        "Compare and evaluate different sorting algorithms",
        "What if we could travel faster than light? Analyze the implications",
        
        # Creative Writing/Storytelling (should prefer llama-3.3)
        "Write a short story about a lonely robot",
        "Create a poem about autumn leaves",
        "Tell me a story about a magical forest",
        "Write a dialogue between two characters meeting for the first time",
        
        # Language Tasks/Translation (should prefer mistral-small)
        "Translate this to Spanish: Hello, how are you?",
        "Rewrite this sentence to be more formal",
        "Summarize the key points of machine learning",
        "Explain how neural networks work",
        
        # Visual/Multimodal (should prefer qwen-2.5)
        "Describe this image in detail",
        "What do you see in this picture?",
        "Analyze this diagram and explain its components",
        "Can you identify the objects in this photo?",
        
        # Quick Facts/Information (should prefer gemini-flash)
        "What is the capital of India?",
        "Who is the current president of the United States?",
        "When did World War II end?",
        "Where is the Great Wall of China located?",
        
        # General Conversation/Chat (should prefer llama-3.3)
        "Hello, how are you today?",
        "What do you think about climate change?",
        "I need advice on choosing a career",
        "Can you help me plan my weekend?",
        
        # Technical Documentation (should prefer mistral-small)
        "How to install Python on Windows?",
        "Explain the steps to deploy a web application",
        "What are the best practices for database design?",
        "Create a tutorial for beginners in programming",
        
        # Mixed/Ambiguous
        "How does quantum computing work?",
        "Tell me about artificial intelligence",
        "What are the benefits of exercise?",
        "Explain blockchain technology",
    ]
    
    print("Testing Enhanced LLM Selection Logic (5 Models)")
    print("=" * 60)
    
    # Track model selection counts
    model_counts = {
        "deepseek-r1": 0,
        "mistral-small": 0,
        "qwen-2.5": 0,
        "gemini-flash": 0,
        "llama-3.3": 0
    }
    
    for i, prompt in enumerate(test_prompts, 1):
        result = selector.select_best_model(prompt)
        model_counts[result['selected_model']] += 1
        
        print(f"\n{i}. Prompt: {prompt}")
        print(f"   Selected: {result['selected_model']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Reasoning: {result['reasoning']}")
        
        # Show top 2 scores for comparison
        sorted_scores = sorted(result['all_scores'].items(), key=lambda x: x[1], reverse=True)
        print(f"   Top scores: {sorted_scores[0][0]}({sorted_scores[0][1]:.2f}), {sorted_scores[1][0]}({sorted_scores[1][1]:.2f})")
    
    print("\n" + "=" * 60)
    print("MODEL SELECTION SUMMARY:")
    print("=" * 60)
    for model, count in model_counts.items():
        percentage = (count / len(test_prompts)) * 100
        print(f"{model}: {count} selections ({percentage:.1f}%)")
    
    print(f"\nTotal prompts tested: {len(test_prompts)}")

def test_specific_scenarios():
    """Test specific edge cases and scenarios"""
    
    selector = LLMSelector()
    
    edge_cases = [
        # Very short prompts
        "Hi",
        "Help",
        "What?",
        
        # Very long prompts
        "I need you to help me understand the complex relationship between quantum mechanics and general relativity, specifically focusing on how these two fundamental theories of physics interact at the Planck scale, and what implications this might have for our understanding of the universe, including potential applications in quantum computing, space travel, and the nature of reality itself.",
        
        # Mixed content
        "Write code to calculate 2+2 and also create a story about it",
        "Translate this math problem: What is 5 * 6?",
        "Show me a picture of a cat and write a poem about it",
        
        # Ambiguous prompts
        "Process this",
        "Make it better",
        "Fix this issue",
    ]
    
    print("\n" + "=" * 60)
    print("TESTING EDGE CASES AND SCENARIOS:")
    print("=" * 60)
    
    for i, prompt in enumerate(edge_cases, 1):
        result = selector.select_best_model(prompt)
        print(f"\n{i}. Prompt: '{prompt}'")
        print(f"   Selected: {result['selected_model']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Reasoning: {result['reasoning']}")

if __name__ == "__main__":
    test_selection_logic()
    test_specific_scenarios()