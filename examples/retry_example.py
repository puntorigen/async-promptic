#!/usr/bin/env python
"""
Example demonstrating the retry functionality in Promptic.

This script shows how to use the retry parameter with the @llm decorator
to handle common LLM API errors gracefully with automatic retries.
"""

import asyncio
import logging
import time
from typing import Dict, Any
from unittest.mock import patch

from async_promptic import Promptic, llm
from litellm.exceptions import RateLimitError, InternalServerError, APIError, Timeout

# Setup logging to see retry attempts
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("retry_example")

# Create a class to track call attempts for testing
class CallCounter:
    def __init__(self):
        self.sync_calls = 0
        self.async_calls = 0

call_counter = CallCounter()

# Create mock functions to simulate API failures and track call attempts
def mock_completion(**kwargs):
    """Mock that fails the first 2 times, then succeeds"""
    call_counter.sync_calls += 1
    print(f"Sync call attempt #{call_counter.sync_calls}")
    
    # Fail the first 2 times
    if call_counter.sync_calls <= 2:
        print(f"  └─ Simulating RateLimitError...")
        raise RateLimitError("Mock rate limit error")
        
    # Succeed on the 3rd attempt
    print(f"  └─ Success!")
    
    # Create a mock response similar to OpenAI API
    class Choice:
        def __init__(self):
            self.message = type('obj', (object,), {
                'content': 'This is a successful response after retries.',
                'role': 'assistant'
            })
            self.finish_reason = "stop"
    
    return type('obj', (object,), {
        'choices': [Choice()],
        'model': 'mock-model',
        'usage': {'total_tokens': 50}
    })

async def mock_acompletion(**kwargs):
    """Async mock that fails the first 2 times, then succeeds"""
    call_counter.async_calls += 1
    print(f"Async call attempt #{call_counter.async_calls}")
    
    # Small delay to simulate async processing
    await asyncio.sleep(0.1)
    
    # Fail the first 2 times
    if call_counter.async_calls <= 2:
        print(f"  └─ Simulating APIError...")
        raise APIError("Mock API error")
    
    # Succeed on the 3rd attempt
    print(f"  └─ Success!")
    
    # Create a mock response similar to OpenAI API
    class Choice:
        def __init__(self):
            self.message = type('obj', (object,), {
                'content': 'This is a successful async response after retries.',
                'role': 'assistant'
            })
            self.finish_reason = "stop"
    
    return type('obj', (object,), {
        'choices': [Choice()],
        'model': 'mock-model',
        'usage': {'total_tokens': 50}
    })

# Apply patch decorators to mock litellm completion functions
@patch('litellm.completion', side_effect=mock_completion)
@patch('litellm.acompletion', side_effect=mock_acompletion)
def run_examples(_mock_acompletion, _mock_completion):
    # Example 1: Default retry (3 attempts)
    print("\n=== Example 1: Default retry (3 attempts) ===")
    
    @llm(retry=True)
    def default_retry_example(topic: str):
        """Give me a one-paragraph explanation about {topic}."""
    
    try:
        result = default_retry_example("artificial intelligence")
        print(f"Result: {result}")
    except Exception as e:
        print(f"Default retry example failed: {type(e).__name__}: {e}")
    
    # Reset call counters
    call_counter.sync_calls = 0
    
    # Example 2: Custom retry attempts (5 attempts)
    print("\n=== Example 2: Custom retry (5 attempts) ===")
    
    @llm(retry=5)
    def custom_retry_example(topic: str):
        """Give me a one-paragraph explanation about {topic}."""
    
    try:
        result = custom_retry_example("space exploration")
        print(f"Result: {result}")
    except Exception as e:
        print(f"Custom retry example failed: {type(e).__name__}: {e}")
    
    # Reset call counters
    call_counter.sync_calls = 0
    
    # Example 3: Disabled retry
    print("\n=== Example 3: Disabled retry ===")
    
    @llm(retry=False)
    def no_retry_example(topic: str):
        """Give me a one-paragraph explanation about {topic}."""
    
    try:
        result = no_retry_example("climate change")
        print(f"Result: {result}")
    except Exception as e:
        print(f"No retry example failed as expected: {type(e).__name__}: {e}")
    
    # Reset call counters
    call_counter.async_calls = 0
    
    # Example 4: Async function with retry
    print("\n=== Example 4: Async function with retry ===")
    
    @llm(retry=True)
    async def async_retry_example(topic: str):
        """Give me a one-paragraph explanation about {topic}."""
    
    try:
        result = asyncio.run(async_retry_example("quantum computing"))
        print(f"Async result: {result}")
    except Exception as e:
        print(f"Async example failed: {type(e).__name__}: {e}")
    
    # Example 5: Class method with retry
    print("\n=== Example 5: Class method with retry ===")
    
    class Assistant:
        def __init__(self, name="AI Assistant"):
            self.name = name
            self.topics_answered = []
        
        @llm(retry=True)
        def answer(self, topic, _result=None):
            """Provide a brief explanation about {topic}.

            This response comes from an assistant.
            """
            self.topics_answered.append(topic)
            if _result:
                return f"{self.name}'s answer: {_result}"
            return _result
    
    # Reset call counters
    call_counter.sync_calls = 0
    
    try:
        assistant = Assistant("Professor Bot")
        result = assistant.answer("renewable energy")
        print(f"Class method result: {result}")
        print(f"Topics answered: {assistant.topics_answered}")
    except Exception as e:
        print(f"Class method example failed: {type(e).__name__}: {e}")

if __name__ == "__main__":
    run_examples()
