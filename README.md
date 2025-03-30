# async-promptic

[![Python Versions](https://img.shields.io/pypi/pyversions/async-promptic)](https://pypi.org/project/async-promptic)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

### 90% of what you need for LLM app development. Nothing you don't.

Async-Promptic is a fork of [Promptic](https://github.com/puntorigen/promptic) that adds full async support. It aims to be the "[requests](https://requests.readthedocs.io/en/latest/)" of LLM development -- the most productive and pythonic way to build LLM applications. It leverages [LiteLLM][litellm], so you're never locked in to an LLM provider and can switch to the latest and greatest with a single line of code. Async-Promptic gets out of your way so you can focus entirely on building features.

> "Perfection is attained, not when there is nothing more to add, but when there is nothing more to take away."

### At a glance

- 🎯 Type-safe structured outputs with Pydantic
- 🤖 Easy-to-build agents with function calling
- 🔄 Streaming support for real-time responses
- 📚 Automatic prompt caching for supported models
- 💾 Built-in conversation memory
- ⚡ **Full async/await support for modern Python applications**

## Installation

```bash
pip install async-promptic
```

## Usage

### Basics

Functions decorated with `@llm` use its docstring as a prompt template. When the function is called, promptic combines the docstring with the function's arguments to generate the prompt and returns the LLM's response.

```py
# examples/basic.py

from async_promptic import llm


@llm
def translate(text, language="Chinese"):
    """Translate '{text}' to {language}"""


print(translate("Hello world!"))
# 您好，世界！

print(translate("Hello world!", language="Spanish"))
# ¡Hola, mundo!


@llm(
    model="claude-3-haiku-20240307",
    system="You are a customer service analyst. Provide clear sentiment analysis with key points.",
)
def analyze_sentiment(text):
    """Analyze the sentiment of this customer feedback: {text}"""


print(analyze_sentiment("The product was okay but shipping took forever"))
# Sentiment: Mixed/Negative
# Key points:
# - Neutral product satisfaction
# - Significant dissatisfaction with shipping time
```

### Class Method Support

Async-Promptic supports using the `@llm` decorator on class methods, allowing you to organize LLM-powered functions within classes for better state management and code organization:

```py
# examples/class_method_example.py

from async_promptic import llm

class LanguageAssistant:
    """A class that demonstrates using @llm decorator with class methods"""
    
    def __init__(self):
        self.translation_count = 0
        
    # Example of an async class method with @llm decorator
    @llm(model="gpt-3.5-turbo")
    async def translate(self, text, target="Spanish", _result=None):
        """Translate the following text: '{text}' into {target}"""
        # Optional post-processing of the LLM result
        self.translation_count += 1
        print(f"Translation #{self.translation_count} completed")
        return f"Translation result: {_result}"
    
    # Example of class method with tools
    @llm(model="gpt-3.5-turbo")
    async def weather_recommendation(self, city, _result=None):
        """
        You are a travel assistant. 
        Provide a brief travel recommendation for {city}.
        Use the fetch_weather tool to check the weather there.
        """
        print(f"Processing recommendation for {city}")
        return f"Processed recommendation: {_result}"
    
    # Register an async tool with the class method
    @weather_recommendation.tool
    async def fetch_weather(self, city):
        """Fetches weather data for a city"""
        return f"Sunny and 75°F in {city}"

    # Simple example without the _result parameter
    @llm(model="gpt-3.5-turbo")
    async def summarize(self, text) -> str:
        """Provide a concise one-sentence summary of: '{text}'"""
        # The method body won't execute
        # The LLM response is returned directly
    
# Usage
async def main():
    assistant = LanguageAssistant()
    
    # Call the class method
    translation = await assistant.translate("Hello world", target="French")
    print(translation)
    # Translation #1 completed
    # Translation result: Bonjour le monde
    
    # Class methods maintain state
    translation = await assistant.translate("How are you?", target="German")
    print(f"Total translations: {assistant.translation_count}")
    # Translation #2 completed
    # Total translations: 2
    
    # Class methods with tools
    recommendation = await assistant.weather_recommendation("Paris")
    print(recommendation)
    # Processing recommendation for Paris
    # Processed recommendation: Paris is currently sunny with temperatures of 75°F...
    
    # Simple example without _result parameter
    summary = await assistant.summarize("Async-Promptic is a fork of Promptic that adds full async support for LLM development. It supports class methods, structured outputs, and function calling.")
    print(summary)
    # A concise one-sentence summary of the text about Async-Promptic and its features.

```

The `_result` parameter in class methods is optional and allows post-processing of the LLM's response. When present, it receives the LLM's output for further processing. This enables you to:

- Transform or validate the LLM output before returning it
- Update class state based on the response
- Log information or track usage metrics
- Chain multiple LLM calls together with shared context

If you don't need to process the LLM output, you can omit the `_result` parameter entirely, as shown in the `summarize` method above. The LLM response will be returned directly.

```py
@llm(model="gpt-3.5-turbo")
def count_words(self, text):
    """Count the number of words in the following text: '{text}'"""
    # No _result parameter, so you won't receive the LLM output here
    return f"The text has approximately {len(text.split())} words"
```

Class method support combines well with Async-Promptic's other features like structured outputs, tools, and streaming, giving you powerful ways to organize complex LLM-powered applications.

### Async Support

Async-Promptic fully supports async/await syntax, making it ideal for modern Python applications using frameworks like FastAPI or asynchronous libraries.

```py
# examples/async_example.py

import asyncio
from async_promptic import llm

# Decorate our async function with the llm decorator
@llm
async def travel_assistant(destination):
    """
    You are a travel assistant. 
    Provide a brief travel recommendation for {destination}.
    Use the fetch_weather tool to check the weather there.
    """

# Register an async tool with the decorated function
@travel_assistant.tool
async def fetch_weather(city):
    """
    Simulates fetching weather data asynchronously
    """
    print(f"Fetching weather for {city}...")
    await asyncio.sleep(1)  # Simulate API call
    return f"Sunny and 75°F in {city}"

# Example of an async function with streaming
@llm(stream=True)
async def stream_story(topic):
    """
    Write a very short story about {topic}.
    """

async def main():
    # Call the async function
    result = await travel_assistant("Tokyo")
    print("Travel Assistant Result:")
    print(result)
    
    # Test the streaming function
    print("Streaming Story about space exploration:")
    generator = await stream_story("space exploration")
    
    # Process the streaming response
    story = ""
    for chunk in generator:
        print(chunk, end="", flush=True)
        story += chunk
    print("\n")

if __name__ == "__main__":
    asyncio.run(main())
```

### Image Support

Promptic supports image inputs through the `ImageBytes` type. By defining an argument as `ImageBytes`, image data is automatically converted to base64 format and sent to the LLM.

```py
# examples/image_support.py

from async_promptic import llm, ImageBytes


@llm(model="gpt-4o")  # Use a vision-capable model
def describe_image(image: ImageBytes):
    """What's in this image?"""


with open("tests/fixtures/ocai-logo.jpeg", "rb") as f:
    image_data = ImageBytes(f.read())

print(describe_image(image_data))
# The image features an illustration of a cheerful orange with glasses sitting on a laptop. There are small, sparkling stars surrounding the orange. Below the illustration, it says "Orange County AI."


@llm(model="gpt-4o")
def analyze_image_feature(image: ImageBytes, feature: str):
    """Tell me about the {feature} in this image in a sentence or less."""


print(analyze_image_feature(image_data, "colors"))
# The image features vibrant orange, green, and black colors against a light background.

```

### Structured Outputs

You can use Pydantic models to ensure the LLM returns data in exactly the structure you expect. Simply define a Pydantic model and use it as the return type annotation on your decorated function. The LLM's response will be automatically validated against your model schema and returned as a Pydantic object.

```py
# examples/structured.py

from pydantic import BaseModel
from async_promptic import llm


class Forecast(BaseModel):
    location: str
    temperature: float
    units: str


@llm
def get_weather(location, units: str = "fahrenheit") -> Forecast:
    """What's the weather for {location} in {units}?"""


print(get_weather("San Francisco", units="celsius"))
# location='San Francisco' temperature=16.0 units='Celsius'

```

Alternatively, you can use JSON Schema dictionaries for more low-level validation:

```py
# examples/json_schema.py

from async_promptic import llm

schema = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "pattern": "^[A-Z][a-z]+$",
            "minLength": 2,
            "maxLength": 20,
        },
        "age": {"type": "integer", "minimum": 0, "maximum": 120},
        "email": {"type": "string", "format": "email"},
    },
    "required": ["name", "age"],
    "additionalProperties": False,
}


@llm(json_schema=schema, system="You generate test data.")
def get_user_info(name: str) -> dict:
    """Get information about {name}"""


print(get_user_info("Alice"))
# {'name': 'Alice', 'age': 25, 'email': 'alice@example.com'}

```

### Agents

Functions decorated with `@llm.tool` become tools that the LLM can invoke to perform actions or retrieve information. The LLM will automatically execute the appropriate tool calls, creating a seamless agent interaction.

```py
# examples/book_meeting.py

from datetime import datetime

from async_promptic import llm


@llm(model="gpt-4o")
def scheduler(command):
    """{command}"""


@scheduler.tool
def get_current_time():
    """Get the current time"""
    print("getting current time")
    return datetime.now().strftime("%I:%M %p")


@scheduler.tool
def add_reminder(task: str, time: str):
    """Add a reminder for a specific task and time"""
    print(f"adding reminder: {task} at {time}")
    return f"Reminder set: {task} at {time}"


@scheduler.tool
def check_calendar(date: str):
    """Check calendar for a specific date"""
    print(f"checking calendar for {date}")
    return f"Calendar checked for {date}: No conflicts found"


cmd = """
What time is it?
Also, can you check my calendar for tomorrow
and set a reminder for a team meeting at 2pm?
"""

print(scheduler(cmd))
# getting current time
# checking calendar for 2023-10-05
# adding reminder: Team meeting at 2023-10-05T14:00:00
# The current time is 3:48 PM. I checked your calendar for tomorrow, and there are no conflicts. I've also set a reminder for your team meeting at 2 PM tomorrow.

```

### Streaming

The streaming feature allows real-time response generation, useful for long-form content or interactive applications:

```py
# examples/streaming.py

from async_promptic import llm


@llm(stream=True)
def write_poem(topic):
    """Write a haiku about {topic}."""


print("".join(write_poem("artificial intelligence")))
# Binary thoughts hum,
# Electron minds awake, learn,
# Future thinking now.

```

### OpenAI Client

Promptic uses [litellm](https://docs.litellm.ai/) by default, but supports any OpenAI-compatible client.

Here's an example on how to use the [langfuse](https://langfuse.com/) OpenAI client to trace function calls.

```py
# examples/langfuse_openai.py

from langfuse.openai import openai
from langfuse.decorators import observe
from async_promptic import Promptic


promptic = Promptic(openai_client=openai.OpenAI())


@observe
@promptic.llm
def greet(name):
    """Greet {name}"""


print(greet("John"))
# Hello, John!

```

### Observability

Promptic integrates with [Weave](https://wandb.ai) to trace function calls and LLM interactions.

<img width="981" alt="Screenshot 2025-02-07 at 6 02 25 PM" src="https://github.com/user-attachments/assets/3ccf4602-3557-455a-838f-7e4b8c2ec21a" />

```py
# examples/weave_integration.py

from async_promptic import Promptic
import weave

# Initialize the weave client with the project name
# it doesn't need to be "promptic"
client = weave.init("promptic")

promptic = Promptic(weave_client=client)


@promptic.llm
def translate(text, language="Chinese"):
    """Translate '{text}' to {language}"""


print(translate("Hello world!"))
# 您好，世界！

print(translate("Hello world!", language="Spanish"))
# ¡Hola, mundo!


@promptic.llm(stream=True)
def write_poem(topic):
    """Write a haiku about {topic}."""


print("".join(write_poem("artificial intelligence")))
# Binary thoughts hum,
# Electron minds awake, learn,
# Future thinking now.

```

### Error Handling and Dry Runs

Dry runs allow you to see which tools will be called and their arguments without invoking the decorated tool functions. You can also enable debug mode for more detailed logging.

```py
# examples/error_handing.py

from async_promptic import llm


@llm(
    system="you are a posh smart home assistant named Jarvis",
    dry_run=True,
    debug=True,
)
def jarvis(command):
    """{command}"""


@jarvis.tool
def turn_light_on():
    """turn light on"""
    return True


@jarvis.tool
def get_current_weather(location: str, unit: str = "fahrenheit"):
    """Get the current weather in a given location"""
    return f"The weather in {location} is 45 degrees {unit}"


print(jarvis("Please turn the light on and check the weather in San Francisco"))
# ...
# [DRY RUN]: function_name = 'turn_light_on' function_args = {}
# [DRY RUN]: function_name = 'get_current_weather' function_args = {'location': 'San Francisco'}
# ...

```

### Resiliency

Promptic has built-in retry support for handling rate limits, temporary API failures, and other common LLM errors. You can enable retry functionality by passing the `retry` parameter to the `@llm` decorator.

```py
# examples/retry_example.py

from async_promptic import llm
from litellm.exceptions import RateLimitError, InternalServerError, APIError, Timeout

# Retry with default settings (3 attempts, exponential backoff)
@llm(retry=True)
def generate_summary(text):
    """Summarize this text in 2-3 sentences: {text}"""


# Specify the number of retry attempts
@llm(retry=5)  # Will retry up to 5 times
def generate_headline(text):
    """Create a catchy headline for this article: {text}"""


# Disable retry functionality
@llm(retry=False)
def generate_tags(text):
    """Generate 5 tags for this content: {text}"""


# All of these functions will handle errors gracefully:
# - RateLimitError
# - InternalServerError
# - APIError
# - Timeout

generate_summary("Long article text here...")  # Will retry up to 3 times
generate_headline("Breaking news story...")    # Will retry up to 5 times
generate_tags("Product description...")        # No retries, will fail immediately
```

The retry mechanism uses the [stamina](https://github.com/hynek/stamina) package under the hood for reliable error handling with exponential backoff. The default retry policy is set to 3 attempts with exponential backoff, but you can customize this behavior by passing a custom `stamina` policy to the `retry` parameter.

### Memory and State Management

By default, each function call is independent and stateless. Setting `memory=True` enables built-in conversation memory, allowing the LLM to maintain context across multiple interactions. Here's a practical example using Gradio to create a web-based chatbot interface:

```py
# examples/memory.py

import gradio as gr
from async_promptic import llm


@llm(memory=True, stream=True)
def assistant(message):
    """{message}"""


def predict(message, history):
    partial_message = ""
    for chunk in assistant(message):
        partial_message += str(chunk)
        yield partial_message


with gr.ChatInterface(title="Promptic Chatbot Demo", fn=predict) as demo:
    # ensure clearing the chat window clears the chat history
    demo.chatbot.clear(assistant.clear)

# demo.launch()
```

---

Note, calling a decorated function will always execute the prompt template. For more direct control over conversations, you can use the `.message()` method to send follow-up messages without re-executing the prompt template:

```py
# examples/direct_messaging.py

from async_promptic import llm


@llm(
    system="You are a knowledgeable history teacher.",
    model="gpt-4o-mini",
    memory=True,
    stream=True,
)
def history_chat(era: str, region: str):
    """Tell me a fun fact about the history of {region} during the {era} period."""


response = history_chat("medieval", "Japan")
for chunk in response:
    print(chunk, end="")

for chunk in history_chat.message(
    "In one sentence, was the most popular person there at the time?"
):
    print(chunk, end="")

for chunk in history_chat.message("In one sentence, who was their main rival?"):
    print(chunk, end="")

```

The `.message()` method is particularly useful when:

- You have a decorated function with parameters but want to ask follow-up questions
- You want to maintain conversation context without re-executing the prompt template
- You need more direct control over the conversation flow while keeping memory intact

---

For custom storage solutions, you can extend the `State` class to implement persistence in any database or storage system:

```py
# examples/state.py

import json
from async_promptic import State, llm


class RedisState(State):
    def __init__(self, redis_client):
        super().__init__()
        self.redis = redis_client
        self.key = "chat_history"

    def add_message(self, message):
        self.redis.rpush(self.key, json.dumps(message))

    def get_messages(self, prompt: str = None, limit: int = None):
        messages = self.redis.lrange(self.key, 0, -1)
        return [json.loads(m) for m in messages][-limit:] if limit else messages

    def clear(self):
        self.redis.delete(self.key)


@llm(state=RedisState(redis_client))
def persistent_chat(message):
    """Chat: {message}"""

```

### Caching

For Anthropic models (Claude), promptic provides intelligent caching control to optimize context window usage and improve performance. By default, caching is enabled but can be disabled if needed. OpenAI models cache by default. Anthropic charges for cache writes, but tokens that are read from the cache are less expensive.

```py
# examples/caching.py

from async_promptic import llm

# imagine these are long legal documents
legal_document, another_legal_document = (
    "a legal document about Sam",
    "a legal document about Jane",
)

system_prompts = [
    "You are a helpful legal assistant",
    "You provide detailed responses based on the provided context",
    f"legal document 1: '{legal_document}'",
    f"legal document 2: '{another_legal_document}'",
]


@llm(
    system=system_prompts,
    cache=True,  # this is the default
)
def legal_chat(message):
    """{message}"""


print(legal_chat("which legal document is about Sam?"))
# The legal document about Sam is "legal document 1."

```

When caching is enabled:

- Long messages (>1KB) are automatically marked as ephemeral to optimize context window usage
- A maximum of 4 message blocks can be cached at once
- System prompts can include explicit cache control

Further reading:

- https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#cache-limitations
- https://docs.litellm.ai/docs/completion/prompt_caching

### Authentication

Authentication can be handled in three ways:

1. Directly via the `api_key` parameter:

```py
from async_promptic import llm

@llm(model="gpt-4o-mini", api_key="your-api-key-here")
def my_function(text):
    """Process this text: {text}"""
```

2. Through environment variables (recommended):

```bash

# OpenAI
export OPENAI_API_KEY=sk-...

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# Google
export GEMINI_API_KEY=...

# Azure OpenAI
export AZURE_API_KEY=...
export AZURE_API_BASE=...
export AZURE_API_VERSION=...
```

3. By setting the API key programmatically via litellm:

```py
from litellm import litellm

litellm.api_key = "your-api-key-here"
```

The supported environment variables correspond to the model provider:

| Provider  | Environment Variable | Model Examples                                                            |
| --------- | -------------------- | ------------------------------------------------------------------------- |
| OpenAI    | `OPENAI_API_KEY`     | gpt-4o, gpt-3.5-turbo                                                     |
| Anthropic | `ANTHROPIC_API_KEY`  | claude-3-haiku-20240307, claude-3-opus-20240229, claude-3-sonnet-20240229 |
| Google    | `GEMINI_API_KEY`     | gemini/gemini-1.5-pro-latest                                              |

## API Reference

### `llm`

The main decorator for creating LLM-powered functions. Can be used as `@llm` or `@llm()` with parameters.

#### Parameters

- `model` (str, optional): The LLM model to use. Defaults to "gpt-4o-mini".
- `system` (str | list[str] | list[dict], optional): System prompt(s) to set context for the LLM. Can be:
  - A single string: `system="You are a helpful assistant"`
  - A list of strings: `system=["You are a helpful assistant", "You speak formally"]`
  - A list of message dictionaries:
    ```python
    system=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Please be concise", "cache_control": {"type": "ephemeral"}},
        {"role": "assistant", "content": "I will be concise"}
    ]
    ```
- `dry_run` (bool, optional): If True, simulates tool calls without executing them. Defaults to False.
- `debug` (bool, optional): If True, enables detailed logging. Defaults to False.
- `memory` (bool, optional): If True, enables conversation memory using the default State implementation. Defaults to False.
- `state` (State, optional): Custom State implementation for memory management. Overrides the `memory` parameter.
- `json_schema` (dict, optional): JSON Schema dictionary for validating LLM outputs. Alternative to using Pydantic models.
- `cache` (bool, optional): If True, enables prompt caching. Defaults to True.
- `create_completion_fn` (callable, optional): Custom completion function to use instead of litellm.completion.
- `weave_client` (WeaveClient, optional): Weave client to use for tracing.
- `**completion_kwargs`: Additional arguments passed directly to the completion function i.e. [litellm.completion](https://docs.litellm.ai/docs/completion/input).

#### Methods

- `tool(fn)`: Decorator method to register a function as a tool that can be called by the LLM.
- `clear()`: Clear all stored messages from memory. Raises ValueError if memory/state is not enabled.

### `State`

Base class for managing conversation memory and state. Can be extended to implement custom storage solutions.

#### Methods

- `add_message(message: dict)`: Add a message to the conversation history.
- `get_messages(prompt: str = None, limit: int = None) -> List[dict]`: Retrieve conversation history, optionally limited to the most recent messages and filtered by a prompt.
- `clear()`: Clear all stored messages.

### `Promptic`

The main class for creating LLM-powered functions and managing conversations.

#### Methods

- `message(message: str, **kwargs)`: Send a message directly to the LLM and get a response. Useful for follow-up questions or direct interactions.
- `completion(messages: list[dict], **kwargs)`: Send a list of messages directly to the LLM and get the raw completion response. Useful for more control over the conversation flow.

  ```python
  from async_promptic import Promptic

  p = Promptic(model="gpt-4o-mini")
  messages = [
      {"role": "user", "content": "What is the capital of France?"},
      {"role": "assistant", "content": "The capital of France is Paris."},
      {"role": "user", "content": "What is its population?"}
  ]
  response = p.completion(messages)
  print(response.choices[0].message.content)
  ```

- `tool(fn)`: Register a function as a tool that can be called by the LLM.
- `clear()`: Clear all stored messages from memory. Raises ValueError if memory/state is not enabled.

#### Example

```py
# examples/api_ref.py

from pydantic import BaseModel
from async_promptic import llm


class Story(BaseModel):
    title: str
    content: str
    style: str
    word_count: int


@llm(
    model="gpt-4o-mini",
    system="You are a creative writing assistant",
    memory=True,
    temperature=0.7,
    max_tokens=800,
    cache=False,
)
def story_assistant(command: str) -> Story:
    """Process this writing request: {command}"""


@story_assistant.tool
def get_writing_style():
    """Get the current writing style preference"""
    return "whimsical and light-hearted"


@story_assistant.tool
def count_words(text: str) -> int:
    """Count words in the provided text"""
    return len(text.split())


story = story_assistant("Write a short story about a magical library")
print(f"Title: {story.title}")
print(f"Style: {story.style}")
print(f"Words: {story.word_count}")
print(story.content)

print(
    story_assistant("Write another story with the same style but about a time traveler")
)

```

## Limitations

`promptic` is a lightweight abstraction layer over [litellm][litellm] and its various LLM providers. As such, there are some provider-specific limitations that are beyond the scope of what the library addresses:

- **Streaming**:
  - Gemini models do not support streaming when using tools/function calls

These limitations reflect the underlying differences between LLM providers and their implementations. For provider-specific features or workarounds, you may need to interact with [litellm][litellm] or the provider's SDK directly.

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md)

[litellm]: https://github.com/BerriAI/litellm
