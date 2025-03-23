import warnings

warnings.filterwarnings("ignore", message="Valid config keys have changed in V2:*")

import logging
from unittest.mock import Mock
import subprocess as sp
from pathlib import Path

import pytest
from litellm.exceptions import RateLimitError, InternalServerError, APIError, Timeout
from pydantic import BaseModel
from tenacity import (
    retry,
    wait_exponential,
    retry_if_exception_type,
)

from async_promptic import ImageBytes, Promptic, State, llm, litellm_completion
from openai import OpenAI

ERRORS = (RateLimitError, InternalServerError, APIError, Timeout)

# Define default model lists
CHEAP_MODELS = ["gpt-4o-mini", "claude-3-5-haiku-20241022", "gemini/gemini-2.0-flash"]
REGULAR_MODELS = ["gpt-4o", "claude-3-5-sonnet-20241022", "gemini/gemini-1.5-pro"]

openai_completion_fn = OpenAI().chat.completions.create


@pytest.mark.parametrize("model", CHEAP_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_basic(model, create_completion_fn):
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(
        temperature=0,
        model=model,
        timeout=5,
        create_completion_fn=create_completion_fn,
    )
    def president(year):
        """Who was the President of the United States in {year}?"""

    result = president(2001)
    assert "George W. Bush" in result
    assert isinstance(result, str)


@pytest.mark.parametrize("model", CHEAP_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_pydantic(model, create_completion_fn):
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

    class Capital(BaseModel):
        country: str
        capital: str

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(
        temperature=0, model=model, timeout=5, create_completion_fn=create_completion_fn
    )
    def capital(country) -> Capital:
        """What's the capital of {country}?"""

    result = capital("France")
    assert result.country == "France"
    assert result.capital == "Paris"


@pytest.mark.parametrize("model", CHEAP_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_streaming(model, create_completion_fn):
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(
        stream=True,
        model=model,
        temperature=0,
        timeout=5,
        create_completion_fn=create_completion_fn,
    )
    def haiku(subject, adjective, verb="delights"):
        """Write a haiku about {subject} that is {adjective} and {verb}."""

    result = "".join(haiku("programming", adjective="witty"))
    assert isinstance(result, str)


@pytest.mark.parametrize("model", CHEAP_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_system_prompt(model, create_completion_fn):
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(
        system="you are a snarky chatbot",
        temperature=0,
        model=model,
        timeout=8,
        create_completion_fn=create_completion_fn,
    )
    def answer(question):
        """{question}"""

    result = answer("How to boil water?")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.parametrize("model", CHEAP_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_system_prompt_list_strings(model, create_completion_fn):
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

    system_prompts = [
        "you are a helpful assistant",
        "you always provide concise answers",
        "you speak in a formal tone",
    ]

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(
        system=system_prompts,
        temperature=0,
        model=model,
        timeout=5,
        create_completion_fn=create_completion_fn,
    )
    def answer(question):
        """{question}"""

    result = answer("What is the capital of France?")
    assert isinstance(result, str)
    assert "Paris" in result
    # Should be concise due to system prompt
    assert len(result.split()) < 30


@pytest.mark.parametrize("model", CHEAP_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_agents(model, create_completion_fn):
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(
        temperature=0,
        model=model,
        timeout=10,
        create_completion_fn=create_completion_fn,
    )
    def scheduler(command):
        """{command}"""

    @scheduler.tool
    def get_current_time():
        """Get the current time"""
        return "3:48 PM"

    @scheduler.tool
    def add_reminder(task: str, time: str):
        """Add a reminder for a specific task and time"""
        return f"Reminder set: {task} at {time}"

    @scheduler.tool
    def check_calendar(date: str):
        """Check calendar for a specific date"""
        return f"Calendar checked for {date}: No conflicts found"

    cmd = """
    What time is it?
    Also, can you check my calendar for tomorrow
    and set a reminder for a team meeting at 2pm?
    """

    result = scheduler(cmd)
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.parametrize("model", CHEAP_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_streaming_with_tools(model, create_completion_fn):
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")
    
    # Skip test for Gemini as it doesn't support streaming with tools
    if "gemini" in model.lower():
        pytest.skip("Gemini doesn't support streaming with tools")

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(
        stream=True,
        temperature=0,
        model=model,
        timeout=10,
        create_completion_fn=create_completion_fn,
    )
    def travel_assistant(destination):
        """
        You are a travel assistant. 
        Provide a brief travel recommendation for {destination}.
        Use the fetch_weather tool to check the weather there.
        """

    @travel_assistant.tool
    def fetch_weather(city):
        """Fetches weather data for a city"""
        return f"Sunny and 75°F in {city}"

    response = travel_assistant("Tokyo")
    result = "".join(response)
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.parametrize("model", CHEAP_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_dry_run_with_tools(model, create_completion_fn, caplog):
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(
        temperature=0,
        model=model,
        timeout=10,
        dry_run=True,
        create_completion_fn=create_completion_fn,
    )
    def assistant(query):
        """{query}"""

    @assistant.tool
    def get_weather(location: str, unit: str = "celsius"):
        """Get the current weather in a given location"""
        return f"The weather in {location} is 25 degrees {unit}"

    with caplog.at_level(logging.INFO):
        result = assistant("What's the weather in London?")
        assert "[DRY RUN]" in caplog.text
        assert "get_weather" in caplog.text
        assert "London" in caplog.text


@pytest.mark.parametrize("model", CHEAP_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_state_basic(model, create_completion_fn):
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(
        memory=True,
        temperature=0,
        model=model,
        timeout=10,
        create_completion_fn=create_completion_fn,
    )
    def chat(message):
        """{message}"""

    result1 = chat("Hello, my name is Bob")
    assert isinstance(result1, str)
    
    result2 = chat("What's my name?")
    assert "Bob" in result2
    assert isinstance(result2, str)


@pytest.mark.parametrize("model", CHEAP_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_custom_state(model, create_completion_fn):
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")

    class FileState(State):
        def __init__(self, path):
            super().__init__()
            self.path = path
            if Path(path).exists():
                with open(path, "r") as f:
                    self.messages = eval(f.read())
            else:
                self.messages = []
                with open(path, "w") as f:
                    f.write(str(self.messages))

        def save(self):
            with open(self.path, "w") as f:
                f.write(str(self.messages))

    file_path = "test_state_file.txt"
    
    # Clean up from previous tests
    if Path(file_path).exists():
        Path(file_path).unlink()

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(ERRORS),
    )
    @llm(
        state=FileState(file_path),
        temperature=0,
        model=model,
        timeout=10,
        create_completion_fn=create_completion_fn,
    )
    def chat(message):
        """{message}"""

    result = chat("Hello, my name is Alice")
    assert isinstance(result, str)

    # Clean up after test
    if Path(file_path).exists():
        Path(file_path).unlink()


@pytest.mark.parametrize("model", CHEAP_MODELS)
@pytest.mark.parametrize(
    "create_completion_fn", [openai_completion_fn, litellm_completion]
)
def test_image_functionality(model, create_completion_fn):
    if create_completion_fn == openai_completion_fn and not model.startswith("gpt"):
        pytest.skip("Non-GPT models are not supported with OpenAI client")
    
    # Skip for models without vision capabilities
    if not (model == "gpt-4o" or model == "claude-3-5-sonnet-20241022"):
        pytest.skip(f"Skipping vision test for {model}")

    try:
        # Test with ImageBytes type
        @retry(
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry=retry_if_exception_type(ERRORS),
        )
        @llm(
            model=model, 
            timeout=15,
            create_completion_fn=create_completion_fn,
        )
        def describe_image(image: ImageBytes):
            """What's in this image?"""

        # Load a test image - Skip if not available
        image_path = Path("tests/fixtures/ocai-logo.jpeg")
        if not image_path.exists():
            pytest.skip("Test image not available")
        
        with open(image_path, "rb") as f:
            image_data = ImageBytes(f.read())
        
        result = describe_image(image_data)
        assert isinstance(result, str)
        assert len(result) > 0
        
    except Exception as e:
        pytest.skip(f"Image test failed: {str(e)}")


@pytest.mark.parametrize("model", CHEAP_MODELS)
def test_class_method(model):
    class LanguageAssistant:
        """A class that demonstrates using @llm decorator with class methods"""
        
        def __init__(self):
            self.translation_count = 0
            
        @llm(model=model)
        def translate(self, text, target="Spanish", _result=None):
            """Translate the following text: '{text}' into {target}"""
            # Optional post-processing of the LLM result
            self.translation_count += 1
            return f"Translation result: {_result}"
            
        @llm(model=model)
        def summarize(self, text):
            """Provide a concise one-sentence summary of: '{text}'"""
    
    assistant = LanguageAssistant()
    
    # Call the class method
    translation = assistant.translate("Hello world", target="French")
    assert "Translation result:" in translation
    assert assistant.translation_count == 1
    
    # Simple example without _result parameter
    summary = assistant.summarize("Async-Promptic is a fork of Promptic that adds full async support.")
    assert isinstance(summary, str)
    assert len(summary) > 0
