import asyncio
from promptic import llm, Promptic

# Create a Promptic instance with our desired settings
promptic = Promptic(model="gpt-3.5-turbo")

# Decorate our main function with the Promptic instance
@promptic.llm
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
@promptic.llm(stream=True)
async def stream_story(topic):
    """
    Write a very short story about {topic}.
    """

async def main():
    # Call the async function
    result = await travel_assistant("Tokyo")
    print("Travel Assistant Result:")
    print(result)
    print("\n---\n")

    # Test the streaming function
    print("Streaming Story about space exploration:")
    generator = await stream_story("space exploration")
    
    # The generator returned is not an async iterator, but a regular one
    story = ""
    for chunk in generator:
        print(chunk, end="", flush=True)
        story += chunk
    print("\n")

if __name__ == "__main__":
    asyncio.run(main())
