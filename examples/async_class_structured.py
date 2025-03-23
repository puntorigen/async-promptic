from pydantic import BaseModel
import asyncio
from async_promptic import llm

class Forecast(BaseModel):
    location: str
    temperature: float
    units: str


class WeatherAssistant:
    def __init__(self, default_units="fahrenheit"):
        self.default_units = default_units
        self.request_count = 0
    
    # Simple approach - no _result parameter needed
    @llm
    async def get_weather(self, location, units: str = None) -> Forecast:
        """What's the weather for {location} in {units}?"""
        # This function body won't be executed
        # The LLM will generate the response directly as a Forecast object


    # Advanced approach with post-processing
    @llm
    async def get_weather_with_processing(self, location, units: str = "fahrenheit", _result=None) -> Forecast:
        """What's the weather for {location} in {units}?"""
        # Track requests
        self.request_count += 1
        print(f"Processing weather request for {location} with units: {units}")
        return _result


async def main():
    # Test with default units
    assistant = WeatherAssistant()
    
    # Test simple approach (no _result parameter)
    print("=== Testing simple approach (no _result parameter) ===")
    forecast = await assistant.get_weather("San Francisco")
    print(f"Forecast type: {type(forecast)}")
    print(f"Forecast data: {forecast}")
    
    # Test advanced approach with post-processing
    print("\n=== Testing advanced approach with post-processing ===")
    processed_forecast = await assistant.get_weather_with_processing("Seattle", units="celsius")
    print(f"Processed forecast type: {type(processed_forecast)}")
    print(f"Processed forecast data: {processed_forecast}")
    print(f"Request count: {assistant.request_count}")


if __name__ == "__main__":
    asyncio.run(main())
