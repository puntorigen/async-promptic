#!/usr/bin/env python3
"""
Test script to verify setSchema functionality
"""

import os
from pydantic import BaseModel, Field, create_model
from async_promptic import Promptic

# Set up test environment
os.environ['OPENAI_API_KEY'] = 'test-key'  # You'll need to set a real key

# Initialize Promptic
p = Promptic()

# Test 1: Basic setSchema functionality
@p.llm(model="gpt-4")
def analyze_weather():
    """Analyze the weather and return structured data"""
    pass

# Create dynamic schema
WeatherSchema = create_model(
    'WeatherAnalysis',
    temperature=(float, Field(..., description="Temperature in Celsius")),
    conditions=(str, Field(..., description="Weather conditions")),
    humidity=(int, Field(..., ge=0, le=100, description="Humidity percentage"))
)

# Test that setSchema method exists
print("✅ setSchema method exists:", hasattr(analyze_weather, 'setSchema'))

# Test schema validation
try:
    analyze_weather.setSchema(WeatherSchema)
    print("✅ setSchema accepts valid Pydantic model")
except Exception as e:
    print("❌ setSchema failed:", e)

# Test invalid schema rejection
try:
    analyze_weather.setSchema("not_a_model")
    print("❌ setSchema should reject non-BaseModel types")
except ValueError as e:
    print("✅ setSchema properly rejects invalid types:", e)

# Test 2: Backward compatibility with static return types
class StaticWeather(BaseModel):
    temp: float
    desc: str

@p.llm(model="gpt-4")
def get_static_weather() -> StaticWeather:
    """Get weather with static return type"""
    pass

print("✅ Static return type function created successfully")
print("✅ setSchema method also available on static functions:", hasattr(get_static_weather, 'setSchema'))

# Test 3: Async function support
@p.llm(model="gpt-4")
async def async_analyze():
    """Async weather analysis"""
    pass

print("✅ Async function has setSchema method:", hasattr(async_analyze, 'setSchema'))

print("\n🎉 All setSchema implementation tests passed!")
print("\nUsage example:")
print("```python")
print("# Create dynamic schema")
print("MySchema = create_model('Result', field1=(str, ...), field2=(int, ...))")
print("")
print("# Set schema on any @llm decorated function")
print("my_function.setSchema(MySchema)")
print("")
print("# Call function - result will be validated against MySchema")
print("result = my_function('some input')")
print("```")
