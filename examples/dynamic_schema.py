from pydantic import BaseModel, Field, create_model
from async_promptic import llm


@llm
def analyze_data(data):
    """Analyze the provided data and return structured results"""


# Example 1: Weather analysis
WeatherSchema = create_model(
    'WeatherAnalysis',
    temperature=(float, Field(..., description="Temperature in Celsius")),
    conditions=(str, Field(..., description="Weather conditions like sunny, rainy, etc.")),
    humidity=(int, Field(..., ge=0, le=100, description="Humidity percentage"))
)

analyze_data.setSchema(WeatherSchema)
result = analyze_data("It's a sunny day, 25°C with 60% humidity")
print("Weather Analysis:", result)
# WeatherAnalysis(temperature=25.0, conditions='sunny', humidity=60)


# Example 2: Financial data analysis - switching schemas dynamically
FinancialSchema = create_model(
    'FinancialAnalysis',
    stock_symbol=(str, Field(..., description="Stock ticker symbol")),
    price=(float, Field(..., gt=0, description="Current stock price")),
    trend=(str, Field(..., description="Price trend: up, down, or stable")),
    recommendation=(str, Field(..., description="Buy, hold, or sell recommendation"))
)

analyze_data.setSchema(FinancialSchema)
result = analyze_data("AAPL is trading at $175.50, up 2.3% today with strong momentum")
print("Financial Analysis:", result)
# FinancialAnalysis(stock_symbol='AAPL', price=175.5, trend='up', recommendation='buy')


# Example 3: Text classification
ClassificationSchema = create_model(
    'TextClassification',
    category=(str, Field(..., description="Primary category of the text")),
    sentiment=(str, Field(..., description="Positive, negative, or neutral")),
    confidence=(float, Field(..., ge=0, le=1, description="Confidence score between 0 and 1")),
    keywords=(list[str], Field(..., description="Key terms found in the text"))
)

analyze_data.setSchema(ClassificationSchema)
result = analyze_data("I absolutely love this new smartphone! The camera quality is amazing and battery life is incredible.")
print("Text Classification:", result)
# TextClassification(category='product review', sentiment='positive', confidence=0.95, keywords=['smartphone', 'camera', 'battery'])
