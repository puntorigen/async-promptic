import asyncio
from promptic import llm, Promptic

# Example of standalone function that uses the decorator (existing approach)
@llm(model="gpt-3.5-turbo")
async def standalone_translation(text, target_language="Spanish"):
    """Translate the following text: '{text}' into {target_language}"""


class LanguageAssistant:
    """A class that demonstrates using @llm decorator with class methods"""
    
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
        self.translation_count = 0
        
    # Example of an async class method with @llm decorator
    @llm(model="gpt-3.5-turbo")
    async def translate(self, text, target="Spanish", _result=None):
        """Translate the following text: '{text}' into {target}"""
        # This method will be called after the LLM returns a result
        # The result from the LLM is passed as the '_result' parameter
        self.translation_count += 1
        print(f"Translation #{self.translation_count} completed")
        return f"Translation result: {_result}"
        
    # Example of a sync class method with @llm decorator
    @llm(model="gpt-3.5-turbo")
    def summarize(self, content, max_words=50, _result=None):
        """Provide a concise summary of the following content in {max_words} words or less: 
        
        {content}
        """
        # This will be called with the result from the LLM in the '_result' parameter
        print(f"Summary generated with max words: {max_words}")
        return _result
        
    # Example of class method with tools
    @llm(model="gpt-3.5-turbo")
    async def weather_recommendation(self, city, _result=None):
        """
        You are a travel assistant. 
        Provide a brief travel recommendation for {city}.
        Use the fetch_weather tool to check the weather there.
        """
        # This method will be called after the LLM returns a result
        # The result from the LLM is passed as the '_result' parameter
        print(f"Processing recommendation for {city}")
        return f"Processed recommendation: {_result}"
    
    # Register an async tool with the class method
    @weather_recommendation.tool
    async def fetch_weather(self, city):
        """
        Fetches weather data for a city
        """
        print(f"Fetching weather for {city}...")
        await asyncio.sleep(1)  # Simulate API call
        return f"Sunny and 75°F in {city}"
        
    # Example of a non-async class method with @llm
    @llm(model="gpt-3.5-turbo")
    def classify_sentiment(self, text, _result=None):
        """
        Analyze the sentiment of the following text and classify it as positive, negative, or neutral:
        
        '{text}'
        
        Provide only the classification without any explanation.
        """
        print(f"Processing sentiment classification for: {text[:20]}...")
        return f"Processed sentiment: {_result}"
        
    # Example of a class method without the result parameter
    @llm(model="gpt-3.5-turbo")
    def count_words(self, text):
        """
        Count the number of words in the following text:
        
        '{text}'
        
        Respond with only a number.
        """
        print(f"Processing word count for: {text[:20]}...")
        # This method doesn't need to use the LLM result
        return f"The text has approximately {len(text.split())} words"
    
    # Test the standalone function (existing approach)
async def main():
    # Test the standalone function (existing approach)
    print("\n=== Testing standalone function (existing approach) ===")
    result = await standalone_translation("Hello world")
    print(f"Standalone translation result: {result}")
    
    # Test class methods with @llm
    print("\n=== Testing class methods with @llm ===")
    assistant = LanguageAssistant()
    
    # Test async class method
    print("\n--- Testing async class method ---")
    translation = await assistant.translate("Hello world", target="French")
    print(f"Class method translation: {translation}")
    
    # Test sync class method
    print("\n--- Testing sync class method ---")
    long_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    as opposed to intelligence displayed by animals and humans. Example tasks in which 
    this is done include speech recognition, computer vision, translation between (natural) 
    languages, as well as other mappings of inputs.
    """
    summary = assistant.summarize(long_text, max_words=30)
    print(f"Class method summary: {summary}")
    
    # Test class method with tools
    print("\n--- Testing class method with tools ---")
    recommendation = await assistant.weather_recommendation("Paris")
    print(f"Weather recommendation: {recommendation}")
    
    # Test non-async class method with @llm
    print("\n--- Testing non-async class method with @llm ---")
    sentiment = assistant.classify_sentiment("I absolutely love this new feature! It works perfectly.")
    print(f"Sentiment classification: {sentiment}")
    
    # Test class method without result parameter
    print("\n--- Testing class method without result parameter ---")
    word_count = assistant.count_words("This is a test sentence to count words in.")
    print(f"Word count: {word_count}")
    
    # Test multiple translations to show state is maintained
    print("\n--- Testing state maintenance in class ---")
    await assistant.translate("How are you?", target="German")
    await assistant.translate("Good morning", target="Italian")
    print(f"Total translations completed: {assistant.translation_count}")


if __name__ == "__main__":
    asyncio.run(main())
