from langchain_anthropic import ChatAnthropic
from decouple import config

model = ChatAnthropic(model="claude-3-5-sonnet-20241022", api_key=config("ANTHROPIC_API_KEY"), temperature=0.5, max_completion_tokens=10)

response = model.invoke("What is the capital of India ?")
print(response)
print(response.content)