from langchain_openai import ChatOpenAI
from decouple import config

model = ChatOpenAI(model="gpt-4", api_key=config("OPENAI_API_KEY"), temperature=0.5, max_completion_tokens=10)

response = model.invoke("What is the capital of India ?")

print(response)
print(response.content)