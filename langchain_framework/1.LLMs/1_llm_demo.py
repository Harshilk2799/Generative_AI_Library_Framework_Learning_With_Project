from langchain_openai import OpenAI
from decouple import config

llm = OpenAI(model="gpt-3.5-turbo-instruct", api_key=config("OPENAI_API_KEY"))

response = llm.invoke("What is the capital of India ?")

print(response)