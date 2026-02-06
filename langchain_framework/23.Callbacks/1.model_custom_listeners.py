from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from decouple import config
from langchain_core.callbacks import BaseCallbackHandler
import logging

class MyLLMListeners(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print("🚀 LLM Call Started")

    def on_llm_end(self, response, **kwargs):
        print("✅ LLM Call Finished")

    def on_llm_error(self, error, **kwargs):
        print("❌ LLM Error:", error)


primary_model = ChatOpenAI(
    api_key=config("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1")



prompt = ChatPromptTemplate([
    ("system", "You are a helpful assistant."),
    ("human", "{query}")
])

chain = prompt | primary_model

response = chain.invoke(
    {"query": "What is the capital of India?"}, 
    config={"callbacks": [MyLLMListeners()]}
)
print(response.content)