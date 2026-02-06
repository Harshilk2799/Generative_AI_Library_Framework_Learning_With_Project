from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from decouple import config
from langchain_core.callbacks import StdOutCallbackHandler

fallback_model1 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    api_key=config("GOOGLE_GEMINI_API_KEY")
)
primary_model = ChatGroq(
    model="deepseek-r1-distill-llama-70b", 
    api_key=config("GROQ_API_KEY")
)

model_with_fallback = primary_model.with_fallbacks([fallback_model1])

prompt = ChatPromptTemplate([
    ("system", "You are a helpful assistant."),
    ("human", "{query}")
])

chain = prompt | model_with_fallback

response = chain.invoke(
    {"query": "What is the capital of India?"},
    config={"callbacks": [StdOutCallbackHandler()]} 
)
print(response.content)