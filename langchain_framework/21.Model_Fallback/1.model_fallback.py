# with_fallbacks() — Backup Model if Primary Fails

# What?
# -> If your primary model fails, it automatically switches to a backup model.

# Why?
# -> Prevents downtime.

# What happens?
# -> Tries Gemini first
# -> If Gemini fails → Groq is used automatically

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from decouple import config

primary_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=config("GOOGLE_GEMINI_API_KEY"))

fallback_model1 = ChatGroq(model="deepseek-r1-distill-llama-70b", api_key=config("GROQ_API_KEY"))

# Way 1
model_with_fallback = primary_model.with_fallbacks([fallback_model1])

prompt = ChatPromptTemplate([
    ("system", "You are a helpful assistant."),
    ("human", "{query}")
])

chain = prompt | model_with_fallback

response = chain.invoke({"query": "What is the capital of India?"})
print(response.content)