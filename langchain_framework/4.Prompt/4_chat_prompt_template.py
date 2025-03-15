from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from decouple import config

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=config("GOOGLE_GEMINI_API_KEY"))

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {domain} expert."),
    ("human", "Explain in simple terms, What is {topic}.")
])

prompt = chat_template.format_messages(domain="python", topic= "Tuple")

response = model.invoke(prompt)

print("Response: ", response.content)