from langchain_google_genai import ChatGoogleGenerativeAI
from decouple import config

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=config("GOOGLE_GEMINI_API_KEY"), temperature=0)

response = model.invoke("Write a 5 lin poem on cricket")

# print(response)
print(response.content)