from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
from decouple import config
from langchain_core.messages import HumanMessage, SystemMessage

# LLM
# llm = GoogleGenerativeAI(model="gemini-1.5-flash", api_key=config("GOOGLE_GEMINI_API_KEY"))

# result = llm.invoke("What is Python language?")
# print(result)


# Chat Models 

# Example 1
# chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=config("GOOGLE_GEMINI_API_KEY"))

# message = """
#     You are an experienced Python Developer and always respond with a python scrip.
#     If the question is not related to python programming, return  the answer out of your scope.

#     How to make a program of Fibonacci series without function with simple example.

# """
# result = chat.invoke(message)
# # print(result)
# print(result.content)


# Example 2

chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=config("GOOGLE_GEMINI_API_KEY"))

system_message = SystemMessage("""
    You are an experienced Python Developer and always respond with a python scrip.
    If the question is not related to python programming, return  the answer out of your scope.
    """)

human_message = HumanMessage("How to make a program of Fibonacci series without function with simple example.")

result = chat.invoke([system_message, human_message])
# print(result)
print(result.content)