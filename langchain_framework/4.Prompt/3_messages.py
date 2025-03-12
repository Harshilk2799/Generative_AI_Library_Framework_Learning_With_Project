from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from decouple import config

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=config("GOOGLE_GEMINI_API_KEY"))

# Example 1
# system_message = SystemMessage("""
#     You are an experienced Python Developer and always respond with a python script.
#     If the question is not related to python programming, return  the answer out of your scope.
#     """)

# human_message = HumanMessage("How to make a program of Fibonacci series without function with simple example.")

# result = model.invoke([system_message, human_message])
# # print(result)
# print(result.content)


# Example 2

messages = [
    SystemMessage(content="You are a helpful assistant!"),
    HumanMessage(content="Tell me about langchain")
]
result = model.invoke(messages)
# print("Answer: ", result.content)
messages.append(AIMessage(content=result.content))
print("Message: ", messages)
