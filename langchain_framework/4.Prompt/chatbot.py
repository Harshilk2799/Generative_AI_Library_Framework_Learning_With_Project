from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from decouple import config

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=config("GOOGLE_GEMINI_API_KEY"))

# while True:
#     user_input = input("You: ")
#     if user_input == "exit":
#         break
#     response = model.invoke(user_input)
#     print("AI: ", response.content)


# Below Code: User, and AI message store into chat_history but don't know which message belong to user, ai.
# chat_history = []
# while True:
#     user_input = input("You: ")
#     chat_history.append(user_input)
#     if user_input == "exit":
#         break
#     print("Chat History: ", chat_history)
#     response = model.invoke(chat_history)
#     chat_history.append(response.content)
#     print("AI: ", response.content)
# print(chat_history)


chat_history = [
    SystemMessage(content="You are a helpful assistant!")
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == "exit":
        break
    print("Chat History: ", chat_history)
    response = model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    print("AI: ", response.content)
print(chat_history)