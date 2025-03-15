from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from decouple import config

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=config("GOOGLE_GEMINI_API_KEY"))

# chat template
chat_template = ChatPromptTemplate([
    ("system", "You are a helpful customer support agent."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])


# load chat history
chat_history = []
with open("chat_history.txt") as f:
    chat_history.extend(f.readlines())
print("Chat History: ", chat_history)


# create prompt
result = chat_template.format_messages(chat_history= chat_history, query= "Where is my refund.")
print(result)

response = model.invoke(result)

print(response.content)