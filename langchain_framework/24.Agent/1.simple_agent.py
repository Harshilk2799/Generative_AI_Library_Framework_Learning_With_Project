from decouple import config
from langchain.agents import create_agent
from langchain_groq.chat_models import ChatGroq

model = ChatGroq(model="llama-3.1-8b-instant", api_key=config("GROQ_API_KEY"))

agent = create_agent(model=model)

response = agent.invoke({"messages": [("human", "What is Python?")]})

print(response["messages"][-1].content)