from decouple import config
from langchain.agents import create_agent
from langchain_groq.chat_models import ChatGroq
from langchain_core.messages import SystemMessage

model = ChatGroq(model="llama-3.1-8b-instant", api_key=config("GROQ_API_KEY"))

# Method 1: String prompt (simplest)
# Method 2: SystemMessage (structured)

# When to use each method 

# 1. string = Simple, static agent.
# 2. SystemMessage = Production chat agent.


# Method 1 = The simplest approach - direct string instruction to the agent.
# agent = create_agent(
#     model=model,
#     system_prompt="You are a helpful assistant. Be concise and accurate in your response."
# )

# Method 2 = Using Langchain's SystemMessage class for more structured prompt handling.
agent = create_agent(
    model=model,
    system_prompt=SystemMessage(content="You are a helpful assistant. Be concise and accurate in your response.")
)

response = agent.invoke({"messages": [("human", "What is Python ?")]})
print(response["messages"][-1].content)