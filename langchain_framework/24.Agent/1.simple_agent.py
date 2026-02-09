from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from decouple import config
from langsmith import Client

search_tool = DuckDuckGoSearchRun()

# Pull the ReAct prompt from Langchain hub
client = Client()
prompt = client.pull_prompt("hwchase17/react")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=config("GOOGLE_GEMINI_API_KEY"))

# Create the ReAct agent manually with the pulled prompt
agent = create_react_agent(
    llm=model,
    tools=[search_tool],
    prompt=prompt
)

# Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool],
    verbose=True
)

response = agent_executor.invoke({"input": "What is python?"})

print(response["output"])

