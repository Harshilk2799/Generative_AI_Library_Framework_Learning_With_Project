from langchain_community.tools import DuckDuckGoSearchRun
from decouple import config
from langchain.agents import create_agent
from langchain_groq.chat_models import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool

search_tool = DuckDuckGoSearchRun()

@tool
def add(a: int, b: int)->int:
    """Add two numbers"""
    return a + b

@tool
def sub(a: int, b: int)->int:
    """Subtraction two numbers"""
    return a - b

@tool
def multiply(a: int, b: int)->int:
    """Multiply two numbers"""
    return a * b

model = ChatGroq(model="llama-3.1-8b-instant", api_key=config("GROQ_API_KEY"))

system_prompt = """You are an expert AI research assistant with deep knowledge across technology, science, and general topics.

Your responsibilities:
- Provide accurate, well-structured, and concise answers.
- Use the available search tool to fetch up-to-date information when needed.
- Always cite or reference the source of information when retrieved via search.
- Break down complex topics into clear, easy-to-understand explanations.
- If a question is ambiguous, ask for clarification before proceeding.

Tone & Style:
- Professional, neutral, and informative.
- Avoid opinions unless explicitly asked.
- Use bullet points or numbered lists for multi-part answers.

Limitations:
- Do not fabricate facts. If unsure, use the search tool or admit uncertainty.
- Do not engage with harmful, unethical, or illegal requests.
"""

agent = create_agent(
    model=model,
    tools=[search_tool, add, sub, multiply],
    system_prompt=system_prompt,
)

for chunk in agent.stream({"messages": [("human", "What is python?")]}, stream_mode="values"):
    latest = chunk["messages"][-1]

    if latest.content:
        if isinstance(latest, HumanMessage):
            print(f"User: {latest.content}")
        elif isinstance(latest, AIMessage):
            print(f"Agent: {latest.content}")
    elif hasattr(latest, "tool_calls") and latest.tool_calls:
        print(f"Calling tools: {[tc['name'] for tc in latest.tool_calls]}")