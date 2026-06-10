from langchain_community.tools import DuckDuckGoSearchRun
from decouple import config
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq.chat_models import ChatGroq
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

checkpointer = MemorySaver()

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

while True:

    user_input = input("You: ")

    if user_input.lower() in ["exit", "bye"]:
        print("Goodbye!")
        break

    agent = create_agent(
        model=model,
        tools=[search_tool, add, sub, multiply],
        system_prompt=system_prompt,
        checkpointer=checkpointer
    )

    response = agent.invoke({"messages": [("human", user_input)]}, config={"configurable": {"thread_id": "thread-1"}})
    # print("Response: ", response)
    print("AI: ",response["messages"][-1].content)

