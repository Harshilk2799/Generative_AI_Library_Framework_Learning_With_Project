from langchain_community.tools import DuckDuckGoSearchRun
from decouple import config
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq.chat_models import ChatGroq
from pydantic import BaseModel, Field
from typing import List

class ContactInfo(BaseModel):
    name: str 
    email: str 
    phone: str 

class ListofContactInfo(BaseModel):
    contacts: List[ContactInfo] = Field(..., description="List of contact information.")

search_tool = DuckDuckGoSearchRun()


# model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=config("GOOGLE_GEMINI_API_KEY"), temperature=0)

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


# STRATEGY 1 — ProviderStrategy
# => Models that natively support structured output: OpenAI GPT-4o, Claude, Gemini


# => Option A: Pass schema directly — LangChain auto-selects ProviderStrategy
# agent = create_agent(
#     model=model,
#     tools=[search_tool],
#     system_prompt=system_prompt,
#     response_format=ListofContactInfo   # Auto-detects → uses ProviderStrategy for Gemini
# )


# Option B: Explicit ProviderStrategy with strict=True (OpenAI supports strict mode)
# agent = create_agent(
#     model=model,
#     tools=[search_tool],
#     system_prompt=system_prompt,
#     response_format=ProviderStrategy(schema=ListofContactInfo, strict=True)   # Auto-detects → uses ProviderStrategy for Gemini
# )



# STRATEGY 2 — ToolStrategy
# => Models that do NOT support native structured output — use tool calling instead

agent = create_agent(
    model=model,
    tools=[search_tool],
    system_prompt=system_prompt,
    response_format=ToolStrategy(
        schema=ListofContactInfo,
        tool_message_content="List of Contact info generated successfully!",
        handle_errors=True  # Auto-retry on validation failures
    )
)

response = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Extract contact info from: Harshil khatri, harshil@gmail.com, 7984944."
    }]
})
print(response)
print("\n\n")
print(response["structured_response"])
print(response["structured_response"].model_dump_json())
