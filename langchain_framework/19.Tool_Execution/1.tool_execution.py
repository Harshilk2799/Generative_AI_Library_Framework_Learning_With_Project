from langchain_core.tools import tool

@tool
def multiply(a: int, b: int)-> int:
    """Given 2 numbers a and b this tool returns their product."""
    return a * b

result = multiply.invoke({"a": 10, "b": 12})
print(result)

# print(multiply.name)
# print(multiply.description)
# print(multiply.args)
# print(multiply.args_schema.model_json_schema())

# Note: Not every LLM has the capability to bind tools. Only a few LLMs have this capability.

# Tool Binding
from langchain_google_genai import ChatGoogleGenerativeAI
from decouple import config
from langchain_core.messages import HumanMessage

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=config("GOOGLE_GEMINI_API_KEY"), temperature=0)

llm_with_tools = model.bind_tools([multiply])
# print(llm_with_tools)


# Tool Calling  
result = llm_with_tools.invoke("Can you multiply 3 with 10.")
# print(result)


# Tool Execution
# final_result = multiply.invoke(result.tool_calls[0]['args'])
# print(final_result)


final_result = multiply.invoke(result.tool_calls[0])
print(final_result)