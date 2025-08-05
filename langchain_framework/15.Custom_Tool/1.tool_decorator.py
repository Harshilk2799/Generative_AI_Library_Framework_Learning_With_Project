from langchain_core.tools import tool

# Method 1: Using @tool decorator

# Step 1: Create a function 
# def multiply(a, b):
#     """Multiply two numbers"""
#     return a * b 

# Step 2: Add Type Hints
# def multiply(a: int, b: int)->int:
#     """Multiply two numbers"""
#     return a * b 

# Step 3: Add tool decorator
@tool
def multiply(a: int, b: int)->int:
    """Multiply two numbers"""
    return a * b 

result = multiply.invoke({"a": 5, "b": 10})
print(result)

print()

print(multiply.name)
print(multiply.description)
print(multiply.args)
print(multiply.args_schema.model_json_schema()) # Tool schema to send to theu LLM
print("="*20)

# from langchain_community.tools import DuckDuckGoSearchRun
# search_tool = DuckDuckGoSearchRun()
# result = search_tool.invoke("Python")

# print(result)
# print(search_tool.name)
# print(search_tool.description)
# print(search_tool.args)