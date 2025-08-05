# Method 2: Using StructureTool

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


def multiply_func(a, b)-> int:
    return a*b

class MultiplyInput(BaseModel):
    a: int = Field(..., description="The first number to add")
    b: int = Field(..., description="The second number to add")

multiply_tool = StructuredTool.from_function(
    func=multiply_func,
    name="multiply",
    description="Multiply two numbers",
    args_schema=MultiplyInput
)

result = multiply_tool.invoke({"a":10, "b": 24})
print(result)
print(multiply_tool.name)
print(multiply_tool.description)
print(multiply_tool.args)