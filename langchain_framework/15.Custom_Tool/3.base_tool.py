# Method 3: Using BaseTool Class

from langchain_core.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field


# arg schema using pydantic
class MultiplyInput(BaseModel):
    a: int = Field(..., description="The first number to add")
    b: int = Field(..., description="The second number to add")


class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiply two numbers"

    args_schema: Type[BaseModel] = MultiplyInput

    def _run(self, a, b)->int:
        return a*b

multiply_tool = MultiplyTool()

result = multiply_tool.invoke({"a": 10, "b": 42})

print(result)

print(multiply_tool.name)
print(multiply_tool.description)
print(multiply_tool.args)