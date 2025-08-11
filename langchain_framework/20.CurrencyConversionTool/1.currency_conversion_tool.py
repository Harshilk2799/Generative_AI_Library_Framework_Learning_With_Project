from langchain_core.tools import tool, InjectedToolArg
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from typing import Annotated
from decouple import config
import requests
import json

# Tool Create
@tool
def get_conversion_factor(base_currency: str, target_currency: str)->float:
    """
    This function fetches the currency conversion factor between a given base currency and target currency.
    """
    url = f"https://v6.exchangerate-api.com/v6/69187d08c39d0a489ad69c7c/pair/{base_currency}/{target_currency}"

    response = requests.get(url)

    return response.json()
# print(get_conversion_factor.invoke({"base_currency": "USD", "target_currency": "INR"}))


# LLM, do not try to fill this argument. I (the developer/runtime) will inject this value after running earlier tools.
@tool
def convert(base_currency_value: int, conversion_rate: float) ->float:
    """
    Given a currency conversion rate this function calculates the target currency value from a given base currency value.
    """
    return base_currency_value * conversion_rate

# print(convert.invoke({"base_currency_value": 10, "conversion_rate": 87.15}))


# Tool Binding
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=config("GOOGLE_GEMINI_API_KEY"), temperature=0)

llm_with_tools = llm.bind_tools([get_conversion_factor, convert])

messages = [HumanMessage(content="What is the conversion factor between USD and INR, and based on that can you convert 10 USD to INR")]

ai_message = llm_with_tools.invoke(messages)

messages.append(ai_message)

print(ai_message.tool_calls)

for tool_call in ai_message.tool_calls:
    # Execute the 1st tool and get the value of conversion rate.
    if tool_call["name"] == "get_conversion_factor":
        tool_message1 = get_conversion_factor.invoke(tool_call)
        # Fetch this conversion rate
        conversion_rate = json.loads(tool_message1.content)["conversion_rate"]
        print(conversion_rate)
        # Append this tool message to messages list 
        messages.append(tool_message1)

    # Execute the 2nd tool using conversion rate from tool 1
    if tool_call["name"] == "convert":
        # Fetch the current args
        tool_call["args"]["conversion_rate"] = conversion_rate
        tool_message2 = convert.invoke(tool_call)
        messages.append(tool_message2)

result = llm_with_tools.invoke(messages).content
print(result)