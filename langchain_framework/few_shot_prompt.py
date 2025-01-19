from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
from decouple import config
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# LLM
# Few Shot Prompt Example
# llm = GoogleGenerativeAI(model="gemini-1.5-flash", api_key=config("GOOGLE_GEMINI_API_KEY"))

# examples = [
#     {
#         "question": "What is 3 × 4?",
#         "solution": """Let's solve this step by step:
#             1. We need to add 3 four times
#             2. 3 + 3 + 3 + 3 = 12
#             Therefore, 3 × 4 = 12"""
#     },
#     {
#         "question": "What is 5 × 6?",
#         "solution": """Let's solve this step by step:
#             1. We need to add 5 six times
#             2. 5 + 5 + 5 + 5 + 5 + 5 = 30
#             Therefore, 5 × 6 = 30"""
#     }
# ]

# example_prompt = PromptTemplate(input_variables=["question", "solution"],
#                                 template="{question}{solution}")

# few_shot_prompt = FewShotPromptTemplate(examples=examples, 
#                                         example_prompt=example_prompt,
#                                         # prefix="You are a math teacher helping students understand multiplication. Solve the following problem step by step:",
#                                         suffix="Question: {question}\nSolution:",
#                                         input_variables=["question"]
#                                     )

# formattedPrompt = few_shot_prompt.format(question="What is 7 × 8?")
# # print(formattedPrompt)


# response = llm.invoke(formattedPrompt)
# print(response)


# Chat Model
# Few Shot Prompt Example
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=config("GOOGLE_GEMINI_API_KEY"))

examples = [
    {
        "question": "What is 3 × 4?",
        "solution": """Let's solve this step by step:
            1. We need to add 3 four times
            2. 3 + 3 + 3 + 3 = 12
            Therefore, 3 × 4 = 12"""
    },
    {
        "question": "What is 5 × 6?",
        "solution": """Let's solve this step by step:
            1. We need to add 5 six times
            2. 5 + 5 + 5 + 5 + 5 + 5 = 30
            Therefore, 5 × 6 = 30"""
    }
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{question}"),
    ("ai", "{solution}")
])

few_shot_prompt = FewShotChatMessagePromptTemplate(examples=examples, 
                                        example_prompt=example_prompt)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful Math problem solver."),
    few_shot_prompt,
    ("human", "{question}")
])

formattedPrompt = final_prompt.format(question="What is 7 × 8?")
# print(formattedPrompt)


response = chat.invoke(formattedPrompt)
print(response.content)