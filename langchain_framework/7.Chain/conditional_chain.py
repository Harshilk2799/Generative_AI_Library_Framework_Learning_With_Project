from langchain_google_genai import ChatGoogleGenerativeAI
from decouple import config
from typing import Literal
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field

chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=config("GOOGLE_GEMINI_API_KEY"))

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Give the sentiment of the feedback")

parser = StrOutputParser()

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}",
    input_variables=["feedback"],
    partial_variables={"format_instruction": parser2.get_format_instructions()}
)

classifier_chain = prompt1 | chat_model | parser2

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

# Syntax
# branch_chain = RunnableBranch(
#     (condition1, chain1),
#     (condition2, chain2),
#     default_chain
# )

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | chat_model | parser),
    (lambda x: x.sentiment == "negative", prompt3 | chat_model | parser),
    RunnableLambda(lambda x:"Could not find sentiment")
)

chain = classifier_chain | branch_chain

print(chain.invoke({'feedback': 'This is a beautiful phone'}))

chain.get_graph().print_ascii()