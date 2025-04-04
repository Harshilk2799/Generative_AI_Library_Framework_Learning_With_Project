from langchain_google_genai import ChatGoogleGenerativeAI
from decouple import config
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnablePassthrough, RunnableBranch

chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=config("GOOGLE_GEMINI_API_KEY"))

prompt1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Summarize the following text \n {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt1, chat_model, parser)

# Syntax
# RunnableBranch(
#     (condition, Runnable),
#     default: A Runnable to run if no condition is met.
# )

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 300, RunnableSequence(prompt2, chat_model, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

response = final_chain.invoke({"topic": "Russia vs Ukrain"})
print(response)