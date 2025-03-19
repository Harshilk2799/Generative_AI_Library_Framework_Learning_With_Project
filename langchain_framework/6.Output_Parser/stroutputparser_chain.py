from langchain_google_genai import ChatGoogleGenerativeAI
from decouple import config
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=config("GOOGLE_GEMINI_API_KEY"))

# 1st Prompt -> Detailed report
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

# 2nd Prompt -> Summary
template2 = PromptTemplate(
    template="Write a 5 line summary on the following text. /n {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = template1 | chat_model | parser | template2 | chat_model | parser

result = chain.invoke({"topic": "Black Hole"})

print("Result: ", result)