from langchain_google_genai import ChatGoogleGenerativeAI
from decouple import config
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=config("GOOGLE_GEMINI_API_KEY"))

prompt1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a 5 pointer summary from the following text. /n {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = prompt1 | chat_model | parser | prompt2 | chat_model | parser

result = chain.invoke({"topic": "Black Hole"})

print("Result: ", result)

chain.get_graph().print_ascii()