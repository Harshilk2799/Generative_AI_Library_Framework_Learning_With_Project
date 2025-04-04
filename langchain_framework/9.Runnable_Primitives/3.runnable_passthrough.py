from langchain_google_genai import ChatGoogleGenerativeAI
from decouple import config
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence, RunnablePassthrough

# Simple Example
# passthrough = RunnablePassthrough()

# print(passthrough.invoke(2))
# print(passthrough.invoke({"name": "harshil"}))


chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=config("GOOGLE_GEMINI_API_KEY"))

prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Explain the following joke - {text}",
    input_variables=["text"]
)

parser = StrOutputParser()


joke_generator_chain = RunnableSequence(prompt1, chat_model, parser)

parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "explaination": RunnableSequence(prompt2, chat_model, parser)
})

final_chain = RunnableSequence(joke_generator_chain, parallel_chain)

response = final_chain.invoke({"topic": "cricket"})
print(response)