from langchain_google_genai import ChatGoogleGenerativeAI
from decouple import config
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda

# Simple Example
# def word_counter(text):
#     return len(text.split())

# runnable_word_counter = RunnableLambda(word_counter)

# print(runnable_word_counter.invoke("Hello! My name is Harshil."))


chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=config("GOOGLE_GEMINI_API_KEY"))

prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

joke_generator_chain = RunnableSequence(prompt1, chat_model, parser)

def word_counter(text):
    return len(text.split())

# First Way
parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "word_count": RunnableLambda(word_counter)
})

# Second Way
# parallel_chain = RunnableParallel({
#     "joke": RunnablePassthrough(),
#     "word_count": RunnableLambda(lambda x: len(x.split()))
# })


final_chain = RunnableSequence(joke_generator_chain, parallel_chain)

response = final_chain.invoke({"topic": "cricket"})
print(response)