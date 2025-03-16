from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from decouple import config

chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=config("GOOGLE_GEMINI_API_KEY"))

# Example 1 - Prompt having No Input Variable
# Way 1
# noInputPrompt = PromptTemplate(input_variables=[], template="Tell me a Python Trick.")

# Way 2
# noInputPrompt = PromptTemplate.from_template("Tell me a Python Trick.")
# print(noInputPrompt)

# formattedNoInputPrompt = noInputPrompt.format()
# print("No Input Prompt: ",formattedNoInputPrompt)

# response = chat_model.invoke(formattedNoInputPrompt)
# print(response.content)




# Example 2 = Prompt having one input variable
# Way 1
# oneInputPrompt = PromptTemplate(input_variables=["language"], template="Tell me a {language} trick")

# Way 2
# oneInputPrompt = PromptTemplate.from_template("Tell me a {language} trick")

# formattedOneInputPrompt = oneInputPrompt.format(language="Java")
# print("One Input Prompt: ",formattedOneInputPrompt)
# response = chat_model.invoke(formattedOneInputPrompt)
# print(response.content)




# Example 3 = Prompt having Multiple input variable
# Way 1
# multipleInputPrompt = PromptTemplate(input_variables=["language", "topic"], template="Tell me a {language} {topic} trick")
# Way 2
multipleInputPrompt = PromptTemplate.from_template("Tell me a {language} {topic} trick")

formattedMultipleInputPrompt = multipleInputPrompt.format(language="Python", topic="Tuple")
print(formattedMultipleInputPrompt)

response = chat_model.invoke(formattedMultipleInputPrompt)
print(response.content)