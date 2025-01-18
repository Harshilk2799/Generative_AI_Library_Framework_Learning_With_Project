from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
from decouple import config
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# LLM
# llm = GoogleGenerativeAI(model="gemini-1.5-flash", api_key=config("GOOGLE_GEMINI_API_KEY"))

# =================== Type 1: String PromptTemplates ===================

# Example 1 = Prompt having no input variable
# Way 1
# noInputPrompt = PromptTemplate(input_variables=[], template="Tell me a Python trick")
# Way 2
# noInputPrompt = PromptTemplate.from_template("Tell me a Python trick")

# formattedNoInputPrompt = noInputPrompt.format()
# print(formattedNoInputPrompt)

# response = llm.invoke(formattedNoInputPrompt)
# print(response)



# Example 2 = Prompt having one input variable
# Way 1
# oneInputPrompt = PromptTemplate(input_variables=["language"], template="Tell me a {language} trick")
# Way 2
# oneInputPrompt = PromptTemplate.from_template("Tell me a {language} trick")

# formattedOneInputPrompt = oneInputPrompt.format(language="Java")
# print(formattedOneInputPrompt)

# response = llm.invoke(formattedOneInputPrompt)
# print(response)



# Example 3 = Prompt having Multiple input variable
# Way 1
# multipleInputPrompt = PromptTemplate(input_variables=["language", "topic"], template="Tell me a {language} {topic} trick")
# Way 2
# multipleInputPrompt = PromptTemplate.from_template("Tell me a {language} {topic} trick")

# formattedMultipleInputPrompt = multipleInputPrompt.format(language="Python", topic="Tuple")
# print(formattedMultipleInputPrompt)

# response = llm.invoke(formattedMultipleInputPrompt)
# print(response)



# Example 4 = No input variable manually
# template = "Tell me a {language} with this {topic} trick"
# promptTemplate = PromptTemplate.from_template(template)

# prompt = promptTemplate.format(language="Python", topic="List")
# print(prompt)

# response = llm.invoke(prompt)
# print(response)



# =================== Type 2: ChatPromptTemplates ===================
# Chat Models 
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder

# Example 1 = Message Prompt Template as tuple
chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=config("GOOGLE_GEMINI_API_KEY"))

# chatPrompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
#     ("human", "{text}")
# ])
# formattedChatPrompt = chatPrompt.format_messages(
#     input_language="English",
#     output_language="Gujarati",
#     text="How are you?"
# )

# response = chat.invoke(formattedChatPrompt)
# print(response.content)



# Example 2 = Using Message Classes
# system_template = "You are a helpful assistant that translates {input_language} to {output_language}."
# human_template = "{text}"

# chatPrompt = ChatPromptTemplate.from_messages([
#   SystemMessagePromptTemplate.from_template(system_template),
#   HumanMessagePromptTemplate.from_template(human_template)  
# ])

# formattedChatPrompt = chatPrompt.format_messages(
#     input_language="English",
#     output_language="Gujarati",
#     text="How are you?"
# )

# response = chat.invoke(formattedChatPrompt)
# print(response.content)


