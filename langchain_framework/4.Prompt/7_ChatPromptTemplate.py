from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from decouple import config

chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=config("GOOGLE_GEMINI_API_KEY"))

# Example 1 = Message Prompt Template as tuple
# chatPrompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
#     ("human", "{text}")
# ])

# formattedChatPrompt = chatPrompt.format_messages(input_language="English", output_language="Hindi", text="How are you?")
# print(formattedChatPrompt)

# response = chat_model.invoke(formattedChatPrompt)
# print(response.content)




# Example 2 = Using Message Classes
system_template = "You are a helpful assistant that translates {input_language} to {output_language}."
human_template = "{text}"

chatPrompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(human_template)
])

formattedChatPrompt = chatPrompt.format_messages(
    input_language="English",
    output_language="Gujarati",
    text="How are you?"
)

response = chat_model.invoke(formattedChatPrompt)
print(response.content)