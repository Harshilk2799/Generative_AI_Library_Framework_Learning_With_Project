from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from decouple import config

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    huggingfacehub_api_token=config("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm=llm)

response = model.invoke("What is Python Programming")
print(response.content)