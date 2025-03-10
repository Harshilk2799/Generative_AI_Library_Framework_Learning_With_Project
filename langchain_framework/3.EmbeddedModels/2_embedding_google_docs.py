from langchain_google_genai import GoogleGenerativeAIEmbeddings
from decouple import config

embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=config("GOOGLE_GEMINI_API_KEY"))

# result = embedding.embed_documents([
#     "What is Python Programming",
#     "What is Langchain Framework"
# ], output_dimensionality=20)

result = embedding.embed_documents([
    "What is Python Programming",
    "What is Langchain Framework"
])

print(result)
# print(len(result[0]))
# print(len(result[1]))