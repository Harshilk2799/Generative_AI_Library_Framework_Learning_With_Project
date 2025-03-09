from langchain_google_genai import GoogleGenerativeAIEmbeddings
from decouple import config

embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=config("GOOGLE_GEMINI_API_KEY"))

# By default, the length of the embedding vector will be 768 for "models/embedding-001".

# result = embedding.embed_query("What is Python Programming", output_dimensionality=20)
result = embedding.embed_query("What is Python Programming")

print(len(result))