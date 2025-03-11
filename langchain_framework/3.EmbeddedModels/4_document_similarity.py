from langchain_google_genai import GoogleGenerativeAIEmbeddings
from decouple import config
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=config("GOOGLE_GEMINI_API_KEY"))

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known for his elegant batting and record-breaking double centuries.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "Tell me about virat kohli"

document_embedding = embedding.embed_documents(documents)

query_embedding = embedding.embed_query(query)

# print(cosine_similarity([query_embedding], document_embedding))

scores = cosine_similarity([query_embedding], document_embedding)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x:x[1])[-1]

print("Query: ", query)
print(documents[index])
print("Similarity Score: ", score)