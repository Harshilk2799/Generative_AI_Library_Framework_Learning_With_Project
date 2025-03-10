from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

document_vector = embedding.embed_documents([
    "What is Python Programming",
    "What is Langchain Framework"
])

# print("Document Vector: ",document_vector)

query_vector = embedding.embed_query("What is Python Programming")

print("Query Vector: ",query_vector)