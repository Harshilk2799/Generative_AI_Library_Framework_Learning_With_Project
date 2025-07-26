from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from decouple import config

docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=config("GOOGLE_GEMINI_API_KEY"))

vectorstore = Chroma.from_documents(
    documents = docs,
    embedding = embedding_model,
    collection_name = "my_collection"
)

# Convert vectorstore into a retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",  # This enables mmr
    search_kwargs={'k': 3, "lambda_mult": 0.15}  # k = top results, lambda_mult = relevance-diversity balance
)
# k: Amount of documents to return (Default: 4)

# fetch_k: Amount of documents to pass to MMR algorithm (Default: 20)

# lambda_mult: Diversity of results returned by MMR;
# 1 for minimum diversity and 0 for maximum. (Default: 0.5)

# retriever = vectorstore.as_retriever(
#     search_type="mmr",  # This enables mmr
#     search_kwargs={'k': 3, "lambda_mult": 0.15, "fetch_k": 10}  # k = top results, lambda_mult = relevance-diversity balance
# )


# retriever = vectorstore.as_retriever(
#     search_type="similarity_score_threshold",  # This enables mmr
#     search_kwargs={'k': 3, "score_threshold": 0.15}  # k = top results, lambda_mult = relevance-diversity balance
# )

query = "What is Langchain?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n---- Result {i+1} ----")
    print(doc.page_content)