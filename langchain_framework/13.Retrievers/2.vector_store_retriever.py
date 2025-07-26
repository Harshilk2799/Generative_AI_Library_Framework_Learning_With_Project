from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from decouple import config

documents = [
    Document(page_content="Langchain helps developers build LLM application easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI providees powerful embedding models.")
]

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=config("GOOGLE_GEMINI_API_KEY"))

vectorstore = Chroma.from_documents(
    documents = documents,
    embedding = embedding_model,
    collection_name = "my_collection"
)

# Convert vectorstore into a retriever
retriever = vectorstore.as_retriever(search_kwargs={'k': 2})

query = "What is Chroma used for?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n---- Result {i+1} ----")
    print(doc.page_content)