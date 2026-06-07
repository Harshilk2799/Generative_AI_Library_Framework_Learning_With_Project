from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from decouple import config
from langchain_milvus import Milvus
from uuid import uuid4

documents = [
    Document(page_content="Langchain helps developers build LLM application easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI providees powerful embedding models.")
]

embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=config("GOOGLE_GEMINI_API_KEY"))

vectorstore= Milvus(
    embedding_function=embedding_model,
    connection_args={
        "uri": config("MILVUS_ENDPOINT"), "token": config("MILVUS_TOKEN"), "db_name": "milvusdb"
    },
    index_params={
        "index_type": "FLAT", "metric_type": "L2"
    },
    consistency_level="Strong",
    drop_old=False,
    collection_name="milvusCollection4",
    # auto_id=True
)

uuids = [str(uuid4()) for _ in range(len(documents))]
vectorstore.add_documents(documents=documents, ids=uuids)
# Convert vectorstore into a retriever
retriever = vectorstore.as_retriever(search_kwargs={'k': 2})

query = "What is Chroma used for?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n---- Result {i+1} ----")
    print(doc.page_content)