from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from decouple import config
from langchain_milvus import Milvus
from uuid import uuid4
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

# Recreate the document objects from the previous data
docs = [
    Document(page_content=(
        """The Grand Canyon is one of the most visited natural wonders in the world.
        Photosynthesis is the process by which green plants convert sunlight into energy.
        Millions of tourists travel to see it every year. The rocks date back millions of years."""
    ), metadata={"source": "Doc1"}),

    Document(page_content=(
        """In medieval Europe, castles were built primarily for defense.
        The chlorophyll in plant cells captures sunlight during photosynthesis.
        Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
    ), metadata={"source": "Doc2"}),

    Document(page_content=(
        """Basketball was invented by Dr. James Naismith in the late 19th century.
        It was originally played with a soccer ball and peach baskets. NBA is now a global league."""
    ), metadata={"source": "Doc3"}),

    Document(page_content=(
        """The history of cinema began in the late 1800s. Silent films were the earliest form.
        Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
        Modern filmmaking involves complex CGI and sound design."""
    ), metadata={"source": "Doc4"})
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
    collection_name="milvusCollectionNew3",
    # auto_id=True
)

uuids = [str(uuid4()) for _ in range(len(docs))]
vectorstore.add_documents(documents=docs, ids=uuids)

base_retriever = vectorstore.as_retriever(search_kwargs={"k":5})

compression_retriever = ContextualCompressionRetriever(
    base_retriever = base_retriever,
    base_compressor = LLMChainExtractor.from_llm(GoogleGenerativeAI(model="gemini-2.5-flash",google_api_key=config("GOOGLE_GEMINI_API_KEY")))
)

query = "What is photosynthesis?"
compressed_results = compression_retriever.invoke(query)

for i, doc in enumerate(compressed_results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)