from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(path="books", glob="*.pdf", loader_cls=PyPDFLoader)

# docs = loader.load()
# print(docs)
# print(len(docs))

# print(docs[1].page_content)
# print(docs[1].metadata)



# Example 1 = load()
# docs = loader.load()
# for doc in docs:
#     print(doc.metadata)

# Example 2 = lazy_load()
docs = loader.lazy_load()

for doc in docs:
    print(doc.metadata)