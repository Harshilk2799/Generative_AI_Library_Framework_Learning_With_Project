from langchain_community.document_loaders import CSVLoader

loader = CSVLoader("Social_Network_Ads.csv")

docs = loader.load()

# print(len(docs))

# print(docs[5].page_content)
print(docs[5].metadata)