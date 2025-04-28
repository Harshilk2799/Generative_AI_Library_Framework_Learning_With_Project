from langchain.text_splitter import CharacterTextSplitter

text = """
The SentenceTransformersTokenTextSplitter is a specialized text splitter for use with the sentence-transformer models. The default behaviour is to split the text into chunks that fit the token window of the sentence transformer model that you would like to use.
To split text and constrain token counts according to the sentence-transformers tokenizer, instantiate a SentenceTransformersTokenTextSplitter. You can optionally specify:
"""

splitter = CharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=0,
    separator=''
)

result = splitter.split_text(text)

print(result)