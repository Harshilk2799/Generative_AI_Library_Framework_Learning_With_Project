from langchain_text_splitters import RecursiveCharacterTextSplitter

# Example 1

# text = """
# My name is Harshil
# I am 25 years old

# I live in Ahmedabad
# How are you
# """

# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=10,
#     chunk_overlap=0
# )

# Example 2

text = """
Space exploration has led to incredible scientific discoveries. From landing on the Moon to exploring Mars, humanity continues to push the boundaries of what’s possible beyond our planet.

These missions have not only expanded our knowledge of the universe but have also contributed to advancements in technology here on Earth. Satellite communications, GPS, and even certain medical imaging techniques trace their roots back to innovations driven by space programs.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0,
)

chunks = splitter.split_text(text)

print(chunks)