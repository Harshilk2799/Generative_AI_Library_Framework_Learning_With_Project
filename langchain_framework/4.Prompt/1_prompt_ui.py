from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import load_prompt
from decouple import config
import streamlit as st

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=config("GOOGLE_GEMINI_API_KEY"))

st.header("Research Tool")

# Static Prompt
# user_input = st.text_input("Enter your prompt: ")   

# if st.button("Summarize"):
#     response = model.invoke(user_input)
#     st.write(response.content)


# Dynamic Prompt
paper_input = st.selectbox("Select Research Paper Name", [
    "Attention is All you need", "BERT: Pre-training of Deep Bidirectional Transformers",
    "GPT-3: Language Models are few-shot learners", "Diffusion models beat GANs on image synthesis"
])

style_input = st.selectbox("Select Explanation Style", ["Beginner-Friendly", "Technical", "Code: Oriented", "Mathematical"])

length_input = st.selectbox("Select Explanation Length", ["Short (1-2 paragraph)", "Medium 3-5 paragraph", "Long (detailed explaination)"])

# Reusable template
template = load_prompt("template.json")

formattedChat = template.format(paper_input=paper_input, style_input=style_input, length_input=length_input)
print(formattedChat)

if st.button("Summarize"):
    result = model.invoke(formattedChat)
    print("Result: ", result.content)
    st.write(result.content)