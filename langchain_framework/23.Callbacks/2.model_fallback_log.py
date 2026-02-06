
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from decouple import config
from langchain_core.callbacks import BaseCallbackHandler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrackingCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        model_name = serialized.get("id", ["unknown"])[-1]
        logger.info(f"🚀 Starting LLM: {model_name}")
    
    def on_llm_error(self, error, **kwargs):
        logger.error(f"❌ LLM Error: {error}")
        logger.info("🔄 Falling back to next model...")

fallback_model1 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    api_key=config("GOOGLE_GEMINI_API_KEY")
)
primary_model = ChatGroq(
    model="deepseek-r1-distill-llama-70b", 
    api_key=config("GROQ_API_KEY")
)

model_with_fallback = primary_model.with_fallbacks([fallback_model1])

prompt = ChatPromptTemplate([
    ("system", "You are a helpful assistant."),
    ("human", "{query}")
])

chain = prompt | model_with_fallback

response = chain.invoke(
    {"query": "What is the capital of India?"}, 
    config={"callbacks": [BaseCallbackHandler()]}
)
print(response.content)