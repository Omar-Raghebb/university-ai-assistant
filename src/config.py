import os
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "data"
FAISS_INDEX_DIR = "faiss_index"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
USE_4BIT_QUANTIZATION = True
LLM_MAX_NEW_TOKENS = 800
LLM_TEMPERATURE = 0.1  
CHUNK_SIZE = 600
CHUNK_OVERLAP = 200
TOP_K_RESULTS = 15  
RERANK_TOP_N = 5     
NGROK_TOKEN = os.getenv("NGROK_TOKEN")
API_KEY = os.getenv("API_KEY", "secret123")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto")