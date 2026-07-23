import os
from dotenv import load_dotenv

load_dotenv()

# paths
DATA_DIR = "data"
FAISS_INDEX_DIR = "faiss_index"

# embedding model - multilingual so it handles Arabic + English
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"

# local LLM - 14B if VRAM allows, else 7B with better prompting
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
USE_4BIT_QUANTIZATION = True
LLM_MAX_NEW_TOKENS = 800
LLM_TEMPERATURE = 0.1  # Lower = more deterministic, better for facts

# text splitting - smaller chunks for better table preservation
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 250

# retrieval
TOP_K_RESULTS = 10  # Retrieve more, re-rank later
RERANK_TOP_N = 4     # How many chunks survive the cross-encoder rerank

# deployment
NGROK_TOKEN = os.getenv("NGROK_TOKEN")
API_KEY = os.getenv("API_KEY", "secret123")

EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto")