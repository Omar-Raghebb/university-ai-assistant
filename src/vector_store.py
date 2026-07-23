import os
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src import config


def get_embedding_model(model_name=config.EMBEDDING_MODEL_NAME):
    # On memory-constrained GPUs (<=8-10GB), keep the embedding model on CPU
    # so the full VRAM budget is available for the LLM. Override via
    # config.EMBEDDING_DEVICE if you have headroom to spare (e.g. 16GB+ card).
    device = config.EMBEDDING_DEVICE
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(chunks, save_path=config.FAISS_INDEX_DIR):
    embeddings = get_embedding_model()
    index = FAISS.from_documents(chunks, embeddings)
    index.save_local(save_path)
    print(f"faiss index saved to {save_path} ({len(chunks)} chunks)")
    return index


def load_vector_store(save_path=config.FAISS_INDEX_DIR):
    if not os.path.isdir(save_path):
        raise FileNotFoundError(f"no index found at {save_path}, run build_vector_store first")
    embeddings = get_embedding_model()
    return FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)


if __name__ == "__main__":
    from src.document_loader import load_pdfs, split_documents

    docs = load_pdfs()
    chunks = split_documents(docs)
    build_vector_store(chunks)