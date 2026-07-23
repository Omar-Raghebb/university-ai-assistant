"""
Exposes the RAG pipeline as a public API using FastAPI + ngrok,
same pattern as NGROK_Kaggle.ipynb. Run this on Kaggle/Colab (where
you have a free GPU), and call it from anywhere using the client
pattern in NGROK_Send_Prompt_Local.ipynb -- e.g. from a lightweight
Streamlit app running on your own laptop.

Usage (in a Kaggle notebook cell):
    !pip install fastapi uvicorn pyngrok -q
    %run deploy_server.py
"""

import threading
import socket
import time

from fastapi import FastAPI, Request, HTTPException
import uvicorn
from pyngrok import ngrok, conf

from src import config
from src.vector_store import load_vector_store
from src.chain import answer_question

app = FastAPI()
_vector_store = None


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = load_vector_store()
    return _vector_store


@app.post("/ask")
async def ask(req: Request):
    if req.headers.get("authorization") != f"Bearer {config.API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    data = await req.json()
    question = data.get("question", "")
    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question' field")

    result = answer_question(get_vector_store(), question)
    return result


def _free_port():
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def start_server():
    if not config.NGROK_TOKEN:
        raise ValueError("NGROK_TOKEN not set. Add it to your .env file.")

    port = _free_port()
    conf.get_default().auth_token = config.NGROK_TOKEN
    public_url = ngrok.connect(port).public_url
    print("Your public URL:", public_url)
    print(f"Send POST requests to: {public_url}/ask")
    print(f"Header: Authorization: Bearer {config.API_KEY}")

    def run():
        uvicorn.run(app, host="0.0.0.0", port=port)

    threading.Thread(target=run, daemon=True).start()
    time.sleep(1)
    return public_url


if __name__ == "__main__":
    start_server()
    # Keep the main thread alive so the background server thread keeps running.
    while True:
        time.sleep(60)
