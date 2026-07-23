"""
Example client for talking to the deployed server (deploy_server.py),
same pattern as NGROK_Send_Prompt_Local.ipynb. Run this from your own
laptop while the model itself runs on Kaggle's GPU behind ngrok.
"""

import requests

NGROK_URL = "ADD_NGROK_URL_HERE"  # e.g. https://xxxx.ngrok-free.app
API_KEY = "secret123"  # must match config.API_KEY on the server side


def ask(question: str):
    response = requests.post(
        f"{NGROK_URL}/ask",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"question": question},
    )
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    result = ask("كام ساعة معتمدة مطلوبة للتخرج؟")
    print("Answer:", result["answer"])
    print("Source page:", result["source_page"])
    print("Confidence:", result["confidence"])
