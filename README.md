# 🚀 [Tips Hindawi](https://www.tipshindawi.com/) Challenge (June–July) 2026

> 🏆 This repository is my official submission for the [ **Tips Hindawi** ](https://www.tipshindawi.com/) **Challenge (June–July) 2026**.
<div align="center">

# University AI Assistant

### RAG System for University Regulations & Graduation Requirements

[![Tips Hindawi Challenge](https://img.shields.io/badge/Tips%20Hindawi-Challenge%202026-blueviolet?style=for-the-badge)](https://www.tipshindawi.com/)
[![Edrak for AI](https://img.shields.io/badge/Organization-Edrak%20for%20AI-orange?style=for-the-badge)](https://edrak4ai.com/en)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Mistral-7B](https://img.shields.io/badge/LLM-Mistral--7B--Instruct-FF7000?style=flat-square)](https://mistral.ai/)
[![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS-4267B2?style=flat-square)](https://github.com/facebookresearch/faiss)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Educational%20%2F%20Portfolio-lightgrey?style=flat-square)](#license)

</div>

---

## Participant

| Field | Value |
|---|---|
| **Full Name** | Omar Mahmoud Mouhamed |
| **Project Name** | University AI Assistant |
| **GitHub Username** | Omar_Raghebb |
| **Challenge Batch** | June–July 2026 |
| **Training Program** | Large Language Models (LLMs) Program |
| **Organization** | [Edrak for AI](https://edrak4ai.com/en) |

This repository is my official submission for the [Tips Hindawi](https://www.tipshindawi.com/) Challenge (June–July 2026).

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Demo](#demo)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [About the Challenge](#about-the-challenge)
- [License](#license)

---

## Overview

An assistant that answers student questions about university regulations and graduation requirements. The system is built entirely on a Retrieval-Augmented Generation (RAG) pipeline using LangChain and a fully local LLM, with fine-tuning, quantization, structured output parsing, and deployment handled end-to-end without any external API dependency.

---

## Features

| Feature | Description |
|---|---|
| **Local RAG pipeline** | No external API calls — retrieval, generation, and reranking all run on locally loaded models |
| **4-bit quantization** | The LLM is loaded in 4-bit (via `bitsandbytes`) instead of fp16 to cut VRAM usage, controlled by a flag in `config.py` |
| **Cross-encoder reranking** | Retrieved chunks are reranked with `cross-encoder/ms-marco-MiniLM-L-6-v2` before being passed to the LLM, with an automatic fallback if the reranker is unavailable |
| **Course-aware retrieval** | Retrieval confidence is boosted for chunks that contain the exact course code mentioned in the question, and further boosted for prerequisite-table chunks on prerequisite-related queries |
| **Structured output** | Every answer is returned as JSON with `answer`, `source_page`, and `confidence` fields, backed by a parser that recovers from malformed or markdown-wrapped model output |
| **Optional LoRA fine-tuning** | A lightweight LoRA pass adjusts the model's answering style (concise, formal, language-matched) — it is not used to teach the model new facts, which is the retrieval layer's job |
| **Deployment** | FastAPI server exposed via ngrok, callable from a local client |
| **Evaluation harness** | A 10-question test set measuring keyword grounding rate and average retrieval confidence |

---

## Technologies Used

<div align="left">

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/-LangChain-1C3C3C?style=flat-square)
![Mistral](https://img.shields.io/badge/-Mistral--7B-FF7000?style=flat-square)
![FAISS](https://img.shields.io/badge/-FAISS-4267B2?style=flat-square)
![FastAPI](https://img.shields.io/badge/-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

</div>

- **Python**
- **LangChain** (`langchain`, `langchain-community`, `langchain-huggingface`, `langchain-classic`)
- **Mistral-7B-Instruct-v0.2** — local LLM, loaded via `transformers`
- **FAISS** — vector store
- **`intfloat/multilingual-e5-large`** — multilingual embedding model
- **sentence-transformers** — `cross-encoder/ms-marco-MiniLM-L-6-v2` for reranking
- **bitsandbytes** — 4-bit quantization
- **peft + trl** — LoRA fine-tuning
- **`pdftotext`** (poppler-utils) — PDF text extraction
- **FastAPI + ngrok** — deployment
- **Streamlit** — UI
- **python-dotenv** — environment variable management

---

## Project Structure

```
├── src/
│   ├── config.py            # model names, chunk size, top-k, ngrok/API keys
│   ├── document_loader.py   # PDF loading + regex preprocessing (course blocks, prereq tables, GPA rules)
│   ├── vector_store.py      # embedding model + FAISS build/load
│   ├── retriever.py         # similarity search + course-code and prereq-table confidence boosting
│   ├── llm.py                # Mistral-7B loading (4-bit) and text generation
│   ├── chain.py              # prompt template + reranking + full RAG pipeline
│   ├── output_parser.py     # robust JSON extraction/cleaning from model output
│   ├── evaluation.py         # keyword-grounding test set and scoring
│   └── finetune.py           # optional LoRA fine-tuning (peft + trl)
├── test_rag.py                # quick manual test against a set of sample questions
├── app.py                     # Streamlit UI
├── deploy_server.py           # FastAPI + ngrok deployment (run on Kaggle)
└── client_example.py          # calls the deployed Kaggle server from your machine
```

---

## Installation

```bash
pip install -r requirements.txt
```

> The LLM and embedding models require a GPU. If your machine doesn't have one, run the project on a Kaggle Notebook (free T4 GPU).

---

## Usage

| Step | Command | Description |
|---|---|---|
| 1 | `python -m src.document_loader` | Test PDF loading and preprocessing |
| 2 | `python -m src.vector_store` | Build the FAISS index |
| 3 | `python -m src.llm` | Test local model loading and quantized generation |
| 4 | `python -m src.chain` | Run a sample question through the full RAG chain |
| 5 | `python test_rag.py` | Quick end-to-end test with a set of sample questions |
| 6 | `python -m src.evaluation` | Run the evaluation harness (keyword grounding rate) |
| 7 | `python -m src.finetune` | (Optional) Run LoRA fine-tuning |
| 8 | `streamlit run app.py` | Launch the UI locally |
| 9 | `python deploy_server.py` (on Kaggle), then `python client_example.py` (locally) | Deploy the server on Kaggle and call it from your machine |

---

## Demo

_Add screenshots, a GIF, or a demo video here._

---

## Results

_Add evaluation results (grounding rate, average confidence, etc.) here after running `src/evaluation.py` against `TEST_SET`._

---

## Future Improvements

- Improve retrieval accuracy with more fine-tuning examples
- Support multiple university regulation PDFs at once
- Replace ngrok with a more permanent deployment option

---

## About the Challenge

This project was developed as part of the [Tips Hindawi](https://www.tipshindawi.com/) Challenge (June–July 2026).

[Tips Hindawi](https://www.tipshindawi.com/) is the internships department of [Edrak for AI](https://edrak4ai.com/en), and the challenge encourages participants to build real-world projects, apply practical skills, and showcase their work through GitHub.

For more information about the challenge, training programs, and upcoming batches, visit the official [Tips Hindawi](https://www.tipshindawi.com/) website.

---

## License

This project is shared for educational and portfolio purposes.

<div align="center">

---

Omar Mahmoud Mouhamed — Tips Hindawi Challenge 2026

</div>
