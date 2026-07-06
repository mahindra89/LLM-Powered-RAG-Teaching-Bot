# Network Security RAG Tutor

An interactive local AI study assistant for network security course material.
The app answers questions, generates practice quizzes, grades responses, and
shows citations from the source documents used to produce each result.

## What It Does

- Indexes local PDF documents into a ChromaDB vector database
- Retrieves relevant source pages for a user question or quiz topic
- Uses Ollama with `qwen2.5:7b-instruct` for local answer generation
- Produces tutor-style explanations with page-level citations
- Generates multiple-choice, true/false, and fill-in-the-blank quiz questions
- Grades quiz answers and explains the correct answer with source references
- Falls back to retrieved source text when the local LLM is unavailable

## Why This Project Matters

This project demonstrates a privacy-preserving Retrieval-Augmented Generation
workflow. Course documents stay on the local machine, while the app still gives
students a modern AI study experience with traceable citations.

It is designed around three practical engineering ideas:

- **Grounded answers:** responses are based on retrieved course material
- **Local-first AI:** documents and LLM calls can remain offline
- **Verifiable learning:** citations make answers auditable instead of opaque

## App Experience

The Streamlit interface includes four sections:

- **Overview:** system status, indexed chunks, source document count, and LLM status
- **Tutor:** ask questions and receive cited explanations
- **Quiz:** generate a short quiz from retrieved context and get immediate grading
- **Learning:** explains the RAG architecture and what the project demonstrates

## Project Structure

```text
.
├── main.py                    # Streamlit app and RAG workflow
├── requirements.txt           # Python dependencies
├── Source/                    # Source PDF documents
├── processed_md/              # Extracted markdown notes
├── ARCHITECTURE.md            # Architecture explanation
├── architecture-diagram.mmd   # Mermaid architecture diagram
└── README.md
```

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start Ollama:

```bash
ollama pull qwen2.5:7b-instruct
ollama serve
```

Run the app:

```bash
python -m streamlit run main.py
```

Open:

```text
http://localhost:8501
```

If the vector database is not initialized, click **Process PDFs** in the
sidebar. The app will read files from `Source/`, extract text, and build the
local ChromaDB index.

## Streamlit Cloud

This repo can be deployed as a Streamlit app.

Use:

```text
Main file path: main.py
```

Note: full LLM generation requires Ollama, which is intended for local use. On
hosted environments without Ollama, the app can still show the interface and
retrieved context, but generated explanations and quizzes need a running local
Ollama service.

## Core Stack

- Streamlit
- ChromaDB
- PyMuPDF
- Ollama
- DuckDuckGo/Wikipedia fallback search

## Future Improvements

- Add document upload from the UI
- Add quiz history and score tracking
- Export quiz attempts to CSV or PDF
- Add source highlighting inside retrieved passages
- Add support for additional local Ollama models
