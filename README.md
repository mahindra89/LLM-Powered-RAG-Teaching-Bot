# Network Security RAG Teaching Bot

A portfolio-ready Streamlit explainer for a local-first Retrieval-Augmented
Generation teaching assistant built around network security course material.
The project shows how lecture slides, textbook excerpts, and homework material
can be converted into a searchable knowledge base for cited answers and quiz
generation.

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
workflow. Course documents stay on the local machine, while the architecture
still supports a modern AI study experience with traceable citations.

It is designed around three practical engineering ideas:

- **Grounded answers:** responses are based on retrieved course material
- **Local-first AI:** documents and LLM calls can remain offline
- **Verifiable learning:** citations make answers auditable instead of opaque

## Public App Experience

The Streamlit interface presents the project as a portfolio-ready explainer with
three sections:

- **Overview:** project idea, source snapshot, design highlights, and downloadable project notes
- **Architecture:** RAG pipeline explanation with the Mermaid architecture diagram
- **Learning:** what the project demonstrates and why the design matters

The underlying code still contains the local Tutor and Quiz workflows for users
who run the project with Ollama and an initialized vector database, but those
interactive workflows are intentionally not shown on the public explainer page.

## Project Structure

```text
.
|-- main.py                    # Streamlit explainer and original RAG workflow code
|-- requirements.txt           # Python dependencies
|-- Source/                    # Source PDF documents
|-- processed_md/              # Extracted markdown notes
|-- ARCHITECTURE.md            # Architecture explanation
|-- architecture-diagram.mmd   # Mermaid architecture diagram
|-- streamlit-architecture-diagram.mmd # Cloud-safe diagram rendered in the app
`-- README.md
```

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
python -m streamlit run main.py
```

Open:

```text
http://localhost:8501
```

The public page is an explainer and does not require Ollama to be running.
Ollama and the vector database are part of the original local RAG workflow
described in the architecture.

## Streamlit Cloud

This repo can be deployed as a Streamlit app.

Use:

```text
Main file path: main.py
```

The public Streamlit page is safe to host as an explainer because it does not
depend on live Ollama calls or an initialized vector database.

## Core Stack

- Streamlit
- ChromaDB
- PyMuPDF
- Ollama
- DuckDuckGo/Wikipedia fallback search

## Future Improvements

- Add a small recorded demo or screenshots of the local Tutor workflow
- Add a sample citation walkthrough
- Add source highlighting inside retrieved passages
- Add support for additional local Ollama models
- Separate explainer UI from the local RAG runtime into dedicated modules
