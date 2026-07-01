# LLM-based Document Q&A System with Vietnamese Support

## Overview

This project is an end-to-end Retrieval-Augmented Generation (RAG) application for asking grounded questions over uploaded documents. It supports PDF, TXT, and DOCX files, with explicit attention to Vietnamese text preprocessing, multilingual embeddings, source citations, and refusal behavior when the answer is not found in the document.

## Features

- Upload PDF, TXT, and DOCX documents.
- Extract text while preserving source metadata.
- Normalize Vietnamese text without removing accents.
- Split documents into citation-friendly chunks.
- Store multilingual embeddings in ChromaDB.
- Retrieve relevant chunks for Vietnamese or English questions.
- Generate grounded answers with an LLM.
- Show answer sources and retrieved evidence.
- Evaluate retrieval with a small Vietnamese QA benchmark.

## Architecture

```text
User Upload
    -> Document Loader
    -> Vietnamese Text Preprocessor
    -> Text Splitter
    -> Embedding Model
    -> ChromaDB Vector Store
    -> Retriever
    -> Prompt Builder
    -> LLM Client
    -> Answer + Sources + Evidence
```

## Project Structure

```text
app/
  main.py
  config.py
  document_loader.py
  text_preprocessor.py
  text_splitter.py
  embedding_model.py
  vector_store.py
  retriever.py
  prompt_template.py
  llm_client.py
  rag_pipeline.py
data/
  raw/
  processed/
  samples/
evaluation/
  questions.csv
  evaluate_retrieval.py
  results.md
tests/
requirements.txt
.env.example
README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in the API key for the LLM provider you want to use.

## Run

```bash
streamlit run app/main.py
```

## Vietnamese Support

The pipeline will normalize Unicode with NFC, preserve Vietnamese accents, avoid aggressive lowercasing, split text using paragraph and sentence boundaries, use multilingual embeddings, and instruct the LLM to answer in the same language as the user question.

## Evaluation

The evaluation pipeline will use `evaluation/questions.csv` to measure Recall@3, Recall@5, citation accuracy, and refusal accuracy.

## Roadmap

- Milestone 1: Project setup.
- Milestone 2: Document loading.
- Milestone 3: Vietnamese preprocessing and chunking.
- Milestone 4: Embeddings and ChromaDB.
- Milestone 5: Retrieval.
- Milestone 6: LLM answer generation.
- Milestone 7: Streamlit demo.
- Milestone 8: Evaluation, tests, and README polish.

## Future Improvements

- Multiple document support.
- Hybrid search.
- Reranking.
- OCR for scanned PDFs.
- FastAPI backend.
- Docker packaging.
