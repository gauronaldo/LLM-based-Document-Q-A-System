# LLM-based Document Q&A System with Vietnamese Support

An end-to-end Retrieval-Augmented Generation (RAG) application for chatting with uploaded PDF, TXT, and DOCX documents. The system supports Vietnamese and English questions, multilingual embeddings, hybrid retrieval, grounded LLM answers, compact citations, and refusal behavior when the document does not contain enough evidence.

## Why This Project

This project is designed as a practical AI Engineering portfolio project. It demonstrates:

- Document ingestion and metadata extraction
- Vietnamese-safe text normalization
- Heading/section-aware chunking
- Multilingual embedding generation
- Hybrid search with vector retrieval and BM25 keyword retrieval
- Optional cross-encoder reranking with a lightweight lexical fallback
- Parent-child retrieval for richer answer context
- MMR retrieval to reduce duplicate evidence
- Query rewriting for short follow-up questions
- Conversational answer synthesis with citations
- Retrieval and answer-level evaluation helpers

## Features

- Upload PDF, TXT, or DOCX documents.
- Extract text with file/page/paragraph metadata.
- Preserve Vietnamese accents and normalize Unicode.
- Split documents into citation-friendly chunks while preserving parent section context.
- Store embeddings in a persistent ChromaDB vector database.
- Combine vector search and keyword/BM25 search.
- Search and boost section titles such as academic headings during retrieval.
- Retrieve a larger candidate pool, rerank it, then return the best diverse chunks.
- Automatically adjust top-k and similarity threshold based on query intent and confidence.
- Cache all chunks/BM25 search inputs after indexing to reduce repeated retrieval latency.
- Run reranking conditionally so simple confident questions skip unnecessary cross-encoder work.
- Compress retrieved context before prompting the LLM to reduce response latency.
- Detect query intent: QA, summary, explanation, comparison, extraction.
- Rewrite follow-up queries using recent chat history for better retrieval.
- Generate natural chatbot-style answers grounded in retrieved evidence.
- Match the answer language to the user's question: English in, English out; Vietnamese in, Vietnamese out.
- Show compact source citations and retrieved evidence.
- Refuse when no relevant evidence is found.
- Evaluate retrieval and answer quality with explainable local metrics.

## Architecture

```text
User Upload
    -> DocumentLoader
    -> VietnameseTextPreprocessor
    -> TextSplitter with section/parent metadata
    -> EmbeddingModel
    -> ChromaDB VectorStore

User Question
    -> Query cleanup
    -> Query intent detection
    -> Query rewriting for follow-ups
    -> Hybrid retrieval: vector + BM25
    -> Section-title matching and boosting
    -> Reranking
    -> MMR selection
    -> Parent context expansion
    -> Prompt Builder
    -> LLMClient
    -> Answer + Sources + Retrieved Evidence
```

## Project Structure

```text
app/
  main.py                  Streamlit UI
  config.py                Environment-based configuration
  document_loader.py        PDF/TXT/DOCX extraction
  text_preprocessor.py      Vietnamese-safe text cleanup
  text_splitter.py          Section-aware chunking with parent metadata
  embedding_model.py        sentence-transformers wrapper
  vector_store.py           ChromaDB wrapper
  keyword_search.py         Tokenization, BM25, lexical similarity
  retriever.py              Hybrid retrieval, reranking, MMR, parent expansion
  reranker.py               Optional cross-encoder reranker and lexical fallback
  query_rewriter.py         Follow-up query rewriting
  query_intent.py           Rule-based query intent detection
  prompt_template.py        Grounded conversational prompts
  llm_client.py             Gemini/OpenAI/Ollama wrapper
  rag_pipeline.py           RAG orchestration
data/
  raw/                      Runtime uploaded documents
  processed/                Optional processed outputs
  samples/                  PDF samples used by evaluation
evaluation/
  README.md                 Dev/holdout/cross-document evaluation workflow
  workflow.json             Machine-readable workflow manifest
  questions.csv             PerezPerez2020 dev/stress-test questions
  questions_w18347.csv      w18347 holdout validation questions
  questions_2302_06590.csv  final RAGAS/cross-document questions
  questions_holdout.csv     Extra holdout-style questions
  debug_config.json         Deterministic debug thresholds
  debug_utils.py            Shared debug/evaluation helpers
  debug_retrieval.py        Retrieval-only debug report
  ab_test_retrieval.py      Retrieval configuration ablation
  debug_generation.py       Gold-context generation debug
  freeze_config.py          Save selected config before holdout
  run_custom_eval.py        Fast custom evaluation before RAGAS
  evaluate.py               Final RAGAS/core evaluator
  cache/                    Generated prediction/RAGAS caches
  results/                  Generated debug and evaluation reports
notebooks/
  debug_existing_pipeline.ipynb
tests/
requirements.txt
.env.example
README.md
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```bash
cp .env.example .env
```

For Gemini:

```env
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash
GEMINI_API_KEY=your_gemini_api_key_here
```

Important retrieval settings:

```env
EMBEDDING_BATCH_SIZE=32
INDEXING_BATCH_SIZE=128
PARENT_CONTEXT_MAX_CHARS=3000
TOP_K=5
SIMILARITY_THRESHOLD=0.3
USE_HYBRID_SEARCH=true
HYBRID_ALPHA=0.7
USE_MMR=true
RERANKER_MODEL=
```

`EMBEDDING_BATCH_SIZE` controls how many chunks are encoded at once while processing a document. For long PDFs, keep it around `16` to `32` so Streamlit stays responsive and memory usage does not spike.

`INDEXING_BATCH_SIZE` controls how many chunks are encoded and written to ChromaDB per indexing step. For very long PDFs, use `64` or `128` so the app does not send thousands of vectors to ChromaDB in a single payload.

`PARENT_CONTEXT_MAX_CHARS` caps the parent section text stored in metadata. This keeps parent-child retrieval useful without duplicating very large text blocks for every chunk.

`HYBRID_ALPHA=0.7` means 70% vector score and 30% BM25 keyword score. Leave `RERANKER_MODEL` empty to use the local lexical fallback. Set it to a sentence-transformers cross-encoder model if you want heavier reranking.

For a no-download local demo, especially when Hugging Face model loading is slow or blocked, use:

```env
EMBEDDING_MODEL=local-hash
RERANKER_MODEL=
```

`local-hash` is a deterministic lexical embedding backend. It is faster and fully offline, but less semantic than the multilingual sentence-transformers model.

Do not commit `.env`.

## Run The App

```bash
streamlit run app/main.py
```

The project includes `.streamlit/config.toml` with `server.fileWatcherType = "none"`. Keep this enabled when using sentence-transformers/PyTorch in Streamlit; it avoids file-watcher issues during model import.

Demo flow:

```text
1. Upload a PDF/TXT/DOCX file.
2. Click Process document.
3. Ask a direct question.
4. Ask a follow-up such as "Explain more" or "giai thich them y do".
5. Inspect the answer, sources, and retrieved evidence.
6. Ask an unrelated question to verify refusal behavior.
```

## How The RAG Pipeline Works

### Ingestion

```text
Document
-> validate file type
-> extract text and metadata
-> normalize Vietnamese/English text
-> split by visible sections/headings when possible
-> create child chunks with parent section metadata
-> generate embeddings
-> store chunks + metadata + vectors in ChromaDB
```

### Question Answering

```text
Question
-> clean query
-> detect intent
-> rewrite follow-up query if needed
-> choose top-k and threshold automatically
-> retrieve candidate chunks using vector search
-> retrieve keyword matches using BM25
-> merge hybrid scores
-> rerank candidates only when the query is complex or scores are close
-> select diverse chunks with MMR
-> expand child chunks to parent section context only when useful
-> compress context to a compact prompt budget
-> build grounded conversational prompt
-> call LLM
-> return answer + citations + evidence
```

## Vietnamese And English Support

- Unicode NFC normalization
- Vietnamese accents are preserved
- No aggressive lowercasing
- No stopword removal
- No stemming
- Paragraph/sentence/section-aware chunking
- Multilingual sentence-transformers model
- Vietnamese and English refusal messages
- Prompt rule: answer in the same language as the current user question

## Evaluation

Use a debug-first workflow. Full RAGAS evaluation is slow, especially with local Ollama judges, so do not run it after every code change.

Recommended loop:

```text
dev set: tune retrieval/citation on PerezPerez2020
-> freeze config
-> holdout validation: custom eval on w18347
-> final RAGAS/cross-document test on 2302.06590v1
-> report profile, document, metrics, and limitations
```

The detailed workflow lives in `evaluation/README.md` and `evaluation/workflow.json`.

Dataset roles:

- `evaluation/questions.csv`: PerezPerez2020 dev / hard stress-test set
- `evaluation/questions_w18347.csv`: w18347 holdout validation set
- `evaluation/questions_2302_06590.csv`: final RAGAS / cross-document report set

Do not tune prompts or retrieval rules from holdout failures. Use holdout results for reporting, not debugging.

### 1. Retrieval-Only Debug

This does not call the LLM and does not call RAGAS.

```bash
python evaluation/debug_retrieval.py --limit 10 --top-k 5
```

To include answerable, hard, and out-of-scope questions in a quick pass:

```bash
python evaluation/debug_retrieval.py --limit 15 --sample-strategy stratified --top-k 5
```

Output:

- `evaluation/results/retrieval_debug_report.csv`

Read this first when the app gives wrong facts. The important columns are `keyword_hit_rate`, `page_hit_at_5`, `top_score`, `status`, and `failure_reason`.

### 2. Retrieval Ablation

Compare current retrieval, multi-query RRF, BGE-M3 embedding, BGE reranker, and full BGE without LLM calls.

```bash
python evaluation/ab_test_retrieval.py --questions evaluation/questions.csv --limit 10 --top-k 5
```

Outputs:

- `evaluation/results/retrieval_ablation_report.csv`

Use this to check whether multi-query, BGE-M3, or BGE reranking is helping or hurting retrieval. The `citation_page_accuracy` column is a retrieval-only proxy, not final answer-level citation correctness.

### 3. Generation Debug With Gold Context

This isolates prompt/model behavior from retrieval. It gives the LLM the expected source pages directly.

```bash
python evaluation/debug_generation.py --sample-size 10
```

Use `--refresh-cache` after prompt/model/config changes:

```bash
python evaluation/debug_generation.py --sample-size 10 --refresh-cache
```

Output:

- `evaluation/results/generation_debug_report.csv`

If gold context fails, fix prompt/model/citation logic. If gold context works but normal RAG fails, fix retrieval.

### 4. Custom Evaluation

This runs the real pipeline but uses deterministic local metrics before RAGAS:

- Keyword Hit@5
- Keyword Recall@5
- Page Recall@5
- Evidence Hit@5
- MRR
- Citation Accuracy
- Refusal Accuracy
- False Refusal Rate
- Average Latency

```bash
python evaluation/run_custom_eval.py --limit 0 --refresh-cache
```

For a faster representative sample:

```bash
python evaluation/run_custom_eval.py --limit 15 --sample-strategy stratified --refresh-cache
```

Outputs:

- `evaluation/results/custom_eval_results.csv`
- `evaluation/results/custom_eval_summary.md`

Generated answers are cached in:

- `evaluation/cache/predictions.csv`

The cache key includes the prompt version, model/provider, embedding model, retrieval settings, and context hash, so prompt/model/config changes do not silently reuse stale generations.

### 5. Final RAGAS Evaluation

Only run this after the fast reports look reasonable.

```bash
python -m pip install -r requirements.txt
python -m evaluation.evaluate --questions evaluation/questions_2302_06590.csv --judge-provider google --judge-model gemini-2.5-flash --sleep-seconds 12 --max-retries 3 --output evaluation/results/ragas_2302_06590.csv --report-output evaluation/results/ragas_2302_06590.md
```

Quick smoke test:

```bash
python -m evaluation.evaluate --questions evaluation/questions_2302_06590.csv --judge-provider google --judge-model gemini-2.5-flash --limit 2 --sleep-seconds 2
```

RAGAS uses Gemini as the judge via `--judge-provider google`. The script maps `GEMINI_API_KEY` to `GOOGLE_API_KEY`, uses the official `google-genai` client, and uses Google embeddings by default through `RAGAS_EMBEDDING_MODEL=gemini-embedding-001`.

To run RAGAS with a local Ollama judge instead of paid API quota:

```bash
ollama pull llama3.1:8b
python -m evaluation.evaluate --questions evaluation/questions_2302_06590.csv --judge-provider ollama --judge-model llama3.1:8b --limit 2 --sleep-seconds 0
```

For Ollama judging, RAGAS uses Ollama's OpenAI-compatible endpoint at `http://localhost:11434/v1` and local Hugging Face embeddings by default. You can override these with:

```env
OLLAMA_OPENAI_BASE_URL=http://localhost:11434/v1
RAGAS_EMBEDDING_PROVIDER=huggingface
RAGAS_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Local judge scores are cheaper and private, but less stable than GPT/Gemini judge scores because smaller local models can be weaker at structured evaluation.

## Tests

```bash
python -m pytest -q
```

Current coverage includes:

- Config loading
- Document loading
- Vietnamese preprocessing
- Section-aware chunking
- Embedding wrapper behavior
- Vector store formatting
- BM25 keyword search
- Hybrid retrieval, reranking, MMR, parent context expansion
- Query rewriting
- Prompt building and language rules
- Query intent detection
- RAG pipeline orchestration
- Retrieval and answer evaluation helpers
- Core RAGAS evaluation reporting

## Current Limitations

- OCR is not implemented for scanned PDFs.
- Cross-encoder reranking is optional and depends on configuring `RERANKER_MODEL`.
- Evaluation uses a local PDF benchmark with RAGAS judge metrics; judge scores still depend on API availability, model choice, and quota.
- Multi-document workflows are basic.
- No FastAPI backend or Docker packaging yet.

## Suggested Upgrades

- Add OCR for scanned Vietnamese PDFs.
- Add document collection management and metadata filters.
- Add a FastAPI backend for production serving.
- Add Docker packaging and deployment instructions.
- Add more RAGAS metrics and judge calibration examples.
- Add reranker model benchmarking.
- Add screenshot/GIF demo assets for the portfolio README.

## CV Summary

Built an LLM-based document Q&A system using RAG, including Vietnamese document preprocessing, section-aware chunking, multilingual embeddings, ChromaDB vector storage, hybrid vector/BM25 retrieval, reranking, MMR evidence selection, parent-child context expansion, grounded LLM answer generation, source citations, and retrieval/answer evaluation.
