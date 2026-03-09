# Learn RAG with Ollama

A minimal, hands-on project to understand **Retrieval-Augmented Generation (RAG)**.

## What is RAG?

LLMs are powerful, but they only know what was in their training data. RAG solves this
by giving the model **your own data** at query time:

```
┌──────────────┐      ┌──────────────────┐      ┌─────────────┐
│  Your query  │─────▶│  Vector Database  │─────▶│  Retrieved  │
│              │      │  (find similar    │      │  chunks     │
└──────────────┘      │   documents)      │      └──────┬──────┘
                      └──────────────────┘             │
                                                       ▼
                                              ┌────────────────┐
                                              │  LLM generates │
                                              │  answer using   │
                                              │  your data as   │
                                              │  context        │
                                              └────────────────┘
```

**Without RAG:** "What is TechFlow's return policy?" → LLM has no idea, makes something up.

**With RAG:** The same question → system finds the relevant document from your JSON,
passes it to the LLM, and you get an accurate, grounded answer.

## Key Concepts

| Concept | What it means |
|---|---|
| **Embedding** | Turning text into a vector (list of numbers) that captures its meaning. Similar texts have similar vectors. |
| **Vector Database** | A database optimized for storing embeddings and finding the closest matches (ChromaDB in our case). |
| **Retrieval** | Searching the vector DB for chunks most relevant to the user's question. |
| **Augmented Generation** | Feeding retrieved chunks as context into the LLM prompt so it answers based on your data. |

## Prerequisites

- [Ollama](https://ollama.com) installed and running
- Python 3.10+

## Quick Start

```bash
# 1. Pull the required Ollama models
ollama pull mxbai-embed-large   # embedding model
ollama pull llama3.2             # LLM for generating answers

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Run the RAG system
python3 rag.py
```

## Project Structure

```
data.json    — Your knowledge base (swap this with your own data)
rag.py       — The RAG pipeline: load → embed → store → retrieve → generate
```

## Using Your Own Data

Edit `data.json`. Each entry needs an `id` and `text` field:

```json
[
  { "id": "1", "text": "Your first document or paragraph..." },
  { "id": "2", "text": "Your second document or paragraph..." }
]
```

**Tips for good results:**
- Keep each `text` entry focused on one topic (a paragraph or two).
- If you have long documents, split them into smaller chunks.
- The more specific each chunk is, the better the retrieval will be.

## How the Code Works

1. **`load_documents()`** — Reads your JSON file.
2. **`build_vector_store()`** — Sends each text to the embedding model, stores the
   resulting vectors in ChromaDB.
3. **`retrieve()`** — Embeds the user's question with the same model, finds the
   top-K most similar documents.
4. **`generate_answer()`** — Builds a prompt with the retrieved context and asks
   the LLM to answer.
