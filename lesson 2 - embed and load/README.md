# Lesson 2 — Embed & Load

Hands-on exploration of what **exactly** goes into ChromaDB and how data looks at every step.

## Run

```bash
cd "lesson 2 - embed and load"
python3 embed_and_load.py
```

The script pauses at each step — press Enter to continue.

## What you'll see

| Step | What happens | What you'll see printed |
|---|---|---|
| **1. LOAD** | Read `data.json` | Raw dicts with `id` and `text` |
| **2. EMBED** | `ollama.embed()` per document | Each embedding: 1024 floats, first/last values |
| **3. STORE** | `collection.add(ids, documents, embeddings)` | The exact 3 parallel lists that go in |
| **4. PEEK** | `collection.get()` | What ChromaDB actually stores per record |
| **5. QUERY** | Embed a question, similarity search | Cosine similarity scores + ranked results |

## Key takeaway

ChromaDB stores **three things** per record:

```
id         →  "1"                              (your unique key)
document   →  "Peter lives in Tai Wai"         (original text, optional)
embedding  →  [-0.037, 0.014, ... 1024 floats] (the vector used for search)
```

When you query, ChromaDB embeds your question into the **same 1024-dimensional space** and finds the stored vectors that are geometrically closest — pure math, no keyword matching.

## Prerequisites

- Ollama running with `mxbai-embed-large` pulled
- `pip install chromadb ollama`
