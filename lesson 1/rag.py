"""
RAG (Retrieval-Augmented Generation) Tutorial

How it works:
  1. LOAD   - Read your JSON data
  2. EMBED  - Turn each text chunk into a vector (list of numbers) using an embedding model
  3. STORE  - Save those vectors in a vector database so we can search them fast
  4. QUERY  - When a question comes in, embed it too, find the most similar chunks
  5. GENERATE - Send the retrieved chunks + question to the LLM for a grounded answer
"""

import json
import os
import sys

os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
import ollama

# Silence broken telemetry (capture() signature bug in this chromadb version)
try:
    from chromadb.telemetry.product import posthog
    posthog.capture = lambda *args, **kwargs: None
except Exception:
    pass

# ---------------------------------------------------------------------------
# Configuration - change these to match your setup
# ---------------------------------------------------------------------------
DATA_FILE = "data.json"
EMBED_MODEL = "mxbai-embed-large"
LLM_MODEL = "llama3.2"
COLLECTION_NAME = "my_knowledge_base"
TOP_K = 3  # how many relevant chunks to retrieve per query


# ---------------------------------------------------------------------------
# Step 1 — Load your JSON data
# ---------------------------------------------------------------------------
def load_documents(path: str) -> list[dict]:
    """
    Expects a JSON array of objects, each with at least an "id" and "text" field.
    You can adapt this to any shape — just make sure you extract a string to embed.
    """
    with open(path) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} documents from {path}")
    return data


# ---------------------------------------------------------------------------
# Step 2 & 3 — Embed documents and store them in a vector database
# ---------------------------------------------------------------------------
def build_vector_store(documents: list[dict]) -> chromadb.Collection:
    """
    Creates a ChromaDB collection and inserts each document.
    ChromaDB persists to disk automatically (./chroma_db folder).
    """
    client = chromadb.PersistentClient(path="./chroma_db")

    # Delete old collection if it exists so we start fresh
    try:
        client.delete_collection(COLLECTION_NAME)
    except ValueError:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    print(f"Embedding {len(documents)} documents with '{EMBED_MODEL}'...")

    ids = []
    texts = []
    embeddings = []

    for doc in documents:
        text = doc["text"]
        response = ollama.embed(model=EMBED_MODEL, input=text)
        embedding = response["embeddings"][0]

        ids.append(doc["id"])
        texts.append(text)
        embeddings.append(embedding)

    collection.add(ids=ids, documents=texts, embeddings=embeddings)
    print(f"Stored {len(ids)} documents in vector store.\n")
    return collection


# ---------------------------------------------------------------------------
# Step 4 — Retrieve the most relevant chunks for a query
# ---------------------------------------------------------------------------
def retrieve(collection: chromadb.Collection, query: str, n_results: int = TOP_K) -> list[str]:
    """
    Embeds the query with the same model, then asks ChromaDB
    for the closest document vectors.
    """
    query_embedding = ollama.embed(model=EMBED_MODEL, input=query)["embeddings"][0]

    n_results = min(n_results, collection.count())
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    return results["documents"][0]  # list of text strings


# ---------------------------------------------------------------------------
# Step 5 — Send context + question to the LLM
# ---------------------------------------------------------------------------
def generate_answer(query: str, context_chunks: list[str]) -> str:
    """
    Builds a prompt that includes the retrieved context,
    then asks the LLM to answer based only on that context.
    """
    print(f"Generating answer using '{LLM_MODEL}' with {len(context_chunks)} context chunks...")
	# print context_chunk
    
    context = "\n\n---\n\n".join(context_chunks)

    prompt = (
        "You are a helpful assistant. Use ONLY the following context to answer the "
        "question. If the context doesn't contain enough information, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


# ---------------------------------------------------------------------------
# Main — interactive loop
# ---------------------------------------------------------------------------
def main():
    documents = load_documents(DATA_FILE)
    collection = build_vector_store(documents)

    print("=" * 60)
    print("RAG system ready!  Ask anything about your data.")
    print("Type 'quit' to exit.\n")

    while True:
        query = input("You: ").strip()
        if not query or query.lower() in ("quit", "exit", "q"):
            break

        # Retrieve
        context_chunks = retrieve(collection, query)
        print(f"\n[Retrieved {len(context_chunks)} relevant chunks]")

        # Generate
        answer = generate_answer(query, context_chunks)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()
