"""
Lesson 2 — Embed & Load: What exactly goes into ChromaDB?

Run:  python3 embed_and_load.py

This script pauses at every step and prints the data so you can see:
  1. LOAD   — the raw JSON documents
  2. EMBED  — what ollama.embed() returns (a vector of 1024 floats)
  3. STORE  — the exact 3 arguments passed to collection.add()
  4. PEEK   — read the data back from ChromaDB to see how it's stored
  5. QUERY  — embed a question, compare distances, get results
"""

import json
import math
import os
import textwrap

os.environ["ANONYMIZED_TELEMETRY"] = "False"

try:
    import posthog
    posthog.capture = lambda *a, **kw: None
except Exception:
    pass

import chromadb
import ollama

EMBED_MODEL = "mxbai-embed-large"
DATA_FILE = "data.json"
COLLECTION_NAME = "lesson2_demo"

SEP = "=" * 70


def pause():
    input("\n⏎  Press Enter to continue ...\n")


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    return dot / (mag_a * mag_b)


# ─────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD: read the JSON file
# ─────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  STEP 1 — LOAD: Read the JSON file")
print(SEP)

with open(DATA_FILE) as f:
    documents = json.load(f)

print(f"\nLoaded {len(documents)} documents from {DATA_FILE}")
print(f"Type: {type(documents).__name__}  (a Python list of dicts)\n")

for doc in documents:
    print(f'  id={doc["id"]}  text="{doc["text"]}"')

print(f"""
┌──────────────────────────────────────────────────────┐
│  Each document is just a dict with "id" and "text".  │
│  This is the RAW INPUT — plain strings, nothing      │
│  special yet.  ChromaDB can't search text directly;  │
│  we need to convert it to numbers first (→ EMBED).   │
└──────────────────────────────────────────────────────┘""")

pause()


# ─────────────────────────────────────────────────────────────────────────
# STEP 2 — EMBED: turn each text into a vector
# ─────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  STEP 2 — EMBED: Turn each text → vector (list of floats)")
print(SEP)

ids_list = []
texts_list = []
embeddings_list = []

for doc in documents:
    text = doc["text"]
    response = ollama.embed(model=EMBED_MODEL, input=text)
    embedding = response["embeddings"][0]

    ids_list.append(doc["id"])
    texts_list.append(text)
    embeddings_list.append(embedding)

    print(f'\n  Document id={doc["id"]}')
    print(f'  Input text : "{text}"')
    print(f"  Embedding  : list of {len(embedding)} floats")
    print(f"  First 8    : {[round(x, 4) for x in embedding[:8]]}")
    print(f"  Last  4    : ...{[round(x, 4) for x in embedding[-4:]]}")

print(f"""
┌──────────────────────────────────────────────────────────────────┐
│  ollama.embed(model="{EMBED_MODEL}", input=text)     │
│  returns: {{"embeddings": [[0.123, -0.456, ...]]}}              │
│                                                                  │
│  Each embedding is a list of {len(embeddings_list[0])} floats.               │
│  These numbers encode the MEANING of the text.                   │
│  Similar texts → similar numbers → close in vector space.        │
└──────────────────────────────────────────────────────────────────┘""")

pause()

# Show similarity between pairs so user can see "close" vs "far"
print("  Cosine similarity between document pairs:\n")
for i in range(len(documents)):
    for j in range(i + 1, len(documents)):
        sim = cosine_similarity(embeddings_list[i], embeddings_list[j])
        t_i = texts_list[i][:40]
        t_j = texts_list[j][:40]
        bar = "█" * int(sim * 30)
        print(f'    doc {ids_list[i]} vs doc {ids_list[j]}  '
              f'sim={sim:.4f}  {bar}')
        print(f'      "{t_i}..."')
        print(f'      "{t_j}..."')
        print()

print(textwrap.dedent("""\
    Higher similarity → texts have more related meaning.
    Notice docs about "Peter" are closer to each other than to "Python".
"""))

pause()


# ─────────────────────────────────────────────────────────────────────────
# STEP 3 — STORE: pass the 3 lists to collection.add()
# ─────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  STEP 3 — STORE: What exactly goes into ChromaDB?")
print(SEP)

print("""
  collection.add() takes 3 parallel lists:

    ┌──────────┬──────────────────────────────────┬──────────────────────┐
    │  ids     │  documents                       │  embeddings          │
    ├──────────┼──────────────────────────────────┼──────────────────────┤""")

for i in range(len(ids_list)):
    id_col = ids_list[i].ljust(8)
    doc_col = texts_list[i][:32].ljust(32)
    emb_col = f"[{embeddings_list[i][0]:.3f}, {embeddings_list[i][1]:.3f}, ... {len(embeddings_list[i])} floats]"
    print(f"    │  {id_col}│  {doc_col}│  {emb_col}")

print("    └──────────┴──────────────────────────────────┴──────────────────────┘")

print(f"""
  In Python this is literally:

    collection.add(
        ids        = {ids_list},
        documents  = {[t[:35]+'...' for t in texts_list]},
        embeddings = [ [0.123, -0.456, ...], ... ]   # {len(ids_list)} lists of {len(embeddings_list[0])} floats
    )
""")

pause()

# Actually store it
client = chromadb.PersistentClient(path="./chroma_db")
try:
    client.delete_collection(COLLECTION_NAME)
except ValueError:
    pass

collection = client.create_collection(name=COLLECTION_NAME)
collection.add(ids=ids_list, documents=texts_list, embeddings=embeddings_list)

print(f"  ✓ Stored {collection.count()} documents in collection '{COLLECTION_NAME}'")

pause()


# ─────────────────────────────────────────────────────────────────────────
# STEP 4 — PEEK: read back from ChromaDB to see what's inside
# ─────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  STEP 4 — PEEK: What does ChromaDB actually store?")
print(SEP)

print("\n  collection.get() returns everything:\n")
everything = collection.get(include=["documents", "embeddings"])

print(f"  Keys: {list(everything.keys())}\n")

for i, doc_id in enumerate(everything["ids"]):
    emb = everything["embeddings"][i]
    doc = everything["documents"][i]
    print(f"  ── Record {doc_id} ──")
    print(f"  id        : {doc_id!r}")
    print(f"  document  : {doc!r}")
    print(f"  embedding : [{emb[0]:.4f}, {emb[1]:.4f}, {emb[2]:.4f}, ... ] ({len(emb)} floats)")
    print()

print(textwrap.dedent("""\
    ChromaDB stores 3 things per record:
      • id         — your unique string key
      • document   — the original text (optional but useful)
      • embedding  — the vector used for similarity search
"""))

pause()


# ─────────────────────────────────────────────────────────────────────────
# STEP 5 — QUERY: embed a question, then search
# ─────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  STEP 5 — QUERY: Embed a question, find closest documents")
print(SEP)

question = "Where does Peter live?"
print(f'\n  Question: "{question}"\n')

q_response = ollama.embed(model=EMBED_MODEL, input=question)
q_embedding = q_response["embeddings"][0]

print(f"  Question embedding: list of {len(q_embedding)} floats")
print(f"  First 8: {[round(x, 4) for x in q_embedding[:8]]}\n")

print("  Cosine similarity between QUESTION and each document:\n")

similarities = []
for i in range(len(ids_list)):
    sim = cosine_similarity(q_embedding, embeddings_list[i])
    similarities.append((sim, ids_list[i], texts_list[i]))

similarities.sort(reverse=True)

for rank, (sim, doc_id, text) in enumerate(similarities, 1):
    bar = "█" * int(sim * 40)
    marker = " ← closest!" if rank == 1 else ""
    print(f"    #{rank}  doc {doc_id}  sim={sim:.4f}  {bar}{marker}")
    print(f'         "{text}"')
    print()

print("  Now let ChromaDB do the same search internally:\n")

results = collection.query(query_embeddings=[q_embedding], n_results=2)

print(f"  collection.query(query_embeddings=[...], n_results=2)")
print(f"  Returns:\n")
print(f"    ids       : {results['ids']}")
print(f"    documents : {results['documents']}")
print(f"    distances : {results['distances']}")

print(f"""
┌────────────────────────────────────────────────────────────────┐
│  ChromaDB did the same cosine comparison internally and        │
│  returned the 2 closest documents.                             │
│                                                                │
│  The QUERY embedding lives in the same 1024-dim space as the   │
│  document embeddings.  "Close" = similar meaning.              │
│                                                                │
│  This is how RETRIEVE works — pure geometry, no keywords.      │
└────────────────────────────────────────────────────────────────┘
""")

print("Done!  Try changing the question or data.json and run again.\n")
