# Lesson 4 — Build an Embedding Model from Scratch

Understand how `mxbai-embed-large` works by building a **mini version** from scratch using PyTorch.

## Run

```bash
cd "lesson 4 - embed model"
python3 build_embed_model.py
```

Pauses at each step — press Enter to continue.

## What you'll build

A 74K-parameter mini transformer encoder that learns sentence embeddings via contrastive learning — the same architecture and training approach as mxbai-embed-large (340M params), just smaller.

## Steps

| Step | What happens |
|---|---|
| **1. Tokenize** | Text → integer IDs (word-level vocabulary) |
| **2. Transformer Encoder** | Token embeddings + positional encoding → 2 self-attention layers → mean pooling → 64-dim vector |
| **3. Triplet Loss** | Push similar sentences together, different sentences apart |
| **4. Training** | 80 epochs on 15 triplets — loss drops to 0 in seconds |
| **5. Evaluation** | Similar pairs get high cosine similarity, different pairs get low |
| **6. Inside the vector** | Inspect the 64 floats, see the L2 norm = 1.0 |
| **7. Comparison** | Side-by-side: our mini model vs mxbai-embed-large |

## Our model vs mxbai-embed-large

| | Mini model | mxbai-embed-large |
|---|---|---|
| Parameters | 74K | 340M |
| Layers | 2 | 24 |
| Embedding dim | 64 | 1024 |
| Tokenizer | word-level, 98 words | WordPiece, ~30K subwords |
| Training data | 15 triplets | 700M+ sentence pairs |
| Training time | ~25 seconds (CPU) | Days (GPU cluster) |

**Same core ideas, just different scale.**

## Prerequisites

- Python 3.10+
- `pip install torch`
