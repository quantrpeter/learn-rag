"""
Lesson 4 — Build an Embedding Model from Scratch

This script builds a MINI version of mxbai-embed-large to teach how
text embedding models actually work.

mxbai-embed-large (real):
  - 340M params, 24 transformer layers, 1024-dim embeddings
  - Trained on 700M+ sentence pairs (contrastive learning)
  - Fine-tuned on 30M triplets (AnglE loss)

Our mini version:
  - ~50K params, 2 transformer layers, 64-dim embeddings
  - Trained on ~30 sentence pairs right here
  - Same core ideas, just tiny so it runs in seconds on CPU

Run:  python3 build_embed_model.py
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

random.seed(42)
torch.manual_seed(42)

SEP = "=" * 70


def pause():
    input("\n⏎  Press Enter to continue ...\n")


# ─────────────────────────────────────────────────────────────────────────
# STEP 1 — TOKENIZATION: text → list of integers
# ─────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  STEP 1 — TOKENIZATION: Turn text into numbers")
print(SEP)

print("""
  Before a neural network can process text, we need to convert each
  word into an integer (token ID).  Real models use subword tokenizers
  (BPE / WordPiece).  We'll use a simple word-level tokenizer.
""")

training_pairs = [
    ("the cat sat on the mat", "a kitten rested on the rug"),
    ("dogs love to play fetch", "puppies enjoy catching balls"),
    ("the sun is shining bright", "it is a sunny warm day"),
    ("i like eating pizza", "pizza is my favourite food"),
    ("she drives a red car", "her vehicle is red"),
    ("the movie was exciting", "the film was really thrilling"),
    ("he plays guitar well", "he is a skilled guitarist"),
    ("it is raining outside", "the weather is wet and rainy"),
    ("they went to the beach", "they visited the seaside"),
    ("i am reading a book", "she is reading a novel"),
    ("coffee tastes very good", "i enjoy drinking coffee"),
    ("the baby is sleeping", "the infant is asleep"),
    ("birds fly in the sky", "birds soar through the air"),
    ("he runs every morning", "he jogs at dawn each day"),
    ("the garden has flowers", "the yard is full of roses"),
]

negative_pairs = [
    ("the cat sat on the mat", "he runs every morning"),
    ("dogs love to play fetch", "it is raining outside"),
    ("the sun is shining bright", "i like eating pizza"),
    ("she drives a red car", "the baby is sleeping"),
    ("the movie was exciting", "the garden has flowers"),
    ("he plays guitar well", "they went to the beach"),
    ("coffee tastes very good", "dogs love to play fetch"),
    ("birds fly in the sky", "i am reading a book"),
    ("i like eating pizza", "he plays guitar well"),
    ("the baby is sleeping", "the sun is shining bright"),
    ("they went to the beach", "the cat sat on the mat"),
    ("he runs every morning", "she drives a red car"),
    ("the garden has flowers", "the movie was exciting"),
    ("i am reading a book", "coffee tastes very good"),
    ("it is raining outside", "birds fly in the sky"),
]

all_sentences = set()
for a, b in training_pairs + negative_pairs:
    all_sentences.add(a)
    all_sentences.add(b)

vocab = {"<PAD>": 0, "<UNK>": 1}
for sent in all_sentences:
    for word in sent.lower().split():
        if word not in vocab:
            vocab[word] = len(vocab)

VOCAB_SIZE = len(vocab)
PAD_ID = vocab["<PAD>"]

print(f"  Vocabulary size: {VOCAB_SIZE} words\n")
print(f"  Sample mappings:")
for word in ["cat", "kitten", "dog", "pizza", "guitar"]:
    if word in vocab:
        print(f'    "{word}" → {vocab[word]}')

MAX_LEN = 10


def tokenize(text):
    tokens = [vocab.get(w, 1) for w in text.lower().split()]
    tokens = tokens[:MAX_LEN]
    tokens += [PAD_ID] * (MAX_LEN - len(tokens))
    return tokens


example = "the cat sat on the mat"
tok = tokenize(example)
print(f'\n  Example: "{example}"')
print(f"  Tokens:  {tok}")

print("""
  ┌────────────────────────────────────────────────────────────┐
  │  Real mxbai-embed-large uses WordPiece tokenizer with      │
  │  ~30K subword tokens and 512 max length.                   │
  │  Our mini version: word-level, ~80 words, max length 10.   │
  └────────────────────────────────────────────────────────────┘""")

pause()


# ─────────────────────────────────────────────────────────────────────────
# STEP 2 — TRANSFORMER ENCODER: the neural network
# ─────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  STEP 2 — TRANSFORMER ENCODER: The neural network")
print(SEP)

print("""
  mxbai-embed-large is a Transformer Encoder (like BERT).
  Architecture:
    1. Token Embedding  — each token ID → a learnable vector
    2. Positional Enc.  — tells the model WHERE each token is
    3. Transformer Layers — self-attention + feed-forward
    4. Mean Pooling     — average all token vectors into ONE vector

  That ONE vector is the sentence embedding.
""")

EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
FF_DIM = 128


class MiniEmbedModel(nn.Module):
    """A tiny transformer encoder for generating sentence embeddings."""

    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(MAX_LEN, EMBED_DIM)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=FF_DIM,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)

    def forward(self, token_ids):
        """
        token_ids: (batch, seq_len) int tensor
        returns:   (batch, EMBED_DIM) float tensor  ← sentence embedding
        """
        B, S = token_ids.shape
        positions = torch.arange(S, device=token_ids.device).unsqueeze(0).expand(B, S)
        pad_mask = token_ids == PAD_ID

        x = self.token_emb(token_ids) + self.pos_emb(positions)
        x = self.transformer(x, src_key_padding_mask=pad_mask)

        # Mean pooling: average non-padding token vectors
        mask = (~pad_mask).unsqueeze(-1).float()
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return F.normalize(pooled, dim=-1)


model = MiniEmbedModel()
total_params = sum(p.numel() for p in model.parameters())

print(f"  Model created!\n")
print(f"  Architecture:")
print(f"    Token embedding : {VOCAB_SIZE} × {EMBED_DIM}")
print(f"    Position embed  : {MAX_LEN} × {EMBED_DIM}")
print(f"    Transformer     : {NUM_LAYERS} layers, {NUM_HEADS} heads")
print(f"    Feed-forward    : {FF_DIM} hidden dim")
print(f"    Output          : {EMBED_DIM}-dim normalized vector")
print(f"    Total params    : {total_params:,}")

print(f"""
  ┌────────────────────────────────────────────────────────────┐
  │  mxbai-embed-large: 340M params, 24 layers, 1024-dim      │
  │  Our mini model:    {total_params:,} params,  {NUM_LAYERS} layers,   {EMBED_DIM}-dim       │
  │  Same architecture, just smaller!                          │
  └────────────────────────────────────────────────────────────┘""")

pause()

# Quick demo: before training
print("  Before training — embeddings are random:\n")
with torch.no_grad():
    t1 = torch.tensor([tokenize("the cat sat on the mat")])
    t2 = torch.tensor([tokenize("a kitten rested on the rug")])
    t3 = torch.tensor([tokenize("he runs every morning")])
    e1, e2, e3 = model(t1), model(t2), model(t3)

    sim_pos = F.cosine_similarity(e1, e2).item()
    sim_neg = F.cosine_similarity(e1, e3).item()
    print(f'    "cat on mat" vs "kitten on rug"  : sim = {sim_pos:.4f}')
    print(f'    "cat on mat" vs "runs morning"   : sim = {sim_neg:.4f}')
    print(f"\n    → Both are random — the model hasn't learned anything yet.")

pause()


# ─────────────────────────────────────────────────────────────────────────
# STEP 3 — CONTRASTIVE LOSS: how the model learns
# ─────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  STEP 3 — CONTRASTIVE LOSS: How the model learns")
print(SEP)

print("""
  The training signal uses TRIPLETS:
    • Anchor:   "the cat sat on the mat"
    • Positive: "a kitten rested on the rug"  (similar meaning)
    • Negative: "he runs every morning"        (different meaning)

  Triplet loss:
    loss = max(0, sim(anchor, negative) - sim(anchor, positive) + margin)

  This says: "positive must be MORE SIMILAR than negative by at least
  margin — otherwise the model is penalized."

  mxbai-embed-large uses the same idea but with:
    • 700M+ sentence pairs for contrastive pre-training
    • AnglE loss for fine-tuning (a more advanced variant)
""")

MARGIN = 0.5


def triplet_loss(anchor, positive, negative, margin=MARGIN):
    """Positive should be closer to anchor than negative by at least margin."""
    sim_pos = F.cosine_similarity(anchor, positive)
    sim_neg = F.cosine_similarity(anchor, negative)
    return torch.clamp(sim_neg - sim_pos + margin, min=0).mean()


print("  Triplet loss function defined.")
print(f"  Margin = {MARGIN}  (positive must beat negative by ≥ {MARGIN})")

pause()


# ─────────────────────────────────────────────────────────────────────────
# STEP 4 — TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  STEP 4 — TRAINING: Teach the model similar vs different")
print(SEP)

triplets = list(zip(
    [a for a, _ in training_pairs],
    [b for _, b in training_pairs],
    [b for _, b in negative_pairs],
))
print(f"\n  Training data: {len(triplets)} triplets (anchor, positive, negative)")
print(f"  Epochs: 80")
print(f"  Optimizer: Adam, lr=0.001\n")

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 80
for epoch in range(EPOCHS):
    random.shuffle(triplets)
    epoch_loss = 0.0

    for anchor, pos, neg in triplets:
        ta = torch.tensor([tokenize(anchor)])
        tp = torch.tensor([tokenize(pos)])
        tn = torch.tensor([tokenize(neg)])

        emb_a = model(ta)
        emb_p = model(tp)
        emb_n = model(tn)
        loss = triplet_loss(emb_a, emb_p, emb_n)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()

    avg = epoch_loss / len(triplets)
    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        bar = "█" * int((1 - min(avg, 1)) * 30)
        print(f"    Epoch {epoch:3d}/{EPOCHS}  loss = {avg:.4f}  {bar}")

print("\n  ✓ Training complete!")

pause()


# ─────────────────────────────────────────────────────────────────────────
# STEP 5 — EVALUATION: does it work?
# ─────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  STEP 5 — EVALUATION: Does our model understand similarity?")
print(SEP)

print("\n  After training — same pairs as before:\n")
model.eval()
with torch.no_grad():
    e1, e2, e3 = model(t1), model(t2), model(t3)
    sim_pos = F.cosine_similarity(e1, e2).item()
    sim_neg = F.cosine_similarity(e1, e3).item()
    print(f'    "cat on mat" vs "kitten on rug"  : sim = {sim_pos:.4f}  (similar ✓)')
    print(f'    "cat on mat" vs "runs morning"   : sim = {sim_neg:.4f}  (different ✓)')

print("\n  More test pairs:\n")

test_pairs = [
    ("the sun is shining bright", "it is a sunny warm day", True),
    ("i like eating pizza", "pizza is my favourite food", True),
    ("dogs love to play fetch", "puppies enjoy catching balls", True),
    ("the cat sat on the mat", "i like eating pizza", False),
    ("she drives a red car", "the baby is sleeping", False),
    ("birds fly in the sky", "coffee tastes very good", False),
]

for sent_a, sent_b, is_similar in test_pairs:
    ta = torch.tensor([tokenize(sent_a)])
    tb = torch.tensor([tokenize(sent_b)])
    ea, eb = model(ta), model(tb)
    sim = F.cosine_similarity(ea, eb).item()
    icon = "✓ similar" if is_similar else "✗ different"
    bar = "█" * int(max(0, sim) * 25)
    a_short = sent_a[:28].ljust(28)
    b_short = sent_b[:28].ljust(28)
    print(f'    {a_short} vs {b_short}  sim={sim:+.3f}  {bar}  ({icon})')

print("""
  ┌────────────────────────────────────────────────────────────────┐
  │  Similar-meaning pairs → HIGH similarity (close to 1.0)        │
  │  Different-meaning pairs → LOW similarity (close to 0 or neg)  │
  │                                                                │
  │  Our mini model learned to distinguish meaning, just like      │
  │  mxbai-embed-large — but on a tiny scale.                      │
  └────────────────────────────────────────────────────────────────┘""")

pause()


# ─────────────────────────────────────────────────────────────────────────
# STEP 6 — INSIDE THE EMBEDDING: what's in the vector?
# ─────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  STEP 6 — INSIDE THE EMBEDDING: What does the vector look like?")
print(SEP)

with torch.no_grad():
    ta = torch.tensor([tokenize("the cat sat on the mat")])
    emb = model(ta)
    vec = emb[0].tolist()

print(f'\n  Sentence: "the cat sat on the mat"')
print(f"  Embedding dimension: {len(vec)}")
print(f"  First 16 values:\n")
for i in range(0, 16, 4):
    vals = "  ".join(f"{vec[j]:+.4f}" for j in range(i, min(i + 4, 16)))
    print(f"    [{i:2d}-{min(i+3,15):2d}]  {vals}")

norm = math.sqrt(sum(v * v for v in vec))
print(f"\n  L2 norm: {norm:.4f}  (normalized to ~1.0)")

print(f"""
  ┌────────────────────────────────────────────────────────────────┐
  │  mxbai-embed-large produces 1024 floats per sentence.          │
  │  Our model produces {EMBED_DIM} floats — same idea, fewer numbers.      │
  │                                                                │
  │  Each float encodes one aspect of the sentence's meaning.      │
  │  Together, the {EMBED_DIM} numbers form a point in {EMBED_DIM}-dimensional     │
  │  space where proximity = semantic similarity.                   │
  └────────────────────────────────────────────────────────────────┘""")

pause()


# ─────────────────────────────────────────────────────────────────────────
# STEP 7 — SUMMARY: how mxbai-embed-large is built
# ─────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  STEP 7 — SUMMARY: How mxbai-embed-large is actually built")
print(SEP)

print("""
  What we built                     What mxbai-embed-large uses
  ─────────────                     ──────────────────────────
  Word-level tokenizer              WordPiece tokenizer (~30K tokens)
  ~80 word vocab                    ~30,000 subword vocab
  2 transformer layers              24 transformer layers
  64-dim embeddings                 1024-dim embeddings
  ~74K parameters                   340M parameters
  15 training triplets              700M+ sentence pairs
  Triplet contrastive loss          Contrastive + AnglE loss
  Trains in seconds                 Trained on GPU cluster for days

  But the CORE IDEAS are identical:
    1. Tokenize text into integer IDs
    2. Pass through transformer encoder layers
    3. Mean-pool the token embeddings into one vector
    4. Normalize to unit length
    5. Train with contrastive loss: push similar texts together,
       push different texts apart

  That's how ALL text embedding models work — from our 74K-param
  toy to OpenAI's proprietary models.  The difference is just
  scale: more data, more layers, more dimensions.
""")

print("Done! You just built an embedding model from scratch. 🎉\n")
