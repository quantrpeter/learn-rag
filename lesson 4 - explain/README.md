# Lesson 4 – Build an Embedding Model (Explainer Video)

Animated video explaining how embedding models like mxbai-embed-large are built from scratch, narrated in Cantonese using Edge TTS.

Covers tokenization, transformer encoder architecture, mean pooling, triplet loss, and training with rich diagrams and animations.

## Prerequisites

```bash
pip install manim manim-voiceover edge-tts
```

## Render

```bash
cd "lesson 4 -explain"

# Medium quality (720p, recommended)
manim render -qm scene.py EmbedModelExplainer

# High quality (1080p)
manim render -qh scene.py EmbedModelExplainer

# Low quality (480p, fast iteration)
manim render -ql scene.py EmbedModelExplainer
```

Output goes to `media/videos/scene/`.

## Files

| File | Purpose |
|---|---|
| `scene.py` | Main Manim scene — all animations, diagrams and voiceover |
| `edge_tts_service.py` | Custom Edge TTS speech service for manim-voiceover |
| `wallpaper1.jpg` | Background image |

## Scenes

1. **Intro** — Title, producer credit (Peter), watermark (香港編程學會)
2. **Big Picture** — Side-by-side comparison: mini model vs mxbai-embed-large
3. **Tokenization** — Animated word → integer ID flow with PAD padding
4. **Architecture** — Full transformer encoder pipeline diagram with data flow animation
5. **Mean Pooling** — Multiple token vectors merging into one sentence vector
6. **Triplet Loss** — 2D vector space with push/pull arrows showing contrastive learning
7. **Training** — Before vs after similarity bar charts
8. **GitHub** — Link to source code
9. **Summary** — 5 core steps (Tokenize, Embed, Transform, Pool, Train)

## Voice

Uses Microsoft Edge TTS voice `zh-HK-HiuMaanNeural` (Cantonese, female).
