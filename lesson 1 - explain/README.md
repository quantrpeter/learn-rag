# Lesson 1 – RAG Explainer Video

Animated video explaining the RAG pipeline, narrated in Cantonese using Edge TTS.

## Prerequisites

```bash
pip install manim manim-voiceover edge-tts
```

## Render

```bash
cd "lesson 1 - explain"

# Medium quality (720p, recommended for preview)
manim render -qm scene.py RAGExplainer

# High quality (1080p)
manim render -qh scene.py RAGExplainer

# Low quality (480p, fast iteration)
manim render -ql scene.py RAGExplainer
```

Output goes to `media/videos/scene/`.

## Files

| File | Purpose |
|---|---|
| `scene.py` | Main Manim scene — all animations and voiceover text |
| `edge_tts_service.py` | Custom Edge TTS speech service for manim-voiceover |

## Voice

Uses Microsoft Edge TTS voice `zh-HK-HiuMaanNeural` (Cantonese, female).
Other available Cantonese voices: `zh-HK-HiuGaaiNeural`, `zh-HK-WanLungNeural`.
