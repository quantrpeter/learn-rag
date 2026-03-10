"""
Lesson 3 – Retrieve & Generate Explainer
Manim CE + Cantonese voiceover (Edge TTS)

Render:
    cd "lesson 3 - explain retrieve"
    manim render -qm scene.py RetrieveGenerateExplainer
"""

from manim import *
from manim_voiceover import VoiceoverScene

from edge_tts_service import EdgeTTSService

# ── colour palette ───────────────────────────────────────────────────────
C_BG = "#0f0f23"
C_BLUE = "#4fc3f7"
C_GREEN = "#81c784"
C_ORANGE = "#ffb74d"
C_PINK = "#f48fb1"
C_PURPLE = "#ce93d8"
C_YELLOW = "#fff176"
C_WHITE = "#e0e0e0"
C_RED = "#ef5350"
C_DIM = "#555577"
C_CYAN = "#80deea"

FONT = "Noto Sans CJK HK"
GITHUB_URL = "github.com/quantrpeter/learn-rag"

# ── code snippets from rag.py ────────────────────────────────────────────

CODE_RETRIEVE = """\
query_embedding = ollama.embed(
    model=EMBED_MODEL, input=query
)["embeddings"][0]

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=n_results
)
return results["documents"][0]"""

CODE_PROMPT = """\
context = "\\n---\\n".join(context_chunks)

prompt = (
    "Use ONLY the following context "
    "to answer the question.\\n"
    f"Context:\\n{context}\\n"
    f"Question: {query}"
)"""

CODE_GENERATE = """\
response = ollama.chat(
    model=LLM_MODEL,
    messages=[{
        "role": "user",
        "content": prompt
    }]
)
return response["message"]["content"]"""


class RetrieveGenerateExplainer(VoiceoverScene):
    """Lesson 3: deep dive into Retrieve & Generate."""

    def construct(self):
        self.set_speech_service(
            EdgeTTSService(voice="zh-HK-HiuMaanNeural", rate="+0%")
        )
        self.camera.background_color = BLACK

        self.bg_image = ImageMobject("wallpaper1.jpg")
        self.bg_image.height = config.frame_height
        self.bg_image.width = config.frame_width
        self.add(self.bg_image)

        self.bg_overlay = Rectangle(
            width=config.frame_width + 0.5,
            height=config.frame_height + 0.5,
            fill_color=BLACK, fill_opacity=0.55, stroke_width=0,
        )
        self.add(self.bg_overlay)

        self.watermark = self.zh(
            "香港編程學會", font_size=18, color="#9999bb"
        ).to_corner(UL, buff=0.3)
        self.add(self.watermark)

        self.scene_intro()
        self.scene_pipeline_recap()
        self.scene_retrieve_flow()
        self.scene_retrieve_code()
        self.scene_generate_flow()
        self.scene_prompt_build()
        self.scene_generate_code()
        self.scene_end_to_end()
        self.scene_github()
        self.scene_summary()

    # ── helpers ───────────────────────────────────────────────────────────

    def zh(self, txt, **kw):
        kw.setdefault("font", FONT)
        kw.setdefault("color", C_WHITE)
        return Text(txt, **kw)

    def en(self, txt, **kw):
        kw.setdefault("color", C_WHITE)
        return Text(txt, **kw)

    def clear(self):
        keep = {self.bg_image, self.bg_overlay, self.watermark}
        to_fade = [m for m in self.mobjects if m not in keep]
        if to_fade:
            self.play(*[FadeOut(m) for m in to_fade], run_time=0.5)

    def make_box(self, label, color, width=2.5, height=0.8, font_size=24):
        rect = RoundedRectangle(
            corner_radius=0.15, width=width, height=height,
            fill_color=color, fill_opacity=0.25, stroke_color=color,
        )
        txt = self.zh(label, font_size=font_size, color=color).move_to(rect)
        return VGroup(rect, txt)

    def make_en_box(self, label, color, width=2.5, height=0.8, font_size=24):
        rect = RoundedRectangle(
            corner_radius=0.15, width=width, height=height,
            fill_color=color, fill_opacity=0.25, stroke_color=color,
        )
        txt = self.en(label, font_size=font_size, color=color).move_to(rect)
        return VGroup(rect, txt)

    def make_code(self, code_str, font_size=18):
        return Code(
            code_string=code_str, language="python",
            formatter_style="monokai", add_line_numbers=False,
            background="rectangle",
            background_config={"stroke_color": C_DIM, "stroke_width": 1},
            paragraph_config={"font_size": font_size},
        )

    def make_sim_bar(self, label, value, max_width=5.0, color=C_CYAN):
        bg = Rectangle(
            width=max_width, height=0.3,
            fill_color=WHITE, fill_opacity=0.05,
            stroke_color=C_DIM, stroke_width=0.5,
        )
        bar = Rectangle(
            width=max_width * value, height=0.3,
            fill_color=color, fill_opacity=0.5, stroke_width=0,
        ).align_to(bg, LEFT)
        lbl = self.en(label, font_size=13, color=C_WHITE).next_to(bg, LEFT, buff=0.15)
        val = self.en(f"{value:.2f}", font_size=14, color=C_YELLOW).next_to(bg, RIGHT, buff=0.15)
        return VGroup(bg, bar, lbl, val)

    # ── Scene 1 — Intro ──────────────────────────────────────────────────

    def scene_intro(self):
        title = self.zh("Retrieve & Generate 深入解析", font_size=46, color=C_BLUE)
        sub = self.zh(
            "了解點樣搵資料同生成答案", font_size=28, color=C_ORANGE
        ).next_to(title, DOWN, buff=0.4)
        lesson = self.en(
            "Lesson 3", font_size=22, color=C_DIM
        ).next_to(sub, DOWN, buff=0.3)
        producer = self.zh(
            "制片人：Peter", font_size=22, color=C_DIM
        ).next_to(lesson, DOWN, buff=0.5)

        with self.voiceover(
            text="大家好！歡迎嚟到第三課。今日我哋深入了解 Retrieve 同 Generate "
            "呢兩個步驟，即係 RAG 入面最核心嘅部分：搵資料同生成答案。"
        ):
            self.play(Write(title), run_time=1.2)
            self.play(FadeIn(sub, shift=UP * 0.2), run_time=0.8)
            self.play(FadeIn(lesson, shift=UP * 0.2), run_time=0.5)
            self.play(FadeIn(producer, shift=UP * 0.2), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 2 — Pipeline Recap ─────────────────────────────────────────

    def scene_pipeline_recap(self):
        heading = self.zh(
            "RAG 五個步驟 — 今日重點", font_size=36, color=C_BLUE
        ).to_edge(UP, buff=0.5)

        steps = [
            ("LOAD", "載入", C_DIM),
            ("EMBED", "向量化", C_DIM),
            ("STORE", "儲存", C_DIM),
            ("RETRIEVE", "檢索", C_PINK),
            ("GENERATE", "生成", C_YELLOW),
        ]

        boxes = []
        for eng, zh_label, col in steps:
            rect = RoundedRectangle(
                corner_radius=0.1, width=1.6, height=0.7,
                fill_color=col, fill_opacity=0.15, stroke_color=col,
            )
            t_en = self.en(eng, font_size=14, color=col)
            t_zh = self.zh(zh_label, font_size=14, color=C_WHITE if col != C_DIM else C_DIM)
            label = VGroup(t_en, t_zh).arrange(DOWN, buff=0.05).move_to(rect)
            boxes.append(VGroup(rect, label))

        row = VGroup(*boxes).arrange(RIGHT, buff=0.45)
        row.next_to(heading, DOWN, buff=0.6)

        arrows = []
        for i in range(len(boxes) - 1):
            a = Arrow(
                boxes[i].get_right(), boxes[i + 1].get_left(),
                buff=0.08, color=C_DIM, stroke_width=2,
                max_tip_length_to_length_ratio=0.2,
            )
            arrows.append(a)

        done_label = self.zh(
            "✓ 已學", font_size=16, color=C_DIM
        ).next_to(VGroup(*boxes[:3]), DOWN, buff=0.3)

        focus_label = self.zh(
            "★ 今日重點", font_size=18, color=C_YELLOW
        ).next_to(VGroup(*boxes[3:]), DOWN, buff=0.3)

        with self.voiceover(
            text="之前兩課我哋學咗 Load、Embed 同 Store，即係點樣將數據載入、"
            "向量化同儲存入 ChromaDB。今日我哋專注最後兩步："
            "Retrieve 搵資料同 Generate 生成答案。呢兩步就係 RAG 嘅核心。"
        ):
            self.play(Write(heading), run_time=0.6)
            for i, box in enumerate(boxes):
                self.play(FadeIn(box, shift=UP * 0.15), run_time=0.3)
                if i < len(arrows):
                    self.play(GrowArrow(arrows[i]), run_time=0.15)
            self.play(FadeIn(done_label), run_time=0.4)
            self.play(
                boxes[3][0].animate.set_fill(opacity=0.4).set_stroke(width=3),
                boxes[4][0].animate.set_fill(opacity=0.4).set_stroke(width=3),
                FadeIn(focus_label, scale=1.2),
                run_time=0.8,
            )

        self.wait(0.3)
        self.clear()

    # ── Scene 3 — Retrieve Flow ──────────────────────────────────────────

    def scene_retrieve_flow(self):
        heading = self.zh(
            "Step 4: RETRIEVE — 檢索相關文檔", font_size=36, color=C_PINK
        ).to_edge(UP, buff=0.4)

        query_box = self.make_en_box(
            "Peter 住喺邊度？", C_YELLOW, width=4, height=0.7
        )
        embed_box = self.make_en_box(
            "Embed Model", C_PURPLE, width=3, height=0.7
        )
        vec_box = self.make_en_box(
            "Query Vector", C_ORANGE, width=3, height=0.7
        )
        db_box = self.make_box(
            "ChromaDB", C_CYAN, width=3, height=0.7
        )
        results_box = self.make_box(
            "Top K 文檔", C_GREEN, width=3, height=0.7
        )

        top_row = VGroup(query_box, embed_box, vec_box).arrange(RIGHT, buff=0.8)
        top_row.move_to(UP * 1.0)
        bot_row = VGroup(db_box, results_box).arrange(RIGHT, buff=1.5)
        bot_row.move_to(DOWN * 1.0)
        db_box.align_to(vec_box, RIGHT)

        arr1 = Arrow(
            query_box.get_right(), embed_box.get_left(),
            buff=0.1, color=C_WHITE, stroke_width=2.5,
        )
        arr2 = Arrow(
            embed_box.get_right(), vec_box.get_left(),
            buff=0.1, color=C_WHITE, stroke_width=2.5,
        )
        arr3 = Arrow(
            vec_box.get_bottom(), db_box.get_top(),
            buff=0.1, color=C_WHITE, stroke_width=2.5,
        )
        arr4 = Arrow(
            db_box.get_left(), results_box.get_right(),
            buff=0.1, color=C_GREEN, stroke_width=2.5,
        )

        note = self.zh(
            "⚠ 一定要用同 STORE 一樣嘅 Embedding Model",
            font_size=18, color=C_RED,
        ).to_edge(DOWN, buff=0.6)

        with self.voiceover(
            text="RETRIEVE 嘅流程係咁嘅。首先，將用戶嘅問題傳入同一個 Embedding 模型，"
            "向量化成一個 Query Vector。然後將呢個 Query Vector 傳入 ChromaDB，"
            "佢會用 cosine similarity 搵出同問題最接近嘅 Top K 個文檔。"
            "注意，一定要用同 STORE 嗰陣同一個 Embedding 模型，否則向量空間唔同，"
            "比較結果就完全冇意義。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(query_box, shift=RIGHT * 0.2), run_time=0.5)
            self.play(GrowArrow(arr1), run_time=0.3)
            self.play(FadeIn(embed_box, shift=RIGHT * 0.2), run_time=0.5)
            self.play(GrowArrow(arr2), run_time=0.3)
            self.play(FadeIn(vec_box, shift=RIGHT * 0.2), run_time=0.5)
            self.play(GrowArrow(arr3), run_time=0.3)
            self.play(FadeIn(db_box, shift=UP * 0.2), run_time=0.5)
            self.play(GrowArrow(arr4), run_time=0.3)
            self.play(FadeIn(results_box, shift=LEFT * 0.2), run_time=0.5)
            self.play(FadeIn(note, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 4 — Retrieve Code ──────────────────────────────────────────

    def scene_retrieve_code(self):
        heading = self.zh(
            "retrieve() 函數代碼", font_size=36, color=C_PINK
        ).to_edge(UP, buff=0.4)

        code_block = self.make_code(CODE_RETRIEVE, font_size=16)
        code_block.move_to(ORIGIN + LEFT * 1.5)

        note1 = self.zh(
            "① 用同一個 model 向量化 query",
            font_size=18, color=C_ORANGE,
        )
        note2 = self.zh(
            "② collection.query 搵最近嘅向量",
            font_size=18, color=C_CYAN,
        )
        note3 = self.zh(
            "③ 返回原文 documents，唔係向量",
            font_size=18, color=C_GREEN,
        )
        notes = VGroup(note1, note2, note3).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        notes.next_to(code_block, RIGHT, buff=0.6)

        with self.voiceover(
            text="我哋睇下 retrieve 函數嘅代碼。"
            "第一行，用 ollama.embed 將 query 向量化，留意用嘅係同一個 EMBED_MODEL。"
            "第二行，用 collection.query 搜尋，傳入 query 向量同 n_results，"
            "即係想返回幾多個最接近嘅結果。"
            "最後返回 results documents，注意返回嘅係原文文字，唔係向量數字。"
            "ChromaDB 幫你保存咗原文，所以可以直接拎返嚟用。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(code_block, shift=UP * 0.2), run_time=0.8)
            self.play(FadeIn(note1, shift=LEFT * 0.2), run_time=0.8)
            self.play(FadeIn(note2, shift=LEFT * 0.2), run_time=0.8)
            self.play(FadeIn(note3, shift=LEFT * 0.2), run_time=0.8)

        self.wait(0.3)
        self.clear()

    # ── Scene 5 — Generate Flow ──────────────────────────────────────────

    def scene_generate_flow(self):
        heading = self.zh(
            "Step 5: GENERATE — 生成答案", font_size=36, color=C_YELLOW
        ).to_edge(UP, buff=0.4)

        chunk1 = self.make_en_box("Context Chunk 1", C_GREEN, width=3.2, height=0.55, font_size=16)
        chunk2 = self.make_en_box("Context Chunk 2", C_GREEN, width=3.2, height=0.55, font_size=16)
        chunk3 = self.make_en_box("Context Chunk 3", C_GREEN, width=3.2, height=0.55, font_size=16)
        chunks = VGroup(chunk1, chunk2, chunk3).arrange(DOWN, buff=0.15)
        chunks.move_to(LEFT * 4 + DOWN * 0.2)

        chunks_label = self.zh(
            "RETRIEVE 搵到嘅文檔", font_size=16, color=C_DIM,
        ).next_to(chunks, UP, buff=0.2)

        query_box = self.make_en_box("User Query", C_YELLOW, width=2.5, height=0.55, font_size=16)
        query_box.next_to(chunks, DOWN, buff=0.3)

        prompt_box = RoundedRectangle(
            corner_radius=0.15, width=3.5, height=2.5,
            fill_color=C_ORANGE, fill_opacity=0.15, stroke_color=C_ORANGE,
        )
        prompt_label = self.zh("Prompt", font_size=22, color=C_ORANGE)
        prompt_items = VGroup(
            self.en("System instruction", font_size=13, color=C_DIM),
            self.en("+ Context chunks", font_size=13, color=C_GREEN),
            self.en("+ User question", font_size=13, color=C_YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        prompt_content = VGroup(prompt_label, prompt_items).arrange(DOWN, buff=0.25)
        prompt_content.move_to(prompt_box)
        prompt_group = VGroup(prompt_box, prompt_content)
        prompt_group.move_to(DOWN * 0.2)

        llm_box = self.make_box("LLM", C_PURPLE, width=2.5, height=0.8)
        llm_box.move_to(RIGHT * 3.5 + UP * 0.3)

        answer_box = self.make_box("答案", C_CYAN, width=2.5, height=0.8)
        answer_box.next_to(llm_box, DOWN, buff=0.7)

        arr1 = Arrow(
            chunks.get_right(), prompt_box.get_left(),
            buff=0.15, color=C_WHITE, stroke_width=2,
        )
        arr1b = Arrow(
            query_box.get_right(), prompt_box.get_left() + DOWN * 0.4,
            buff=0.15, color=C_WHITE, stroke_width=2,
        )
        arr2 = Arrow(
            prompt_box.get_right(), llm_box.get_left(),
            buff=0.15, color=C_WHITE, stroke_width=2,
        )
        arr3 = Arrow(
            llm_box.get_bottom(), answer_box.get_top(),
            buff=0.15, color=C_CYAN, stroke_width=2.5,
        )

        with self.voiceover(
            text="到 GENERATE 步驟。我哋拎到 Retrieve 返嚟嘅文檔之後，"
            "要將佢哋同用戶嘅問題合併成一個 Prompt。"
            "呢個 Prompt 包含三個部分：系統指令、context 文檔、同用戶問題。"
            "然後將成個 Prompt 傳俾 LLM，佢就會根據 context 生成答案。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(
                FadeIn(chunks_label),
                FadeIn(chunk1, shift=RIGHT * 0.2),
                FadeIn(chunk2, shift=RIGHT * 0.2),
                FadeIn(chunk3, shift=RIGHT * 0.2),
                run_time=0.6,
            )
            self.play(FadeIn(query_box, shift=RIGHT * 0.2), run_time=0.4)
            self.play(GrowArrow(arr1), GrowArrow(arr1b), run_time=0.4)
            self.play(FadeIn(prompt_group, shift=RIGHT * 0.2), run_time=0.6)
            self.play(GrowArrow(arr2), run_time=0.3)
            self.play(FadeIn(llm_box, shift=RIGHT * 0.2), run_time=0.5)
            self.play(GrowArrow(arr3), run_time=0.3)
            self.play(FadeIn(answer_box, shift=UP * 0.2), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 6 — Prompt Construction ────────────────────────────────────

    def scene_prompt_build(self):
        heading = self.zh(
            "Prompt 點樣構建？", font_size=36, color=C_ORANGE
        ).to_edge(UP, buff=0.4)

        prompt_lines = [
            ("System:", "You are a helpful assistant.", C_DIM, C_WHITE),
            ("指令:", "Use ONLY the following context", C_RED, C_RED),
            ("", "to answer the question.", C_RED, C_RED),
            ("", "", None, None),
            ("Context:", '"Peter lives in Tai Wai"', C_GREEN, C_GREEN),
            ("", "---", C_DIM, C_DIM),
            ("", '"Peter Cheung is chairman..."', C_GREEN, C_GREEN),
            ("", "", None, None),
            ("Question:", "Peter 住喺邊度？", C_YELLOW, C_YELLOW),
        ]

        prompt_texts = VGroup()
        for prefix, content, pcol, ccol in prompt_lines:
            if pcol is None:
                prompt_texts.add(self.en(" ", font_size=4))
                continue
            parts = VGroup()
            if prefix:
                parts.add(self.en(prefix, font_size=15, color=pcol))
            parts.add(self.en(content, font_size=15, color=ccol))
            parts.arrange(RIGHT, buff=0.15)
            prompt_texts.add(parts)

        prompt_texts.arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        prompt_rect = SurroundingRectangle(
            prompt_texts, buff=0.3, color=C_ORANGE,
            corner_radius=0.1, stroke_width=1.5,
        )
        prompt_title = self.en(
            "prompt", font_size=16, color=C_ORANGE
        ).next_to(prompt_rect, UP, buff=0.1, aligned_edge=LEFT)
        prompt_group = VGroup(prompt_title, prompt_rect, prompt_texts)
        prompt_group.move_to(ORIGIN + LEFT * 1.5 + DOWN * 0.3)

        key_note = self.zh(
            "「Use ONLY the following context」",
            font_size=20, color=C_RED,
        )
        key_explain = self.zh(
            "→ 限制 LLM 只用提供嘅資料回答",
            font_size=18, color=C_WHITE,
        )
        key_explain2 = self.zh(
            "→ 大幅減少幻覺（亂噏）問題",
            font_size=18, color=C_GREEN,
        )
        key_group = VGroup(key_note, key_explain, key_explain2).arrange(
            DOWN, aligned_edge=LEFT, buff=0.2,
        )
        key_group.next_to(prompt_group, RIGHT, buff=0.5)

        with self.voiceover(
            text="Prompt 嘅構建好重要。最上面係系統指令，明確話俾 LLM 知 "
            "Use ONLY the following context to answer the question。"
            "呢句係關鍵，限制咗 LLM 只可以用我哋提供嘅 context 嚟回答，"
            "唔好自己作答案。中間係 context，即係 Retrieve 搵到嘅文檔，"
            "用分隔線隔開。最後係用戶嘅問題。"
            "咁樣嘅 prompt 設計可以大幅減少幻覺問題，"
            "即係 LLM 唔會再亂噏。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(Create(prompt_rect), FadeIn(prompt_title), run_time=0.5)
            for line in prompt_texts:
                self.play(FadeIn(line, shift=RIGHT * 0.1), run_time=0.3)
            self.play(
                FadeIn(key_note, shift=LEFT * 0.2),
                run_time=0.6,
            )
            self.play(FadeIn(key_explain, shift=LEFT * 0.2), run_time=0.5)
            self.play(FadeIn(key_explain2, shift=LEFT * 0.2), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 7 — Generate Code ──────────────────────────────────────────

    def scene_generate_code(self):
        heading = self.zh(
            "generate_answer() 函數代碼", font_size=36, color=C_YELLOW
        ).to_edge(UP, buff=0.4)

        code_prompt = self.make_code(CODE_PROMPT, font_size=15)
        code_chat = self.make_code(CODE_GENERATE, font_size=15)

        code_prompt.move_to(LEFT * 3.2 + DOWN * 0.5)
        code_chat.move_to(RIGHT * 2.8 + DOWN * 0.5)

        label1 = self.zh(
            "① 構建 Prompt", font_size=18, color=C_ORANGE,
        ).next_to(code_prompt, UP, buff=0.2)
        label2 = self.zh(
            "② 傳俾 LLM", font_size=18, color=C_PURPLE,
        ).next_to(code_chat, UP, buff=0.2)

        arr = Arrow(
            code_prompt.get_right(), code_chat.get_left(),
            buff=0.2, color=C_WHITE, stroke_width=2,
        )

        note = self.zh(
            "LLM_MODEL = llama3.2（唔係 embedding model！）",
            font_size=16, color=C_DIM,
        ).to_edge(DOWN, buff=0.5)

        with self.voiceover(
            text="代碼分兩部分。左邊，先用 join 將所有 context chunks 用分隔線合併，"
            "然後構建完整嘅 prompt 字串。"
            "右邊，用 ollama.chat 將 prompt 傳俾 LLM 生成答案。"
            "注意，呢度用嘅係 LLM_MODEL，即係 llama3.2，"
            "唔係 embedding model。Embedding model 係用嚟向量化，"
            "LLM 先係用嚟生成答案。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(label1), FadeIn(code_prompt, shift=UP * 0.2), run_time=0.8)
            self.play(GrowArrow(arr), run_time=0.4)
            self.play(FadeIn(label2), FadeIn(code_chat, shift=UP * 0.2), run_time=0.8)
            self.play(FadeIn(note, shift=UP * 0.1), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 8 — End-to-End Demo ────────────────────────────────────────

    def scene_end_to_end(self):
        heading = self.zh(
            "完整示範", font_size=36, color=C_BLUE
        ).to_edge(UP, buff=0.4)

        q_text = self.zh(
            '問：「Peter 住喺邊度？」', font_size=24, color=C_YELLOW,
        ).next_to(heading, DOWN, buff=0.4)

        step_r = self.zh("RETRIEVE", font_size=18, color=C_PINK)
        step_r_box = SurroundingRectangle(
            step_r, buff=0.1, color=C_PINK, corner_radius=0.08, stroke_width=1.5,
        )
        step_r_group = VGroup(step_r_box, step_r)
        step_r_group.next_to(q_text, DOWN, buff=0.4).shift(LEFT * 4)

        results = [
            ('"Peter lives in Tai Wai"', "0.77", C_GREEN),
            ('"Peter Cheung is chairman..."', "0.46", C_BLUE),
            ('"Hong Kong is in southern..."', "0.39", C_ORANGE),
        ]
        result_items = VGroup()
        for text, score, color in results:
            t = self.en(f"{text}  ({score})", font_size=13, color=color)
            result_items.add(t)
        result_items.arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        result_items.next_to(step_r_group, RIGHT, buff=0.3)

        arr_r = Arrow(
            step_r_group.get_right(), result_items.get_left(),
            buff=0.1, color=C_WHITE, stroke_width=2,
        )

        step_g = self.zh("GENERATE", font_size=18, color=C_YELLOW)
        step_g_box = SurroundingRectangle(
            step_g, buff=0.1, color=C_YELLOW, corner_radius=0.08, stroke_width=1.5,
        )
        step_g_group = VGroup(step_g_box, step_g)
        step_g_group.next_to(step_r_group, DOWN, buff=0.8)

        prompt_preview = VGroup(
            self.en("prompt = instruction", font_size=12, color=C_DIM),
            self.en("      + context (top 3)", font_size=12, color=C_GREEN),
            self.en("      + question", font_size=12, color=C_YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        prompt_preview.next_to(step_g_group, RIGHT, buff=0.3)

        arr_g = Arrow(
            step_g_group.get_right(), prompt_preview.get_left(),
            buff=0.1, color=C_WHITE, stroke_width=2,
        )

        llm_label = self.en(
            "→ ollama.chat(llama3.2)", font_size=14, color=C_PURPLE,
        ).next_to(prompt_preview, RIGHT, buff=0.3)

        answer = self.zh(
            '答：「Peter 住喺大圍」 ✓', font_size=26, color=C_GREEN,
        ).to_edge(DOWN, buff=0.8)

        answer_note = self.zh(
            "答案完全基於你嘅數據，唔係 LLM 自己作", font_size=16, color=C_DIM,
        ).next_to(answer, DOWN, buff=0.2)

        with self.voiceover(
            text="最後嚟一個完整示範。用戶問 Peter 住喺邊度。"
            "Retrieve 步驟將問題向量化，喺 ChromaDB 搵到三個最相關嘅文檔。"
            "最相似嘅係 Peter lives in Tai Wai，相似度 0.77。"
            "然後 Generate 步驟將呢三個文檔加埋問題構建成 prompt，"
            "傳俾 llama3.2。"
            "LLM 根據 context 回答：Peter 住喺大圍。"
            "答案完全基於你提供嘅數據，唔係 LLM 自己作嘅。"
        ):
            self.play(Write(heading), run_time=0.5)
            self.play(FadeIn(q_text, shift=DOWN * 0.2), run_time=0.6)
            self.play(FadeIn(step_r_group, shift=RIGHT * 0.2), run_time=0.4)
            self.play(GrowArrow(arr_r), run_time=0.3)
            for item in result_items:
                self.play(FadeIn(item, shift=RIGHT * 0.1), run_time=0.3)
            self.play(FadeIn(step_g_group, shift=RIGHT * 0.2), run_time=0.4)
            self.play(GrowArrow(arr_g), run_time=0.3)
            self.play(FadeIn(prompt_preview, shift=RIGHT * 0.1), run_time=0.5)
            self.play(FadeIn(llm_label, shift=RIGHT * 0.1), run_time=0.4)
            self.play(
                FadeIn(answer, shift=UP * 0.3),
                run_time=0.8,
            )
            self.play(FadeIn(answer_note, shift=UP * 0.1), run_time=0.4)

        self.wait(0.3)
        self.clear()

    # ── Scene 9 — GitHub ─────────────────────────────────────────────────

    def scene_github(self):
        heading = self.zh(
            "完整代碼", font_size=40, color=C_BLUE
        ).to_edge(UP, buff=0.6)

        url_text = self.en(GITHUB_URL, font_size=26, color=C_GREEN)
        folder = self.en(
            "lesson 1 - overall / rag.py",
            font_size=22, color=C_ORANGE,
        ).next_to(url_text, DOWN, buff=0.35)

        funcs = self.en(
            "retrieve()  &  generate_answer()",
            font_size=20, color=C_PINK,
        ).next_to(folder, DOWN, buff=0.25)

        url_group = VGroup(url_text, folder, funcs).move_to(ORIGIN)
        box = SurroundingRectangle(
            url_group, buff=0.4, color=C_BLUE,
            corner_radius=0.15, stroke_width=2,
        )

        with self.voiceover(
            text="完整嘅 source code 可以喺 GitHub 搵到。"
            "去 lesson 1 overall 文件夾嘅 rag.py，"
            "入面嘅 retrieve 同 generate_answer 函數就係今日講嘅所有內容。"
            "大家記得去睇下，自己跑一次。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(Create(box), FadeIn(url_text), run_time=1)
            self.play(FadeIn(folder, shift=UP * 0.2), run_time=0.8)
            self.play(FadeIn(funcs, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 10 — Summary ───────────────────────────────────────────────

    def scene_summary(self):
        heading = self.zh(
            "總結", font_size=44, color=C_BLUE
        ).to_edge(UP, buff=0.6)

        bullets = VGroup(
            self.zh(
                "•  RETRIEVE：向量化問題 → cosine similarity → Top K 文檔",
                font_size=22, color=C_PINK,
            ),
            self.zh(
                "•  一定要用同 STORE 一樣嘅 Embedding Model",
                font_size=22, color=C_RED,
            ),
            self.zh(
                "•  GENERATE：context + query → 構建 prompt → LLM 生成",
                font_size=22, color=C_YELLOW,
            ),
            self.zh(
                "•  Prompt 要明確限制 LLM 只用 context 回答",
                font_size=22, color=C_ORANGE,
            ),
            self.zh(
                "•  兩個 model：Embed Model 向量化，LLM 生成答案",
                font_size=22, color=C_PURPLE,
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        bullets.move_to(ORIGIN)

        thanks = self.zh("多謝收睇！", font_size=52, color=C_YELLOW)

        with self.voiceover(
            text="總結一下。RETRIEVE 就係將問題向量化，用 cosine similarity "
            "搵出最相關嘅 Top K 個文檔。記住一定要用同 Store 一樣嘅 embedding model。"
            "GENERATE 就係將搵到嘅文檔同問題合併成 prompt，傳俾 LLM 生成答案。"
            "Prompt 設計好重要，要明確限制 LLM 只用 context 回答。"
            "另外要記住，成個 RAG 系統用咗兩個 model：embedding model 負責向量化，"
            "LLM 負責生成答案，唔好搞混。"
        ):
            self.play(Write(heading), run_time=0.6)
            for b in bullets:
                self.play(FadeIn(b, shift=RIGHT * 0.3), run_time=0.5)

        self.wait(0.5)
        self.clear()

        with self.voiceover(
            text="多謝收睇！大家記得自己跑一次 rag.py，親身體驗下 Retrieve 同 Generate 嘅效果。"
            "下一課見！"
        ):
            self.play(Write(thanks), run_time=1)

        self.wait(1)
        self.play(FadeOut(thanks), run_time=0.8)
