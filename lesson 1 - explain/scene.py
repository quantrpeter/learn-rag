"""
Lesson 1 – RAG Overall Explainer
Manim CE + Cantonese voiceover (Edge TTS)

Render:
    cd "lesson 1 - explain"
    manim render -qm scene.py RAGExplainer
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

FONT = "Noto Sans CJK HK"
GITHUB_URL = "github.com/quantrpeter/learn-rag"

# ── code snippets from rag.py ────────────────────────────────────────────

CODE_LOAD = """\
def load_documents(path):
    with open(path) as f:
        data = json.load(f)
    return data"""

CODE_EMBED = """\
for doc in documents:
    text = doc["text"]
    response = ollama.embed(
        model=EMBED_MODEL, input=text
    )
    embedding = response["embeddings"][0]"""

CODE_STORE = """\
client = chromadb.PersistentClient(
    path="./chroma_db"
)
collection = client.create_collection(
    name=COLLECTION_NAME
)
collection.add(
    ids=ids, documents=texts,
    embeddings=embeddings
)"""

CODE_RETRIEVE = """\
query_embedding = ollama.embed(
    model=EMBED_MODEL, input=query
)["embeddings"][0]

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=TOP_K
)"""

CODE_GENERATE = """\
context = "\\n---\\n".join(context_chunks)
prompt = (
    "Use ONLY the following context "
    "to answer ...\\n"
    f"Context:\\n{context}\\n"
    f"Question: {query}"
)
response = ollama.chat(
    model=LLM_MODEL,
    messages=[{"role": "user",
               "content": prompt}]
)"""


class RAGExplainer(VoiceoverScene):
    """Single scene that explains RAG end-to-end in Cantonese."""

    def construct(self):
        self.set_speech_service(
            EdgeTTSService(voice="zh-HK-HiuMaanNeural", rate="+0%")
        )
        self.camera.background_color = C_BG

        self.watermark = self.zh(
            "香港編程學會", font_size=18, color=C_DIM
        ).to_corner(UL, buff=0.3)
        self.add(self.watermark)

        self.scene_intro()
        self.scene_problem()
        self.scene_rag_idea()
        self.scene_pipeline()
        self.scene_github()
        self.scene_summary()

    # ── text helpers ─────────────────────────────────────────────────────

    def zh(self, txt, **kw):
        kw.setdefault("font", FONT)
        kw.setdefault("color", C_WHITE)
        return Text(txt, **kw)

    def en(self, txt, **kw):
        kw.setdefault("color", C_WHITE)
        return Text(txt, **kw)

    def clear(self):
        to_fade = [m for m in self.mobjects if m is not self.watermark]
        if to_fade:
            self.play(*[FadeOut(m) for m in to_fade], run_time=0.5)

    def make_box(self, label, color, width=2.5, height=0.8):
        rect = RoundedRectangle(
            corner_radius=0.15,
            width=width,
            height=height,
            fill_color=color,
            fill_opacity=0.25,
            stroke_color=color,
        )
        txt = self.zh(label, font_size=24, color=color).move_to(rect)
        return VGroup(rect, txt)

    def make_code(self, code_str, font_size=18):
        return Code(
            code_string=code_str,
            language="python",
            formatter_style="monokai",
            add_line_numbers=False,
            background="rectangle",
            background_config={
                "stroke_color": C_DIM,
                "stroke_width": 1,
            },
            paragraph_config={
                "font_size": font_size,
            },
        )

    # ── Scene 1 — Title / Intro ──────────────────────────────────────────

    def scene_intro(self):
        title = self.zh("RAG 入門教學", font_size=56, color=C_BLUE)
        sub_en = self.en(
            "Retrieval-Augmented Generation", font_size=26, color=C_WHITE
        ).next_to(title, DOWN, buff=0.4)
        sub_zh = self.zh(
            "檢索增強生成", font_size=32, color=C_ORANGE
        ).next_to(sub_en, DOWN, buff=0.3)
        producer = self.zh(
            "制片人：Peter", font_size=24, color=C_DIM
        ).next_to(sub_zh, DOWN, buff=0.7)

        with self.voiceover(
            text="大家好！今日我哋嚟學下 RAG，即係 Retrieval Augmented Generation，"
            "中文叫做「檢索增強生成」。"
        ):
            self.play(Write(title), run_time=1.5)
            self.play(FadeIn(sub_en, shift=UP * 0.2), run_time=0.8)
            self.play(FadeIn(sub_zh, shift=UP * 0.2), run_time=0.8)
            self.play(FadeIn(producer, shift=UP * 0.2), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 2 — The Problem ────────────────────────────────────────────

    def scene_problem(self):
        heading = self.zh(
            "LLM 嘅問題", font_size=40, color=C_BLUE
        ).to_edge(UP, buff=0.6)

        llm_box = self.make_box("LLM 大型語言模型", C_PURPLE, width=5, height=1)
        llm_box.shift(UP * 0.5)

        q_text = self.zh(
            "問：「Peter 住喺邊度？」", font_size=26, color=C_YELLOW
        ).next_to(llm_box, DOWN, buff=0.6)

        a_bad = self.zh(
            "答：「我唔知道 … 或者亂噏」", font_size=26, color=C_PINK
        ).next_to(q_text, DOWN, buff=0.4)

        cross_mark = self.zh("✗", font_size=36, color=C_RED).next_to(
            a_bad, RIGHT, buff=0.3
        )

        with self.voiceover(
            text="首先，大型語言模型，即係 LLM，雖然好叻，但係佢只係知道訓練數據入面嘅嘢。"
            "如果你問佢你自己嘅資料，例如「Peter 住喺邊度」，佢係唔知道㗎，"
            "仲可能會亂噏添。呢個就係所謂嘅幻覺問題。"
        ):
            self.play(Write(heading), run_time=0.8)
            self.play(FadeIn(llm_box, shift=DOWN * 0.2), run_time=1)
            self.play(FadeIn(q_text, shift=UP * 0.2), run_time=1)
            self.play(FadeIn(a_bad, shift=UP * 0.2), run_time=1)
            self.play(FadeIn(cross_mark, scale=1.5), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 3 — RAG Idea ───────────────────────────────────────────────

    def scene_rag_idea(self):
        heading = self.zh(
            "RAG 嘅做法", font_size=40, color=C_BLUE
        ).to_edge(UP, buff=0.6)

        data_box = self.make_box("你嘅數據", C_GREEN, width=2.8)
        search_box = self.make_box("搜尋相關資料", C_ORANGE, width=3.2)
        llm_box = self.make_box("LLM 生成答案", C_PURPLE, width=3.2)

        row = VGroup(data_box, search_box, llm_box).arrange(RIGHT, buff=1.2)
        row.move_to(ORIGIN + UP * 0.2)

        arr1 = Arrow(
            data_box.get_right(), search_box.get_left(),
            buff=0.15, color=C_WHITE, stroke_width=3,
        )
        arr2 = Arrow(
            search_box.get_right(), llm_box.get_left(),
            buff=0.15, color=C_WHITE, stroke_width=3,
        )

        answer = self.zh(
            "答：「Peter 住喺大圍」 ✓", font_size=26, color=C_GREEN
        ).next_to(row, DOWN, buff=0.8)

        with self.voiceover(
            text="RAG 嘅做法好簡單。喺回答問題之前，先喺你自己嘅數據入面搵到最相關嘅資料，"
            "然後將呢啲資料傳俾 LLM，等佢參考住嚟回答。咁樣答案就會準確好多，唔使再靠估。"
        ):
            self.play(Write(heading), run_time=0.8)
            self.play(FadeIn(data_box, shift=RIGHT * 0.2), run_time=0.8)
            self.play(GrowArrow(arr1), run_time=0.5)
            self.play(FadeIn(search_box, shift=RIGHT * 0.2), run_time=0.8)
            self.play(GrowArrow(arr2), run_time=0.5)
            self.play(FadeIn(llm_box, shift=RIGHT * 0.2), run_time=0.8)
            self.play(FadeIn(answer, shift=UP * 0.2), run_time=0.8)

        self.wait(0.3)
        self.clear()

    # ── Scene 4 — Pipeline + Code Walkthrough ────────────────────────────

    def scene_pipeline(self):
        heading = self.zh(
            "RAG 五個步驟", font_size=36, color=C_BLUE
        ).to_edge(UP, buff=0.3)

        steps = [
            ("LOAD", "載入", C_GREEN),
            ("EMBED", "向量化", C_ORANGE),
            ("STORE", "儲存", C_PURPLE),
            ("RETRIEVE", "檢索", C_PINK),
            ("GENERATE", "生成", C_YELLOW),
        ]

        boxes = []
        for eng, zh_label, col in steps:
            rect = RoundedRectangle(
                corner_radius=0.1, width=1.6, height=0.7,
                fill_color=col, fill_opacity=0.2, stroke_color=col,
            )
            t_en = self.en(eng, font_size=14, color=col)
            t_zh = self.zh(zh_label, font_size=14, color=C_WHITE)
            label = VGroup(t_en, t_zh).arrange(DOWN, buff=0.05).move_to(rect)
            boxes.append(VGroup(rect, label))

        row = VGroup(*boxes).arrange(RIGHT, buff=0.45)
        row.next_to(heading, DOWN, buff=0.35)

        arrows = []
        for i in range(len(boxes) - 1):
            a = Arrow(
                boxes[i].get_right(), boxes[i + 1].get_left(),
                buff=0.08, color=C_WHITE, stroke_width=2,
                max_tip_length_to_length_ratio=0.2,
            )
            arrows.append(a)

        # ── Overview: build the pipeline ─────────────────────────────────
        self.play(Write(heading), run_time=0.6)

        with self.voiceover(
            text="RAG 嘅流程總共有五個步驟，我哋逐個嚟睇下佢哋嘅代碼。"
        ):
            for i, box in enumerate(boxes):
                self.play(FadeIn(box, shift=UP * 0.15), run_time=0.35)
                if i < len(arrows):
                    self.play(GrowArrow(arrows[i]), run_time=0.2)

        # ── Code walkthrough per step ────────────────────────────────────
        code_snippets = [CODE_LOAD, CODE_EMBED, CODE_STORE, CODE_RETRIEVE, CODE_GENERATE]
        code_voiceovers = [
            "第一步 LOAD。load_documents 函數好簡單，用 Python 嘅 json.load "
            "讀入一個 JSON 文件。入面每個 entry 都有 id 同 text 兩個字段。",

            "第二步 EMBED，向量化。我哋用 ollama.embed 呢個函數，傳入 embedding "
            "模型名同文字，佢就會返回一個向量。我哋逐個文檔咁做，將每個都向量化。",

            "第三步 STORE，儲存。用 ChromaDB 建立一個 PersistentClient，即係持久化嘅客戶端。"
            "然後建立一個 collection，用 add 方法將所有 id、文字同向量存入去。",

            "第四步 RETRIEVE，檢索。首先將用戶嘅問題用同一個 embedding 模型向量化，"
            "然後用 collection.query 喺資料庫入面搵出最接近嘅 TOP K 個文檔。",

            "最後一步 GENERATE，生成答案。將搵到嘅文檔合併做 context，放入 prompt。"
            "明確話俾 LLM 知要根據 context 嚟回答，然後用 ollama.chat 生成最終答案。",
        ]

        code_center = DOWN * 1.2
        prev_code = None

        for i, (box, snippet, vo) in enumerate(
            zip(boxes, code_snippets, code_voiceovers)
        ):
            code_block = self.make_code(snippet)
            code_block.move_to(code_center)

            with self.voiceover(text=vo):
                self.play(
                    box[0].animate.set_fill(opacity=0.55).set_stroke(width=3),
                    run_time=0.35,
                )
                if prev_code:
                    self.play(
                        FadeOut(prev_code, shift=LEFT * 0.3),
                        FadeIn(code_block, shift=RIGHT * 0.3),
                        run_time=0.6,
                    )
                else:
                    self.play(FadeIn(code_block, shift=UP * 0.2), run_time=0.6)
                prev_code = code_block

            self.play(
                box[0].animate.set_fill(opacity=0.2).set_stroke(width=1),
                run_time=0.25,
            )

        # ── Full pipeline highlight ──────────────────────────────────────
        if prev_code:
            self.play(FadeOut(prev_code), run_time=0.3)

        with self.voiceover(
            text="成個流程就係：載入數據，向量化，儲存，檢索，最後生成答案。"
            "RETRIEVE 搵到嘅資料會直接傳俾 GENERATE，呢個就係 RAG 嘅核心。"
        ):
            self.play(
                *[box[0].animate.set_fill(opacity=0.5) for box in boxes],
                *[a.animate.set_color(C_YELLOW) for a in arrows],
                run_time=1.2,
            )
            self.wait(1)
            self.play(
                *[box[0].animate.set_fill(opacity=0.2) for box in boxes],
                *[a.animate.set_color(C_WHITE) for a in arrows],
                run_time=0.8,
            )

        self.wait(0.3)
        self.clear()

    # ── Scene 5 — GitHub callout ─────────────────────────────────────────

    def scene_github(self):
        heading = self.zh(
            "完整代碼", font_size=40, color=C_BLUE
        ).to_edge(UP, buff=0.6)

        url_text = self.en(
            GITHUB_URL, font_size=26, color=C_GREEN
        )
        folder = self.en(
            "lesson 1 - overall / rag.py", font_size=22, color=C_ORANGE
        ).next_to(url_text, DOWN, buff=0.35)

        url_group = VGroup(url_text, folder).move_to(ORIGIN)

        box = SurroundingRectangle(
            url_group, buff=0.4, color=C_BLUE,
            corner_radius=0.15, stroke_width=2,
        )

        with self.voiceover(
            text="如果想睇完整嘅 source code，可以去 GitHub 搵 quantrpeter 嘅 "
            "learn-rag repo。入面嘅 lesson 1 overall 文件夾就有晒所有代碼，"
            "包括 rag.py 同 data.json，大家記得去睇下。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(Create(box), FadeIn(url_text), run_time=1)
            self.play(FadeIn(folder, shift=UP * 0.2), run_time=0.8)

        self.wait(0.3)
        self.clear()

    # ── Scene 6 — Summary ────────────────────────────────────────────────

    def scene_summary(self):
        heading = self.zh(
            "總結", font_size=44, color=C_BLUE
        ).to_edge(UP, buff=0.6)

        bullets = VGroup(
            self.zh("•  RAG = 檢索 + 生成", font_size=28, color=C_GREEN),
            self.zh(
                "•  令 LLM 可以用你自己嘅數據回答問題",
                font_size=28, color=C_ORANGE,
            ),
            self.zh(
                "•  五步：LOAD → EMBED → STORE → RETRIEVE → GENERATE",
                font_size=22, color=C_PURPLE,
            ),
            self.zh(
                "•  解決 LLM 幻覺問題，答案更準確",
                font_size=28, color=C_PINK,
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.45)
        bullets.move_to(ORIGIN)

        thanks = self.zh("多謝收睇！", font_size=52, color=C_YELLOW)

        with self.voiceover(
            text="總結一下。RAG 即係檢索加生成，令 LLM 可以利用你自己嘅數據嚟回答問題。"
            "成個流程有五個步驟：載入數據、向量化、儲存、檢索、生成答案。"
            "有咗 RAG，LLM 嘅答案會準確好多，唔使再靠估喇。"
        ):
            self.play(Write(heading), run_time=0.6)
            for b in bullets:
                self.play(FadeIn(b, shift=RIGHT * 0.3), run_time=0.6)

        self.wait(0.5)
        self.clear()

        with self.voiceover(
            text="多謝收睇！記得去 GitHub 睇完整代碼，同埋繼續跟住學下一課喇。"
        ):
            self.play(Write(thanks), run_time=1)

        self.wait(1)
        self.play(FadeOut(thanks), run_time=0.8)
