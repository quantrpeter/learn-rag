"""
Lesson 2 – Embed & Load Explainer
Manim CE + Cantonese voiceover (Edge TTS)

Render:
    cd "lesson 2 - explain"
    manim render -qm scene.py EmbedLoadExplainer
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

# ── code snippets ────────────────────────────────────────────────────────

CODE_LOAD = """\
with open("data.json") as f:
    documents = json.load(f)

# documents = [
#   {"id": "1", "text": "Peter Cheung..."},
#   {"id": "2", "text": "Peter lives..."},
# ]"""

CODE_EMBED = """\
response = ollama.embed(
    model="mxbai-embed-large",
    input="Peter lives in Tai Wai"
)
embedding = response["embeddings"][0]
# embedding = [-0.037, 0.014, ... ]
# len(embedding) == 1024"""

CODE_STORE = """\
collection.add(
    ids        = ["1", "2", "3", "4"],
    documents  = ["Peter Cheung...", ...],
    embeddings = [[-0.009, ...], ...]
)
# 3 parallel lists, same length"""

CODE_QUERY = """\
q_emb = ollama.embed(
    model="mxbai-embed-large",
    input="Where does Peter live?"
)["embeddings"][0]

results = collection.query(
    query_embeddings=[q_emb],
    n_results=2
)"""


class EmbedLoadExplainer(VoiceoverScene):
    """Lesson 2: deep dive into Embed & Load — what goes into ChromaDB."""

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
        self.scene_load()
        self.scene_embed()
        self.scene_similarity()
        self.scene_store()
        self.scene_query()
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

    def make_box(self, label, color, width=2.5, height=0.8):
        rect = RoundedRectangle(
            corner_radius=0.15, width=width, height=height,
            fill_color=color, fill_opacity=0.25, stroke_color=color,
        )
        txt = self.zh(label, font_size=24, color=color).move_to(rect)
        return VGroup(rect, txt)

    def make_code(self, code_str, font_size=18):
        return Code(
            code_string=code_str, language="python",
            formatter_style="monokai", add_line_numbers=False,
            background="rectangle",
            background_config={"stroke_color": C_DIM, "stroke_width": 1},
            paragraph_config={"font_size": font_size},
        )

    def make_doc_card(self, doc_id, text, color, width=5.5):
        rect = RoundedRectangle(
            corner_radius=0.12, width=width, height=0.55,
            fill_color=color, fill_opacity=0.15, stroke_color=color,
            stroke_width=1.5,
        )
        id_t = self.en(f"id: {doc_id}", font_size=14, color=color)
        txt_t = self.en(f'"{text}"', font_size=14, color=C_WHITE)
        content = VGroup(id_t, txt_t).arrange(RIGHT, buff=0.4).move_to(rect)
        return VGroup(rect, content)

    def make_sim_bar(self, label, value, max_width=5.0, color=C_CYAN):
        bg = Rectangle(
            width=max_width, height=0.3,
            fill_color=WHITE, fill_opacity=0.05,
            stroke_color=C_DIM, stroke_width=0.5,
        )
        bar = Rectangle(
            width=max_width * value, height=0.3,
            fill_color=color, fill_opacity=0.5,
            stroke_width=0,
        ).align_to(bg, LEFT)
        lbl = self.en(label, font_size=13, color=C_WHITE).next_to(bg, LEFT, buff=0.15)
        val = self.en(f"{value:.2f}", font_size=14, color=C_YELLOW).next_to(bg, RIGHT, buff=0.15)
        return VGroup(bg, bar, lbl, val)

    # ── Scene 1 — Intro ──────────────────────────────────────────────────

    def scene_intro(self):
        title = self.zh("Embed & Load 深入解析", font_size=50, color=C_BLUE)
        sub = self.zh(
            "了解數據點樣存入 ChromaDB", font_size=28, color=C_ORANGE
        ).next_to(title, DOWN, buff=0.4)
        lesson = self.en(
            "Lesson 2", font_size=22, color=C_DIM
        ).next_to(sub, DOWN, buff=0.3)
        producer = self.zh(
            "制片人：Peter", font_size=22, color=C_DIM
        ).next_to(lesson, DOWN, buff=0.5)

        with self.voiceover(
            text="大家好！歡迎嚟到第二課。今日我哋深入了解 Embed 同 Load 步驟，"
            "一步一步睇清楚數據係點樣從文字變成向量，再存入 ChromaDB。"
        ):
            self.play(Write(title), run_time=1.2)
            self.play(FadeIn(sub, shift=UP * 0.2), run_time=0.8)
            self.play(FadeIn(lesson, shift=UP * 0.2), run_time=0.5)
            self.play(FadeIn(producer, shift=UP * 0.2), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 2 — LOAD: raw JSON data ────────────────────────────────────

    def scene_load(self):
        heading = self.zh(
            "Step 1: LOAD — 載入數據", font_size=36, color=C_GREEN
        ).to_edge(UP, buff=0.4)

        docs = [
            ("1", "Peter Cheung is chairman of HKPS", C_CYAN),
            ("2", "Peter lives in Tai Wai", C_CYAN),
            ("3", "Python is a popular programming language", C_ORANGE),
            ("4", "Hong Kong is in southern China", C_ORANGE),
        ]
        cards = VGroup(*[
            self.make_doc_card(d, t, c) for d, t, c in docs
        ]).arrange(DOWN, buff=0.15)
        cards.next_to(heading, DOWN, buff=0.5).shift(LEFT * 1.8)

        code_block = self.make_code(CODE_LOAD, font_size=15)
        code_block.next_to(cards, RIGHT, buff=0.5)

        json_label = self.en(
            "data.json", font_size=18, color=C_GREEN
        ).next_to(cards, UP, buff=0.2)

        with self.voiceover(
            text="首先係 LOAD，載入數據。我哋嘅數據存喺一個 JSON 文件入面。"
            "每個文檔都有兩個字段：id 係唯一標識符，text 就係文檔嘅內容。"
            "呢四個就係我哋嘅原始數據，全部都係純文字，未經任何處理。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(json_label, shift=DOWN * 0.1), run_time=0.4)
            for card in cards:
                self.play(FadeIn(card, shift=RIGHT * 0.2), run_time=0.4)
            self.play(FadeIn(code_block, shift=UP * 0.2), run_time=0.8)

        note = self.zh(
            "ChromaDB 搜尋唔到純文字 → 需要 EMBED", font_size=20, color=C_YELLOW
        ).next_to(cards, DOWN, buff=0.4).shift(RIGHT * 1)

        with self.voiceover(
            text="但係問題嚟喇，ChromaDB 係一個向量資料庫，佢搜尋唔到純文字。"
            "所以我哋需要下一步：EMBED，將文字轉成數字。"
        ):
            self.play(FadeIn(note, shift=UP * 0.2), run_time=0.8)

        self.wait(0.3)
        self.clear()

    # ── Scene 3 — EMBED: text → vector ───────────────────────────────────

    def scene_embed(self):
        heading = self.zh(
            "Step 2: EMBED — 文字變向量", font_size=36, color=C_ORANGE
        ).to_edge(UP, buff=0.4)

        text_box = self.make_box("Peter lives in Tai Wai", C_CYAN, width=5, height=0.7)
        model_box = self.make_box("mxbai-embed-large", C_PURPLE, width=4, height=0.7)
        text_box.move_to(UP * 1.8 + LEFT * 2.5)
        model_box.next_to(text_box, DOWN, buff=0.7)

        arr1 = Arrow(
            text_box.get_bottom(), model_box.get_top(),
            buff=0.1, color=C_WHITE, stroke_width=2.5,
        )

        vec_nums = self.en(
            "[-0.037, 0.014, -0.006, -0.046, -0.007, ...]",
            font_size=16, color=C_YELLOW,
        )
        vec_label = self.zh("1024 個浮點數", font_size=18, color=C_PINK)
        vec_group = VGroup(vec_nums, vec_label).arrange(DOWN, buff=0.15)
        vec_group.next_to(model_box, DOWN, buff=0.7)

        vec_rect = SurroundingRectangle(
            vec_group, buff=0.2, color=C_ORANGE,
            corner_radius=0.1, stroke_width=1.5,
        )

        arr2 = Arrow(
            model_box.get_bottom(), vec_rect.get_top(),
            buff=0.1, color=C_WHITE, stroke_width=2.5,
        )

        code_block = self.make_code(CODE_EMBED, font_size=14)
        code_block.to_edge(RIGHT, buff=0.4).shift(DOWN * 0.3)

        with self.voiceover(
            text="EMBED 步驟就係將文字變成向量。我哋用 ollama.embed 函數，"
            "傳入模型名 mxbai-embed-large 同文字內容。"
            "佢會返回一個向量，即係一組 1024 個浮點數。"
            "呢啲數字就代表咗文字嘅意思。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(text_box, shift=DOWN * 0.2), run_time=0.6)
            self.play(GrowArrow(arr1), run_time=0.4)
            self.play(FadeIn(model_box, shift=DOWN * 0.2), run_time=0.6)
            self.play(GrowArrow(arr2), run_time=0.4)
            self.play(
                FadeIn(vec_group, shift=UP * 0.2),
                Create(vec_rect),
                run_time=0.8,
            )
            self.play(FadeIn(code_block, shift=LEFT * 0.3), run_time=0.8)

        meaning = self.zh(
            "意思相近嘅文字 → 向量相近", font_size=22, color=C_GREEN
        ).to_edge(DOWN, buff=0.5)

        with self.voiceover(
            text="重點係：意思相近嘅文字，佢哋嘅向量都會好接近。"
            "好似「Peter 住喺大圍」同「Peter Cheung 係主席」，"
            "因為都係講 Peter，所以佢哋嘅向量會比較相似。"
        ):
            self.play(FadeIn(meaning, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 4 — Cosine Similarity ──────────────────────────────────────

    def scene_similarity(self):
        heading = self.zh(
            "向量相似度 (Cosine Similarity)", font_size=34, color=C_CYAN
        ).to_edge(UP, buff=0.4)

        pairs = [
            ("doc 1 vs 2", 0.65, "Peter × Peter", C_GREEN),
            ("doc 2 vs 4", 0.53, "Peter × HK", C_BLUE),
            ("doc 1 vs 3", 0.51, "Peter × Python", C_ORANGE),
            ("doc 1 vs 4", 0.49, "HKPS × HK", C_BLUE),
            ("doc 2 vs 3", 0.37, "Peter × Python", C_PINK),
            ("doc 3 vs 4", 0.34, "Python × HK", C_PINK),
        ]

        bars = VGroup()
        for label, val, desc, color in pairs:
            b = self.make_sim_bar(f"{label}  {desc}", val, max_width=5.5, color=color)
            bars.add(b)
        bars.arrange(DOWN, buff=0.25).move_to(ORIGIN + DOWN * 0.2)

        with self.voiceover(
            text="而家我哋計算所有文檔對之間嘅 cosine similarity。"
            "數值越高代表兩段文字嘅意思越相似。"
            "你睇，兩個關於 Peter 嘅文檔相似度有 0.65，排第一。"
            "但 Peter 同 Python 嘅文檔只有 0.37，低好多。"
            "呢個就證明咗 embedding 真係可以捕捉到文字嘅語義。"
        ):
            self.play(Write(heading), run_time=0.6)
            for b in bars:
                self.play(FadeIn(b, shift=RIGHT * 0.2), run_time=0.4)

        self.wait(0.5)
        self.clear()

    # ── Scene 5 — STORE: what goes into ChromaDB ─────────────────────────

    def scene_store(self):
        heading = self.zh(
            "Step 3: STORE — 存入 ChromaDB", font_size=36, color=C_PURPLE
        ).to_edge(UP, buff=0.4)

        col_ids = self.en("ids", font_size=18, color=C_GREEN)
        col_docs = self.en("documents", font_size=18, color=C_ORANGE)
        col_embs = self.en("embeddings", font_size=18, color=C_PINK)
        headers = VGroup(col_ids, col_docs, col_embs).arrange(RIGHT, buff=2.0)
        headers.next_to(heading, DOWN, buff=0.5)

        h_line = Line(
            headers.get_left() + LEFT * 0.3 + DOWN * 0.2,
            headers.get_right() + RIGHT * 0.3 + DOWN * 0.2,
            color=C_DIM, stroke_width=1,
        )

        rows_data = [
            ('"1"', '"Peter Cheung is..."', "[-0.009, -0.016, ...]"),
            ('"2"', '"Peter lives in..."', "[-0.037, 0.014, ...]"),
            ('"3"', '"Python is a popular..."', "[0.025, -0.015, ...]"),
            ('"4"', '"Hong Kong is..."', "[-0.032, 0.026, ...]"),
        ]

        rows = VGroup()
        for r_id, r_doc, r_emb in rows_data:
            t_id = self.en(r_id, font_size=15, color=C_GREEN)
            t_doc = self.en(r_doc, font_size=15, color=C_WHITE)
            t_emb = self.en(r_emb, font_size=14, color=C_PINK)
            row = VGroup(t_id, t_doc, t_emb).arrange(RIGHT, buff=1.4)
            rows.add(row)
        rows.arrange(DOWN, buff=0.3)
        rows.next_to(h_line, DOWN, buff=0.3)

        for row in rows:
            row[0].align_to(col_ids, LEFT)
            row[1].align_to(col_docs, LEFT)
            row[2].align_to(col_embs, LEFT)

        table_rect = SurroundingRectangle(
            VGroup(headers, h_line, rows), buff=0.25,
            color=C_DIM, corner_radius=0.1, stroke_width=1,
        )

        code_block = self.make_code(CODE_STORE, font_size=15)
        code_block.to_edge(DOWN, buff=0.35)

        with self.voiceover(
            text="到 STORE 步驟，我哋要將數據存入 ChromaDB。"
            "collection.add 需要三個平行嘅列表。"
            "第一個係 ids，每個文檔嘅唯一標識符。"
            "第二個係 documents，即係原文內容。"
            "第三個係 embeddings，就係啱啱生成嘅向量。"
            "呢三個列表要一一對應，長度一樣。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(Create(table_rect), run_time=0.5)
            self.play(
                FadeIn(col_ids), FadeIn(col_docs), FadeIn(col_embs),
                Create(h_line),
                run_time=0.6,
            )
            for row in rows:
                self.play(FadeIn(row, shift=RIGHT * 0.15), run_time=0.35)
            self.play(FadeIn(code_block, shift=UP * 0.2), run_time=0.8)

        self.wait(0.3)
        self.clear()

    # ── Scene 6 — QUERY: embed question + search ─────────────────────────

    def scene_query(self):
        heading = self.zh(
            "QUERY — 向量搜尋示範", font_size=36, color=C_YELLOW
        ).to_edge(UP, buff=0.4)

        q_box = self.make_box("Where does Peter live?", C_YELLOW, width=5.5, height=0.7)
        q_box.next_to(heading, DOWN, buff=0.4)

        model_label = self.en(
            "embed →", font_size=16, color=C_DIM
        )
        q_vec = self.en(
            "[-0.014, -0.018, -0.040, ...]  1024 floats",
            font_size=14, color=C_YELLOW,
        )
        embed_row = VGroup(model_label, q_vec).arrange(RIGHT, buff=0.3)
        embed_row.next_to(q_box, DOWN, buff=0.35)

        compare_label = self.zh(
            "同每個文檔比較 cosine similarity：", font_size=20, color=C_WHITE,
        ).next_to(embed_row, DOWN, buff=0.4)

        results = [
            ('#1  "Peter lives in Tai Wai"', 0.77, C_GREEN),
            ('#2  "Peter Cheung is chairman..."', 0.46, C_BLUE),
            ('#3  "Python is a popular..."', 0.39, C_ORANGE),
            ('#4  "Hong Kong is in southern..."', 0.39, C_PINK),
        ]

        bars = VGroup()
        for label, val, color in results:
            b = self.make_sim_bar(label, val, max_width=4.5, color=color)
            bars.add(b)
        bars.arrange(DOWN, buff=0.2).next_to(compare_label, DOWN, buff=0.3)

        winner_label = self.zh(
            "← 最接近！", font_size=18, color=C_GREEN,
        ).next_to(bars[0], RIGHT, buff=0.2)

        code_block = self.make_code(CODE_QUERY, font_size=13)
        code_block.to_edge(RIGHT, buff=0.3).shift(DOWN * 1.5)

        with self.voiceover(
            text="最後嚟做個查詢示範。問題係 Where does Peter live。"
            "首先將問題用同一個 embedding 模型向量化，變成 1024 個浮點數。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(q_box, shift=DOWN * 0.2), run_time=0.6)
            self.play(FadeIn(embed_row, shift=RIGHT * 0.2), run_time=0.6)

        with self.voiceover(
            text="然後同所有文檔嘅向量比較 cosine similarity。"
            "你睇，Peter lives in Tai Wai 嘅相似度最高，有 0.77。"
            "第二名係 Peter Cheung is chairman，有 0.46。"
            "其他文檔就低好多，只有 0.39。"
            "ChromaDB 嘅 collection.query 就係內部做呢個比較，自動返回最接近嘅文檔。"
        ):
            self.play(FadeIn(compare_label, shift=UP * 0.1), run_time=0.4)
            for b in bars:
                self.play(FadeIn(b, shift=RIGHT * 0.2), run_time=0.35)
            self.play(FadeIn(winner_label, scale=1.3), run_time=0.4)
            self.play(FadeIn(code_block, shift=LEFT * 0.3), run_time=0.8)

        self.wait(0.3)
        self.clear()

    # ── Scene 7 — GitHub ─────────────────────────────────────────────────

    def scene_github(self):
        heading = self.zh(
            "完整代碼", font_size=40, color=C_BLUE
        ).to_edge(UP, buff=0.6)

        url_text = self.en(GITHUB_URL, font_size=26, color=C_GREEN)
        folder = self.en(
            "lesson 2 - embed and load / embed_and_load.py",
            font_size=20, color=C_ORANGE,
        ).next_to(url_text, DOWN, buff=0.35)

        url_group = VGroup(url_text, folder).move_to(ORIGIN)
        box = SurroundingRectangle(
            url_group, buff=0.4, color=C_BLUE,
            corner_radius=0.15, stroke_width=2,
        )

        with self.voiceover(
            text="完整嘅 source code 可以喺 GitHub 搵到。"
            "去 lesson 2 embed and load 文件夾，入面嘅 embed_and_load.py "
            "會一步一步 print 出所有中間數據，大家記得自己跑一次。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(Create(box), FadeIn(url_text), run_time=1)
            self.play(FadeIn(folder, shift=UP * 0.2), run_time=0.8)

        self.wait(0.3)
        self.clear()

    # ── Scene 8 — Summary ────────────────────────────────────────────────

    def scene_summary(self):
        heading = self.zh(
            "總結", font_size=44, color=C_BLUE
        ).to_edge(UP, buff=0.6)

        bullets = VGroup(
            self.zh(
                "•  LOAD：讀入 JSON — 每個文檔有 id 同 text",
                font_size=24, color=C_GREEN,
            ),
            self.zh(
                "•  EMBED：文字 → 1024 個浮點數（向量）",
                font_size=24, color=C_ORANGE,
            ),
            self.zh(
                "•  ChromaDB 存三樣嘢：id、document、embedding",
                font_size=24, color=C_PURPLE,
            ),
            self.zh(
                "•  查詢：問題向量化 → cosine similarity → 搵最近文檔",
                font_size=24, color=C_YELLOW,
            ),
            self.zh(
                "•  語義搜尋，唔係關鍵字搜尋！",
                font_size=24, color=C_PINK,
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        bullets.move_to(ORIGIN)

        thanks = self.zh("多謝收睇！", font_size=52, color=C_YELLOW)

        with self.voiceover(
            text="總結一下。LOAD 就係讀入 JSON 數據，每個文檔有 id 同 text。"
            "EMBED 就係用 embedding 模型將文字變成 1024 個浮點數嘅向量。"
            "ChromaDB 儲存三樣嘢：id、原文、同向量。"
            "查詢嘅時候，將問題向量化，用 cosine similarity 搵最接近嘅文檔。"
            "記住，呢個係語義搜尋，唔係關鍵字搜尋！"
        ):
            self.play(Write(heading), run_time=0.6)
            for b in bullets:
                self.play(FadeIn(b, shift=RIGHT * 0.3), run_time=0.5)

        self.wait(0.5)
        self.clear()

        with self.voiceover(
            text="多謝收睇！記得自己跑一次 embed_and_load.py，親身體驗下數據嘅變化。"
        ):
            self.play(Write(thanks), run_time=1)

        self.wait(1)
        self.play(FadeOut(thanks), run_time=0.8)
