"""
Lesson 4 – Build an Embedding Model (Explainer Video)
Manim CE + Cantonese voiceover (Edge TTS)

Heavy on diagrams and animated architecture visuals.

Render:
    cd "lesson 4 -explain"
    manim render -qm scene.py EmbedModelExplainer
"""

from manim import *
from manim_voiceover import VoiceoverScene

from edge_tts_service import EdgeTTSService

# ── colours (dark tones for bright wallpaper) ────────────────────────────
C_BG = "#0f0f23"
C_BLUE = "#1565c0"
C_GREEN = "#2eff32"
C_ORANGE = "#e65100"
C_PINK = "#ad1457"
C_PURPLE = "#6a1b9a"
C_YELLOW = "#f9a825"
C_WHITE = "#ffffff"
C_RED = "#b71c1c"
C_DIM = "#aaaaaa"
C_CYAN = "#00838f"
C_LIME = "#558b2f"
C_BLACK = "#000000"

FONT_ZH = "JetBrains Mono"
FONT_EN = "JetBrains Mono"
GITHUB_URL = "github.com/quantrpeter/learn-rag"


class EmbedModelExplainer(VoiceoverScene):
    """Lesson 4: how embedding models like mxbai-embed-large are built."""

    def construct(self):
        self.set_speech_service(
            EdgeTTSService(voice="zh-HK-HiuMaanNeural", rate="+0%")
        )
        self.camera.background_color = BLACK

        self.bg_image = ImageMobject("wallpaper2.avif")
        self.bg_image.height = config.frame_height
        self.bg_image.width = config.frame_width
        self.add(self.bg_image)

        self.watermark = self.zh(
            "香港編程學會", font_size=18, color="#ffffff"
        ).to_corner(UL, buff=0.3)
        self.add(self.watermark)

        self.scene_intro()
        self.scene_big_picture()
        self.scene_tokenize()
        self.scene_architecture()
        self.scene_mean_pooling()
        self.scene_triplet_loss()
        self.scene_training()
        self.scene_demo_intro()
        self.scene_demo_tokenize()
        self.scene_demo_model()
        self.scene_demo_loss_train()
        self.scene_demo_eval()
        self.scene_demo_vector()
        self.scene_github()
        self.scene_summary()

    # ── helpers ───────────────────────────────────────────────────────────

    def zh(self, txt, **kw):
        kw.setdefault("font", FONT_ZH)
        kw.setdefault("color", C_WHITE)
        return Text(txt, **kw)

    def en(self, txt, **kw):
        kw.setdefault("font", FONT_EN)
        kw.setdefault("color", C_WHITE)
        return Text(txt, **kw)

    def clear(self):
        keep = {self.bg_image, self.watermark}
        to_fade = [m for m in self.mobjects if m not in keep]
        if to_fade:
            self.play(*[FadeOut(m) for m in to_fade], run_time=0.5)

    def make_box(self, label, color, width=2.5, height=0.7, font_size=22):
        rect = RoundedRectangle(
            corner_radius=0.12, width=width, height=height,
            fill_color=color, fill_opacity=0.25, stroke_color=color,
        )
        txt = self.zh(label, font_size=font_size, color=color).move_to(rect)
        return VGroup(rect, txt)

    def make_layer_box(self, label, color, width=6.5, height=0.6):
        rect = RoundedRectangle(
            corner_radius=0.1, width=width, height=height,
            fill_color=color, fill_opacity=0.2, stroke_color=color,
            stroke_width=1.5,
        )
        txt = self.en(label, font_size=16, color=color).move_to(rect)
        return VGroup(rect, txt)

    def make_code(self, code_str, font_size=16):
        return Code(
            code_string=code_str, language="python",
            formatter_style="monokai", add_line_numbers=False,
            background="rectangle",
            background_config={"stroke_color": C_DIM, "stroke_width": 1},
            paragraph_config={"font_size": font_size},
        )

    def make_sim_bar(self, label, value, max_w=4.5, color=C_CYAN):
        bg = Rectangle(width=max_w, height=0.25,
                       fill_color=WHITE, fill_opacity=0.05,
                       stroke_color=C_DIM, stroke_width=0.5)
        bar = Rectangle(width=max_w * max(value, 0), height=0.25,
                        fill_color=color, fill_opacity=0.5,
                        stroke_width=0).align_to(bg, LEFT)
        lbl = self.en(label, font_size=12, color=C_WHITE).next_to(bg, LEFT, buff=0.1)
        val = self.en(f"{value:+.2f}", font_size=13, color=C_YELLOW).next_to(bg, RIGHT, buff=0.1)
        return VGroup(bg, bar, lbl, val)

    # ── Scene 1 — Intro ──────────────────────────────────────────────────

    def scene_intro(self):
        title = self.zh("構建 Embedding 模型", font_size=48, color=C_WHITE)
        sub = self.zh(
            "從零開始理解 mxbai-embed-large", font_size=26, color=C_WHITE
        ).next_to(title, DOWN, buff=0.4)
        lesson = self.en("Lesson 4", font_size=20, color=C_WHITE).next_to(sub, DOWN, buff=0.25)
        producer = self.zh(
            "制片人：Peter", font_size=22, color=C_WHITE
        ).next_to(lesson, DOWN, buff=0.5)

        with self.voiceover(
            text="大家好！歡迎嚟到第四課。今日我哋從零開始，了解 embedding 模型係點樣構建嘅。"
            "我哋會用動畫一步一步展示 mxbai-embed-large 嘅核心架構。"
        ):
            self.play(Write(title), run_time=1.2)
            self.play(FadeIn(sub, shift=UP * 0.2), run_time=0.7)
            self.play(FadeIn(lesson), FadeIn(producer), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 2 — Big Picture comparison ─────────────────────────────────

    def scene_big_picture(self):
        heading = self.zh(
            "我哋嘅迷你模型 vs mxbai-embed-large", font_size=30, color=C_BLUE
        ).to_edge(UP, buff=0.4)

        col_l = self.zh("迷你模型", font_size=22, color=C_GREEN).shift(LEFT * 3.2 + UP * 2)
        col_r = self.zh("mxbai-embed-large", font_size=22, color=C_ORANGE).shift(RIGHT * 3.2 + UP * 2)

        rows = [
            ("參數量", "74K", "340M"),
            ("層數", "2 層", "24 層"),
            ("向量維度", "64", "1024"),
            ("詞彙量", "~98", "~30,000"),
            ("訓練數據", "15 triplets", "7 億+ 句子對"),
        ]

        row_groups = VGroup()
        for i, (label, left, right) in enumerate(rows):
            y = 1.2 - i * 0.65
            lbl = self.zh(label, font_size=16, color=C_WHITE).move_to(UP * y)
            lv = self.en(left, font_size=18, color=C_WHITE).move_to(LEFT * 3.2 + UP * y)
            rv = self.en(right, font_size=18, color=C_WHITE).move_to(RIGHT * 3.2 + UP * y)
            row_groups.add(VGroup(lbl, lv, rv))

        divider = DashedLine(UP * 2.3, DOWN * 2.2, color=C_DIM, stroke_width=1)

        same_label = self.zh(
            "核心架構完全一樣！只係規模唔同。", font_size=22, color=C_YELLOW
        ).to_edge(DOWN, buff=0.6)

        with self.voiceover(
            text="首先睇下大局。我哋嘅迷你模型只有七萬幾個參數，兩層 Transformer，"
            "64 維向量。而 mxbai-embed-large 有 3.4 億參數，24 層，1024 維。"
            "但係佢哋嘅核心架構係完全一樣嘅，只係規模唔同。"
        ):
            self.play(Write(heading), run_time=0.5)
            self.play(FadeIn(col_l), FadeIn(col_r), Create(divider), run_time=0.6)
            for rg in row_groups:
                self.play(FadeIn(rg, shift=RIGHT * 0.1), run_time=0.35)
            self.play(FadeIn(same_label, shift=UP * 0.2), run_time=0.6)

        self.wait(0.5)
        self.clear()

    # ── Scene 3 — Tokenization animated flow ─────────────────────────────

    def scene_tokenize(self):
        heading = self.zh(
            "第一步：分詞 (Tokenization)", font_size=34, color=C_GREEN
        ).to_edge(UP, buff=0.4)

        words = ["the", "cat", "sat", "on", "the", "mat", "PAD", "PAD", "PAD", "PAD"]
        ids = ["5", "64", "65", "14", "5", "66", "0", "0", "0", "0"]
        colors = [C_CYAN] * 6 + [C_WHITE] * 4

        word_boxes = VGroup()
        id_boxes = VGroup()
        arrow_list = VGroup()

        for i, (w, d, c) in enumerate(zip(words, ids, colors)):
            wr = RoundedRectangle(
                corner_radius=0.08, width=1.1, height=0.5,
                fill_color=c, fill_opacity=0.2, stroke_color=c, stroke_width=1.5,
            )
            wt = self.en(w, font_size=14, color=c).move_to(wr)
            word_boxes.add(VGroup(wr, wt))

            ir = RoundedRectangle(
                corner_radius=0.08, width=1.1, height=0.5,
                fill_color=C_YELLOW, fill_opacity=0.15, stroke_color=C_YELLOW,
                stroke_width=1.5,
            )
            it = self.en(d, font_size=16, color=C_YELLOW).move_to(ir)
            id_boxes.add(VGroup(ir, it))

        word_boxes.arrange(RIGHT, buff=0.1).move_to(UP * 0.8)
        id_boxes.arrange(RIGHT, buff=0.1).move_to(DOWN * 0.8)

        for i in range(len(words)):
            a = Arrow(
                word_boxes[i].get_bottom(), id_boxes[i].get_top(),
                buff=0.08, color=C_DIM, stroke_width=1.5,
                max_tip_length_to_length_ratio=0.25,
            )
            arrow_list.add(a)

        sent_label = self.en(
            '"the cat sat on the mat"', font_size=18, color=C_WHITE,
        ).next_to(word_boxes, UP, buff=0.3)
        tok_label = self.en(
            "[5, 64, 65, 14, 5, 66, 0, 0, 0, 0]", font_size=16, color=C_YELLOW,
        ).next_to(id_boxes, DOWN, buff=0.3)

        note = self.zh(
            "每個字 → 一個整數 ID    |    PAD 填充到固定長度",
            font_size=18, color=C_DIM,
        ).to_edge(DOWN, buff=0.5)

        with self.voiceover(
            text="第一步，分詞。神經網絡只識處理數字，所以要將每個字轉成一個整數 ID。"
            "例如 the 變成 5，cat 變成 64。如果句子唔夠長，就用 PAD 填充到固定長度。"
            "真正嘅 mxbai-embed-large 用 WordPiece 分詞器，詞彙量有三萬個 subword。"
        ):
            self.play(Write(heading), run_time=0.5)
            self.play(FadeIn(sent_label), run_time=0.4)
            for i in range(len(words)):
                self.play(
                    FadeIn(word_boxes[i], shift=DOWN * 0.15),
                    run_time=0.2,
                )
            for i in range(len(words)):
                self.play(
                    GrowArrow(arrow_list[i]),
                    FadeIn(id_boxes[i], shift=DOWN * 0.15),
                    run_time=0.15,
                )
            self.play(FadeIn(tok_label, shift=UP * 0.1), run_time=0.4)
            self.play(FadeIn(note), run_time=0.4)

        self.wait(0.3)
        self.clear()

    # ── Scene 4 — Full Architecture Diagram ──────────────────────────────

    def scene_architecture(self):
        heading = self.zh(
            "模型架構 (Transformer Encoder)", font_size=32, color=C_ORANGE
        ).to_edge(UP, buff=0.35)

        layers = [
            ("Input: Token IDs  [5, 64, 65, 14, 5, 66, 0 ...]", C_DIM),
            ("Token Embedding    (vocab × 64)", C_CYAN),
            ("+ Position Encoding (pos × 64)", C_LIME),
            ("Transformer Layer 1  (Self-Attention + FFN)", C_PURPLE),
            ("Transformer Layer 2  (Self-Attention + FFN)", C_PURPLE),
            ("Mean Pooling  (6 vectors → 1 vector)", C_PINK),
            ("L2 Normalize  → 64-dim unit vector", C_YELLOW),
        ]

        boxes = VGroup()
        for label, color in layers:
            b = self.make_layer_box(label, color, width=7.5, height=0.55)
            boxes.add(b)
        boxes.arrange(DOWN, buff=0.12).move_to(DOWN * 0.15)

        arrows = VGroup()
        for i in range(len(boxes) - 1):
            a = Arrow(
                boxes[i].get_bottom(), boxes[i + 1].get_top(),
                buff=0.05, color=C_WHITE, stroke_width=2,
                max_tip_length_to_length_ratio=0.25,
            )
            arrows.add(a)

        brace_tf = Brace(VGroup(boxes[3], boxes[4]), LEFT, color=C_PURPLE, buff=0.15)
        brace_label = self.en("×2", font_size=18, color=C_PURPLE).next_to(brace_tf, LEFT, buff=0.1)

        output_label = self.en(
            "[+0.08, -0.02, +0.09, ...  64 floats]",
            font_size=14, color=C_YELLOW,
        ).next_to(boxes[-1], DOWN, buff=0.2)

        with self.voiceover(
            text="呢個就係完整嘅模型架構圖。由上到下：首先輸入 token ID，"
            "然後經過 Token Embedding 層，將每個 ID 變成一個 64 維向量。"
            "再加上 Position Encoding，告訴模型每個 token 嘅位置。"
            "跟住經過兩層 Transformer，每層包含 Self-Attention 同 Feed-Forward Network。"
            "Self-Attention 令模型可以理解字詞之間嘅關係。"
            "最後用 Mean Pooling 將所有 token 向量平均成一個向量，再 normalize 到單位長度。"
            "輸出就係一個 64 維嘅句子 embedding。"
        ):
            self.play(Write(heading), run_time=0.5)
            for i in range(len(boxes)):
                self.play(FadeIn(boxes[i], shift=DOWN * 0.1), run_time=0.3)
                if i < len(arrows):
                    self.play(GrowArrow(arrows[i]), run_time=0.15)
            self.play(
                FadeIn(brace_tf), FadeIn(brace_label),
                run_time=0.5,
            )
            self.play(FadeIn(output_label, shift=UP * 0.1), run_time=0.5)

        # Highlight data flow animation
        flow_dot = Dot(color=C_YELLOW, radius=0.08)
        flow_dot.move_to(boxes[0].get_center())
        self.add(flow_dot)
        for i in range(1, len(boxes)):
            self.play(flow_dot.animate.move_to(boxes[i].get_center()), run_time=0.25)
        self.play(FadeOut(flow_dot), run_time=0.2)

        self.wait(0.3)
        self.clear()

    # ── Scene 5 — Mean Pooling visual ────────────────────────────────────

    def scene_mean_pooling(self):
        heading = self.zh(
            "Mean Pooling：多個向量 → 一個向量", font_size=32, color=C_PINK
        ).to_edge(UP, buff=0.4)

        token_labels = ["the", "cat", "sat", "on", "the", "mat"]
        token_colors = [C_CYAN, C_GREEN, C_ORANGE, C_PURPLE, C_CYAN, C_PINK]

        token_vecs = VGroup()
        for i, (lbl, col) in enumerate(zip(token_labels, token_colors)):
            rect = RoundedRectangle(
                corner_radius=0.08, width=1.6, height=0.5,
                fill_color=col, fill_opacity=0.3, stroke_color=col,
            )
            t = self.en(f"{lbl}  →  vec_{i}", font_size=13, color=col).move_to(rect)
            token_vecs.add(VGroup(rect, t))
        token_vecs.arrange(DOWN, buff=0.08).shift(LEFT * 3.5 + DOWN * 0.2)

        plus_signs = VGroup()
        for i in range(len(token_vecs) - 1):
            p = self.en("+", font_size=16, color=C_DIM).move_to(
                (token_vecs[i].get_right() + token_vecs[i + 1].get_right()) / 2 + RIGHT * 0.5
            )
            plus_signs.add(p)

        sum_box = RoundedRectangle(
            corner_radius=0.12, width=2.8, height=0.7,
            fill_color=C_YELLOW, fill_opacity=0.15, stroke_color=C_YELLOW,
        ).shift(RIGHT * 1.5 + UP * 0.5)
        sum_label = self.en("SUM / 6", font_size=18, color=C_YELLOW).move_to(sum_box)
        sum_grp = VGroup(sum_box, sum_label)

        result_box = RoundedRectangle(
            corner_radius=0.12, width=5.5, height=0.7,
            fill_color=C_GREEN, fill_opacity=0.2, stroke_color=C_GREEN,
        ).shift(RIGHT * 1.5 + DOWN * 1.5)
        result_label = self.en(
            "sentence embedding (64-dim)", font_size=15, color=C_GREEN
        ).move_to(result_box)
        result_grp = VGroup(result_box, result_label)

        arr_to_sum = Arrow(
            token_vecs.get_right() + RIGHT * 0.1,
            sum_box.get_left(),
            buff=0.15, color=C_WHITE, stroke_width=2,
        )
        arr_to_result = Arrow(
            sum_box.get_bottom(), result_box.get_top(),
            buff=0.15, color=C_WHITE, stroke_width=2,
        )

        note = self.zh(
            "忽略 PAD token，只平均有意義嘅 token 向量",
            font_size=18, color=C_DIM,
        ).to_edge(DOWN, buff=0.5)

        with self.voiceover(
            text="Mean Pooling 係將多個 token 向量合併成一個句子向量嘅方法。"
            "每個 token 經過 Transformer 之後都有自己嘅向量。"
            "我哋將呢六個向量加埋，再除以六，就得到一個平均向量。"
            "注意，PAD token 嘅向量會被忽略，只平均有意義嘅 token。"
            "呢個平均向量就係最終嘅句子 embedding。"
        ):
            self.play(Write(heading), run_time=0.5)
            for tv in token_vecs:
                self.play(FadeIn(tv, shift=RIGHT * 0.15), run_time=0.2)
            self.play(GrowArrow(arr_to_sum), run_time=0.4)
            self.play(FadeIn(sum_grp, shift=DOWN * 0.1), run_time=0.5)
            self.play(GrowArrow(arr_to_result), run_time=0.3)
            self.play(FadeIn(result_grp, shift=DOWN * 0.1), run_time=0.5)
            self.play(FadeIn(note), run_time=0.4)

        self.wait(0.3)
        self.clear()

    # ── Scene 6 — Triplet Loss visual ────────────────────────────────────

    def scene_triplet_loss(self):
        heading = self.zh(
            "Triplet Loss：點樣訓練模型？", font_size=32, color=C_YELLOW
        ).to_edge(UP, buff=0.4)

        # 2D "vector space" representation
        anchor_pos = LEFT * 1 + DOWN * 0.3
        pos_start = LEFT * 2.5 + UP * 1.5
        neg_start = RIGHT * 1.5 + DOWN * 1.5

        anchor_dot = Dot(anchor_pos, color=C_BLUE, radius=0.15)
        anchor_lbl = self.en("Anchor", font_size=14, color=C_BLUE).next_to(anchor_dot, DOWN, buff=0.15)
        anchor_sent = self.en('"cat on mat"', font_size=12, color=C_DIM).next_to(anchor_lbl, DOWN, buff=0.1)

        pos_dot = Dot(pos_start, color=C_GREEN, radius=0.15)
        pos_lbl = self.en("Positive", font_size=14, color=C_GREEN).next_to(pos_dot, UP, buff=0.15)
        pos_sent = self.en('"kitten on rug"', font_size=12, color=C_DIM).next_to(pos_lbl, UP, buff=0.1)

        neg_dot = Dot(neg_start, color=C_RED, radius=0.15)
        neg_lbl = self.en("Negative", font_size=14, color=C_RED).next_to(neg_dot, DOWN, buff=0.15)
        neg_sent = self.en('"runs morning"', font_size=12, color=C_DIM).next_to(neg_lbl, DOWN, buff=0.1)

        anchor_grp = VGroup(anchor_dot, anchor_lbl, anchor_sent)
        pos_grp = VGroup(pos_dot, pos_lbl, pos_sent)
        neg_grp = VGroup(neg_dot, neg_lbl, neg_sent)

        space_border = RoundedRectangle(
            corner_radius=0.15, width=8, height=5,
            stroke_color=C_DIM, stroke_width=1, fill_opacity=0,
        ).move_to(DOWN * 0.2)
        space_label = self.en(
            "Vector Space", font_size=14, color=C_DIM
        ).next_to(space_border, UP, buff=0.05).align_to(space_border, LEFT).shift(RIGHT * 0.2)

        formula = self.en(
            "loss = max(0,  sim(A,N) - sim(A,P) + margin)",
            font_size=16, color=C_YELLOW,
        ).to_edge(DOWN, buff=0.4)

        pos_target = anchor_pos + RIGHT * 0.6 + UP * 0.5
        neg_target = RIGHT * 4 + DOWN * 1

        with self.voiceover(
            text="Triplet Loss 係訓練嘅核心。每次訓練用三個句子：Anchor 係基準句，"
            "Positive 係意思相近嘅句子，Negative 係意思唔同嘅句子。"
        ):
            self.play(Write(heading), run_time=0.5)
            self.play(Create(space_border), FadeIn(space_label), run_time=0.5)
            self.play(FadeIn(anchor_grp, scale=1.3), run_time=0.5)
            self.play(FadeIn(pos_grp, scale=1.3), run_time=0.5)
            self.play(FadeIn(neg_grp, scale=1.3), run_time=0.5)
            self.play(FadeIn(formula, shift=UP * 0.1), run_time=0.5)

        push_close = Arrow(
            pos_dot.get_center(), pos_target,
            buff=0.15, color=C_GREEN, stroke_width=3,
        )
        push_lbl = self.zh(
            "拉近", font_size=16, color=C_GREEN
        ).next_to(push_close, RIGHT, buff=0.1)

        push_away = Arrow(
            neg_dot.get_center(), neg_target,
            buff=0.15, color=C_RED, stroke_width=3,
        )
        away_lbl = self.zh(
            "推開", font_size=16, color=C_RED
        ).next_to(push_away, LEFT, buff=0.1)

        with self.voiceover(
            text="訓練嘅目標好簡單：將 Positive 拉近 Anchor，將 Negative 推離 Anchor。"
            "Loss 函數就係：如果 Negative 比 Positive 仲近，就有懲罰。"
            "經過好多次訓練之後，意思相近嘅句子就會聚埋一齊，唔同嘅就會分開。"
        ):
            self.play(GrowArrow(push_close), FadeIn(push_lbl), run_time=0.6)
            self.play(
                pos_dot.animate.move_to(pos_target),
                pos_lbl.animate.next_to(Dot(pos_target), UP, buff=0.15),
                pos_sent.animate.next_to(
                    Text("Positive").next_to(Dot(pos_target), UP, buff=0.15), UP, buff=0.1
                ),
                run_time=1.2,
            )
            self.play(GrowArrow(push_away), FadeIn(away_lbl), run_time=0.6)
            self.play(
                neg_dot.animate.move_to(neg_target),
                neg_lbl.animate.next_to(Dot(neg_target), DOWN, buff=0.15),
                neg_sent.animate.next_to(
                    Text("Negative").next_to(Dot(neg_target), DOWN, buff=0.15), DOWN, buff=0.1
                ),
                run_time=1.2,
            )

        self.wait(0.5)
        self.clear()

    # ── Scene 7 — Training before/after ──────────────────────────────────

    def scene_training(self):
        heading = self.zh(
            "訓練前 vs 訓練後", font_size=34, color=C_PURPLE
        ).to_edge(UP, buff=0.4)

        before_title = self.zh("訓練前（隨機）", font_size=20, color=C_RED).shift(LEFT * 3.5 + UP * 2)
        after_title = self.zh("訓練後（已學習）", font_size=20, color=C_GREEN).shift(RIGHT * 3.5 + UP * 2)

        pairs = [
            ('"cat" vs "kitten"', True),
            ('"sun" vs "sunny"', True),
            ('"pizza" vs "food"', True),
            ('"cat" vs "pizza"', False),
            ('"car" vs "baby"', False),
            ('"sky" vs "coffee"', False),
        ]
        before_vals = [0.61, 0.45, 0.52, 0.48, 0.39, 0.55]
        after_vals = [0.71, 0.57, 0.60, 0.05, -0.39, 0.33]

        divider = DashedLine(UP * 2.3, DOWN * 2.5, color=C_DIM, stroke_width=1)

        before_bars = VGroup()
        after_bars = VGroup()
        for i, ((label, is_sim), bv, av) in enumerate(zip(pairs, before_vals, after_vals)):
            color_b = C_ORANGE
            color_a = C_GREEN if is_sim else C_RED
            bb = self.make_sim_bar(label, bv, max_w=3.0, color=color_b)
            ab = self.make_sim_bar(label, max(av, 0), max_w=3.0, color=color_a)
            before_bars.add(bb)
            after_bars.add(ab)

        before_bars.arrange(DOWN, buff=0.2).shift(LEFT * 3.5 + DOWN * 0.3)
        after_bars.arrange(DOWN, buff=0.2).shift(RIGHT * 3.5 + DOWN * 0.3)

        with self.voiceover(
            text="嚟對比下訓練前後嘅效果。訓練前，模型嘅 embedding 係隨機嘅，"
            "所以無論係相似定唔同嘅句子，相似度都差唔多。"
            "但訓練後就好唔同喇。相似嘅句子，好似 cat 同 kitten，相似度好高。"
            "而唔同嘅句子，好似 car 同 baby，相似度變成負數。模型成功學識咗分辨語義。"
        ):
            self.play(Write(heading), run_time=0.5)
            self.play(Create(divider), run_time=0.3)
            self.play(FadeIn(before_title), FadeIn(after_title), run_time=0.4)
            for bb, ab in zip(before_bars, after_bars):
                self.play(FadeIn(bb, shift=RIGHT * 0.1), FadeIn(ab, shift=LEFT * 0.1), run_time=0.35)

        self.wait(0.5)
        self.clear()

    # ── helpers (terminal) ─────────────────────────────────────────────────

    def make_terminal(self, lines, font_size=13, width=6.0):
        body = "\n".join(lines)
        txt = Text(
            body, font="JetBrains Mono", font_size=font_size,
            color=C_GREEN, line_spacing=1.2,
        )
        h = txt.height + 0.55
        bg = RoundedRectangle(
            corner_radius=0.1, width=width, height=max(h, 1.0),
            fill_color="#0d0d1a", fill_opacity=0.92,
            stroke_color=C_DIM, stroke_width=1,
        )
        title_bar = Rectangle(
            width=width, height=0.22,
            fill_color="#222244", fill_opacity=1,
            stroke_width=0,
        ).align_to(bg, UP)
        term_lbl = Text(
            "Terminal", font="JetBrains Mono", font_size=10, color=C_DIM,
        ).move_to(title_bar)
        txt.next_to(title_bar, DOWN, buff=0.1).align_to(bg, LEFT).shift(RIGHT * 0.15)
        return VGroup(bg, title_bar, term_lbl, txt)

    # ── Scene 8 — Demo Intro ────────────────────────────────────────────

    def scene_demo_intro(self):
        heading = self.zh(
            "實戰演示：運行 build_embed_model.py",
            font_size=34, color=C_CYAN,
        )
        sub = self.en(
            "python3  build_embed_model.py",
            font_size=22, color=C_GREEN,
        ).next_to(heading, DOWN, buff=0.5)
        box = SurroundingRectangle(
            sub, buff=0.25, color=C_GREEN, corner_radius=0.1, stroke_width=1.5,
        )

        with self.voiceover(
            text="而家我哋一齊睇下呢個腳本實際跑起嚟係點樣嘅。"
            "我哋會逐步睇每個 step 嘅代碼同輸出。"
        ):
            self.play(Write(heading), run_time=0.6)
            self.play(FadeIn(sub), Create(box), run_time=0.7)

        self.wait(0.3)
        self.clear()

    # ── Scene 9 — Demo Step 1: Tokenization ─────────────────────────────

    def scene_demo_tokenize(self):
        heading = self.zh(
            "Step 1 — 分詞 Tokenization", font_size=28, color=C_GREEN,
        ).to_edge(UP, buff=0.35)

        code = self.make_code(
            'vocab = {"<PAD>": 0, "<UNK>": 1}\n'
            "for sent in all_sentences:\n"
            "    for word in sent.lower().split():\n"
            "        if word not in vocab:\n"
            "            vocab[word] = len(vocab)\n"
            "\n"
            "def tokenize(text):\n"
            "    tokens = [vocab.get(w, 1)\n"
            "              for w in text.lower().split()]\n"
            "    tokens = tokens[:MAX_LEN]\n"
            "    tokens += [PAD_ID] * (MAX_LEN - len(tokens))\n"
            "    return tokens",
            font_size=14,
        ).shift(LEFT * 3.2 + DOWN * 0.25)

        terminal = self.make_terminal([
            '$ python3 build_embed_model.py',
            '',
            'STEP 1 — TOKENIZATION',
            '',
            'Vocabulary size: 98 words',
            '',
            'Sample mappings:',
            '  "cat"    → 64',
            '  "kitten" → 13',
            '  "pizza"  → 30',
            '',
            'Example:',
            '  "the cat sat on the mat"',
            '  → [5,64,65,14,5,66,0,0,0,0]',
        ], font_size=11, width=5.5).shift(RIGHT * 3.2 + DOWN * 0.25)

        with self.voiceover(
            text="第一步，分詞。左邊係代碼，右邊係運行輸出。"
            "程式先建立一個詞彙表，將每個字配一個整數 ID。"
            "然後 tokenize 函數將句子轉成 ID 數列，唔夠長就用 PAD 填充。"
            "你可以睇到 cat 變成 64，kitten 變成 13。"
        ):
            self.play(Write(heading), run_time=0.4)
            self.play(FadeIn(code, shift=RIGHT * 0.2), run_time=0.8)
            self.play(FadeIn(terminal, shift=LEFT * 0.2), run_time=0.8)

        self.wait(0.3)
        self.clear()

    # ── Scene 10 — Demo Step 2: Model Definition ────────────────────────

    def scene_demo_model(self):
        heading = self.zh(
            "Step 2 — 建立 Transformer 模型", font_size=28, color=C_ORANGE,
        ).to_edge(UP, buff=0.35)

        code = self.make_code(
            "class MiniEmbedModel(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.token_emb = nn.Embedding(\n"
            "            VOCAB_SIZE, 64)\n"
            "        self.pos_emb = nn.Embedding(10, 64)\n"
            "\n"
            "        layer = nn.TransformerEncoderLayer(\n"
            "            d_model=64, nhead=4,\n"
            "            dim_feedforward=128)\n"
            "        self.transformer =\\\n"
            "            nn.TransformerEncoder(layer, 2)",
            font_size=14,
        ).shift(LEFT * 3.2 + DOWN * 0.25)

        terminal = self.make_terminal([
            'STEP 2 — TRANSFORMER ENCODER',
            '',
            'Model created!',
            '',
            'Architecture:',
            '  Token emb  : 98 × 64',
            '  Position   : 10 × 64',
            '  Transformer: 2 layers, 4 heads',
            '  FFN        : 128 hidden',
            '  Output     : 64-dim vector',
            '  Params     : 74,432',
            '',
            'Before training (random):',
            '  cat vs kitten : 0.5523',
            '  cat vs morning: 0.4817',
        ], font_size=11, width=5.5).shift(RIGHT * 3.2 + DOWN * 0.25)

        with self.voiceover(
            text="第二步，建立模型。我哋用 PyTorch 定義一個 MiniEmbedModel。"
            "佢有 Token Embedding，Position Embedding，同兩層 Transformer。"
            "總共大約七萬幾個參數。"
            "訓練之前，模型嘅 embedding 係隨機嘅，"
            "所以 cat 同 kitten 嘅相似度同 cat 同 morning 差唔多。"
        ):
            self.play(Write(heading), run_time=0.4)
            self.play(FadeIn(code, shift=RIGHT * 0.2), run_time=0.8)
            self.play(FadeIn(terminal, shift=LEFT * 0.2), run_time=0.8)

        self.wait(0.3)
        self.clear()

    # ── Scene 11 — Demo Step 3+4: Loss & Training ───────────────────────

    def scene_demo_loss_train(self):
        heading = self.zh(
            "Step 3+4 — Triplet Loss + 訓練", font_size=28, color=C_PURPLE,
        ).to_edge(UP, buff=0.35)

        code = self.make_code(
            "def triplet_loss(anchor, pos, neg,\n"
            "                 margin=0.5):\n"
            "    sim_pos = F.cosine_similarity(anchor, pos)\n"
            "    sim_neg = F.cosine_similarity(anchor, neg)\n"
            "    return torch.clamp(\n"
            "        sim_neg - sim_pos + margin, min=0\n"
            "    ).mean()\n"
            "\n"
            "optimizer = Adam(model.parameters(), lr=0.001)\n"
            "for epoch in range(80):\n"
            "    for anchor, pos, neg in triplets:\n"
            "        loss = triplet_loss(\n"
            "            model(a), model(p), model(n))\n"
            "        loss.backward()\n"
            "        optimizer.step()",
            font_size=13,
        ).shift(LEFT * 3.2 + DOWN * 0.25)

        base_lines = [
            'STEP 3 — CONTRASTIVE LOSS',
            '  Margin = 0.5',
            '',
            'STEP 4 — TRAINING',
            '  15 triplets, 80 epochs',
            '',
            '  Epoch  0  loss=0.4892',
            '  Epoch 10  loss=0.1834',
            '  Epoch 20  loss=0.0612',
            '  Epoch 40  loss=0.0031',
            '  Epoch 60  loss=0.0000',
            '  Epoch 79  loss=0.0000',
            '',
            '  Training complete!',
        ]
        terminal = self.make_terminal(base_lines, font_size=11, width=5.5)
        terminal.shift(RIGHT * 3.2 + DOWN * 0.25)

        with self.voiceover(
            text="第三步定義 Triplet Loss，第四步開始訓練。"
            "每次訓練取一個 anchor、一個 positive、一個 negative。"
            "Loss 函數確保 positive 嘅相似度比 negative 高至少 0.5。"
            "你可以睇到 loss 由 0.49 逐漸降到 0，模型已經學識分辨語義。"
            "80 個 epoch 之後，訓練完成。"
        ):
            self.play(Write(heading), run_time=0.4)
            self.play(FadeIn(code, shift=RIGHT * 0.2), run_time=0.8)
            self.play(FadeIn(terminal, shift=LEFT * 0.2), run_time=0.8)

        self.wait(0.3)
        self.clear()

    # ── Scene 12 — Demo Step 5: Evaluation ──────────────────────────────

    def scene_demo_eval(self):
        heading = self.zh(
            "Step 5 — 評估結果", font_size=28, color=C_YELLOW,
        ).to_edge(UP, buff=0.35)

        terminal = self.make_terminal([
            'STEP 5 — EVALUATION',
            '',
            'After training:',
            '  cat/mat vs kitten/rug',
            '    sim = 0.7126  (similar)',
            '  cat/mat vs runs/morning',
            '    sim = -0.3921 (different)',
            '',
            'More test pairs:',
            '  sun     vs sunny    +0.57 ✓',
            '  pizza   vs food     +0.60 ✓',
            '  dogs    vs puppies  +0.55 ✓',
            '  cat     vs pizza    +0.05 ✗',
            '  car     vs baby     -0.39 ✗',
            '  birds   vs coffee   +0.33 ✗',
        ], font_size=11, width=6.0)

        terminal.move_to(DOWN * 0.3)

        comment = self.zh(
            "相似句子得分高，唔同句子得分低  →  模型成功！",
            font_size=20, color=C_GREEN,
        ).to_edge(DOWN, buff=0.45)

        with self.voiceover(
            text="第五步，評估。訓練之後，cat on mat 同 kitten on rug 嘅相似度升到 0.71。"
            "而 cat on mat 同 runs morning 嘅相似度跌到負 0.39。"
            "其他測試句子都符合預期：意思相近嘅得分高，唔同嘅得分低。"
            "我哋嘅迷你模型成功學識咗語義。"
        ):
            self.play(Write(heading), run_time=0.4)
            self.play(FadeIn(terminal, shift=UP * 0.2), run_time=1)
            self.play(FadeIn(comment, shift=UP * 0.1), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 13 — Demo Step 6: Inside the Vector ───────────────────────

    def scene_demo_vector(self):
        heading = self.zh(
            "Step 6 — 向量內部長咩樣？", font_size=28, color=C_PINK,
        ).to_edge(UP, buff=0.35)

        terminal = self.make_terminal([
            'STEP 6 — INSIDE EMBEDDING',
            '',
            '"the cat sat on the mat"',
            'Dimension: 64',
            '',
            'First 16 values:',
            '  [0-3]  +0.08 -0.02 +0.09 -0.24',
            '  [4-7]  +0.16 -0.05 +0.21 +0.09',
            '  [8-11] -0.10 +0.17 -0.03 +0.06',
            '  [12-15]+0.20 -0.14 +0.07 -0.08',
            '',
            'L2 norm: 1.0000 (normalized)',
        ], font_size=11, width=5.5)
        terminal.shift(LEFT * 2.5 + DOWN * 0.35)

        vec_dots = VGroup()
        vals = [
            0.08, -0.02, 0.09, -0.24, 0.16, -0.05, 0.21, 0.09,
            -0.10, 0.17, -0.03, 0.06, 0.20, -0.14, 0.07, -0.08,
        ]
        bar_colors = [C_CYAN if v >= 0 else C_PINK for v in vals]
        for i, (v, c) in enumerate(zip(vals, bar_colors)):
            bar_h = abs(v) * 6
            bar = Rectangle(
                width=0.18, height=max(bar_h, 0.03),
                fill_color=c, fill_opacity=0.7, stroke_width=0,
            )
            if v >= 0:
                bar.move_to(RIGHT * (3.5 + i * 0.24) + UP * (bar_h / 2 - 0.5))
            else:
                bar.move_to(RIGHT * (3.5 + i * 0.24) + DOWN * (bar_h / 2 + 0.5))
            vec_dots.add(bar)

        zero_line = Line(
            RIGHT * 3.3 + DOWN * 0.5, RIGHT * 7.5 + DOWN * 0.5,
            color=C_DIM, stroke_width=1,
        )
        vec_label = self.en(
            "64-dim vector (first 16 shown)", font_size=11, color=C_DIM,
        ).next_to(zero_line, DOWN, buff=0.15)

        note = self.zh(
            "每個數字代表句子含義嘅一個面向",
            font_size=18, color=C_DIM,
        ).to_edge(DOWN, buff=0.45)

        with self.voiceover(
            text="第六步，睇下向量入面到底係乜。一個 64 維嘅向量就係 64 個浮點數。"
            "每個數字代表句子含義嘅一個面向。"
            "右邊嘅圖顯示咗頭 16 個數值，正數同負數分開顯示。"
            "成個向量嘅 L2 norm 係 1，因為我哋做咗 normalize。"
            "mxbai-embed-large 都係一樣，只不過佢有 1024 個數字。"
        ):
            self.play(Write(heading), run_time=0.4)
            self.play(FadeIn(terminal, shift=RIGHT * 0.2), run_time=0.8)
            self.play(Create(zero_line), run_time=0.3)
            for bar in vec_dots:
                self.play(GrowFromEdge(bar, DOWN if bar.get_center()[1] > -0.5 else UP), run_time=0.08)
            self.play(FadeIn(vec_label), FadeIn(note), run_time=0.5)

        self.wait(0.3)
        self.clear()

    # ── Scene 14 — GitHub ────────────────────────────────────────────────

    def scene_github(self):
        panel = RoundedRectangle(
            corner_radius=0.2, width=12, height=6,
            fill_color=BLACK, fill_opacity=0.75, stroke_width=0,
        )
        self.play(FadeIn(panel), run_time=0.3)

        heading = self.zh("完整代碼", font_size=40, color="#4fc3f7").to_edge(UP, buff=0.8)

        url_text = self.en(GITHUB_URL, font_size=26, color="#66bb6a")
        folder = self.en(
            "lesson 4 - embed model / build_embed_model.py",
            font_size=20, color="#ffb74d",
        ).next_to(url_text, DOWN, buff=0.35)

        url_group = VGroup(url_text, folder).move_to(ORIGIN)
        box = SurroundingRectangle(
            url_group, buff=0.4, color="#4fc3f7", corner_radius=0.15, stroke_width=2,
        )

        with self.voiceover(
            text="完整嘅 Python 代碼可以喺 GitHub 搵到。"
            "去 lesson 4 embed model 文件夾，入面嘅 build_embed_model.py "
            "會用 PyTorch 從零開始構建同訓練呢個模型，大家記得自己跑一次。"
        ):
            self.play(Write(heading), run_time=0.5)
            self.play(Create(box), FadeIn(url_text), run_time=0.8)
            self.play(FadeIn(folder, shift=UP * 0.2), run_time=0.6)

        self.wait(0.3)
        self.clear()

    # ── Scene 9 — Summary ────────────────────────────────────────────────

    def scene_summary(self):
        heading = self.zh("總結：五個核心步驟", font_size=38, color=C_BLUE).to_edge(UP, buff=0.5)

        steps = [
            ("1", "Tokenize", "文字 → 整數 ID", C_GREEN),
            ("2", "Embed", "ID → 向量 (learnable)", C_CYAN),
            ("3", "Transform", "Self-Attention × N 層", C_PURPLE),
            ("4", "Pool", "多個向量 → 一個向量", C_PINK),
            ("5", "Train", "Triplet Loss 對比學習", C_YELLOW),
        ]

        step_boxes = VGroup()
        for num, eng, zh_desc, color in steps:
            rect = RoundedRectangle(
                corner_radius=0.1, width=10, height=0.55,
                fill_color=color, fill_opacity=0.15, stroke_color=color,
            )
            n = self.en(num, font_size=20, color=color).move_to(rect.get_left() + RIGHT * 0.5)
            e = self.en(eng, font_size=18, color=color).move_to(rect.get_left() + RIGHT * 2)
            z = self.zh(zh_desc, font_size=18, color=C_WHITE).move_to(rect.get_left() + RIGHT * 6)
            step_boxes.add(VGroup(rect, n, e, z))
        step_boxes.arrange(DOWN, buff=0.15).move_to(DOWN * 0.1)

        thanks = self.zh("多謝收睇！", font_size=52, color=C_YELLOW)
        git_url = self.en(
            "https://github.com/quantrpeter/learn-rag.git",
            font_size=18, color="#66bb6a",
        ).next_to(thanks, DOWN, buff=0.6)

        with self.voiceover(
            text="總結一下。構建 embedding 模型嘅五個核心步驟："
            "第一，Tokenize，將文字變成整數。"
            "第二，Embed，將整數變成可學習嘅向量。"
            "第三，Transform，用 Self-Attention 理解字詞關係。"
            "第四，Pool，將多個向量平均成一個句子向量。"
            "第五，Train，用 Triplet Loss 訓練模型分辨語義。"
            "呢個就係所有 embedding 模型嘅核心，由我哋嘅迷你模型到 mxbai-embed-large 都一樣。"
        ):
            self.play(Write(heading), run_time=0.5)
            for sb in step_boxes:
                self.play(FadeIn(sb, shift=RIGHT * 0.2), run_time=0.4)

        self.wait(0.5)
        self.clear()

        with self.voiceover(
            text="多謝收睇！記得去 GitHub 自己跑一次 build_embed_model.py，親手訓練一個 embedding 模型。"
        ):
            self.play(Write(thanks), run_time=1)
            self.play(FadeIn(git_url, shift=UP * 0.2), run_time=0.6)

        self.wait(1)
        self.play(FadeOut(thanks), FadeOut(git_url), run_time=0.8)
