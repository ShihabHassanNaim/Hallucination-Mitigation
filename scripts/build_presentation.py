"""
Build a thesis defense presentation slides for the
Hallucination Detection Pipeline project.

Outputs a single .pptx file at the project root.
"""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
PROJECT_NAME = "Hallucination Detection Pipeline"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "Hallucination_Presentation.pptx"

DARK_BG = RGBColor(0x0E, 0x1B, 0x2C)          # deep navy
ACCENT = RGBColor(0x2E, 0x86, 0xDE)           # bright blue
ACCENT2 = RGBColor(0xF5, 0xA6, 0x23)          # amber
LIGHT = RGBColor(0xF4, 0xF6, 0xFA)            # near-white
GREY = RGBColor(0x9C, 0xA3, 0xB1)
SUCCESS = RGBColor(0x54, 0xC4, 0x7A)
DANGER = RGBColor(0xE5, 0x53, 0x53)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def style_text(shape_or_tf, text, *, size=20, bold=False, color=LIGHT, align=PP_ALIGN.LEFT):
    """Apply text to a shape or text frame. Accepts both for convenience."""
    tf = getattr(shape_or_tf, "text_frame", shape_or_tf)
    tf.word_wrap = True
    # Clear existing paragraphs.
    for _ in range(len(tf.paragraphs) - 1):
        p = tf.paragraphs[-1]
        p._p.getparent().remove(p._p)
    if tf.paragraphs:
        p = tf.paragraphs[0]
        # Remove all runs from the paragraph.
        for r in list(p.runs):
            r._r.getparent().remove(r._r)
    else:
        p = tf.add_paragraph()
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = "Calibri"
    return p


def _to_tf(obj):
    """Return the text_frame of a shape or the text_frame itself."""
    return getattr(obj, "text_frame", obj)


def add_bullets(obj, items, *, size=18, color=LIGHT, bullet_color=ACCENT):
    tf = _to_tf(obj)
    tf.word_wrap = True
    # Clear existing paragraphs.
    for _ in range(len(tf.paragraphs) - 1):
        p = tf.paragraphs[-1]
        p._p.getparent().remove(p._p)
    if tf.paragraphs:
        p_first = tf.paragraphs[0]
        for r in list(p_first.runs):
            r._r.getparent().remove(r._r)
    else:
        p_first = tf.add_paragraph()
    paragraphs = [p_first]
    for item in items[1:]:
        paragraphs.append(tf.add_paragraph())

    for i, item in enumerate(items):
        p = paragraphs[i]
        p.alignment = PP_ALIGN.LEFT
        p.level = 0
        dot_run = p.add_run()
        dot_run.text = "•  "
        dot_run.font.size = Pt(size)
        dot_run.font.color.rgb = bullet_color
        dot_run.font.bold = True
        dot_run.font.name = "Calibri"
        text_run = p.add_run()
        text_run.text = item
        text_run.font.size = Pt(size)
        text_run.font.color.rgb = color
        text_run.font.name = "Calibri"


def add_background(slide, color=DARK_BG):
    bg = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5)
    )
    bg.line.fill.background()
    bg.fill.solid()
    bg.fill.fore_color.rgb = color
    bg.shadow.inherit = False
    return bg


def add_title_bar(slide, title, subtitle=None):
    # Accent strip
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(0.9)
    )
    bar.line.fill.background()
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.shadow.inherit = False

    tf = slide.shapes.add_textbox(Inches(0.5), Inches(0.12), Inches(12.3), Inches(0.7))
    style_text(tf, title, size=30, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))
    if subtitle:
        tf2 = slide.shapes.add_textbox(Inches(0.5), Inches(0.55), Inches(12.3), Inches(0.4))
        style_text(tf2, subtitle, size=14, color=RGBColor(0xE6, 0xEF, 0xFA))


def add_caption(slide, text, *, left=0.5, top=7.05, width=12.3, size=11, color=GREY):
    tf = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(0.3))
    style_text(tf, text, size=size, color=color)


def add_page_number(slide, n, total):
    tf = slide.shapes.add_textbox(Inches(12.3), Inches(7.1), Inches(0.9), Inches(0.3))
    style_text(tf, f"{n} / {total}", size=10, color=GREY, align=PP_ALIGN.RIGHT)


def add_section_title(slide, title):
    tf = slide.shapes.add_textbox(Inches(0.5), Inches(1.1), Inches(12.3), Inches(0.7))
    style_text(tf, title, size=28, bold=True, color=ACCENT2)


def add_card(slide, left, top, width, height, *, fill_color=None, line_color=None):
    card = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height
    )
    if fill_color is None:
        card.fill.background()
    else:
        card.fill.solid()
        card.fill.fore_color.rgb = fill_color
    if line_color is None:
        card.line.fill.background()
    else:
        card.line.color.rgb = line_color
        card.line.width = Pt(0.75)
    card.shadow.inherit = False
    return card


# ---------------------------------------------------------------------------
# Slides
# ---------------------------------------------------------------------------
def add_slide_title(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    add_background(slide, DARK_BG)

    # Decorative side bar
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.45), Inches(7.5))
    bar.line.fill.background()
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.shadow.inherit = False

    # Title
    tf = slide.shapes.add_textbox(Inches(0.9), Inches(1.6), Inches(11.5), Inches(1.4))
    style_text(tf, "Hallucination Detection Pipeline", size=46, bold=True, color=LIGHT)

    tf = slide.shapes.add_textbox(Inches(0.9), Inches(2.9), Inches(11.5), Inches(0.7))
    style_text(
        tf,
        "A 13-Step Verification Architecture for Retrieval-Augmented Generation",
        size=22, color=ACCENT2, bold=False,
    )

    # Divider
    div = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.9), Inches(3.65), Inches(3.0), Inches(0.05)
    )
    div.line.fill.background()
    div.fill.solid()
    div.fill.fore_color.rgb = ACCENT
    div.shadow.inherit = False

    # Subtitle
    tf = slide.shapes.add_textbox(Inches(0.9), Inches(3.85), Inches(11.5), Inches(0.5))
    style_text(
        tf,
        "Real HuggingFace models  ·  31 tests passing  ·  Flask + FastAPI UI",
        size=18, color=LIGHT,
    )

    # Author block
    tf = slide.shapes.add_textbox(Inches(0.9), Inches(5.4), Inches(11.5), Inches(0.45))
    style_text(tf, "Master's Thesis Defense", size=18, color=LIGHT, bold=True)
    tf = slide.shapes.add_textbox(Inches(0.9), Inches(5.85), Inches(11.5), Inches(0.4))
    style_text(tf, "Shihab Hassan Naim", size=16, color=GREY)
    tf = slide.shapes.add_textbox(Inches(0.9), Inches(6.25), Inches(11.5), Inches(0.4))
    style_text(tf, "github.com/ShihabHassanNaim/Hallucination-Mitigation", size=14, color=ACCENT2)


def add_slide_agenda(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_title_bar(slide, "Agenda", "What we will cover today")
    add_section_title(slide, "Roadmap")

    items = [
        ("1.  Motivation & Problem", "Why hallucination is a hard, real problem"),
        ("2.  Background & Related Work", "RAG, NLI, claim-level verification"),
        ("3.  Goals & Contributions", "Specific, measurable outcomes"),
        ("4.  System Architecture", "The 13-step pipeline, one-to-one with the diagram"),
        ("5.  Implementation Details", "Models, retrieval, fusion, regeneration"),
        ("6.  User Interface", "Flask demo, REST API, project explorer"),
        ("7.  Evaluation", "Metrics, datasets (HaluEval + TruthfulQA), results"),
        ("8.  Comparison with Prior Work", "Why our design wins"),
        ("9.  Limitations & Future Work", "Honest discussion"),
        ("10. Conclusion", "Take-aways and Q&A"),
    ]
    for i, (head, sub) in enumerate(items):
        col = i % 2
        row = i // 2
        left = Inches(0.7 + col * 6.2)
        top = Inches(1.95 + row * 1.0)
        add_card(slide, left, top, Inches(6.0), Inches(0.85), fill_color=RGBColor(0x16, 0x2A, 0x40))
        tf = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.08), Inches(5.7), Inches(0.4))
        style_text(tf, head, size=16, bold=True, color=ACCENT2)
        tf = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.43), Inches(5.7), Inches(0.4))
        style_text(tf, sub, size=12, color=LIGHT)


def add_slide_motivation(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_title_bar(slide, "Motivation", "Hallucination is the bottleneck for trustworthy RAG")
    add_section_title(slide, "The problem we are solving")

    # Left: bullets
    add_card(slide, Inches(0.5), Inches(1.95), Inches(7.5), Inches(4.7), fill_color=RGBColor(0x16, 0x2A, 0x40))
    tf = slide.shapes.add_textbox(Inches(0.8), Inches(2.1), Inches(7.0), Inches(4.4))
    add_bullets(tf, [
        "LLMs invent facts — even when grounding context is provided.",
        "A single wrong token can poison a downstream answer.",
        "Generic logprob-based confidence is poorly calibrated.",
        "Faithfulness ≠ factuality — both must be verified.",
        "Key insight: catch hallucinations at the claim level, not the answer level.",
        "Goal: a system that can PASS, REGENERATE, or ABSTAIN — and explain why.",
    ], size=18)

    # Right: stats
    add_card(slide, Inches(8.3), Inches(1.95), Inches(4.5), Inches(4.7), fill_color=RGBColor(0x16, 0x2A, 0x40))
    f = slide.shapes.add_textbox(Inches(8.5), Inches(2.1), Inches(4.1), Inches(0.5))
    style_text(f, "Why now?", size=18, bold=True, color=ACCENT2)
    bullets = [
        "HaluEval & TruthfulQA expose the gap in modern LLM answers.",
        "Enterprises want verifiable, citation-backed outputs.",
        "No open-source pipeline unifies detection + regen + abduction.",
    ]
    add_bullets(
        slide.shapes.add_textbox(Inches(8.5), Inches(2.6), Inches(4.1), Inches(3.0)).text_frame,
        bullets, size=14,
    )

    add_caption(slide, "Supporting papers: HaluEval · TruthfulQA · LLM-Check · QuaLLM-KG (see /Paper)")


def add_slide_related_work(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_title_bar(slide, "Background & Related Work", "Where this work sits")
    add_section_title(slide, "Three families of approaches")

    families = [
        ("Self-check", "Logprobs, entropy, self-consistency", "Cheap, but poorly calibrated"),
        ("RAG w/ abstractive checks", "Re-read prompts, self-ask", "Hallucinates confidently"),
        ("NLI / FactScore style", "Atomic claims vs evidence", "Most accurate, slow & fragmented"),
    ]
    for i, (head, tech, note) in enumerate(families):
        left = Inches(0.5 + i * 4.3)
        top = Inches(2.0)
        add_card(slide, left, top, Inches(4.05), Inches(4.4), fill_color=RGBColor(0x16, 0x2A, 0x40))
        tf = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.2), Inches(3.6), Inches(0.5))
        style_text(tf, head, size=20, bold=True, color=ACCENT2)
        tf = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.85), Inches(3.6), Inches(0.5))
        style_text(tf, tech, size=14, color=LIGHT)
        tf = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(1.4), Inches(3.6), Inches(2.0))
        style_text(tf, note, size=14, color=GREY)

    # Our place
    add_card(slide, Inches(0.5), Inches(6.6), Inches(12.3), Inches(0.55), fill_color=ACCENT)
    tf = slide.shapes.add_textbox(Inches(0.7), Inches(6.65), Inches(12.0), Inches(0.5))
    style_text(
        tf,
        "Our contribution: unify all three — logprob confidence + RAG context + claim-level NLI — "
        "with explicit early detection, evidence-guided regeneration, and a 3-way gate.",
        size=14, bold=True, color=RGBColor(0x0E, 0x1B, 0x2C),
    )


def add_slide_goals(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_title_bar(slide, "Goals & Contributions", "Specific, measurable outcomes")
    add_section_title(slide, "What this project delivers")

    goals = [
        ("G1", "13-step pipeline", "One-to-one mapping with the architecture diagram, every step a real Python module."),
        ("G2", "Real HF models", "Llama 3.1, Qwen 3, BGE-M3, BGE-Reranker, DeBERTa-v3 NLI — no toy stubs."),
        ("G3", "Claim-level verification", "Atomic claim extraction + NLI per claim, not per sentence."),
        ("G4", "Three-way gate", "PASS / REGEN / ABSTAIN with calibrated confidence (EEDC + temperature scaling)."),
        ("G5", "Two user surfaces", "FastAPI (port 8000) + Flask (port 5000) with live metrics & project tree."),
        ("G6", "Reproducible evaluation", "31 unit tests, metrics on HaluEval + TruthfulQA, comparison report."),
    ]
    for i, (tag, title, body) in enumerate(goals):
        col = i % 2
        row = i // 2
        left = Inches(0.5 + col * 6.3)
        top = Inches(1.95 + row * 1.55)
        add_card(slide, left, top, Inches(6.1), Inches(1.4), fill_color=RGBColor(0x16, 0x2A, 0x40))
        # tag badge
        badge = slide.shapes.add_shape(MSO_SHAPE.OVAL, left + Inches(0.15), top + Inches(0.35), Inches(0.7), Inches(0.7))
        badge.fill.solid()
        badge.fill.fore_color.rgb = ACCENT
        badge.line.fill.background()
        badge.shadow.inherit = False
        t = slide.shapes.add_textbox(left + Inches(0.15), top + Inches(0.42), Inches(0.7), Inches(0.55))
        style_text(t, tag, size=16, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.CENTER)
        # title
        tf = slide.shapes.add_textbox(left + Inches(1.0), top + Inches(0.15), Inches(5.0), Inches(0.4))
        style_text(tf, title, size=16, bold=True, color=ACCENT2)
        # body
        tf = slide.shapes.add_textbox(left + Inches(1.0), top + Inches(0.55), Inches(5.0), Inches(0.85))
        style_text(tf, body, size=13, color=LIGHT)


def add_slide_architecture_steps(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_title_bar(slide, "System Architecture", "The 13-step pipeline, end-to-end")
    add_section_title(slide, "From user question to verified answer")

    steps = [
        ("01", "Query Understanding", "GLiNER + spaCy + Llama 3.1"),
        ("02", "Grounded Generation", "Llama 3.1 / Qwen 3"),
        ("03", "Early Detection", "BGE-M3 cosine + XGBoost"),
        ("04", "Confidence Assessment", "EEDC + temperature scaling"),
        ("05", "Atomic Claim Extraction", "rule-based + LLM splitter"),
        ("06", "Hybrid Retrieval", "BGE-M3 + FAISS + BM25"),
        ("07", "Evidence Reranking", "BGE-Reranker-v2 cross-encoder"),
        ("08", "NLI Verification", "DeBERTa-v3-MNLI (multilingual)"),
        ("09", "Confidence Fusion", "weighted multi-score fusion"),
        ("10", "Evidence-Guided Regen", "Llama 3.1 (verified-only)"),
        ("11", "Final Verification Gate", "PASS / REGEN / ABSTAIN"),
        ("12", "Explainable Response", "JSON + Markdown + HTML"),
        ("13", "Evaluation", "HRR, CCR, VSR, BLEU, ROUGE-L, BERTScore, ROC-AUC, MRR, nDCG"),
    ]

    # 3 columns × 5 rows (last row has 1 cell)
    cols = 3
    cell_w = 4.05
    cell_h = 0.85
    for idx, (num, name, model) in enumerate(steps):
        col = idx % cols
        row = idx // cols
        left = Inches(0.5 + col * (cell_w + 0.15))
        top = Inches(1.95 + row * (cell_h + 0.1))
        add_card(slide, left, top, Inches(cell_w), Inches(cell_h), fill_color=RGBColor(0x16, 0x2A, 0x40))
        # number
        t = slide.shapes.add_textbox(left + Inches(0.1), top + Inches(0.18), Inches(0.7), Inches(0.5))
        style_text(t, num, size=18, bold=True, color=ACCENT2)
        # name
        t = slide.shapes.add_textbox(left + Inches(0.85), top + Inches(0.08), Inches(3.0), Inches(0.4))
        style_text(t, name, size=14, bold=True, color=LIGHT)
        # model
        t = slide.shapes.add_textbox(left + Inches(0.85), top + Inches(0.45), Inches(3.0), Inches(0.4))
        style_text(t, model, size=10, color=GREY)


def add_slide_architecture_flow(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_title_bar(slide, "Pipeline Flow", "How the 13 steps connect")
    add_section_title(slide, "One cycle, two feedback loops")

    # Phase 1: Generation
    add_card(slide, Inches(0.5), Inches(1.95), Inches(4.0), Inches(4.9), fill_color=RGBColor(0x16, 0x2A, 0x40))
    t = slide.shapes.add_textbox(Inches(0.7), Inches(2.05), Inches(3.6), Inches(0.5))
    style_text(t, "Phase 1 — Generate", size=18, bold=True, color=ACCENT2)
    add_bullets(
        slide.shapes.add_textbox(Inches(0.7), Inches(2.6), Inches(3.6), Inches(4.0)).text_frame,
        ["1. Query Understanding", "2. Grounded Generation", "3. Early Detection", "4. Confidence Assessment"],
        size=14,
    )

    # Phase 2: Verify
    add_card(slide, Inches(4.7), Inches(1.95), Inches(4.0), Inches(4.9), fill_color=RGBColor(0x16, 0x2A, 0x40))
    t = slide.shapes.add_textbox(Inches(4.9), Inches(2.05), Inches(3.6), Inches(0.5))
    style_text(t, "Phase 2 — Verify", size=18, bold=True, color=ACCENT2)
    add_bullets(
        slide.shapes.add_textbox(Inches(4.9), Inches(2.6), Inches(3.6), Inches(4.0)).text_frame,
        ["5. Claim Extraction", "6. Hybrid Retrieval", "7. Evidence Reranking", "8. NLI Verification", "9. Confidence Fusion"],
        size=14,
    )

    # Phase 3: Decide
    add_card(slide, Inches(8.9), Inches(1.95), Inches(3.9), Inches(4.9), fill_color=RGBColor(0x16, 0x2A, 0x40))
    t = slide.shapes.add_textbox(Inches(9.1), Inches(2.05), Inches(3.5), Inches(0.5))
    style_text(t, "Phase 3 — Decide", size=18, bold=True, color=ACCENT2)
    add_bullets(
        slide.shapes.add_textbox(Inches(9.1), Inches(2.6), Inches(3.5), Inches(4.0)).text_frame,
        ["10. Evidence-Guided Regen", "11. Final Gate (PASS/REGEN/ABSTAIN)", "12. Explainable Response", "13. Evaluation"],
        size=14,
    )

    # Arrows / feedback
    add_card(slide, Inches(0.5), Inches(6.95), Inches(12.3), Inches(0.4), fill_color=ACCENT)
    t = slide.shapes.add_textbox(Inches(0.7), Inches(6.97), Inches(12.0), Inches(0.4))
    style_text(
        t,
        "Feedback: low confidence → regen with verified-only context. Persistent disagreement → ABSTAIN.",
        size=13, bold=True, color=RGBColor(0x0E, 0x1B, 0x2C),
    )


def add_slide_implementation(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_title_bar(slide, "Implementation Details", "Inside the modules")
    add_section_title(slide, "Notable technical choices")

    highlights = [
        ("Hybrid retrieval", "Reciprocal Rank Fusion of dense (BGE-M3 + FAISS) and sparse (BM25) — robust to paraphrase and keyword mismatch."),
        ("EEDC calibration", "Expected Entropy of Disagreement across Claims, then Platt + temperature scaling for the 3-way gate."),
        ("Atomic claim splitter", "Rule-based first (deterministic), LLM (Llama 3.1) fallback only when sentence splitting is ambiguous."),
        ("Evidence-guided regen", "Only verified claims survive — the regenerator is conditioned on the supported subgraph."),
        ("Real models, no stubs", "Every wrapper loads its HF checkpoint; production code never falls back to a mock."),
        ("Configuration-as-code", "configs/default.yaml + HALLUC_* env overrides — one source of truth."),
    ]
    for i, (head, body) in enumerate(highlights):
        col = i % 2
        row = i // 2
        left = Inches(0.5 + col * 6.3)
        top = Inches(1.95 + row * 1.55)
        add_card(slide, left, top, Inches(6.1), Inches(1.4), fill_color=RGBColor(0x16, 0x2A, 0x40))
        tf = slide.shapes.add_textbox(left + Inches(0.25), top + Inches(0.15), Inches(5.7), Inches(0.4))
        style_text(tf, head, size=16, bold=True, color=ACCENT2)
        tf = slide.shapes.add_textbox(left + Inches(0.25), top + Inches(0.55), Inches(5.7), Inches(0.85))
        style_text(tf, body, size=13, color=LIGHT)


def add_slide_models(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_title_bar(slide, "Models Used", "HuggingFace checkpoints at a glance")
    add_section_title(slide, "One row per pipeline responsibility")

    rows = [
        ("Query understanding", "knowledgator/gliner-base-v1.0, en_core_web_sm, Llama 3.1 8B"),
        ("Generator", "meta-llama/Meta-Llama-3.1-8B-Instruct (alt: Qwen/Qwen3-8B)"),
        ("Dense embeddings", "BAAI/bge-m3"),
        ("Hallucination classifier", "XGBoost (sklearn-logreg fallback)"),
        ("Cross-encoder reranker", "BAAI/bge-reranker-v2-m3"),
        ("NLI verifier", "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"),
        ("Sparse retrieval", "rank_bm25 (pure-Python BM25)"),
        ("Generation metrics", "bert-score, rouge-score, nltk"),
    ]
    # Header
    add_card(slide, Inches(0.5), Inches(1.95), Inches(12.3), Inches(0.5), fill_color=ACCENT)
    t = slide.shapes.add_textbox(Inches(0.7), Inches(2.0), Inches(5.0), Inches(0.4))
    style_text(t, "Component", size=14, bold=True, color=RGBColor(0x0E, 0x1B, 0x2C))
    t = slide.shapes.add_textbox(Inches(6.0), Inches(2.0), Inches(6.8), Inches(0.4))
    style_text(t, "HuggingFace checkpoint", size=14, bold=True, color=RGBColor(0x0E, 0x1B, 0x2C))

    for i, (comp, mdl) in enumerate(rows):
        top = Inches(2.5 + i * 0.5)
        add_card(slide, Inches(0.5), top, Inches(12.3), Inches(0.45),
                 fill_color=RGBColor(0x16, 0x2A, 0x40) if i % 2 == 0 else RGBColor(0x12, 0x22, 0x35))
        t = slide.shapes.add_textbox(Inches(0.7), top + Inches(0.05), Inches(5.0), Inches(0.4))
        style_text(t, comp, size=13, color=ACCENT2)
        t = slide.shapes.add_textbox(Inches(6.0), top + Inches(0.05), Inches(6.8), Inches(0.4))
        style_text(t, mdl, size=13, color=LIGHT)


def add_slide_ui(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_title_bar(slide, "User Interface", "Demo, API, and project explorer")
    add_section_title(slide, "Three views on the same pipeline")

    # Flask demo
    add_card(slide, Inches(0.5), Inches(1.95), Inches(6.0), Inches(2.5), fill_color=RGBColor(0x16, 0x2A, 0x40))
    t = slide.shapes.add_textbox(Inches(0.7), Inches(2.05), Inches(5.6), Inches(0.4))
    style_text(t, "Flask Demo (port 5000)", size=18, bold=True, color=ACCENT2)
    add_bullets(
        slide.shapes.add_textbox(Inches(0.7), Inches(2.5), Inches(5.6), Inches(2.0)).text_frame,
        [
            "Query box + 13-step trace panel",
            "Per-claim verdicts and retrieved evidence",
            "Live / degraded badge (model status)",
            "Dataset selector on Accuracy & Evaluation cards",
        ],
        size=13,
    )

    # FastAPI
    add_card(slide, Inches(6.8), Inches(1.95), Inches(6.0), Inches(2.5), fill_color=RGBColor(0x16, 0x2A, 0x40))
    t = slide.shapes.add_textbox(Inches(7.0), Inches(2.05), Inches(5.6), Inches(0.4))
    style_text(t, "FastAPI (port 8000)", size=18, bold=True, color=ACCENT2)
    add_bullets(
        slide.shapes.add_textbox(Inches(7.0), Inches(2.5), Inches(5.6), Inches(2.0)).text_frame,
        [
            "GET  /health        · liveness",
            "POST /index         · rebuild index",
            "POST /query         · run 13-step pipeline",
            "GET  /metrics       · aggregate metrics",
            "GET  /structure     · project tree JSON",
        ],
        size=13,
    )

    # Project tree
    add_card(slide, Inches(0.5), Inches(4.6), Inches(12.3), Inches(2.35), fill_color=RGBColor(0x16, 0x2A, 0x40))
    t = slide.shapes.add_textbox(Inches(0.7), Inches(4.7), Inches(11.9), Inches(0.4))
    style_text(t, "Project Explorer (/structure)", size=18, bold=True, color=ACCENT2)
    add_bullets(
        slide.shapes.add_textbox(Inches(0.7), Inches(5.15), Inches(11.9), Inches(2.0)).text_frame,
        [
            "Live, depth-aware view of the codebase — backed by GET /api/structure",
            "Hidden-file toggle, depth selector, live filter",
            "Same JSON surface used by the CLI tree view and the README diagram",
        ],
        size=14,
    )


def add_slide_evaluation_metrics(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_title_bar(slide, "Evaluation", "What we measure, and how")
    add_section_title(slide, "10 metric categories — same as the diagram")

    metrics = [
        ("HRR", "Hallucination Rejection Rate — fraction of bad answers caught."),
        ("CCR", "Correct Claim Recall — share of gold claims covered."),
        ("VSR", "Verification Success Rate — claims supported by evidence."),
        ("BLEU / ROUGE-L", "Lexical overlap with gold answer."),
        ("BERTScore", "Semantic similarity using BERT embeddings."),
        ("ROC-AUC", "Detector calibration across thresholds."),
        ("MRR / nDCG", "Retrieval ranking quality."),
        ("P / R / F1", "Claim-level detection report."),
        ("End-to-end accuracy", "Final answer matches gold."),
        ("Confidence calibration", "EEDC vs. observed correctness."),
    ]
    for i, (head, body) in enumerate(metrics):
        col = i % 2
        row = i // 2
        left = Inches(0.5 + col * 6.3)
        top = Inches(1.95 + row * 0.95)
        add_card(slide, left, top, Inches(6.1), Inches(0.85), fill_color=RGBColor(0x16, 0x2A, 0x40))
        tf = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.1), Inches(5.7), Inches(0.35))
        style_text(tf, head, size=15, bold=True, color=ACCENT2)
        tf = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.43), Inches(5.7), Inches(0.4))
        style_text(tf, body, size=12, color=LIGHT)


def add_slide_evaluation_results(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_title_bar(slide, "Evaluation Results", "Snapshot from the latest full run")
    add_section_title(slide, "Synthetic QA set (small subset) — illustrative")

    # Numbers from data/metrics.json
    rows = [
        ("Verification Success Rate", "1.000", SUCCESS),
        ("Retrieval Recall @ 5", "0.797", ACCENT2),
        ("Retrieval MRR", "0.797", ACCENT2),
        ("Retrieval nDCG", "0.797", ACCENT2),
        ("Retrieval Precision @ 5", "0.431", ACCENT2),
        ("End-to-end accuracy", "0.344", GREY),
        ("Mean confidence", "0.428", ACCENT),
        ("Claim P / R / F1", "0.240", GREY),
    ]
    cols = 4
    cell_w = 3.0
    cell_h = 1.55
    for i, (name, val, color) in enumerate(rows):
        col = i % cols
        row = i // cols
        left = Inches(0.5 + col * (cell_w + 0.15))
        top = Inches(1.95 + row * (cell_h + 0.15))
        add_card(slide, left, top, Inches(cell_w), Inches(cell_h), fill_color=RGBColor(0x16, 0x2A, 0x40))
        t = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.2), Inches(cell_w - 0.4), Inches(0.4))
        style_text(t, name, size=14, color=LIGHT)
        t = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.65), Inches(cell_w - 0.4), Inches(0.7))
        style_text(t, val, size=30, bold=True, color=color)

    # Take-away
    add_card(slide, Inches(0.5), Inches(5.95), Inches(12.3), Inches(1.0), fill_color=ACCENT)
    t = slide.shapes.add_textbox(Inches(0.7), Inches(6.05), Inches(12.0), Inches(0.5))
    style_text(
        t,
        "Take-away:  retrieval is strong (≈0.80), VSR is 1.0, claim-level P/R/F1 is the weak link — "
        "motivating our evidence-guided regen and per-claim verdict reporting.",
        size=14, bold=True, color=RGBColor(0x0E, 0x1B, 0x2C),
    )

    add_caption(slide, "Source: data/metrics.json (aggregated synthetic subset). Full benchmark numbers in Paper/.")


def add_slide_datasets(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_title_bar(slide, "Datasets", "HaluEval + TruthfulQA")
    add_section_title(slide, "Two real benchmarks, side-by-side")

    cards = [
        ("HaluEval", "Large-scale hallucination benchmark (~35k samples).", "halueval_qa.jsonl · halueval_corpus.jsonl"),
        ("TruthfulQA", "Truthfulness benchmark for common misconceptions.", "truthfulqa_qa.jsonl · truthfulqa_corpus.jsonl"),
    ]
    for i, (name, desc, files) in enumerate(cards):
        left = Inches(0.5 + i * 6.3)
        top = Inches(1.95)
        add_card(slide, left, top, Inches(6.1), Inches(3.4), fill_color=RGBColor(0x16, 0x2A, 0x40))
        t = slide.shapes.add_textbox(left + Inches(0.25), top + Inches(0.2), Inches(5.7), Inches(0.5))
        style_text(t, name, size=22, bold=True, color=ACCENT2)
        t = slide.shapes.add_textbox(left + Inches(0.25), top + Inches(0.85), Inches(5.7), Inches(1.5))
        style_text(t, desc, size=14, color=LIGHT)
        t = slide.shapes.add_textbox(left + Inches(0.25), top + Inches(2.2), Inches(5.7), Inches(1.0))
        style_text(t, files, size=12, color=GREY)

    # Cross-dataset comparison callout
    add_card(slide, Inches(0.5), Inches(5.55), Inches(12.3), Inches(1.4), fill_color=ACCENT)
    t = slide.shapes.add_textbox(Inches(0.7), Inches(5.65), Inches(12.0), Inches(0.5))
    style_text(t, "Cross-dataset comparison", size=18, bold=True, color=RGBColor(0x0E, 0x1B, 0x2C))
    t = slide.shapes.add_textbox(Inches(0.7), Inches(6.1), Inches(12.0), Inches(0.8))
    style_text(
        t,
        "Frontend /evaluation page shows a side-by-side card with one column per dataset plus a Δ column. "
        "Run: python -m scripts.evaluate_datasets to populate.",
        size=13, color=RGBColor(0x0E, 0x1B, 0x2C),
    )


def add_slide_comparison(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_title_bar(slide, "Comparison with Prior Work", "Why this design wins")
    add_section_title(slide, "Side-by-side with the surveyed systems")

    headers = ["System", "Detection", "Citation", "Regen", "Gate", "Open"]
    rows = [
        ("SelfCheckGPT", "✓", "✗", "✗", "✗", "✓"),
        ("FActScore", "✓", "✓", "✗", "✗", "✓"),
        ("RAGAS", "✓", "✗", "✗", "✗", "✓"),
        ("LLM-Check", "✓", "✓", "✗", "✗", "✗"),
        ("QuaLLM-KG", "✓", "✓", "✗", "✗", "✗"),
        ("This work", "✓", "✓", "✓", "✓", "✓"),
    ]
    col_widths = [2.4, 2.0, 2.0, 2.0, 2.0, 2.0]
    left = 0.5
    top = 1.95
    # Header row
    x = Inches(left)
    for i, h in enumerate(headers):
        add_card(slide, x, Inches(top), Inches(col_widths[i]), Inches(0.55), fill_color=ACCENT)
        t = slide.shapes.add_textbox(x + Inches(0.1), Inches(top + 0.08), Inches(col_widths[i] - 0.2), Inches(0.4))
        style_text(t, h, size=14, bold=True, color=RGBColor(0x0E, 0x1B, 0x2C), align=PP_ALIGN.CENTER)
        x = Inches(left + sum(col_widths[:i + 1]))
    # Data rows
    for r_idx, row in enumerate(rows):
        row_top = 2.5 + r_idx * 0.55
        x = Inches(left)
        for i, val in enumerate(row):
            color = RGBColor(0x16, 0x2A, 0x40) if r_idx % 2 == 0 else RGBColor(0x12, 0x22, 0x35)
            add_card(slide, x, Inches(row_top), Inches(col_widths[i]), Inches(0.5), fill_color=color)
            text_color = LIGHT
            if val == "✓":
                text_color = SUCCESS
            elif val == "✗":
                text_color = DANGER
            t = slide.shapes.add_textbox(x + Inches(0.1), Inches(row_top + 0.08), Inches(col_widths[i] - 0.2), Inches(0.4))
            align = PP_ALIGN.LEFT if i == 0 else PP_ALIGN.CENTER
            is_last = r_idx == len(rows) - 1
            bold = is_last or i == 0
            text_color = ACCENT2 if is_last else text_color
            style_text(t, val, size=13, color=text_color, bold=bold, align=align)
            x = Inches(left + sum(col_widths[:i + 1]))

    add_caption(slide, "Comparison table source: Paper/comparison_table.csv")


def add_slide_limitations(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_title_bar(slide, "Limitations & Future Work", "Honest discussion")
    add_section_title(slide, "Where we are constrained — and what comes next")

    limits = [
        ("L1", "Latency", "13 serial steps add 5–15 s per query. Future: batch, cache, and async retrieval."),
        ("L2", "GPU dependence", "Llama 3.1 8B needs ~16 GB VRAM. Future: distillation + 4-bit quantization."),
        ("L3", "Claim splitter", "Accuracy on nested claims is uneven. Future: stronger LLM-only splitter with feedback."),
        ("L4", "Hallucination classifier", "XGBoost uses engineered features. Future: trained end-to-end with NLI signal."),
        ("L5", "Evaluation coverage", "Synthetic QA is small. Future: full runs on HaluEval + TruthfulQA + GSM8K."),
        ("L6", "Multilingual evidence", "NLI is multilingual, retrieval is mostly English. Future: cross-lingual index."),
    ]
    for i, (tag, head, body) in enumerate(limits):
        col = i % 2
        row = i // 2
        left = Inches(0.5 + col * 6.3)
        top = Inches(1.95 + row * 1.55)
        add_card(slide, left, top, Inches(6.1), Inches(1.4), fill_color=RGBColor(0x16, 0x2A, 0x40))
        badge = slide.shapes.add_shape(MSO_SHAPE.OVAL, left + Inches(0.15), top + Inches(0.35), Inches(0.7), Inches(0.7))
        badge.fill.solid()
        badge.fill.fore_color.rgb = DANGER
        badge.line.fill.background()
        badge.shadow.inherit = False
        t = slide.shapes.add_textbox(left + Inches(0.15), top + Inches(0.42), Inches(0.7), Inches(0.55))
        style_text(t, tag, size=16, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.CENTER)
        t = slide.shapes.add_textbox(left + Inches(1.0), top + Inches(0.15), Inches(5.0), Inches(0.4))
        style_text(t, head, size=16, bold=True, color=ACCENT2)
        t = slide.shapes.add_textbox(left + Inches(1.0), top + Inches(0.55), Inches(5.0), Inches(0.85))
        style_text(t, body, size=13, color=LIGHT)


def add_slide_conclusion(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_title_bar(slide, "Conclusion", "Take-aways")
    add_section_title(slide, "What we built, what we learned")

    # Left: contributions
    add_card(slide, Inches(0.5), Inches(1.95), Inches(6.1), Inches(4.8), fill_color=RGBColor(0x16, 0x2A, 0x40))
    t = slide.shapes.add_textbox(Inches(0.7), Inches(2.05), Inches(5.7), Inches(0.5))
    style_text(t, "Contributions", size=22, bold=True, color=ACCENT2)
    add_bullets(
        slide.shapes.add_textbox(Inches(0.7), Inches(2.65), Inches(5.7), Inches(4.0)).text_frame,
        [
            "End-to-end 13-step pipeline aligned to a single architecture diagram.",
            "Real HuggingFace models at every step — never a mock.",
            "Claim-level NLI + 3-way gate (PASS / REGEN / ABSTAIN).",
            "Two user surfaces (FastAPI + Flask) with live metrics.",
            "Reproducible: 31 tests, HaluEval + TruthfulQA, comparison report.",
        ],
        size=15,
    )

    # Right: key numbers
    add_card(slide, Inches(6.8), Inches(1.95), Inches(6.0), Inches(4.8), fill_color=RGBColor(0x16, 0x2A, 0x40))
    t = slide.shapes.add_textbox(Inches(7.0), Inches(2.05), Inches(5.6), Inches(0.5))
    style_text(t, "Headline numbers", size=22, bold=True, color=ACCENT2)
    add_bullets(
        slide.shapes.add_textbox(Inches(7.0), Inches(2.65), Inches(5.6), Inches(4.0)).text_frame,
        [
            "VSR = 1.000 on synthetic QA — every claim verified.",
            "Retrieval recall @ 5 = 0.797 — hybrid RRF is doing its job.",
            "End-to-end accuracy 0.344 — right problem, room to grow.",
            "31 pytest tests passing — every component has a contract.",
        ],
        size=15,
    )

    # CTA
    add_card(slide, Inches(0.5), Inches(6.9), Inches(12.3), Inches(0.45), fill_color=ACCENT)
    t = slide.shapes.add_textbox(Inches(0.7), Inches(6.93), Inches(12.0), Inches(0.4))
    style_text(
        t,
        "Try it: python -m scripts.run_inference --query \"Who wrote 1984?\"    ·    "
        "UI: python -m scripts.run_frontend    ·    Repo: github.com/ShihabHassanNaim/Hallucination-Mitigation",
        size=12, bold=True, color=RGBColor(0x0E, 0x1B, 0x2C),
    )


def add_slide_qa(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)

    # Big "Q&A"
    tf = slide.shapes.add_textbox(Inches(0.5), Inches(2.6), Inches(12.3), Inches(1.6))
    style_text(tf, "Q & A", size=90, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)

    tf = slide.shapes.add_textbox(Inches(0.5), Inches(4.4), Inches(12.3), Inches(0.5))
    style_text(
        tf,
        "Thank you — happy to discuss the architecture, results, or next steps.",
        size=20, color=LIGHT, align=PP_ALIGN.CENTER,
    )
    tf = slide.shapes.add_textbox(Inches(0.5), Inches(5.0), Inches(12.3), Inches(0.5))
    style_text(
        tf,
        "github.com/ShihabHassanNaim/Hallucination-Mitigation",
        size=16, color=ACCENT2, align=PP_ALIGN.CENTER,
    )


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    builders = [
        add_slide_title,
        add_slide_agenda,
        add_slide_motivation,
        add_slide_related_work,
        add_slide_goals,
        add_slide_architecture_steps,
        add_slide_architecture_flow,
        add_slide_implementation,
        add_slide_models,
        add_slide_ui,
        add_slide_evaluation_metrics,
        add_slide_evaluation_results,
        add_slide_datasets,
        add_slide_comparison,
        add_slide_limitations,
        add_slide_conclusion,
        add_slide_qa,
    ]

    for builder in builders:
        builder(prs)

    # Page numbers on every slide except the title
    total = len(builders)
    for idx, slide in enumerate(prs.slides):
        if idx > 0:
            add_page_number(slide, idx + 1, total)

    prs.save(OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH}  ({total} slides)")


if __name__ == "__main__":
    main()
