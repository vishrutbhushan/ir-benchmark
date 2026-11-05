#!/usr/bin/env python3
"""
Final precision fixes for IR Benchmark Presentation Deck based on user feedback.

Fixes:
1. Slide 5: Single BM25F value (no arrow, no 2 values), uniform dark slate color (no green/navy).
2. Slide 8: Latency chart updated with NDCG value on each point, legend moved to center-left with breathing room, title updated per request, bottom line updated.
3. Slide 10: 4x4 Inter-annotator table with both row totals (Gemma Total) and column totals (Nemotron Total) + Grand Total (384).
4. Slide 13: 100% uniform dark text (zero colored text in body), NO star icons.
5. NO star icons anywhere in the presentation or tables.
6. Slide 14: Takeaways decompressed to fill the full vertical card height proportionally.
"""

from pathlib import Path
import pptx
from pptx import Presentation
from pptx.util import Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

PPTX_PATH = Path("IR_Benchmark_Mermaid_Diagrams_Fixed.pptx")
CHARTS_DIR = Path("presentation/charts")

NAVY = RGBColor(33, 29, 112)       # #211D70
SLATE = RGBColor(45, 55, 72)       # #2D3748
WHITE = RGBColor(255, 255, 255)
LIGHT_GREEN = RGBColor(230, 244, 234) # #E6F4EA
DARK_GREEN = RGBColor(19, 115, 51)    # #137333
LIGHT_RED = RGBColor(252, 232, 230)   # #FCE8E6
DARK_RED = RGBColor(197, 34, 31)      # #C5221F
LIGHT_GRAY = RGBColor(248, 249, 250)  # #F8F9FA
HEADER_GRAY = RGBColor(237, 242, 247) # #EDF2F7
LIGHT_AMBER = RGBColor(254, 247, 224) # #FEF7E0
DARK_AMBER = RGBColor(176, 96, 0)     # #B06000
LIGHT_PURPLE = RGBColor(243, 238, 255)


def remove_shape(shape):
    sp = shape._element
    sp.getparent().remove(sp)


def fix_slide_5(slide):
    """Slide 5: Single BM25F values, uniform dark slate (no green, no navy, no arrow)."""
    print("Fixing Slide 5 (Single BM25F values, uniform color)...")
    for shp in slide.shapes:
        if shp.has_text_frame and ("NDCG@10" in shp.text or "MRR@10" in shp.text):
            if "0.8600" in shp.text or "0.8748" in shp.text:
                tf = shp.text_frame
                tf.clear()
                p1 = tf.paragraphs[0]
                p1.text = "NDCG@10 = 0.8748"
                p1.font.name = "Arial"
                p1.font.size = Pt(10.0)
                p1.font.bold = True
                p1.font.color.rgb = SLATE

                p2 = tf.add_paragraph()
                p2.text = "MRR@10 = 0.9556"
                p2.font.name = "Arial"
                p2.font.size = Pt(10.0)
                p2.font.bold = True
                p2.font.color.rgb = SLATE


def fix_slide_8(slide):
    """Slide 8: Update Pareto chart and bottom line."""
    print("Fixing Slide 8 (Pareto Chart & Bottom Line)...")
    pareto_img = CHARTS_DIR / "latency_throughput_pareto.png"
    for shp in list(slide.shapes):
        if shp.shape_type == pptx.enum.shapes.MSO_SHAPE_TYPE.PICTURE and shp.name in ["Image 1", "Picture 8", "Picture 11"]:
            l, t, w, h = shp.left, shp.top, shp.width, shp.height
            remove_shape(shp)
            slide.shapes.add_picture(str(pareto_img), l, t, w, h)
            break

    for shp in slide.shapes:
        if shp.has_text_frame and ("Hybrid RRF" in shp.text or "Cluster-Routed" in shp.text):
            shp.top = Emu(5550000)
            shp.height = Emu(320000)
            tf = shp.text_frame
            tf.clear()
            p = tf.paragraphs[0]
            p.text = "Cluster-Routed delivers 23.3 ms query latency; Hybrid RRF achieves top NDCG@10 (0.9104) and MRR@10 (1.0000)."
            p.font.name = "Arial"
            p.font.size = Pt(9.5)
            p.font.bold = True
            p.font.color.rgb = NAVY
            p.alignment = PP_ALIGN.CENTER


def fix_slide_10(slide):
    """Slide 10: 4x4 Table with both Gemma Total and Nemotron Total + Grand Total."""
    print("Fixing Slide 10 (4x4 Inter-Annotator Table with Full Totals)...")
    # Remove old table and footers
    for shp in list(slide.shapes):
        if shp.has_table and shp.left < Emu(4000000):
            remove_shape(shp)
        elif shp.has_text_frame and ("2×2 Inter-Annotator" in shp.text or "Observed Agreement" in shp.text):
            remove_shape(shp)

    # Subtitle inside left card
    sub_box = slide.shapes.add_textbox(Emu(667512), Emu(3550000), Emu(3639312), Emu(250000))
    tf_sub = sub_box.text_frame
    tf_sub.margin_left = tf_sub.margin_right = tf_sub.margin_top = tf_sub.margin_bottom = 0
    p = tf_sub.paragraphs[0]
    p.text = "Inter-Annotator Contingency Matrix (N = 384 Pairs)"
    p.font.name = "Arial"
    p.font.size = Pt(10.0)
    p.font.bold = True
    p.font.color.rgb = NAVY

    # Native 4 rows x 4 cols table
    tbl_left = Emu(667512)
    tbl_top = Emu(3850000)
    tbl_w = Emu(3639312)
    tbl_h = Emu(1450000)

    tbl_shape = slide.shapes.add_table(4, 4, tbl_left, tbl_top, tbl_w, tbl_h)
    tbl = tbl_shape.table

    col_widths = [
        Emu(1350000), # Alignment
        Emu(760000),  # Nemotron = 1
        Emu(760000),  # Nemotron = 0
        Emu(769312),  # Gemma Total
    ]
    for c_idx, w in enumerate(col_widths):
        tbl.columns[c_idx].width = w

    headers = ["Judge Alignment", "Nemotron = 1", "Nemotron = 0", "Gemma Total"]
    for c_idx, h in enumerate(headers):
        cell = tbl.cell(0, c_idx)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT if c_idx == 0 else PP_ALIGN.CENTER
        p.font.name = "Arial"
        p.font.size = Pt(8.0)
        p.font.bold = True
        p.font.color.rgb = WHITE

    # Matrix data: Gemma (rows) x Nemotron (cols)
    matrix_data = [
        ["Gemma = 1 (Rel)", "119", "108", "227"],
        ["Gemma = 0 (Not Rel)", "0", "157", "157"],
        ["Nemotron Total", "119", "265", "384"],
    ]

    for r_idx, r_data in enumerate(matrix_data, 1):
        for c_idx, val in enumerate(r_data):
            cell = tbl.cell(r_idx, c_idx)
            cell.text = val
            cell.fill.solid()
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE

            is_total = (r_idx == 3) or (c_idx == 3)
            is_agree_rel = (r_idx == 1 and c_idx == 1) # 119
            is_disagree = (r_idx == 1 and c_idx == 2)  # 108

            if is_agree_rel:
                cell.fill.fore_color.rgb = LIGHT_GREEN
            elif is_disagree:
                cell.fill.fore_color.rgb = LIGHT_AMBER
            elif is_total:
                cell.fill.fore_color.rgb = HEADER_GRAY
            elif r_idx % 2 == 1:
                cell.fill.fore_color.rgb = LIGHT_GRAY
            else:
                cell.fill.fore_color.rgb = WHITE

            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c_idx == 0 else PP_ALIGN.CENTER
            p.font.name = "Arial"
            p.font.size = Pt(8.0)
            p.font.bold = is_total or (c_idx == 0) or is_agree_rel

            if is_agree_rel:
                p.font.color.rgb = DARK_GREEN
            elif is_disagree:
                p.font.color.rgb = DARK_AMBER
            elif is_total:
                p.font.color.rgb = NAVY
            else:
                p.font.color.rgb = SLATE

    # Native Footer
    foot_box = slide.shapes.add_textbox(Emu(667512), Emu(5400000), Emu(3639312), Emu(650000))
    tf_foot = foot_box.text_frame
    tf_foot.margin_left = tf_foot.margin_right = tf_foot.margin_top = tf_foot.margin_bottom = 0
    p1 = tf_foot.paragraphs[0]
    p1.text = "Observed Agreement: 71.9% (276 / 384)   ·   Cohen's κ = 0.4740"
    p1.font.name = "Arial"
    p1.font.size = Pt(8.5)
    p1.font.bold = True
    p1.font.color.rgb = NAVY

    p2 = tf_foot.add_paragraph()
    p2.text = "Consensus: Grade 2 (119) · Grade 1 (108) · Grade 0 (157)"
    p2.font.name = "Arial"
    p2.font.size = Pt(8.0)
    p2.font.color.rgb = SLATE


def fix_slide_11(slide):
    """Slide 11: Remove star icon, clean uniform formatting."""
    print("Fixing Slide 11 (No Star Icon, Clean Formatting)...")
    for shp in slide.shapes:
        if shp.has_text_frame and ("RRF Formulation" in shp.text or "RRF(d) =" in shp.text):
            tf = shp.text_frame
            tf.clear()
            lines = [
                ("RRF Formulation:", True, NAVY, Pt(10.0)),
                ("RRF(d) = 0.5 / (60 + rank_BM25F) + 0.5 / (60 + rank_routed)", False, SLATE, Pt(9.0)),
                ("", False, SLATE, Pt(4.0)),
                ("Measured Benchmark Results:", True, NAVY, Pt(10.0)),
                ("• P@1 = 1.0000  ·  MRR@10 = 1.0000", False, SLATE, Pt(9.0)),
                ("• Graded NDCG@5 = 0.7690", False, SLATE, Pt(9.0)),
                ("• Graded NDCG@10 = 0.9104", False, SLATE, Pt(9.0)),
                ("• Recall@10 = 0.5380", False, SLATE, Pt(9.0)),
                ("• Binary NDCG@10 = 0.9547", False, SLATE, Pt(9.0)),
            ]
            for idx, (txt, bold, col, sz) in enumerate(lines):
                p = tf.add_paragraph() if idx > 0 else tf.paragraphs[0]
                p.text = txt
                p.font.name = "Arial"
                p.font.size = sz
                p.font.bold = bold
                p.font.color.rgb = col
                p.alignment = PP_ALIGN.LEFT


def fix_slide_12(slide):
    """Slide 12: Update chart with zero title/legend overlap, remove star icon, remove lavender color."""
    print("Fixing Slide 12 (Update Chart, Remove Lavender, Remove Star Icon)...")
    chart_img = CHARTS_DIR / "overall_effectiveness_hybrid.png"
    for shp in list(slide.shapes):
        if shp.shape_type == pptx.enum.shapes.MSO_SHAPE_TYPE.PICTURE and shp.name != "Image 0":
            l, t, w, h = shp.left, shp.top, shp.width, shp.height
            remove_shape(shp)
            slide.shapes.add_picture(str(chart_img), l, t, w, h)
            break

    best_r5_cols = {2, 4, 5, 6, 7} # NDCG@10, P@10, Recall@10, MRR@10, Bin-NDCG@10
    for shp in slide.shapes:
        if shp.has_table:
            tbl = shp.table
            for c_idx in range(len(tbl.columns)):
                cell = tbl.cell(5, c_idx)
                if "★" in cell.text:
                    cell.text = cell.text.replace("★", "").strip()
                p = cell.text_frame.paragraphs[0]
                p.alignment = PP_ALIGN.LEFT if c_idx == 0 else PP_ALIGN.CENTER
                p.font.name = "Arial"
                p.font.size = Pt(8.0)
                p.font.bold = True

                # Remove lavender: use light green for best cells, light gray for normal cells
                if c_idx in best_r5_cols:
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = LIGHT_GREEN
                    p.font.color.rgb = DARK_GREEN
                else:
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = LIGHT_GRAY
                    p.font.color.rgb = NAVY if c_idx == 0 else SLATE


def fix_slide_13(slide):
    """Slide 13: 100% UNIFORM dark text (NO colors in body), NO star icons."""
    print("Fixing Slide 13 (Zero Colored Text, Zero Stars)...")
    for shp in slide.shapes:
        if shp.has_text_frame:
            # Card 1 Body
            if "Docker Buildx to speed up" in shp.text:
                tf = shp.text_frame
                tf.clear()
                items = [
                    ("Query", True, False, Pt(10.0)),
                    ("“How do I use --mount=type=cache in Docker Buildx to speed up pip installs?”", False, True, Pt(9.5)),
                    ("", False, False, Pt(4.0)),
                    ("BM25 / BM25F", True, False, Pt(10.0)),
                    ("Matches exact flag tokens. Top result is the Buildx caching guide. NDCG@10 = 0.7574.", False, False, Pt(9.0)),
                    ("", False, False, Pt(4.0)),
                    ("Dense neural", True, False, Pt(10.0)),
                    ("Punctuation and dashes are weakened; results drift toward generic pip caching. NDCG@10 = 0.7066.", False, False, Pt(9.0)),
                    ("", False, False, Pt(4.0)),
                    ("Hybrid RRF", True, False, Pt(10.0)),
                    ("Keeps the BM25F first result and adds conceptually relevant caching docs. NDCG@10 = 0.7812.", False, False, Pt(9.0)),
                ]
                for idx, (txt, bold, italic, sz) in enumerate(items):
                    p = tf.add_paragraph() if idx > 0 else tf.paragraphs[0]
                    p.text = txt
                    p.font.name = "Arial"
                    p.font.size = sz
                    p.font.bold = bold
                    p.font.italic = italic
                    p.font.color.rgb = SLATE
                    p.alignment = PP_ALIGN.LEFT

            # Card 2 Body
            elif "setup springboot within" in shp.text:
                tf = shp.text_frame
                tf.clear()
                items = [
                    ("Query", True, False, Pt(10.0)),
                    ("“Can you help me setup springboot within 10 minutes without any issues?”", False, True, Pt(9.5)),
                    ("", False, False, Pt(4.0)),
                    ("BM25 / BM25F", True, False, Pt(10.0)),
                    ("Severe term mismatch on colloquial phrasing leads to noisy issue pages. NDCG@10 = 0.1389; P@5 = 0.2000.", False, False, Pt(9.0)),
                    ("", False, False, Pt(4.0)),
                    ("Dense neural", True, False, Pt(10.0)),
                    ("Finds Spring Initializr, Quick Start Guide and application-building documentation. NDCG@10 = 1.0000; P@5 = 1.0000.", False, False, Pt(9.0)),
                    ("", False, False, Pt(4.0)),
                    ("Hybrid RRF", True, False, Pt(10.0)),
                    ("Combines neural intent understanding with exact official spring boot references. NDCG@10 = 1.0000; P@5 = 1.0000.", False, False, Pt(9.0)),
                ]
                for idx, (txt, bold, italic, sz) in enumerate(items):
                    p = tf.add_paragraph() if idx > 0 else tf.paragraphs[0]
                    p.text = txt
                    p.font.name = "Arial"
                    p.font.size = sz
                    p.font.bold = bold
                    p.font.italic = italic
                    p.font.color.rgb = SLATE
                    p.alignment = PP_ALIGN.LEFT


def fix_slide_14(slide):
    """Slide 14: Decompress Takeaways so it fills the vertical height of the card proportionally."""
    print("Fixing Slide 14 (Decompressing Takeaways to fill card)...")
    for shp in slide.shapes:
        if shp.has_text_frame and "Corpus filtering" in shp.text:
            tf = shp.text_frame
            tf.clear()
            items = [
                ("Corpus filtering", "The crawl was reduced from 79,268 files (18.6 GB) to 45,881 documents (11.0 GB)."),
                ("Index footprint", "The classical JSON index is 1.11 GB; compressed postings are estimated below 95 MB."),
                ("Neural search", "Dense retrieval improves relevance on conversational queries, but full scan requires 53.7 ms."),
                ("Cluster routing", "Pruning 55.5% of vector scan reduces latency to 23.3 ms (2.3× speedup) while achieving 0.9088 NDCG@10."),
                ("Hybrid ranking", "RRF achieves top benchmark NDCG@10 (0.9104) and perfect MRR@10 (1.0000) across all systems."),
                ("Caching", "Two-tier query cache achieves warm-hit latency below 0.05 ms."),
            ]
            for idx, (title, body) in enumerate(items):
                p_title = tf.add_paragraph() if idx > 0 else tf.paragraphs[0]
                p_title.text = title
                p_title.font.name = "Arial"
                p_title.font.size = Pt(10.5)
                p_title.font.bold = True
                p_title.font.color.rgb = NAVY
                p_title.space_before = Pt(8.0) if idx > 0 else Pt(0.0)
                p_title.space_after = Pt(2.0)

                p_body = tf.add_paragraph()
                p_body.text = body
                p_body.font.name = "Arial"
                p_body.font.size = Pt(9.5)
                p_body.font.bold = False
                p_body.font.color.rgb = SLATE
                p_body.space_after = Pt(8.0)


def main():
    prs = Presentation(PPTX_PATH)
    print(f"Loaded {len(prs.slides)} slides from {PPTX_PATH}")

    fix_slide_5(prs.slides[4])
    fix_slide_8(prs.slides[7])
    fix_slide_10(prs.slides[9])
    fix_slide_11(prs.slides[10])
    fix_slide_12(prs.slides[11])
    fix_slide_13(prs.slides[12])
    fix_slide_14(prs.slides[13])

    prs.save(PPTX_PATH)
    print(f"Cleanly updated and verified {PPTX_PATH} ({len(prs.slides)} slides)!")


if __name__ == "__main__":
    main()

