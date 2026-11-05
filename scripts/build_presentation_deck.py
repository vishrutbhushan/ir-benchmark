#!/usr/bin/env python3
"""
Build the official BITS Pilani IR Benchmark Presentation Deck.

Output: presentation/IR_Benchmark_Presentation.pptx
Format: 16:9 Widescreen (13.333" x 7.5")
Theme: BITS Pilani Academic Template
"""

from __future__ import annotations

import os
from pathlib import Path
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# Paths
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PPTX = ROOT / "presentation" / "IR_Benchmark_Presentation.pptx"
OUTPUT_PPTX.parent.mkdir(parents=True, exist_ok=True)
CHARTS_DIR = ROOT / "presentation" / "charts"
TEMPLATE_DIR = ROOT / "extracted_template"
CREST_PATH = TEMPLATE_DIR / "image_0.png"
BANNER_PATH = TEMPLATE_DIR / "image_1.png"

# Colors
C_NAVY = RGBColor(33, 29, 112)       # #211D70
C_GOLD = RGBColor(234, 152, 48)      # #EA9830
C_CRIMSON = RGBColor(234, 63, 38)    # #EA3F26
C_SKYBLUE = RGBColor(114, 195, 232)  # #72C3E8
C_SLATE = RGBColor(45, 55, 72)       # #2D3748
C_LIGHT_BG = RGBColor(248, 249, 250)# #F8F9FA
C_CARD_BG = RGBColor(255, 255, 255) # #FFFFFF
C_BORDER = RGBColor(226, 232, 240)   # #E2E8F0
C_WHITE = RGBColor(255, 255, 255)
C_MUTED = RGBColor(113, 128, 150)


def create_deck():
    prs = pptx.Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def add_header(slide, title_text, category_text="CS F469: Information Retrieval | Technical IR Benchmark"):
        # Top banner band
        header_box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.15))
        header_box.fill.solid()
        header_box.fill.fore_color.rgb = C_NAVY
        header_box.line.color.rgb = C_NAVY

        # Gold accent stripe
        stripe = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(1.15), Inches(13.333), Inches(0.08))
        stripe.fill.solid()
        stripe.fill.fore_color.rgb = C_GOLD
        stripe.line.color.rgb = C_GOLD

        # Header Title
        tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.15), Inches(10.5), Inches(0.95))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_top = Inches(0)
        tf.margin_bottom = Inches(0)
        tf.margin_left = Inches(0)

        p_cat = tf.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.name = "Calibri"
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = C_SKYBLUE

        p_title = tf.add_paragraph()
        p_title.text = title_text
        p_title.font.name = "Calibri"
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = C_WHITE

        # Crest logo on right
        if CREST_PATH.exists():
            slide.shapes.add_picture(str(CREST_PATH), Inches(12.2), Inches(0.12), height=Inches(0.95))

        # Bottom footer
        foot_box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(7.15), Inches(13.333), Inches(0.35))
        foot_box.fill.solid()
        foot_box.fill.fore_color.rgb = C_LIGHT_BG
        foot_box.line.color.rgb = C_BORDER

        ftb = slide.shapes.add_textbox(Inches(0.8), Inches(7.18), Inches(11.7), Inches(0.3))
        ftf = ftb.text_frame
        ftf.word_wrap = True
        p_ft = ftf.paragraphs[0]
        p_ft.text = "BITS Pilani — Group 7 Final Project Presentation | Classical BM25 vs. Dense Neural vs. Hybrid RRF"
        p_ft.font.name = "Calibri"
        p_ft.font.size = Pt(9.5)
        p_ft.font.color.rgb = C_MUTED

    # =========================================================================
    # SLIDE 1: Title Slide (Full BITS Branding)
    # =========================================================================
    slide1 = prs.slides.add_slide(blank_layout)

    bg1 = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = C_NAVY
    bg1.line.color.rgb = C_NAVY

    band = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(4.7), Inches(13.333), Inches(0.15))
    band.fill.solid()
    band.fill.fore_color.rgb = C_GOLD
    band.line.color.rgb = C_GOLD

    if CREST_PATH.exists():
        slide1.shapes.add_picture(str(CREST_PATH), Inches(0.9), Inches(0.8), height=Inches(1.5))

    if BANNER_PATH.exists():
        slide1.shapes.add_picture(str(BANNER_PATH), Inches(9.5), Inches(0.5), width=Inches(3.2))

    tbox = slide1.shapes.add_textbox(Inches(0.9), Inches(2.3), Inches(11.5), Inches(2.2))
    ttf = tbox.text_frame
    ttf.word_wrap = True

    p0 = ttf.paragraphs[0]
    p0.text = "CS F469: Information Retrieval — Final Project"
    p0.font.name = "Calibri"
    p0.font.size = Pt(14)
    p0.font.bold = True
    p0.font.color.rgb = C_SKYBLUE

    p1 = ttf.add_paragraph()
    p1.text = "BM25 vs. Neural Retrieval on Technical Documents"
    p1.font.name = "Calibri"
    p1.font.size = Pt(32)
    p1.font.bold = True
    p1.font.color.rgb = C_WHITE

    p2 = ttf.add_paragraph()
    p2.text = "Architectural Evaluation, Cluster Pruning, and Hybrid Reciprocal Rank Fusion (RRF)"
    p2.font.name = "Calibri"
    p2.font.size = Pt(18)
    p2.font.color.rgb = C_GOLD

    abox = slide1.shapes.add_textbox(Inches(0.9), Inches(5.1), Inches(11.5), Inches(1.8))
    atf = abox.text_frame
    atf.word_wrap = True

    pa1 = atf.paragraphs[0]
    pa1.text = "Project Group 7:"
    pa1.font.name = "Calibri"
    pa1.font.size = Pt(13)
    pa1.font.bold = True
    pa1.font.color.rgb = C_WHITE

    pa2 = atf.add_paragraph()
    pa2.text = "Vishrut Bhushan (h20250143)   •   Bhagavath M S (h20250167)   •   Ajay B (h20250142)   •   Jalluri Mahesh Kumar (h20250061)"
    pa2.font.name = "Calibri"
    pa2.font.size = Pt(13.5)
    pa2.font.color.rgb = C_SKYBLUE

    pa3 = atf.add_paragraph()
    pa3.text = "Department of Computer Science & Information Systems, BITS Pilani"
    pa3.font.name = "Calibri"
    pa3.font.size = Pt(12)
    pa3.font.italic = True
    pa3.font.color.rgb = C_WHITE

    # =========================================================================
    # SLIDE 2: Problem Statement & Research Objectives
    # =========================================================================
    slide2 = prs.slides.add_slide(blank_layout)
    add_header(slide2, "1. Motivation & Technical Information Retrieval Challenges")

    c1 = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.3))
    c1.fill.solid()
    c1.fill.fore_color.rgb = C_CARD_BG
    c1.line.color.rgb = C_BORDER
    tf1 = c1.text_frame
    tf1.word_wrap = True

    p = tf1.paragraphs[0]
    p.text = "Why Technical Documentation Differs from Standard IR"
    p.font.name = "Calibri"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    points1 = [
        ("Exact Syntax & Identifier Tokens", "Technical docs contain CLI flags (`--mount=type=cache`), API endpoints, error signatures, and semantic version strings (`v1.35.7`) where aggressive stemming fails."),
        ("Multi-Zone Hierarchical Structure", "Information is non-uniformly distributed across `<title>`, `<h1>..<h3>`, `<code>` snippets, and prose `<body>` tags."),
        ("Vocabulary Mismatch vs. Rigid Matching", "Conversational symptoms ('cannot connect to database') require semantic abstraction, while syntax queries require zero-tolerance exact token matching."),
        ("Scalability Constraints", "Bi-encoder dense models suffer from $O(N)$ full-corpus dot products, making unpruned neural search prohibitively expensive for enterprise search."),
    ]
    for title, desc in points1:
        p_t = tf1.add_paragraph()
        p_t.text = f"•  {title}:"
        p_t.font.name = "Calibri"
        p_t.font.size = Pt(11.5)
        p_t.font.bold = True
        p_t.font.color.rgb = C_SLATE
        p_d = tf1.add_paragraph()
        p_d.text = f"    {desc}"
        p_d.font.name = "Calibri"
        p_d.font.size = Pt(10.5)
        p_d.font.color.rgb = C_MUTED

    c2 = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3))
    c2.fill.solid()
    c2.fill.fore_color.rgb = C_CARD_BG
    c2.line.color.rgb = C_BORDER
    tf2 = c2.text_frame
    tf2.word_wrap = True

    p = tf2.paragraphs[0]
    p.text = "Key Research Questions & Project Objectives"
    p.font.name = "Calibri"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    points2 = [
        ("RQ1 — Classical Lexical vs. Dense Neural", "Quantify exact trade-offs between inverted index BM25/BM25F and bi-encoder MiniLM embeddings on a production-scale corpus."),
        ("RQ2 — Sub-Linear Search via Cluster Pruning", "Implement unsupervised K-Means++ document clustering (Manning Ch. 7.1.6) to prune candidate passages by 75% and evaluate latency."),
        ("RQ3 — Hybrid Fusion via RRF", "Overcome score distribution mismatch using Reciprocal Rank Fusion (SIGIR 2009) to deliver superior ranking across both syntax and conceptual queries."),
        ("RQ4 — Query Caching Architecture", "Deploy a generic two-tier LRU query cache (Manning Ch. 19) to achieve sub-millisecond repeat latency."),
    ]
    for title, desc in points2:
        p_t = tf2.add_paragraph()
        p_t.text = f"•  {title}:"
        p_t.font.name = "Calibri"
        p_t.font.size = Pt(11.5)
        p_t.font.bold = True
        p_t.font.color.rgb = C_CRIMSON
        p_d = tf2.add_paragraph()
        p_d.text = f"    {desc}"
        p_d.font.name = "Calibri"
        p_d.font.size = Pt(10.5)
        p_d.font.color.rgb = C_MUTED

    # =========================================================================
    # SLIDE 3: Corpus Acquisition & 5-Stage Cleaning Pipeline
    # =========================================================================
    slide3 = prs.slides.add_slide(blank_layout)
    add_header(slide3, "2. Corpus Acquisition & 5-Stage Preprocessing Pipeline")

    stages = [
        ("Stage 1", "Raw Web Crawl", "18.6 GB (~20 GB)\n79,275 Raw Files\nCommon Crawl WARC"),
        ("Stage 2", "Boilerplate Strip", "Trafilatura & JusText\nStrip nav, footer, ads\nPreserve code & tags"),
        ("Stage 3", "Zone Extraction", "Structured Split:\nTitle, Headings,\nBody, Code Blocks"),
        ("Stage 4", "Quality Filter", "Deduplication, min words,\nEnglish language detect,\nMalformed HTML prune"),
        ("Stage 5", "Final Clean DB", "45,881 Pristine Docs\n1.12 GB SQLite DB\nFull schema indexed"),
    ]

    for idx, (st, name, desc) in enumerate(stages):
        bx = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8 + idx * 2.4), Inches(1.5), Inches(2.2), Inches(2.0))
        bx.fill.solid()
        bx.fill.fore_color.rgb = C_CARD_BG
        bx.line.color.rgb = C_NAVY if idx == 4 else C_BORDER
        btf = bx.text_frame
        btf.word_wrap = True

        p0 = btf.paragraphs[0]
        p0.text = st
        p0.font.name = "Calibri"
        p0.font.size = Pt(10)
        p0.font.bold = True
        p0.font.color.rgb = C_GOLD

        p1 = btf.add_paragraph()
        p1.text = name
        p1.font.name = "Calibri"
        p1.font.size = Pt(12)
        p1.font.bold = True
        p1.font.color.rgb = C_NAVY

        p2 = btf.add_paragraph()
        p2.text = desc
        p2.font.name = "Calibri"
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = C_SLATE

        if idx < 4:
            arr = slide3.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(3.05 + idx * 2.4), Inches(2.35), Inches(0.12), Inches(0.25))
            arr.fill.solid()
            arr.fill.fore_color.rgb = C_GOLD
            arr.line.fill.background()

    cd1 = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(3.7), Inches(5.6), Inches(3.2))
    cd1.fill.solid()
    cd1.fill.fore_color.rgb = C_CARD_BG
    cd1.line.color.rgb = C_BORDER
    tf_cd1 = cd1.text_frame
    tf_cd1.word_wrap = True

    p = tf_cd1.paragraphs[0]
    p.text = "Corpus Diversity & Scale Highlights"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    facts1 = [
        "Initial Raw Crawl: 18.6 GB uncompressed across 79,275 HTML files.",
        "Local Extracted Disk Footprint: 11.0 GB in data/00_corpus/documents/.",
        "Filtered Production Corpus: 45,881 unique documents in 1.12 GB SQLite DB.",
        "27 Engineering Ecosystems: PyTorch, Kubernetes, PostgreSQL, Linux Kernel, Envoy, AWS, GitHub, etc.",
        "Vocabulary Size: 280,000 unique terms after domain-aware tokenization.",
    ]
    for f in facts1:
        pf = tf_cd1.add_paragraph()
        pf.text = f"•  {f}"
        pf.font.name = "Calibri"
        pf.font.size = Pt(10.5)
        pf.font.color.rgb = C_SLATE

    cd2 = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(3.7), Inches(5.7), Inches(3.2))
    cd2.fill.solid()
    cd2.fill.fore_color.rgb = C_CARD_BG
    cd2.line.color.rgb = C_BORDER
    tf_cd2 = cd2.text_frame
    tf_cd2.word_wrap = True

    p = tf_cd2.paragraphs[0]
    p.text = "Why Zone Separation Matters (Course: Manning Ch. 6.1)"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    facts2 = [
        "Title Zone: High-density semantic signal (e.g., 'Docker Buildx cache mounts').",
        "Headings Zone: Hierarchical section context (H1..H4) capturing document structure.",
        "Code Zone: Syntax-pure tokens (`sysctl.conf`, `kubectl apply -f`, `optimizer.step()`).",
        "Body Zone: Descriptive explanations, error messages, and walkthrough prose.",
        "Enables multi-field probabilistic scoring (BM25F) rather than treating docs as flat text.",
    ]
    for f in facts2:
        pf = tf_cd2.add_paragraph()
        pf.text = f"•  {f}"
        pf.font.name = "Calibri"
        pf.font.size = Pt(10.5)
        pf.font.color.rgb = C_SLATE

    # =========================================================================
    # SLIDE 4: Unsupervised Neural Clustering Discovers Ground-Truth Categories
    # =========================================================================
    slide4 = prs.slides.add_slide(blank_layout)
    add_header(slide4, "3. Domain Discovery: Unsupervised K-Means++ vs. Ground Truth")

    c_left = slide4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(5.2), Inches(5.3))
    c_left.fill.solid()
    c_left.fill.fore_color.rgb = C_CARD_BG
    c_left.line.color.rgb = C_BORDER
    tfl = c_left.text_frame
    tfl.word_wrap = True

    p = tfl.paragraphs[0]
    p.text = "Unsupervised Semantic Clustering (Manning Ch. 16/17)"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    c_points = [
        ("No Category Labels Provided", "K-Means++ ($k=8$) was run purely over normalized Title + Headings semantic vectors ($d=1024$)."),
        ("Natural Domain Alignment", "The clustering algorithm rediscovered the exact functional domains of the engineering ecosystem:"),
        ("• Cluster 4 (93.4% Purity)", "DevOps & Version Control (GitHub, GitLab, Git-SCM)"),
        ("• Cluster 3 (81.2% Purity)", "AI / Machine Learning (PyTorch, NVIDIA CUDA, Ray, Hugging Face)"),
        ("• Cluster 7 (74.8% Purity)", "Relational & Distributed DBs (PostgreSQL, MySQL, ClickHouse)"),
        ("• Cluster 6 (68.5% Purity)", "Observability & Containers (Grafana, Kubernetes, Linkerd)"),
        ("• Cluster 0 (62.1% Purity)", "OS, Networking & Security (Linux Kernel, Envoy, OpenSSL)"),
        ("Takeaway", "Dense embeddings capture domain ontology organically, providing the foundation for cluster pruning."),
    ]
    for t, d in c_points:
        pt = tfl.add_paragraph()
        pt.text = f"{t}: {d}" if d else t
        pt.font.name = "Calibri"
        pt.font.size = Pt(10.5)
        pt.font.color.rgb = C_SLATE

    chart4 = CHARTS_DIR / "cluster_category_alignment.png"
    if chart4.exists():
        slide4.shapes.add_picture(str(chart4), Inches(6.3), Inches(1.5), width=Inches(6.2))

    # =========================================================================
    # SLIDE 5: Classical Architecture: Flat BM25 vs. Multi-Field BM25F
    # =========================================================================
    slide5 = prs.slides.add_slide(blank_layout)
    add_header(slide5, "4. Classical Lexical Retrieval: Inverted Index & BM25F")

    ci1 = slide5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.3))
    ci1.fill.solid()
    ci1.fill.fore_color.rgb = C_CARD_BG
    ci1.line.color.rgb = C_BORDER
    tfi1 = ci1.text_frame
    tfi1.word_wrap = True

    p = tfi1.paragraphs[0]
    p.text = "Inverted Index Architecture (Manning Ch. 4 & 5)"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    pts_inv = [
        ("Single-Pass In-Memory Index (SPIMI)", "Built across 45,881 docs, generating an inverted index over 280,000 terms."),
        ("Serialization Overhead Analysis", "The 1.11 GB file size is due to uncompressed plaintext JSON serialization (string doc IDs, repeated dictionary keys)."),
        ("Production Compression Potential", "Standard IR index compression (VByte / Elias-Fano / Lucene) compresses this exact postings list to < 95 MB (a 91% reduction)."),
        ("Memory Footprint & Speed", "Resident memory: ~850 MB RAM; query latency: 5.3 ms for flat BM25, 8.2 ms for BM25F."),
    ]
    for t, d in pts_inv:
        pt = tfi1.add_paragraph()
        pt.text = f"•  {t}:"
        pt.font.name = "Calibri"
        pt.font.size = Pt(11)
        pt.font.bold = True
        pt.font.color.rgb = C_SLATE
        pd = tfi1.add_paragraph()
        pd.text = f"    {d}"
        pd.font.name = "Calibri"
        pd.font.size = Pt(10)
        pd.font.color.rgb = C_MUTED

    ci2 = slide5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3))
    ci2.fill.solid()
    ci2.fill.fore_color.rgb = C_CARD_BG
    ci2.line.color.rgb = C_BORDER
    tfi2 = ci2.text_frame
    tfi2.word_wrap = True

    p = tfi2.paragraphs[0]
    p.text = "Probability Ranking Principle & BM25F (Manning Ch. 11)"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    pts_bm25f = [
        ("The Flaw of Linear Zone Combination", "Summing separate BM25 scores per zone distorts TF saturation because terms in multiple zones saturate independently."),
        ("BM25F Solution — Effective TF First", "BM25F computes effective term frequency before applying non-linear saturation:"),
        ("$$\\tilde{tf}_{t,d} = \\sum_{z} w_z \\frac{tf_{t,z,d}}{(1 - b_z) + b_z \\frac{L_{z,d}}{AL_z}}$$", "Field weights: Title (w=3.0), Headings (w=2.0), Code (w=2.0), Body (w=1.0)."),
        ("Empirical Performance Boost", "BM25F raises NDCG@10 from 0.6727 to 0.7089 (+5.4%) and MRR from 0.8667 to 0.8889."),
        ("Preserves Rare Syntax", "Excels at CLI flags, class names, and code syntax where neural models suffer semantic drift."),
    ]
    for t, d in pts_bm25f:
        pt = tfi2.add_paragraph()
        pt.text = f"•  {t}:"
        pt.font.name = "Calibri"
        pt.font.size = Pt(11)
        pt.font.bold = True
        pt.font.color.rgb = C_CRIMSON
        pd = tfi2.add_paragraph()
        pd.text = f"    {d}"
        pd.font.name = "Calibri"
        pd.font.size = Pt(10)
        pd.font.color.rgb = C_MUTED

    # =========================================================================
    # SLIDE 6: Neural Passage Extraction & Dense Vector Architecture
    # =========================================================================
    slide6 = prs.slides.add_slide(blank_layout)
    add_header(slide6, "5. Dense Neural Retrieval: Bi-Encoder Embeddings")

    c_neu1 = slide6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.3))
    c_neu1.fill.solid()
    c_neu1.fill.fore_color.rgb = C_CARD_BG
    c_neu1.line.color.rgb = C_BORDER
    tf_n1 = c_neu1.text_frame
    tf_n1.word_wrap = True

    p = tf_n1.paragraphs[0]
    p.text = "Dense Bi-Encoder Architecture & Representation"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    pts_neu1 = [
        ("Semantic Passage Windowing", "HTML documents chunked via sliding window (256 tokens, 64 overlap) across Title, Headings, Body, Code."),
        ("Transformer Backbone", "all-MiniLM-L6-v2 produces 384-dimensional dense vectors normalized to unit hypersphere."),
        ("True Neural Pipeline Footprint (2.89 GB)", "• Vector Arrays (.npy across 4 zones): 615 MB\n• Transformer Model Weights: 1.20 GB\n• Passage Windows SQLite DB: 919 MB\n• Cluster Representation: 157 MB"),
        ("Hardware Requirements", "Requires ~2.1 GB resident VRAM during inference (2.5× higher memory footprint than Classical)."),
    ]
    for t, d in pts_neu1:
        pt = tf_n1.add_paragraph()
        pt.text = f"•  {t}:"
        pt.font.name = "Calibri"
        pt.font.size = Pt(11)
        pt.font.bold = True
        pt.font.color.rgb = C_SLATE
        pd = tf_n1.add_paragraph()
        pd.text = f"    {d}"
        pd.font.name = "Calibri"
        pd.font.size = Pt(10)
        pd.font.color.rgb = C_MUTED

    c_neu2 = slide6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3))
    c_neu2.fill.solid()
    c_neu2.fill.fore_color.rgb = C_CARD_BG
    c_neu2.line.color.rgb = C_BORDER
    tf_n2 = c_neu2.text_frame
    tf_n2.word_wrap = True

    p = tf_n2.paragraphs[0]
    p.text = "Strengths & The Dense Scalability Bottleneck"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    pts_neu2 = [
        ("High Semantic Effectiveness", "Achieves NDCG@10 = 0.8036 and P@5 = 0.8133 on full corpus scan, dramatically outperforming BM25 on conversational queries."),
        ("Resolves Vocabulary Mismatch", "Maps symptom phrases ('high latency under load') to technical causes ('connection pooling', 'TCP keepalive') without keyword overlap."),
        ("The O(N) Compute Wall", "Exhaustive dot-product scan across 45,881 vectors takes 248.0 ms per query (only 4.0 QPS throughput)."),
        ("The Engineering Need", "Full dense scan is too slow for production search engines, necessitating cluster pruning and routing."),
    ]
    for t, d in pts_neu2:
        pt = tf_n2.add_paragraph()
        pt.text = f"•  {t}:"
        pt.font.name = "Calibri"
        pt.font.size = Pt(11)
        pt.font.bold = True
        pt.font.color.rgb = C_GOLD
        pd = tf_n2.add_paragraph()
        pd.text = f"    {d}"
        pd.font.name = "Calibri"
        pd.font.size = Pt(10)
        pd.font.color.rgb = C_MUTED

    # =========================================================================
    # SLIDE 7: Cluster Pruning: Scalable Sub-Linear Neural Routing
    # =========================================================================
    slide7 = prs.slides.add_slide(blank_layout)
    add_header(slide7, "6. Cluster Pruning: Sub-Linear Search & 12.5x Latency Speedup")

    c_cp1 = slide7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.3))
    c_cp1.fill.solid()
    c_cp1.fill.fore_color.rgb = C_CARD_BG
    c_cp1.line.color.rgb = C_BORDER
    tf_cp1 = c_cp1.text_frame
    tf_cp1.word_wrap = True

    p = tf_cp1.paragraphs[0]
    p.text = "Cluster Pruning Theory (Manning Ch. 7.1.6)"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    pts_cp1 = [
        ("Problem Statement", "Computing cosine similarity between query vector q and all N=45,881 documents takes 248 ms, failing SLAs."),
        ("Preprocessing Phase", "Partition corpus into k=8 clusters using K-Means++. Store centroid vectors (only 70 KB)."),
        ("Query Phase (Centroid Routing)", "1. Compute similarity between q and 8 centroids.\n2. Select top-b closest clusters (b=2).\n3. Restrict passage scan strictly to documents in selected clusters."),
        ("Search Space Pruning", "Prunes 75% of documents (scans ~11,470 passages instead of 45,881)."),
    ]
    for t, d in pts_cp1:
        pt = tf_cp1.add_paragraph()
        pt.text = f"•  {t}:"
        pt.font.name = "Calibri"
        pt.font.size = Pt(11)
        pt.font.bold = True
        pt.font.color.rgb = C_SLATE
        pd = tf_cp1.add_paragraph()
        pd.text = f"    {d}"
        pd.font.name = "Calibri"
        pd.font.size = Pt(10)
        pd.font.color.rgb = C_MUTED

    c_cp2 = slide7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3))
    c_cp2.fill.solid()
    c_cp2.fill.fore_color.rgb = C_CARD_BG
    c_cp2.line.color.rgb = C_BORDER
    tf_cp2 = c_cp2.text_frame
    tf_cp2.word_wrap = True

    p = tf_cp2.paragraphs[0]
    p.text = "Measured Routing Latency & Throughput Gains"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    pts_cp2 = [
        ("Full Dense Scan Latency", "248.0 ms per query (Throughput: 4.0 queries/sec)."),
        ("Cluster Routed Latency", "19.8 ms per query (Throughput: 50.5 queries/sec)."),
        ("Speedup Factor", "12.5× wall-clock acceleration achieved directly through cluster pruning!"),
        ("Centroid Routing Overhead", "< 0.4 ms to evaluate query against all 8 cluster centroids."),
        ("Top-Rank Precision Retention", "Maintains P@1 = 0.8000 and MRR = 0.9000 while cutting search compute by 4x."),
        ("Production Significance", "Enables real-time interactive search on modest hardware without GPU clusters."),
    ]
    for t, d in pts_cp2:
        pt = tf_cp2.add_paragraph()
        pt.text = f"•  {t}:"
        pt.font.name = "Calibri"
        pt.font.size = Pt(11)
        pt.font.bold = True
        pt.font.color.rgb = C_CRIMSON
        pd = tf_cp2.add_paragraph()
        pd.text = f"    {d}"
        pd.font.name = "Calibri"
        pd.font.size = Pt(10)
        pd.font.color.rgb = C_MUTED

    # =========================================================================
    # SLIDE 8: Two-Tier Query Caching Architecture
    # =========================================================================
    slide8 = prs.slides.add_slide(blank_layout)
    add_header(slide8, "7. Query Result Caching: Sub-Millisecond Repeat Latency")

    c_cac = slide8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(5.4), Inches(5.3))
    c_cac.fill.solid()
    c_cac.fill.fore_color.rgb = C_CARD_BG
    c_cac.line.color.rgb = C_BORDER
    tf_cac = c_cac.text_frame
    tf_cac.word_wrap = True

    p = tf_cac.paragraphs[0]
    p.text = "Search Engine Caching Architecture (Manning Ch. 19)"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    pts_cac = [
        ("Two-Tier Caching Design", "1. Tier-1 LRU Query Result Cache: Stores normalized query string -> top-k document IDs and scores.\n2. Tier-2 Query Embedding Cache: Caches dense semantic vectors to avoid transformer forward pass."),
        ("Thread-Safe Implementation", "LRUQueryCache in query_processor.py using collections.OrderedDict protected by a re-entrant mutex lock."),
        ("Cold Miss Latencies", "BM25 (5.3 ms), BM25F (8.2 ms), Routed (19.8 ms), Dense (248.0 ms)."),
        ("Cache Hit Response Time", "< 0.05 ms across all engines (Throughput: >25,000 QPS)."),
        ("Speedup Range", "132× to 6,200× latency reduction on repeated and warm queries."),
    ]
    for t, d in pts_cac:
        pt = tf_cac.add_paragraph()
        pt.text = f"•  {t}:"
        pt.font.name = "Calibri"
        pt.font.size = Pt(11)
        pt.font.bold = True
        pt.font.color.rgb = C_SLATE
        pd = tf_cac.add_paragraph()
        pd.text = f"    {d}"
        pd.font.name = "Calibri"
        pd.font.size = Pt(10)
        pd.font.color.rgb = C_MUTED

    chart8 = CHARTS_DIR / "cache_latency_speedup.png"
    if chart8.exists():
        slide8.shapes.add_picture(str(chart8), Inches(6.5), Inches(1.5), width=Inches(6.0))

    # =========================================================================
    # SLIDE 9: Evaluation Methodology: Dual-LLM Judges & Graded Consensus
    # =========================================================================
    slide9 = prs.slides.add_slide(blank_layout)
    add_header(slide9, "8. Evaluation Methodology: Cranfield Paradigm & Dual LLMs")

    cm1 = slide9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.3))
    cm1.fill.solid()
    cm1.fill.fore_color.rgb = C_CARD_BG
    cm1.line.color.rgb = C_BORDER
    tf_m1 = cm1.text_frame
    tf_m1.word_wrap = True

    p = tf_m1.paragraphs[0]
    p.text = "Cranfield Paradigm & TREC Pooling (Manning Ch. 8)"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    pts_m1 = [
        ("15 Diverse Evaluation Queries", "10 complex technical syntax/config queries + 5 conversational troubleshooting queries covering 9 distinct IR query categories."),
        ("Depth-k Heterogeneous Pooling", "Pooled top-10 retrieved documents across all 4 candidate paradigms (BM25, BM25F, Routed, Dense)."),
        ("Candidate Pool Scale", "Formed 384 unique (q, d) pairs, ensuring zero system bias in the judged relevance pool."),
        ("Standardized Evaluation Metrics", "NDCG@5, NDCG@10, P@1, P@3, P@5, P@10, Recall@10, Pooled MAP@10, and MRR@10."),
    ]
    for t, d in pts_m1:
        pt = tf_m1.add_paragraph()
        pt.text = f"•  {t}:"
        pt.font.name = "Calibri"
        pt.font.size = Pt(11)
        pt.font.bold = True
        pt.font.color.rgb = C_SLATE
        pd = tf_m1.add_paragraph()
        pd.text = f"    {d}"
        pd.font.name = "Calibri"
        pd.font.size = Pt(10)
        pd.font.color.rgb = C_MUTED

    cm2 = slide9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3))
    cm2.fill.solid()
    cm2.fill.fore_color.rgb = C_CARD_BG
    cm2.line.color.rgb = C_BORDER
    tf_m2 = cm2.text_frame
    tf_m2.word_wrap = True

    p = tf_m2.paragraphs[0]
    p.text = "Dual LLM Adjudication & Agreement (Manning Ch. 8.2)"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    pts_m2 = [
        ("Judge 1 — Nemotron-3.5-Bolt (e4b)", "Strict engineering judge requiring exact code/syntax alignment (119 positive judgments)."),
        ("Judge 2 — Gemma-4 (12b)", "Lenient context-aware judge recognizing partial solutions and troubleshooting advice (227 positive judgments)."),
        ("Inter-Annotator Agreement", "Raw Agreement: 71.88% | Cohen's Kappa: κ = 0.4740 (Moderate-to-substantial agreement on difficult technical passages)."),
        ("3-Tier Graded Consensus", "• Grade 2 (Strict Consensus): 119 documents (Both judges agree R)\n• Grade 1 (Lenient Positive): 108 documents (1 judge R, 1 NR)\n• Grade 0 (Consensus Irrelevant): 157 documents (Both agree NR)"),
    ]
    for t, d in pts_m2:
        pt = tf_m2.add_paragraph()
        pt.text = f"•  {t}:"
        pt.font.name = "Calibri"
        pt.font.size = Pt(11)
        pt.font.bold = True
        pt.font.color.rgb = C_CRIMSON
        pd = tf_m2.add_paragraph()
        pd.text = f"    {d}"
        pd.font.name = "Calibri"
        pd.font.size = Pt(10)
        pd.font.color.rgb = C_MUTED

    # =========================================================================
    # SLIDE 10: Overall Effectiveness Comparison Across Paradigms
    # =========================================================================
    slide10 = prs.slides.add_slide(blank_layout)
    add_header(slide10, "9. Comparative Benchmark Results Across 5 Paradigms")

    # Embed the comprehensive 7-metric chart across top
    chart10 = CHARTS_DIR / "overall_effectiveness_hybrid.png"
    if chart10.exists():
        slide10.shapes.add_picture(str(chart10), Inches(0.8), Inches(1.35), width=Inches(11.73))

    # Add a summary metrics banner card at the bottom
    c_sum = slide10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.65), Inches(11.73), Inches(1.35))
    c_sum.fill.solid()
    c_sum.fill.fore_color.rgb = C_CARD_BG
    c_sum.line.color.rgb = C_BORDER
    tf_sum = c_sum.text_frame
    tf_sum.word_wrap = True

    p = tf_sum.paragraphs[0]
    p.text = "Key Benchmark Metrics Summary Across Paradigms:"
    p.font.name = "Calibri"
    p.font.size = Pt(11.5)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    p_tab = tf_sum.add_paragraph()
    p_tab.text = "• Flat BM25: P@1=0.7333, P@5=0.7067, NDCG@5=0.7188, NDCG@10=0.6727, MAP@10=0.3978, MRR@10=0.8667\n" \
                 "• BM25F (Fields): P@1=0.8000, P@5=0.7333, NDCG@5=0.7503, NDCG@10=0.7089 (+5.4%), MAP@10=0.4124, MRR@10=0.8889\n" \
                 "• Dense Neural: P@1=0.9333, P@5=0.8133, NDCG@5=0.8420, NDCG@10=0.8036, MAP@10=0.4634, MRR@10=0.9667\n" \
                 "• ★ Hybrid RRF: P@1=1.0000, P@5=0.8533, NDCG@5=0.8931, NDCG@10=0.8235, MAP@10=0.4864, MRR@10=1.0000"
    p_tab.font.name = "Consolas"
    p_tab.font.size = Pt(9.5)
    p_tab.font.color.rgb = C_SLATE

    # =========================================================================
    # SLIDE 11: The Hybrid Solution: Reciprocal Rank Fusion (RRF)
    # =========================================================================
    slide11 = prs.slides.add_slide(blank_layout)
    add_header(slide11, "10. The Hybrid Solution: Reciprocal Rank Fusion (RRF)")

    c_rrf1 = slide11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.3))
    c_rrf1.fill.solid()
    c_rrf1.fill.fore_color.rgb = C_CARD_BG
    c_rrf1.line.color.rgb = C_BORDER
    tf_r1 = c_rrf1.text_frame
    tf_r1.word_wrap = True

    p = tf_r1.paragraphs[0]
    p.text = "Why Rank Fusion Outperforms Linear Score Mixing"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    pts_rrf1 = [
        ("The Score Incompatibility Problem", "BM25 scores are unbounded [0, inf) depending on document lengths and IDF. Dense cosine similarities are bounded [-1, 1]. Min-max normalization is fragile to score distribution outliers."),
        ("Reciprocal Rank Fusion Formulation (SIGIR 2009)", "RRF scores documents based strictly on rank position, making it immune to score scaling distortions:"),
        ("$$RRF(d) = \\sum_{m \\in \\{\\text{Dense}, \\text{BM25F}\\}} \\frac{w_m}{k + \\text{rank}_m(d)}$$", "Constant k=20 prevents top-ranked outliers from dominating; weights w_dense=0.7, w_bm25f=0.3 reflect neural ranking strength."),
        ("Mutual Verification Effect", "Documents ranked moderately high by both systems jump to Rank 1; documents hallucinated by one system are demoted."),
    ]
    for t, d in pts_rrf1:
        pt = tf_r1.add_paragraph()
        pt.text = f"•  {t}:"
        pt.font.name = "Calibri"
        pt.font.size = Pt(11)
        pt.font.bold = True
        pt.font.color.rgb = C_SLATE
        pd = tf_r1.add_paragraph()
        pd.text = f"    {d}"
        pd.font.name = "Calibri"
        pd.font.size = Pt(10)
        pd.font.color.rgb = C_MUTED

    c_rrf2 = slide11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3))
    c_rrf2.fill.solid()
    c_rrf2.fill.fore_color.rgb = C_CARD_BG
    c_rrf2.line.color.rgb = C_BORDER
    tf_r2 = c_rrf2.text_frame
    tf_r2.word_wrap = True

    p = tf_r2.paragraphs[0]
    p.text = "Measured Hybrid Breakthroughs Across All Cutoffs"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    pts_rrf2 = [
        ("Perfect First-Result Hit (P@1 = 1.0000, MRR = 1.0000)", "Across all 15 benchmark queries, Weighted RRF places a verified relevant document at Rank 1 (100% first-hit success rate)."),
        ("Top-Rank Precision Retention", "• P@1: 1.0000 (+6.7% over Dense, +20.0% over BM25F)\n• P@3: 0.9333 (+6.7% over Dense, +13.3% over BM25F)\n• P@5: 0.8533 (+4.0% over Dense, +12.0% over BM25F)"),
        ("Discounted Cumulative Gain", "• NDCG@5: 0.8931 (Surpasses Dense at 0.8420 and BM25F at 0.7503)\n• NDCG@10: 0.8235 (Highest overall retrieval quality)"),
        ("Mean Average Precision", "MAP@10 = 0.4864 (Exceeds BM25F at 0.4124 and Dense at 0.4634)."),
        ("Conclusion", "Weighted RRF is the definitive winner for technical documentation retrieval."),
    ]
    for t, d in pts_rrf2:
        pt = tf_r2.add_paragraph()
        pt.text = f"•  {t}:"
        pt.font.name = "Calibri"
        pt.font.size = Pt(11)
        pt.font.bold = True
        pt.font.color.rgb = C_GOLD
        pd = tf_r2.add_paragraph()
        pd.text = f"    {d}"
        pd.font.name = "Calibri"
        pd.font.size = Pt(10)
        pd.font.color.rgb = C_MUTED

    # =========================================================================
    # SLIDE 12: Query Deep Dive: Lexical Syntax vs. Conversational Semantics
    # =========================================================================
    slide12 = prs.slides.add_slide(blank_layout)
    add_header(slide12, "11. Qualitative Query Analysis: Syntax vs. Conversational")

    cq1 = slide12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.3))
    cq1.fill.solid()
    cq1.fill.fore_color.rgb = C_CARD_BG
    cq1.line.color.rgb = C_BORDER
    tf_q1 = cq1.text_frame
    tf_q1.word_wrap = True

    p = tf_q1.paragraphs[0]
    p.text = "Case Study 1: Exact Syntax Query (Q05)"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    pq1 = tf_q1.add_paragraph()
    pq1.text = "Query: \"How do I use --mount=type=cache in Docker Buildx to speed up pip installs?\""
    pq1.font.name = "Calibri"
    pq1.font.size = Pt(11)
    pq1.font.italic = True
    pq1.font.color.rgb = C_CRIMSON

    pts_cs1 = [
        ("BM25 / BM25F Behavior (WIN)", "Matches exact code token `--mount=type=cache` and `Buildx` in Code and Title zones. Top result is the exact Buildx caching guide (Score: 28.4). NDCG@10 = 0.7574."),
        ("Dense Neural Behavior (WEAK)", "Embeddings lose punctuation/dashes (`--mount=type=cache` collapses into generic semantic space). Top results drift toward generic python pip caching and Dockerfile layers. NDCG@10 = 0.7066."),
        ("Hybrid RRF Behavior (WIN)", "Preserves Rank 1 from BM25F while enriching lower ranks with conceptually relevant Docker build guides."),
    ]
    for t, d in pts_cs1:
        pt = tf_q1.add_paragraph()
        pt.text = f"•  {t}:"
        pt.font.name = "Calibri"
        pt.font.size = Pt(10.5)
        pt.font.bold = True
        pt.font.color.rgb = C_SLATE
        pd = tf_q1.add_paragraph()
        pd.text = f"    {d}"
        pd.font.name = "Calibri"
        pd.font.size = Pt(9.5)
        pd.font.color.rgb = C_MUTED

    cq2 = slide12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3))
    cq2.fill.solid()
    cq2.fill.fore_color.rgb = C_CARD_BG
    cq2.line.color.rgb = C_BORDER
    tf_q2 = cq2.text_frame
    tf_q2.word_wrap = True

    p = tf_q2.paragraphs[0]
    p.text = "Case Study 2: Conversational Setup Query (Q11)"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    pq2 = tf_q2.add_paragraph()
    pq2.text = "Query: \"Can you help me setup springboot within 10 minutes without any issues?\""
    pq2.font.name = "Calibri"
    pq2.font.size = Pt(11)
    pq2.font.italic = True
    pq2.font.color.rgb = C_GOLD

    pts_cs2 = [
        ("BM25 / BM25F Behavior (FAIL)", "Suffers from severe term mismatch: no official doc contains 'without any issues' or '10 minutes'. Ranks noisy issues pages mentioning 'issues'. NDCG@10 = 0.1389, P@5 = 0.2000."),
        ("Dense Neural Behavior (WIN)", "Captures user intent: understands 'setup quickly' -> 'Spring Initializr', 'Quick Start Guide', and 'Building an Application with Spring Boot'. NDCG@10 = 1.0000, P@5 = 1.0000."),
        ("Hybrid RRF Behavior (WIN)", "Propels the dense semantic matches to top ranks, yielding a flawless NDCG@10 = 1.0000."),
    ]
    for t, d in pts_cs2:
        pt = tf_q2.add_paragraph()
        pt.text = f"•  {t}:"
        pt.font.name = "Calibri"
        pt.font.size = Pt(10.5)
        pt.font.bold = True
        pt.font.color.rgb = C_SLATE
        pd = tf_q2.add_paragraph()
        pd.text = f"    {d}"
        pd.font.name = "Calibri"
        pd.font.size = Pt(9.5)
        pd.font.color.rgb = C_MUTED

    # =========================================================================
    # SLIDE 13: System Trade-Offs, Engineering Guidelines & Conclusion
    # =========================================================================
    slide13 = prs.slides.add_slide(blank_layout)
    add_header(slide13, "12. Architectural Trade-Offs, Guidelines & Conclusion")

    # Embed the updated two-panel storage chart on the left
    chart13 = CHARTS_DIR / "index_size_breakdown.png"
    if chart13.exists():
        slide13.shapes.add_picture(str(chart13), Inches(0.8), Inches(1.5), width=Inches(6.0))

    # Right: Practical Guidelines & Takeaways
    c_end = slide13.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.1), Inches(1.5), Inches(5.4), Inches(5.3))
    c_end.fill.solid()
    c_end.fill.fore_color.rgb = C_CARD_BG
    c_end.line.color.rgb = C_BORDER
    tf_end = c_end.text_frame
    tf_end.word_wrap = True

    p = tf_end.paragraphs[0]
    p.text = "Production Recommendations & Final Conclusions"
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    guidelines = [
        ("Corpus Filtering (18.6 GB -> 1.12 GB)", "Filtering boilerplate and non-technical noise reduced 79k raw files to 45.8k high-signal documents, preserving critical code blocks."),
        ("Classical Index Serialization Reality", "Uncompressed plaintext JSON takes 1.11 GB; standard production binary compression (VByte/Lucene) achieves < 95 MB."),
        ("Neural Pipeline Requires 2.6x More Disk & 2.5x RAM", "The full neural system requires 2.89 GB on disk and 2.1 GB VRAM, but cluster pruning cuts evaluation latency by 12.5x (19.8 ms)."),
        ("Deploy Weighted RRF for State-of-the-Art Quality", "Combining BM25F and Dense rankings via Weighted RRF achieves MRR = 1.0000 and NDCG@10 = 0.8235, strictly beating all individual models."),
        ("Multi-Tier Caching Eliminates SLA Bottlenecks", "LRU query result and embedding caching delivers <0.05 ms latency for repeated queries, supporting >25,000 QPS."),
    ]
    for t, d in guidelines:
        pt = tf_end.add_paragraph()
        pt.text = f"✔  {t}:"
        pt.font.name = "Calibri"
        pt.font.size = Pt(10)
        pt.font.bold = True
        pt.font.color.rgb = C_NAVY
        pd = tf_end.add_paragraph()
        pd.text = f"    {d}"
        pd.font.name = "Calibri"
        pd.font.size = Pt(9)
        pd.font.color.rgb = C_SLATE

    prs.save(str(OUTPUT_PPTX))
    print(f"Presentation deck successfully created at: {OUTPUT_PPTX}")
    print(f"Total slides: {len(prs.slides)}")


if __name__ == "__main__":
    create_deck()
