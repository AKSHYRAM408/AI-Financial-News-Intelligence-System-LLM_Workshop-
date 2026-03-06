"""
app.py — Premium Streamlit dashboard for the Stock Market AI Summarizer.

Pipeline:
  Query → News API → BERT Similarity Scoring → Select Top Articles → LLM Summary → Dashboard
"""

import logging
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

from news_fetcher import fetch_market_news
from processor import clean_and_merge_articles
from embedder import rank_articles_by_relevance
from llm import generate_market_report, answer_investor_question, direct_llm_query

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Stock Market Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Premium Dark Mode CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

    /* ─── Global Reset & Dark Theme ─── */
    :root {
        --bg-primary: #0a0e1a;
        --bg-secondary: #111827;
        --bg-card: rgba(17, 24, 39, 0.8);
        --bg-glass: rgba(255, 255, 255, 0.03);
        --border-glass: rgba(255, 255, 255, 0.08);
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --accent-blue: #3b82f6;
        --accent-purple: #8b5cf6;
        --accent-green: #10b981;
        --accent-amber: #f59e0b;
        --accent-red: #ef4444;
        --accent-cyan: #06b6d4;
        --gradient-primary: linear-gradient(135deg, #3b82f6, #8b5cf6);
        --gradient-green: linear-gradient(135deg, #10b981, #06b6d4);
        --gradient-amber: linear-gradient(135deg, #f59e0b, #ef4444);
        --shadow-glow: 0 0 30px rgba(59, 130, 246, 0.15);
        --shadow-card: 0 4px 24px rgba(0, 0, 0, 0.3);
    }

    .stApp {
        background: var(--bg-primary) !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
    }

    .stApp > header { background: transparent !important; }

    /* Hide default Streamlit elements */
    #MainMenu, footer, .stDeployButton { display: none !important; }

    /* ─── Typography (scoped to avoid breaking Streamlit internals) ─── */
    .stApp {
        font-family: 'Inter', -apple-system, sans-serif !important;
        color: var(--text-primary) !important;
    }

    .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2,
    .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6,
    .stMarkdown li, .stMarkdown span {
        font-family: 'Inter', sans-serif !important;
        color: var(--text-primary) !important;
    }

    .stCaption, .stCaption p {
        color: var(--text-muted) !important;
    }

    /* Alert / error / warning text */
    .stAlert p {
        color: inherit !important;
    }

    /* ─── Hero Section ─── */
    .hero-container {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
        position: relative;
    }

    .hero-logo {
        font-size: 3.5rem;
        margin-bottom: 0.5rem;
        filter: drop-shadow(0 0 20px rgba(59, 130, 246, 0.4));
        animation: float 3s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: var(--text-secondary) !important;
        font-weight: 400;
        letter-spacing: 0.02em;
    }

    /* ─── Pipeline Visualization ─── */
    .pipeline-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 0.4rem;
        padding: 1rem 0;
        flex-wrap: wrap;
    }

    .pipeline-node {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        border: 1px solid;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }

    .pipeline-node:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }

    .node-news {
        background: rgba(59, 130, 246, 0.1);
        border-color: rgba(59, 130, 246, 0.3);
        color: #60a5fa !important;
    }

    .node-bert {
        background: rgba(139, 92, 246, 0.1);
        border-color: rgba(139, 92, 246, 0.3);
        color: #a78bfa !important;
    }

    .node-filter {
        background: rgba(6, 182, 212, 0.1);
        border-color: rgba(6, 182, 212, 0.3);
        color: #22d3ee !important;
    }

    .node-llm {
        background: rgba(16, 185, 129, 0.1);
        border-color: rgba(16, 185, 129, 0.3);
        color: #34d399 !important;
    }

    .node-ui {
        background: rgba(245, 158, 11, 0.1);
        border-color: rgba(245, 158, 11, 0.3);
        color: #fbbf24 !important;
    }

    .pipeline-arrow {
        color: var(--text-muted) !important;
        font-size: 1rem;
        opacity: 0.5;
    }

    /* ─── Glass Card ─── */
    .glass-card {
        background: var(--bg-glass);
        border: 1px solid var(--border-glass);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(20px);
        box-shadow: var(--shadow-card);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        border-color: rgba(255, 255, 255, 0.12);
        box-shadow: var(--shadow-glow);
    }

    .glass-card-header {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ─── Metrics Row ─── */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 0.8rem;
        margin: 1rem 0;
    }

    .metric-card {
        background: var(--bg-glass);
        border: 1px solid var(--border-glass);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(59, 130, 246, 0.3);
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.1);
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace !important;
        margin-bottom: 0.2rem;
    }

    .metric-label {
        font-size: 0.75rem;
        color: var(--text-muted) !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
    }

    .metric-blue .metric-value { color: #60a5fa !important; }
    .metric-purple .metric-value { color: #a78bfa !important; }
    .metric-green .metric-value { color: #34d399 !important; }
    .metric-amber .metric-value { color: #fbbf24 !important; }

    /* ─── Status Badges ─── */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    .badge-success {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399 !important;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .badge-bert {
        background: rgba(139, 92, 246, 0.15);
        color: #a78bfa !important;
        border: 1px solid rgba(139, 92, 246, 0.3);
    }

    .badge-info {
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa !important;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }

    .badge-warn {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24 !important;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    /* ─── Source Pills ─── */
    .source-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: var(--text-secondary) !important;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 500;
        margin: 0.15rem;
        transition: all 0.2s ease;
    }

    .source-pill:hover {
        background: rgba(59, 130, 246, 0.1);
        border-color: rgba(59, 130, 246, 0.3);
    }

    /* ─── Relevance Bars ─── */
    .relevance-item {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        padding: 0.6rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }

    .relevance-item:last-child { border-bottom: none; }

    .relevance-score {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem;
        font-weight: 700;
        min-width: 48px;
        text-align: right;
    }

    .relevance-bar-bg {
        flex: 1;
        height: 6px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 3px;
        overflow: hidden;
    }

    .relevance-bar-fill {
        height: 100%;
        border-radius: 3px;
        background: linear-gradient(90deg, #8b5cf6, #3b82f6);
        transition: width 0.8s ease;
    }

    .relevance-headline {
        font-size: 0.82rem;
        color: var(--text-secondary) !important;
        max-width: 400px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* ─── Report Section Cards ─── */
    .report-section {
        background: var(--bg-glass);
        border: 1px solid var(--border-glass);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
        transition: all 0.3s ease;
    }

    .report-section:hover {
        border-color: rgba(255, 255, 255, 0.12);
        box-shadow: 0 2px 15px rgba(0, 0, 0, 0.2);
    }

    .report-section-title {
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .report-section-body {
        font-size: 0.9rem;
        line-height: 1.7;
        color: var(--text-secondary) !important;
    }

    /* ─── Q&A Section ─── */
    .qa-answer-box {
        background: rgba(59, 130, 246, 0.05);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-left: 3px solid #3b82f6;
        padding: 1.2rem 1.5rem;
        border-radius: 0 12px 12px 0;
        margin-top: 0.8rem;
        color: var(--text-secondary) !important;
        line-height: 1.7;
    }

    /* ─── Disclaimer ─── */
    .disclaimer-box {
        background: rgba(245, 158, 11, 0.05);
        border: 1px solid rgba(245, 158, 11, 0.15);
        border-left: 3px solid #f59e0b;
        padding: 1rem 1.2rem;
        font-size: 0.8rem;
        color: var(--text-muted) !important;
        border-radius: 0 8px 8px 0;
        margin-top: 1.5rem;
    }

    /* ─── Direct Answer Box ─── */
    .direct-answer-box {
        background: rgba(6, 182, 212, 0.05);
        border: 1px solid rgba(6, 182, 212, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        color: var(--text-secondary) !important;
        line-height: 1.7;
    }

    /* ─── Footer ─── */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem;
        color: var(--text-muted) !important;
        font-size: 0.75rem;
        letter-spacing: 0.03em;
    }

    .footer-links {
        display: flex;
        justify-content: center;
        gap: 1.5rem;
        margin-top: 0.5rem;
        font-size: 0.7rem;
    }

    .footer-links span {
        color: var(--text-muted) !important;
        opacity: 0.6;
    }

    /* ─── Input Styling ─── */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
        padding: 0.7rem 1rem !important;
        font-size: 0.95rem !important;
        transition: all 0.3s ease !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: rgba(59, 130, 246, 0.5) !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1) !important;
    }

    .stTextInput > div > div > input::placeholder {
        color: var(--text-muted) !important;
        opacity: 0.7;
    }

    /* ─── Button Styling ─── */
    .stButton > button {
        border-radius: 12px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        transition: all 0.3s ease !important;
        border: none !important;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3) !important;
    }

    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 25px rgba(59, 130, 246, 0.5) !important;
        transform: translateY(-1px);
    }

    .stButton > button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        color: var(--text-primary) !important;
    }

    .stButton > button[kind="secondary"]:hover {
        background: rgba(255, 255, 255, 0.1) !important;
        border-color: rgba(59, 130, 246, 0.4) !important;
    }

    /* ─── Expander Styling (modern data-testid selectors) ─── */
    [data-testid="stExpander"] {
        background: var(--bg-glass) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: 12px !important;
        overflow: hidden;
    }

    [data-testid="stExpander"] summary {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        padding: 0.8rem 1rem !important;
    }

    [data-testid="stExpander"] summary:hover {
        background: rgba(255, 255, 255, 0.04) !important;
    }

    [data-testid="stExpander"] summary span {
        color: var(--text-primary) !important;
    }

    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        background: rgba(255, 255, 255, 0.02) !important;
        border-top: 1px solid var(--border-glass) !important;
    }

    /* ─── Status Widget ─── */
    [data-testid="stStatus"] {
        background: var(--bg-glass) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: 12px !important;
    }

    /* ─── Divider ─── */
    hr {
        border-color: rgba(255, 255, 255, 0.06) !important;
        margin: 1.5rem 0 !important;
    }

    /* ─── Spinner ─── */
    .stSpinner > div {
        border-top-color: #3b82f6 !important;
    }

    /* ─── Scrollbar ─── */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }

    ::-webkit-scrollbar-track {
        background: transparent;
    }

    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 3px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.2);
    }

    /* ─── Fade-in animation ─── */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(15px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .fade-in {
        animation: fadeInUp 0.5s ease forwards;
    }

    .fade-in-delay-1 { animation-delay: 0.1s; opacity: 0; }
    .fade-in-delay-2 { animation-delay: 0.2s; opacity: 0; }
    .fade-in-delay-3 { animation-delay: 0.3s; opacity: 0; }

    /* ─── Pulse animation for live indicator ─── */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    .live-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #10b981;
        margin-right: 6px;
        animation: pulse 2s ease-in-out infinite;
    }

    /* ─── Section Divider ─── */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
        margin: 2rem 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
defaults = {
    "report": None,
    "cleaned_text": "",
    "articles_count": 0,
    "last_query": "",
    "qa_answer": "",
    "sources": [],
    "direct_answer": "",
    "ranked_articles": [],
    "bert_used": False,
    "bert_top_score": 0.0,
    "pipeline_time": "",
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ---------------------------------------------------------------------------
# Hero Section
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-container fade-in">
        <div class="hero-logo">📈</div>
        <div class="hero-title">Stock Market Intelligence</div>
        <div class="hero-subtitle">Real-time news analysis powered by BERT embeddings & AI</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Pipeline visualization
st.markdown(
    """
    <div class="pipeline-container fade-in fade-in-delay-1">
        <span class="pipeline-node node-news">📰 News API</span>
        <span class="pipeline-arrow">→</span>
        <span class="pipeline-node node-bert">🧠 BERT Embeddings</span>
        <span class="pipeline-arrow">→</span>
        <span class="pipeline-node node-filter">🔍 Similarity Filter</span>
        <span class="pipeline-arrow">→</span>
        <span class="pipeline-node node-llm">🤖 LLM Analysis</span>
        <span class="pipeline-arrow">→</span>
        <span class="pipeline-node node-ui">✨ Dashboard</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Input section
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="glass-card fade-in fade-in-delay-2">
        <div class="glass-card-header">
            🔍 &nbsp;Market Intelligence Query
        </div>
        <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.5rem;">
            Search stocks, sectors, or market topics — AI will fetch news, rank by relevance, and generate insights.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col_input, col_btn = st.columns([4, 1], gap="medium")

with col_input:
    query = st.text_input(
        "Market Query",
        placeholder="e.g. NIFTY today, Tesla earnings, RBI interest rate, IT sector...",
        label_visibility="collapsed",
    )

with col_btn:
    generate_btn = st.button("🚀 Generate Report", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Report section icons
# ---------------------------------------------------------------------------
SECTION_ICONS = {
    "Executive Summary": "📋",
    "Market Overview": "🌐",
    "Sector Performance": "📊",
    "Key Companies": "🏢",
    "Macro Signals": "📡",
    "Risk Signals": "⚠️",
    "Outlook": "🔮",
    "Disclaimer": "📜",
}

SECTION_COLORS = {
    "Executive Summary": "#3b82f6",
    "Market Overview": "#06b6d4",
    "Sector Performance": "#8b5cf6",
    "Key Companies": "#10b981",
    "Macro Signals": "#f59e0b",
    "Risk Signals": "#ef4444",
    "Outlook": "#a78bfa",
    "Disclaimer": "#64748b",
}

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------
if generate_btn:
    if not query or not query.strip():
        st.error("⚠️ Please enter a market query or topic to search for.")
    else:
        # Reset state
        for key in defaults:
            st.session_state[key] = defaults[key]
        st.session_state.last_query = query.strip()

        start_time = datetime.now()

        try:
            # ── Step 1: Fetch news via API ──
            with st.status("📰 Step 1/4 — Fetching news articles from API...", expanded=True) as status:
                st.write(f"🔎 Searching NEWSDATA.io for: **{query.strip()}**")
                articles = fetch_market_news(user_query=query.strip())
                st.session_state.articles_count = len(articles)

                if not articles:
                    status.update(label="⚠️ No articles found — using AI knowledge", state="complete")
                    st.warning(
                        "No news articles found from the API. "
                        "Falling back to AI's own knowledge to answer your query."
                    )

            # Fallback: no articles → ask LLM directly
            if not articles:
                with st.status("🤖 Asking AI directly...", expanded=True) as status:
                    st.write(f"Generating answer for: **{query.strip()}**")
                    direct_answer = direct_llm_query(query.strip())
                    st.session_state.direct_answer = direct_answer
                    elapsed = (datetime.now() - start_time).total_seconds()
                    st.session_state.pipeline_time = f"{elapsed:.1f}s"
                    status.update(label="✅ AI response ready", state="complete")
            else:
                # Collect sources
                sources = list({a.get("source", "") for a in articles if a.get("source")})
                st.session_state.sources = sources
                st.write(f"✅ Found **{len(articles)}** article(s) from **{len(sources)}** source(s)")
                status.update(label=f"✅ Fetched {len(articles)} article(s)", state="complete")

                # ── Step 2: BERT Embedding & Similarity Scoring ──
                with st.status("🧠 Step 2/4 — Generating BERT embeddings & scoring...", expanded=True) as status:
                    st.write(f"🔄 Encoding {len(articles)} articles with `sentence-transformers/all-MiniLM-L6-v2`")
                    st.write(f"📐 Computing cosine similarity against query: **{query.strip()}**")

                    ranked = rank_articles_by_relevance(
                        query=query.strip(),
                        articles=articles,
                        top_k=min(5, len(articles)),
                        min_score=0.1,
                    )

                    if ranked:
                        st.session_state.ranked_articles = ranked
                        st.session_state.bert_used = True
                        st.session_state.bert_top_score = ranked[0][1]
                        articles_for_llm = [art for art, _ in ranked]

                        st.write(f"✅ Selected **{len(ranked)}** most relevant articles:")
                        for i, (art, score) in enumerate(ranked, 1):
                            st.write(f"  {i}. **{score:.0%}** — {art.get('headline', 'N/A')[:80]}")

                        status.update(
                            label=f"✅ Top {len(ranked)} articles ranked by BERT similarity",
                            state="complete",
                        )
                    else:
                        import embedder
                        err_detail = embedder.last_error or "Unknown error"
                        st.warning(f"BERT ranking failed: **{err_detail}**. Using all articles.")
                        articles_for_llm = articles
                        status.update(label="⚠️ BERT failed — using all articles", state="complete")

                # ── Step 3: Similarity Filtering ──
                with st.status("🔍 Step 3/4 — Filtering by similarity threshold...", expanded=True) as status:
                    original_count = len(articles)
                    filtered_count = len(articles_for_llm)
                    removed = original_count - filtered_count

                    if st.session_state.bert_used:
                        st.write(f"📊 Applied similarity threshold: **≥ 10%**")
                        st.write(f"✅ Kept **{filtered_count}** articles, filtered out **{removed}** low-relevance articles")
                    else:
                        st.write(f"ℹ️ No BERT filtering applied — using all **{filtered_count}** articles")

                    status.update(label=f"✅ {filtered_count} articles passed filter", state="complete")

                # ── Step 4: Process + LLM Report ──
                with st.status("🧹 Step 4a/4 — Cleaning & processing text...", expanded=True) as status:
                    cleaned_text = clean_and_merge_articles(articles_for_llm)
                    if not cleaned_text:
                        status.update(label="❌ Processing failed", state="error")
                        st.error("Text processing produced empty output.")
                        st.stop()

                    st.session_state.cleaned_text = cleaned_text
                    st.write(f"✅ Processed **{len(cleaned_text):,}** characters from **{len(articles_for_llm)}** article(s)")
                    status.update(label="✅ Text processed", state="complete")

                with st.status("🤖 Step 4b/4 — LLM generating market analysis...", expanded=True) as status:
                    st.write("🧠 Sending to Mistral AI for deep market analysis...")
                    report = generate_market_report(cleaned_text)
                    st.session_state.report = report
                    elapsed = (datetime.now() - start_time).total_seconds()
                    st.session_state.pipeline_time = f"{elapsed:.1f}s"
                    status.update(label="✅ Market analysis report generated", state="complete")

        except Exception as exc:
            logger.error("Pipeline error: %s", exc, exc_info=True)
            st.error(f"An unexpected error occurred: {exc}")

# ---------------------------------------------------------------------------
# Display direct AI answer (fallback)
# ---------------------------------------------------------------------------
if st.session_state.direct_answer:
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="glass-card fade-in">
            <div class="glass-card-header">
                <span class="badge badge-info">🤖 AI Direct Answer</span>
                <span style="font-size: 0.75rem; color: #64748b; margin-left: auto;">No live news data available</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(f'Query: "{st.session_state.last_query}"')

    st.markdown(
        f'<div class="direct-answer-box fade-in fade-in-delay-1">{st.session_state.direct_answer}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="disclaimer-box fade-in fade-in-delay-2">'
        "<strong>📜 Note:</strong> "
        "This answer is based on the AI's general knowledge, not on live news data. "
        "For informational purposes only — not investment advice."
        "</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Display report
# ---------------------------------------------------------------------------
if st.session_state.report:
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Metrics Dashboard ──
    bert_score_display = f"{st.session_state.bert_top_score:.0%}" if st.session_state.bert_used else "N/A"
    bert_status = "Active" if st.session_state.bert_used else "Skipped"
    filtered_count = len(st.session_state.ranked_articles) if st.session_state.bert_used else st.session_state.articles_count

    st.markdown(
        f"""
        <div class="metrics-grid fade-in">
            <div class="metric-card metric-blue">
                <div class="metric-value">{st.session_state.articles_count}</div>
                <div class="metric-label">Articles Fetched</div>
            </div>
            <div class="metric-card metric-purple">
                <div class="metric-value">{bert_score_display}</div>
                <div class="metric-label">Top BERT Score</div>
            </div>
            <div class="metric-card metric-green">
                <div class="metric-value">{filtered_count}</div>
                <div class="metric-label">Articles Analyzed</div>
            </div>
            <div class="metric-card metric-amber">
                <div class="metric-value">{st.session_state.pipeline_time}</div>
                <div class="metric-label">Pipeline Time</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Status Badges ──
    meta_col1, meta_col2 = st.columns(2)
    with meta_col1:
        badges = '<span class="badge badge-success"><span class="live-dot"></span>Report Ready</span> '
        if st.session_state.bert_used:
            badges += '<span class="badge badge-bert">🧠 BERT Filtered</span>'
        st.markdown(badges, unsafe_allow_html=True)
    with meta_col2:
        st.caption(
            f'Query: "{st.session_state.last_query}" · '
            f"{st.session_state.articles_count} article(s) analysed"
        )

    # ── Show sources ──
    if st.session_state.sources:
        source_pills = " ".join(
            f'<span class="source-pill">📰 {s}</span>' for s in st.session_state.sources
        )
        st.markdown(f"**Sources:** {source_pills}", unsafe_allow_html=True)

    # ── Show BERT-ranked articles with visual scores ──
    if st.session_state.ranked_articles:
        with st.expander("🧠 BERT Relevance Scores — Top Matched Articles", expanded=False):
            for i, (art, score) in enumerate(st.session_state.ranked_articles, 1):
                bar_pct = int(score * 100)
                score_color = "#10b981" if score >= 0.5 else "#f59e0b" if score >= 0.3 else "#ef4444"
                st.markdown(
                    f"""
                    <div class="relevance-item">
                        <span class="relevance-score" style="color: {score_color} !important;">{score:.0%}</span>
                        <div class="relevance-bar-bg">
                            <div class="relevance-bar-fill" style="width: {bar_pct}%; background: linear-gradient(90deg, {score_color}, {score_color}88);"></div>
                        </div>
                        <span class="relevance-headline">{art.get('headline', 'N/A')}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if art.get("source"):
                    st.caption(f"  ↳ {art['source']} · {art.get('published_time', '')}")

    st.markdown("")

    report = st.session_state.report

    # ── Render each report section as styled cards ──
    sections_to_show = [
        "Executive Summary",
        "Market Overview",
        "Sector Performance",
        "Key Companies",
        "Macro Signals",
        "Risk Signals",
        "Outlook",
    ]

    for idx, section in enumerate(sections_to_show):
        icon = SECTION_ICONS.get(section, "📄")
        color = SECTION_COLORS.get(section, "#3b82f6")
        content = report.get(section, "No data available.")

        with st.expander(f"{icon}  {section}", expanded=(section == "Executive Summary")):
            st.markdown(content)

    # ── Disclaimer ──
    disclaimer = report.get(
        "Disclaimer",
        "This report is for informational purposes only and does not constitute "
        "investment advice. Past performance is not indicative of future results.",
    )
    st.markdown(
        f'<div class="disclaimer-box"><strong>📜 Disclaimer:</strong> {disclaimer}</div>',
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------
    # Q&A Section
    # -------------------------------------------------------------------
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="glass-card">
            <div class="glass-card-header">💬 &nbsp;Ask a Follow-up Question</div>
            <div style="font-size: 0.85rem; color: #94a3b8;">
                Ask any market question — the AI will answer using the most relevant articles from your search.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    qa_col_input, qa_col_btn = st.columns([4, 1], gap="medium")

    with qa_col_input:
        investor_question = st.text_input(
            "Your Question",
            placeholder="e.g. Which stocks should I watch today? What's driving the market?",
            label_visibility="collapsed",
            key="qa_input",
        )

    with qa_col_btn:
        ask_btn = st.button("💡 Get Answer", use_container_width=True, type="secondary")

    if ask_btn:
        if not investor_question or not investor_question.strip():
            st.error("⚠️ Please type a question.")
        else:
            with st.spinner("🤖 Analyzing articles and generating answer..."):
                answer = answer_investor_question(
                    st.session_state.cleaned_text,
                    investor_question.strip(),
                )
                st.session_state.qa_answer = answer

    if st.session_state.qa_answer:
        st.markdown(
            f'<div class="qa-answer-box fade-in">{st.session_state.qa_answer}</div>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("")
st.markdown(
    """
    <div class="footer">
        <div>AI Stock Market Intelligence</div>
        <div class="footer-links">
            <span>News API</span>
            <span>•</span>
            <span>BERT Embeddings</span>
            <span>•</span>
            <span>Similarity Filtering</span>
            <span>•</span>
            <span>Mistral AI</span>
            <span>•</span>
            <span>Streamlit</span>
        </div>
        <div style="margin-top: 0.5rem; opacity: 0.5;">For educational and informational purposes only</div>
    </div>
    """,
    unsafe_allow_html=True,
)
