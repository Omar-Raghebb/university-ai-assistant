"""
Streamlit UI for the University AI Assistant.
Run with: streamlit run app.py
"""

import streamlit as st
from src.document_loader import load_pdfs, split_documents
from src.vector_store import build_vector_store, load_vector_store
from src.chain import answer_question

st.set_page_config(
    page_title="Sadat Academy Assistant",
    page_icon="🏢",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS — Modern Premium Dark Academic Portal (unified system)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg: #0f1117;
    --surface: #181b24;
    --surface-hover: #1e2230;
    --border: #262a38;
    --border-light: #383d52;
    --text: #e9ebf2;
    --text-secondary: #8a92a8;
    --text-tertiary: #5a6178;

    /* single brand accent */
    --accent: #d4a843;
    --accent-hover: #e8bc5a;
    --accent-soft: rgba(212,168,67,0.12);

    /* semantic Q/A colors — used ONLY for question/answer distinction */
    --q-color: #5b8def;
    --q-soft: rgba(91,141,239,0.10);
    --a-color: #34c98f;
    --a-soft: rgba(52,201,143,0.10);

    /* confidence semantics */
    --success: #34c98f;
    --success-soft: rgba(52,201,143,0.12);
    --warning: #f0a83d;
    --warning-soft: rgba(240,168,61,0.12);
    --danger: #ef5b5b;
    --danger-soft: rgba(239,91,91,0.12);

    --radius-lg: 16px;
    --radius: 12px;
    --radius-sm: 9px;
    --shadow: 0 6px 28px rgba(0,0,0,0.4);
    --shadow-sm: 0 2px 10px rgba(0,0,0,0.25);
    --space-1: 6px;
    --space-2: 10px;
    --space-3: 16px;
    --space-4: 24px;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.stApp { background: var(--bg) !important; }

.block-container {
    padding: 3.75rem 1.25rem 6rem 1.25rem !important;
    max-width: 720px;
}

/* ─── Header ─── */
.header-wrap {
    background: linear-gradient(135deg, #171b28 0%, #0f1117 100%);
    border-bottom: 1px solid var(--border);
    padding: 18px 0;
    margin: 0 -1.25rem;
    position: sticky;
    top: 3.75rem; /* sits just below Streamlit's own toolbar so it isn't clipped */
    z-index: 100;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
}

.header-inner {
    max-width: 720px;
    margin: 0 auto;
    padding: 0 1.25rem;
    display: flex;
    align-items: center;
    gap: 14px;
}

.header-icon {
    width: 42px;
    height: 42px;
    background: linear-gradient(135deg, var(--accent) 0%, #b5893a 100%);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 21px;
    box-shadow: 0 4px 14px rgba(212,168,67,0.25);
    flex-shrink: 0;
}

.header-title {
    color: var(--text);
    font-size: 17px;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.2px;
    line-height: 1.25;
}

.header-sub {
    color: var(--text-secondary);
    font-size: 12px;
    margin: 2px 0 0 0;
    font-weight: 400;
}

/* ─── Breadcrumb ─── */
.breadcrumb {
    padding: 18px 0 2px 0;
    font-size: 12px;
    color: var(--text-tertiary);
    margin-bottom: 18px;
}
.breadcrumb .current { color: var(--accent); font-weight: 600; }

/* ─── Welcome ─── */
.welcome-title {
    font-size: 24px;
    font-weight: 700;
    color: var(--text);
    margin: 4px 0 6px 0;
    letter-spacing: -0.4px;
}
.welcome-sub {
    font-size: 14px;
    color: var(--text-secondary);
    margin: 0 0 22px 0;
    line-height: 1.6;
    max-width: 520px;
}

/* ─── Chip Buttons ─── */
.chip-btn > button {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 20px !important;
    padding: 8px 16px !important;
    font-size: 12.5px !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    transition: all 0.18s ease !important;
    box-shadow: none !important;
    width: 100%;
}
.chip-btn > button:hover {
    background: var(--accent-soft) !important;
    border-color: var(--accent) !important;
    color: var(--accent-hover) !important;
    transform: translateY(-1px);
}
.chip-btn > button:active { transform: translateY(0); }

/* ─── Chat Cards ─── */
.chat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    margin-bottom: 14px;
    overflow: hidden;
    box-shadow: var(--shadow-sm);
    animation: fadeIn 0.35s ease forwards;
}

.card-header {
    padding: 13px 18px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 10px;
}
.q-header { background: var(--q-soft); }
.a-header { background: var(--a-soft); }

.avatar {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    font-weight: 700;
    flex-shrink: 0;
}
.avatar-q { background: var(--q-color); color: #0f1117; }
.avatar-a { background: var(--a-color); color: #0f1117; }

.card-label {
    font-size: 10.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.1px;
}
.q-header .card-label { color: var(--q-color); }
.a-header .card-label { color: var(--a-color); }

.card-body {
    padding: 18px;
    font-size: 15px;
    color: var(--text);
    line-height: 1.7;
}
.q-body { font-weight: 500; }

/* ─── Meta Bar ─── */
.meta-bar {
    padding: 11px 18px;
    background: rgba(255,255,255,0.015);
    border-top: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}

.pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 11px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    border: 1px solid transparent;
}
.pill-high { background: var(--success-soft); color: var(--success); border-color: rgba(52,201,143,0.2); }
.pill-medium { background: var(--warning-soft); color: var(--warning); border-color: rgba(240,168,61,0.2); }
.pill-low { background: var(--danger-soft); color: var(--danger); border-color: rgba(239,91,91,0.2); }
.pill-source { background: var(--accent-soft); color: var(--accent); border-color: rgba(212,168,67,0.2); }

/* ─── Source Cards ─── */
.source-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 13px 15px;
    margin-bottom: 8px;
    transition: border-color 0.18s ease;
}
.source-card:hover { border-color: var(--border-light); }
.source-title {
    font-size: 11.5px;
    font-weight: 700;
    color: var(--accent);
    margin-bottom: 5px;
}
.source-text {
    font-size: 12.5px;
    color: var(--text-secondary);
    line-height: 1.6;
}

/* ─── Expander ─── */
[data-testid="stExpander"] > details {
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background: var(--surface);
    overflow: hidden;
}
[data-testid="stExpander"] > details > summary {
    padding: 11px 15px;
    font-size: 12.5px;
    font-weight: 600;
    color: var(--text-secondary);
    background: transparent;
    border-bottom: 1px solid var(--border);
}
[data-testid="stExpander"] > details > summary:hover {
    color: var(--text);
    background: rgba(255,255,255,0.02);
}
[data-testid="stExpander"] > details > div {
    padding: 13px 15px;
    background: transparent;
}

/* ─── Input ─── */
[data-testid="stChatInput"] {
    background: var(--surface) !important;
    border-top: 1px solid var(--border) !important;
    padding: 12px 18px !important;
}
[data-testid="stChatInput"] textarea {
    font-family: 'Inter', sans-serif !important;
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 13px 16px !important;
    font-size: 14px !important;
    color: var(--text) !important;
    box-shadow: var(--shadow-sm) !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-soft) !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: var(--text-tertiary) !important; }

/* ─── Sidebar ─── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, #b5893a 100%);
    color: #0f1117 !important;
    border: none;
    border-radius: var(--radius-sm);
    font-weight: 700;
    font-size: 13px;
    padding: 11px 18px;
    width: 100%;
    transition: all 0.18s ease;
    box-shadow: 0 4px 12px rgba(212,168,67,0.2);
}
[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(212,168,67,0.3);
}

/* ─── Empty State ─── */
.empty-wrap { text-align: center; padding: 70px 20px; color: var(--text-secondary); }
.empty-icon {
    width: 68px;
    height: 68px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 30px;
    margin: 0 auto 18px auto;
}
.empty-title { font-size: 16px; font-weight: 700; color: var(--text); margin-bottom: 8px; }
.empty-desc { font-size: 13.5px; line-height: 1.6; max-width: 380px; margin: 0 auto; }
.empty-desc code {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 2px 6px;
    font-size: 12px;
    color: var(--accent);
}

/* ─── Divider ─── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, var(--border) 50%, transparent 100%);
    margin: 28px 0;
}

/* ─── Animations ─── */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ─── Scrollbar ─── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-light); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #4a4f68; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="header-wrap">
    <div class="header-inner">
        <div class="header-icon">🎓</div>
        <div>
            <p class="header-title">Sadat Academy Assistant</p>
            <p class="header-sub">University Regulations & Academic Policies</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<h3 style='margin-bottom:4px;'>⚙️ System</h3>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:13px; color:#8a92a8; margin-bottom:20px;'>"
        "Rebuild the knowledge base when new documents are added.</p>",
        unsafe_allow_html=True,
    )

    if st.button("🔄 Rebuild Index", use_container_width=True):
        with st.spinner("Processing documents..."):
            docs = load_pdfs()
            chunks = split_documents(docs)
            build_vector_store(chunks)
        st.success("✅ Index updated")
        st.balloons()

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:12px; color:#5a6178; text-align:center;'>Sadat Academy Assistant v1.1</p>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Load Vector Store
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_vector_store():
    return load_vector_store()

try:
    vector_store = get_vector_store()
    store_ready = True
except FileNotFoundError:
    store_ready = False

# ---------------------------------------------------------------------------
# Main Content
# ---------------------------------------------------------------------------
if not store_ready:
    st.markdown("""
    <div class="empty-wrap">
        <div class="empty-icon">📂</div>
        <p class="empty-title">Index Not Ready</p>
        <p class="empty-desc">Place your academic regulation PDFs in the <code>data/</code> folder,
        then click <strong>Rebuild Index</strong> from the sidebar.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Breadcrumb
st.markdown("""
<div class="breadcrumb">
    Home &nbsp;›&nbsp; Student Services &nbsp;›&nbsp; <span class="current">Sadat Academy Assistant</span>
</div>
""", unsafe_allow_html=True)

# Welcome
st.markdown('<p class="welcome-title">How can I help you today?</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="welcome-sub">Ask about graduation requirements, course prerequisites, GPA policies, '
    'or any academic inquiry. I\'ll search the official handbook and cite the exact source.</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ---------------------------------------------------------------------------
# Example Chips
# ---------------------------------------------------------------------------
EXAMPLES = [
    "What are the graduation requirements?",
    "Prerequisites for CS 3301?",
    "How many credit hours are required?",
    "What is the minimum GPA?",
    "Specialization tracks?",
    "Transfer credit policy?",
]

if not st.session_state.messages:
    st.markdown(
        '<p style="font-size:12.5px; color:#8a92a8; margin-bottom:10px;">Try asking:</p>',
        unsafe_allow_html=True,
    )
    for row_start in range(0, len(EXAMPLES), 3):
        cols = st.columns(3)
        for i in range(3):
            idx = row_start + i
            if idx < len(EXAMPLES):
                with cols[i]:
                    st.markdown('<div class="chip-btn">', unsafe_allow_html=True)
                    if st.button(EXAMPLES[idx], key=f"chip_{idx}", use_container_width=True):
                        st.session_state.pending_question = EXAMPLES[idx]
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Process Pending Question (from chip click)
# ---------------------------------------------------------------------------
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("🔍 Searching regulations..."):
        result = answer_question(vector_store, question)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "confidence": result["confidence"],
        "source_page": result["source_page"],
        "sources": result["sources"],
    })
    st.rerun()

# ---------------------------------------------------------------------------
# Display History
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="chat-card">
            <div class="card-header q-header">
                <div class="avatar avatar-q">Q</div>
                <span class="card-label">Student Question</span>
            </div>
            <div class="card-body q-body">{msg['content']}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        conf = msg.get("confidence", "unknown")
        if conf == "high":
            pill_class, pill_text, pill_icon = "pill-high", "High Confidence", "✓"
        elif conf == "medium":
            pill_class, pill_text, pill_icon = "pill-medium", "Medium Confidence", "~"
        elif conf == "low":
            pill_class, pill_text, pill_icon = "pill-low", "Low Confidence", "!"
        else:
            pill_class, pill_text, pill_icon = "pill-source", "Unknown", "?"

        source_page = msg.get("source_page", "—")

        st.markdown(f"""
        <div class="chat-card">
            <div class="card-header a-header">
                <div class="avatar avatar-a">A</div>
                <span class="card-label">Answer</span>
            </div>
            <div class="card-body">{msg['content']}</div>
            <div class="meta-bar">
                <span class="pill {pill_class}">{pill_icon} {pill_text}</span>
                <span class="pill pill-source">📄 Page {source_page}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if msg.get("sources"):
            with st.expander("📚 Sources & References"):
                for i, src in enumerate(msg["sources"][:3], start=1):
                    filename = src["source"].split("/")[-1].split("\\")[-1]
                    page_num = src.get("page", "—")
                    content_preview = src["content"][:280]
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-title">📄 Source {i} — {filename} · Page {page_num}</div>
                        <div class="source-text">{content_preview}...</div>
                    </div>
                    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
question = st.chat_input("Type your question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("🔍 Searching regulations..."):
        result = answer_question(vector_store, question)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "confidence": result["confidence"],
        "source_page": result["source_page"],
        "sources": result["sources"],
    })
    st.rerun()