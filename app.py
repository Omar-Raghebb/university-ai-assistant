"""
Streamlit UI for the University AI Assistant.
Run with: streamlit run app.py

NOTE: this loads the local LLM (Mistral-7B, optionally 4-bit
quantized) the first time a question is asked, so it needs a GPU.
"""

import streamlit as st

from src.document_loader import load_pdfs, split_documents
from src.vector_store import build_vector_store, load_vector_store
from src.chain import answer_question

st.set_page_config(
    page_title="المساعد الأكاديمي الذكي",
    page_icon="🎓",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Design: formal student-portal look. Quiet navy/white, no gradients or
# decoration, bordered panels, breadcrumb-style page header, structured
# Q&A record instead of chat bubbles -- the register of a real university
# system page, not a consumer chat app.
# ---------------------------------------------------------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root { --navy: #1B2A4A; --navy-deep: #14213A; --line: #D8DCE3; --bg: #FFFFFF; --panel: #F7F8FA; --ink: #24272E; --muted: #6B7280; --gold: #A9822F; --high: #1F7A4D; --medium: #A9822F; --low: #B23B2E; }
html, body, [class*="css"] { font-family: 'Tajawal', sans-serif; }
.stApp { background: var(--bg); }
.block-container { padding-top: 0 !important; max-width: 760px; }
.portal-topbar { background: var(--navy); border-bottom: 3px solid var(--gold); padding: 14px 22px; margin: -1rem -1rem 0 -1rem; display: flex; align-items: center; gap: 10px; }
.portal-topbar-crest { width: 34px; height: 34px; border: 1.5px solid rgba(255,255,255,0.5); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 16px; }
.portal-topbar-text { color: #FFFFFF; }
.portal-topbar-name { font-size: 15px; font-weight: 700; margin: 0; line-height: 1.3; }
.portal-topbar-sub { font-size: 11.5px; color: rgba(255,255,255,0.65); margin: 0; }
.portal-breadcrumb { background: var(--panel); border-bottom: 1px solid var(--line); padding: 10px 22px; margin: 0 -1rem 20px -1rem; font-size: 12.5px; color: var(--muted); }
.portal-breadcrumb b { color: var(--navy); font-weight: 700; }
.portal-page-title { font-size: 19px; font-weight: 700; color: var(--ink); margin: 4px 0 2px 0; }
.portal-page-desc { font-size: 13px; color: var(--muted); margin: 0 0 18px 0; }
[data-testid="stSidebar"] { background: var(--panel); border-left: 1px solid var(--line); }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: var(--navy) !important; font-weight: 700 !important; font-size: 15px !important; }
[data-testid="stSidebar"] .stButton>button { background: var(--navy); color: #FFFFFF !important; border: none; border-radius: 4px; font-weight: 700; font-size: 13.5px; padding: 9px 14px; width: 100%; }
[data-testid="stSidebar"] .stButton>button:hover { background: var(--navy-deep); }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] li { font-size: 13px !important; color: var(--muted) !important; }
.qa-record { border: 1px solid var(--line); border-radius: 6px; margin-bottom: 14px; overflow: hidden; }
.qa-question { background: var(--panel); padding: 10px 16px; font-size: 13.5px; color: var(--ink); font-weight: 500; border-bottom: 1px solid var(--line); direction: rtl; text-align: right; }
.qa-question-label { font-size: 10.5px; font-weight: 700; color: var(--muted); letter-spacing: 0.5px; text-transform: uppercase; display: block; margin-bottom: 3px; }
.qa-answer { padding: 14px 16px; font-size: 14px; color: var(--ink); line-height: 1.8; direction: rtl; text-align: right; }
.qa-answer-label { font-size: 10.5px; font-weight: 700; color: var(--navy); letter-spacing: 0.5px; text-transform: uppercase; display: block; margin-bottom: 6px; }
.qa-meta { padding: 8px 16px; background: #FCFCFD; border-top: 1px solid var(--line); display: flex; align-items: center; gap: 10px; direction: rtl; }
.status-tag { display: inline-block; padding: 2px 10px; border-radius: 3px; font-size: 11.5px; font-weight: 700; border: 1px solid; }
.status-high { color: var(--high); border-color: var(--high); background: rgba(31,122,77,0.06); }
.status-medium { color: var(--medium); border-color: var(--medium); background: rgba(169,130,47,0.06); }
.status-low { color: var(--low); border-color: var(--low); background: rgba(178,59,46,0.06); }
.status-unknown { color: var(--muted); border-color: var(--line); background: var(--panel); }
.ref-item { border-right: 3px solid var(--navy); background: var(--panel); padding: 10px 14px; margin-bottom: 8px; direction: rtl; text-align: right; border-radius: 3px; }
.ref-source { font-size: 12px; font-weight: 700; color: var(--navy); margin-bottom: 3px; }
.ref-text { font-size: 12.5px; color: var(--muted); line-height: 1.6; }
[data-testid="stChatInput"] textarea { font-family: 'Tajawal', sans-serif !important; direction: rtl; border: 1px solid var(--line) !important; border-radius: 6px !important; }
.portal-empty { border: 1px dashed var(--line); border-radius: 6px; padding: 30px 20px; text-align: center; color: var(--muted); font-size: 13.5px; }
</style>
""", unsafe_allow_html=True)

# ---- Top bar + breadcrumb ----
st.markdown("""
<div class="portal-topbar">
    <div class="portal-topbar-crest">🎓</div>
    <div class="portal-topbar-text">
        <p class="portal-topbar-name">أكاديمية السادات للعلوم الإدارية</p>
        <p class="portal-topbar-sub">بوابة الخدمات الطلابية</p>
    </div>
</div>
<div class="portal-breadcrumb">الرئيسية &nbsp;›&nbsp; الخدمات الطلابية &nbsp;›&nbsp; <b>المساعد الأكاديمي الذكي</b></div>
""", unsafe_allow_html=True)

st.markdown('<p class="portal-page-title">المساعد الأكاديمي الذكي</p>', unsafe_allow_html=True)
st.markdown('<p class="portal-page-desc">اسأل عن اللوائح الدراسية ومتطلبات التخرج، وستحصل على إجابة موثّقة بالمصدر والصفحة.</p>', unsafe_allow_html=True)

# ---- Sidebar ----
with st.sidebar:
    st.markdown("## إعداد الفهرس")

    if st.button("بناء / إعادة بناء الفهرس"):
        with st.spinner("جاري تحميل ومعالجة المستندات..."):
            docs = load_pdfs()
            chunks = split_documents(docs)
            build_vector_store(chunks)
        st.success("تم بناء الفهرس بنجاح.")

# ---- Load vector store ----
@st.cache_resource
def get_vector_store():
    return load_vector_store()

try:
    vector_store = get_vector_store()
    store_ready = True
except FileNotFoundError:
    store_ready = False
    st.markdown(
        '<div class="portal-empty">لا يوجد فهرس جاهز بعد. ضع ملفات PDF في مجلد '
        '<code>data/</code> ثم اضغط "بناء / إعادة بناء الفهرس" من القائمة الجانبية.</div>',
        unsafe_allow_html=True,
    )

# ---- History as formal Q&A records ----
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="qa-record">
            <div class="qa-question"><span class="qa-question-label">السؤال</span>{msg['content']}</div>
        """, unsafe_allow_html=True)
    else:
        pill_class = {"high": "status-high", "medium": "status-medium", "low": "status-low"}.get(
            msg.get("confidence"), "status-unknown"
        )
        pill_label = {"high": "ثقة عالية", "medium": "ثقة متوسطة", "low": "ثقة منخفضة"}.get(
            msg.get("confidence"), "غير محددة"
        )
        st.markdown(f"""
            <div class="qa-answer"><span class="qa-answer-label">الإجابة</span>{msg['content']}</div>
            <div class="qa-meta">
                <span class="status-tag {pill_class}">{pill_label}</span>
                <span style="font-size:12px; color:#6B7280;">الصفحات المرجعية: {msg.get('source_page', '—')}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if msg.get("sources"):
            with st.expander("عرض المراجع"):
                for i, src in enumerate(msg["sources"], start=1):
                    filename = src["source"].split("/")[-1].split("\\")[-1]
                    st.markdown(f"""
                    <div class="ref-item">
                        <div class="ref-source">مرجع {i} — {filename}، صفحة {src['page']}</div>
                        <div class="ref-text">{src['content'][:220]}...</div>
                    </div>
                    """, unsafe_allow_html=True)

# ---- Input ----
if store_ready:
    question = st.chat_input("اكتب سؤالك هنا...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("جاري البحث في اللوائح وإعداد الإجابة..."):
            result = answer_question(vector_store, question)

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "confidence": result["confidence"],
            "source_page": result["source_page"],
            "sources": result["sources"],
        })
        st.rerun()