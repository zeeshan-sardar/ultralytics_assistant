"""Streamlit chat interface for the Ultralytics Code Assistant."""

import time

import streamlit as st
from dotenv import load_dotenv

import config

load_dotenv()

st.set_page_config(
    page_title="Ultralytics Code Assistant",
    page_icon="🦾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .source-card {
        background: #1e1e2e;
        border-left: 3px solid #0078d4;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }
    .similarity-badge {
        background: #0078d4;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def render_source_cards(source_list: list[dict]):
    for src in source_list:
        st.markdown(f"""
<div class="source-card">
    <b>{src['name']}</b>&nbsp;
    <span class="similarity-badge">score: {src['score']:.3f}</span><br>
    <code>{src['file_path']}</code> · {src['chunk_type']}
</div>
""", unsafe_allow_html=True)
        with st.expander(f"View source: `{src['name']}`"):
            st.code(src["source"], language="python")


@st.cache_resource(show_spinner="Loading embedding model…")
def load_retriever():
    from retriever import Retriever
    return Retriever()


if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_query_time" not in st.session_state:
    st.session_state.last_query_time = 0.0
if "processing" not in st.session_state:
    st.session_state.processing = False
if "prefill_question" not in st.session_state:
    st.session_state.prefill_question = ""

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    top_k = st.slider("Code chunks to retrieve per query", min_value=2, max_value=10, value=6)
    show_sources = st.checkbox("Show retrieved source code", value=True)

    st.divider()
    st.markdown("### 💡 Example Questions")

    for q in [
        "How does YOLO handle non-maximum suppression (NMS)?",
        "How do I train a custom YOLOv8 model on my dataset?",
        "What does the BasePredictor class do?",
        "How does the data augmentation pipeline work?",
        "How can I export a YOLO model to ONNX format?",
    ]:
        if st.button(q, use_container_width=True, key=f"ex_{q[:20]}"):
            if not st.session_state.processing:
                st.session_state.prefill_question = q

    st.divider()
    missing = config.validate()
    for key in missing:
        st.error(f"⚠️ `{key}` not set in .env")

st.markdown("# 🦾 Ultralytics Code Assistant")
st.caption("Ask questions about Ultralytics YOLO to get source code grounded answers.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources") and show_sources:
            with st.expander("📎 Retrieved source chunks", expanded=False):
                render_source_cards(message["sources"])

user_question = st.chat_input("Ask about Ultralytics YOLO source code…")
if not user_question and st.session_state.prefill_question:
    user_question = st.session_state.prefill_question
    st.session_state.prefill_question = ""

if user_question and not st.session_state.processing:
    st.session_state.processing = True

    MIN_GAP_SECONDS = 6
    elapsed = time.time() - st.session_state.last_query_time
    if elapsed < MIN_GAP_SECONDS and st.session_state.last_query_time > 0:
        with st.spinner(f"⏳ Waiting {MIN_GAP_SECONDS - elapsed:.1f}s between queries…"):
            time.sleep(MIN_GAP_SECONDS - elapsed)

    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.chat_message("assistant"):
        with st.spinner("Searching codebase…"):
            try:
                retriever = load_retriever()
                search_results = retriever.search(user_question, top_k=top_k)
                context_text = retriever.format_results_as_context(search_results)
            except Exception as error:
                st.error(f"Retrieval failed: {error}")
                st.session_state.processing = False
                st.stop()

        source_data = [
            {"name": r.name, "file_path": r.file_path, "chunk_type": r.chunk_type,
             "source": r.source, "score": r.score}
            for r in search_results
        ]

        if show_sources and source_data:
            with st.expander("📎 Retrieved source chunks", expanded=False):
                render_source_cards(source_data)

        from generator import stream_answer

        answer_placeholder = st.empty()
        full_answer = ""
        for token in stream_answer(user_question, context_text):
            full_answer += token
            answer_placeholder.markdown(full_answer + "▌")
        answer_placeholder.markdown(full_answer)

    st.session_state.last_query_time = time.time()
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer,
        "sources": source_data,
    })
    st.session_state.processing = False
