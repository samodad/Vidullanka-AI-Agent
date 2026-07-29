"""
Vidullanka PLC - AI Plant Assistant
------------------------------------
Streamlit + LangChain + Groq powered chat assistant that answers questions using the
official Standard Operating Procedures (SOPs) of the Wembiyagoda (WMB) and
Batathota (BTO) Mini Hydro Power Plants.

Run with:
    streamlit run app.py
"""

import os
import time
import streamlit as st
from dotenv import load_dotenv

from rag_engine import (
    PLANTS,
    AVAILABLE_MODELS,
    build_or_load_vectorstore,
    ask_question,
)

load_dotenv()

# ----------------------------------------------------------------------------------
# Page config + styling
# ----------------------------------------------------------------------------------

st.set_page_config(
    page_title="Vidullanka Plant AI Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* Overall page background */
.stApp {
    background: linear-gradient(180deg, #0b1e2d 0%, #0e2a3d 100%);
}

/* Default body/markdown text color across the whole app (not just chat) */
.stApp, .stApp p, .stApp li, .stApp span, .stMarkdown, .stMarkdown p {
    color: #eaf6ff;
}
h1, h2, h3, h4, h5 { color: #ffffff !important; }

/* Chat input box - force a solid white box with dark text so typed text is
   always readable, no matter how Streamlit themes the underlying component. */
div[data-testid="stChatInput"] {
    background-color: #ffffff !important;
    border: 2px solid #38bdf8 !important;
    border-radius: 12px !important;
}
div[data-testid="stChatInput"] textarea,
div[data-testid="stChatInput"] input {
    background-color: #ffffff !important;
    color: #0b1e2d !important;
    -webkit-text-fill-color: #0b1e2d !important;
    caret-color: #0b1e2d !important;
    font-weight: 500;
}
div[data-testid="stChatInput"] textarea::placeholder,
div[data-testid="stChatInput"] input::placeholder {
    color: #64748b !important;
    -webkit-text-fill-color: #64748b !important;
    opacity: 1 !important;
}
/* Send button inside the chat input */
div[data-testid="stChatInput"] button {
    background-color: #0369a1 !important;
}
div[data-testid="stChatInput"] button svg {
    fill: #ffffff !important;
}

/* Hero header */
.hero {
    background: linear-gradient(120deg, #0f766e 0%, #0369a1 60%, #0c4a6e 100%);
    border-radius: 18px;
    padding: 2rem 2.2rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    border: 1px solid rgba(255,255,255,0.08);
}
.hero h1 {
    color: #ffffff;
    font-size: 2.1rem;
    margin-bottom: 0.3rem;
    font-weight: 800;
    letter-spacing: -0.5px;
}
.hero p {
    color: #d7f4f0;
    font-size: 1.02rem;
    margin: 0;
}
.badge-row { margin-top: 0.9rem; }
.badge {
    display: inline-block;
    background: rgba(255,255,255,0.14);
    color: #eafffb;
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 0.82rem;
    margin-right: 8px;
    border: 1px solid rgba(255,255,255,0.25);
}

/* Chat bubbles */
.stChatMessage {
    border-radius: 14px !important;
    background: rgba(255,255,255,0.045) !important;
    border: 1px solid rgba(255,255,255,0.08);
}

/* ---- Force bright, high-contrast text everywhere inside a chat message ---- */
.stChatMessage, .stChatMessage p, .stChatMessage li, .stChatMessage span,
.stChatMessage div, .stChatMessage td, .stChatMessage th,
.stChatMessage h1, .stChatMessage h2, .stChatMessage h3,
.stChatMessage h4, .stChatMessage h5, .stChatMessage h6 {
    color: #f5fbff !important;
}
.stChatMessage strong, .stChatMessage b {
    color: #ffffff !important;
    font-weight: 700 !important;
}
.stChatMessage em, .stChatMessage i {
    color: #bff2ea !important;
}
.stChatMessage a {
    color: #7dd3fc !important;
}
.stChatMessage code {
    color: #ffe9a8 !important;
    background: rgba(255,255,255,0.08) !important;
    padding: 1px 5px;
    border-radius: 4px;
}
.stChatMessage hr {
    border-color: rgba(255,255,255,0.15) !important;
}

/* Tables inside answers (the SOP step tables) */
.stChatMessage table {
    color: #f5fbff !important;
    border-collapse: collapse;
    width: 100%;
}
.stChatMessage table th {
    background: rgba(56,189,248,0.18) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    padding: 8px 10px !important;
}
.stChatMessage table td {
    border: 1px solid rgba(255,255,255,0.12) !important;
    padding: 8px 10px !important;
}
.stChatMessage table tr:nth-child(even) td {
    background: rgba(255,255,255,0.035) !important;
}

/* Markdown container Streamlit wraps text in */
[data-testid="stChatMessageContent"] * {
    color: #f5fbff !important;
}
[data-testid="stChatMessageContent"] strong {
    color: #ffffff !important;
}

/* Quick action buttons */
div[data-testid="stButton"] > button {
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.15);
    background: rgba(255,255,255,0.04);
    color: #e5f6ff;
    font-weight: 500;
    transition: all 0.15s ease-in-out;
}
div[data-testid="stButton"] > button:hover {
    border-color: #38bdf8;
    color: #38bdf8;
    background: rgba(56,189,248,0.08);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0a1a26;
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* Source expander */
.source-box {
    background: rgba(255,255,255,0.05);
    border-left: 3px solid #38bdf8;
    padding: 0.6rem 0.9rem;
    border-radius: 6px;
    margin-bottom: 0.5rem;
    font-size: 0.86rem;
}

footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------------
# Hero header
# ----------------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>⚡ Vidullanka PLC — AI Plant Assistant</h1>
        <p>Instant, SOP-grounded guidance for Wembiyagoda &amp; Batathota Mini Hydro Power Plants —
        startup/shutdown steps, emergency procedures, safety rules and equipment troubleshooting,
        answered straight from the official SOP documents.</p>
        <div class="badge-row">
            <span class="badge">🏞️ Wembiyagoda MHPP</span>
            <span class="badge">⚙️ Batathota MHPP</span>
            <span class="badge">🔒 Answers grounded in official SOPs</span>
            <span class="badge">⚡ Powered by Groq</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------------
# Sidebar - configuration
# ----------------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # --- API key ---------------------------------------------------------------
    st.markdown("#### Groq API Key")
    env_key = os.getenv("GROQ_API_KEY", "")
    try:
        secrets_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        secrets_key = ""
    default_key = env_key or secrets_key

    api_key_input = st.text_input(
        "Enter your key (or set GROQ_API_KEY in .env / secrets.toml)",
        value="",
        type="password",
        placeholder="gsk_...",
        help="Get a free key at https://console.groq.com/keys",
    )
    groq_api_key = api_key_input or default_key

    if groq_api_key:
        st.success("API key loaded ✅", icon="✅")
    else:
        st.warning("No API key yet — enter one above to start chatting.", icon="⚠️")

    st.markdown("---")

    # --- Model selection ---------------------------------------------------------
    st.markdown("#### Model")
    model_label = st.selectbox("Groq model", list(AVAILABLE_MODELS.keys()), index=0)
    model_name = AVAILABLE_MODELS[model_label]

    st.markdown("---")

    # --- Plant selection ---------------------------------------------------------
    st.markdown("#### Plant scope")
    plant_choice = st.radio(
        "Search SOPs from:",
        options=["BOTH", "WMB", "BTO"],
        format_func=lambda code: {
            "BOTH": "🌐 Both plants",
            "WMB": f"{PLANTS['WMB']['emoji']} Wembiyagoda only",
            "BTO": f"{PLANTS['BTO']['emoji']} Batathota only",
        }[code],
    )

    st.markdown("---")
    top_k = st.slider("How many SOP passages to retrieve", min_value=3, max_value=10, value=5)
    show_sources = st.checkbox("Show source passages under each answer", value=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        rebuild = st.button("🔄 Rebuild index", use_container_width=True)
    with col_b:
        clear_chat = st.button("🗑️ Clear chat", use_container_width=True)

    st.markdown("---")
    st.caption(
        "Built with Streamlit, LangChain, FAISS and Groq. "
        "Answers are generated from the plants' official SOP PDFs only."
    )

if clear_chat:
    st.session_state.messages = []

# ----------------------------------------------------------------------------------
# Build / load the vector store (cached so it only happens once)
# ----------------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _get_vectorstore(_force: bool = False):
    return build_or_load_vectorstore(force_rebuild=_force)

if rebuild:
    _get_vectorstore.clear()

with st.spinner("🔧 Indexing SOP documents (first run only, may take a minute)..."):
    try:
        vectorstore = _get_vectorstore(rebuild)
        index_error = None
    except Exception as e:
        vectorstore = None
        index_error = str(e)

if index_error:
    st.error(
        f"Couldn't build/load the SOP index: {index_error}\n\n"
        "Make sure WMB_SOP.pdf and BTO_SOP.pdf are inside the `data/` folder."
    )
    st.stop()

# ----------------------------------------------------------------------------------
# Quick action buttons
# ----------------------------------------------------------------------------------

st.markdown("##### 💡 Quick questions")
quick_cols = st.columns(4)
quick_questions = [
    ("🚀 Startup procedure", "What is the standard startup procedure for the machine?"),
    ("🛑 Emergency stop", "What is the emergency stop procedure?"),
    ("🔥 Electrical fire", "What should I do in case of an electrical fire?"),
    ("💧 Water leakage in turbine", "What is the procedure for abnormal water leakage in the turbine?"),
]
quick_clicked = None
for col, (label, question) in zip(quick_cols, quick_questions):
    if col.button(label, use_container_width=True):
        quick_clicked = question

st.markdown("")

# ----------------------------------------------------------------------------------
# Chat state
# ----------------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of dicts: {role, content, sources}

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="⚡" if msg["role"] == "assistant" else None):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources") and show_sources:
            with st.expander(f"📄 Sources ({len(msg['sources'])} passages)"):
                for s in msg["sources"]:
                    st.markdown(
                        f'<div class="source-box"><b>{s.plant_name}</b> · Page {s.page}<br>{s.text[:400]}...</div>',
                        unsafe_allow_html=True,
                    )

# ----------------------------------------------------------------------------------
# Handle new input (typed or quick-action)
# ----------------------------------------------------------------------------------

user_input = st.chat_input("Ask about startup steps, safety procedures, troubleshooting...")
final_question = quick_clicked or user_input

if final_question:
    if not groq_api_key:
        st.error("⚠️ Please enter your Groq API key in the sidebar first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": final_question, "sources": []})
    with st.chat_message("user"):
        st.markdown(final_question)

    # Build short chat history for the LLM (role, content) tuples
    history_pairs = [(m["role"], m["content"]) for m in st.session_state.messages[:-1]]

    with st.chat_message("assistant", avatar="⚡"):
        placeholder = st.empty()
        placeholder.markdown("🤔 Checking the SOP documents...")
        try:
            result = ask_question(
                vectorstore=vectorstore,
                api_key=groq_api_key,
                model_name=model_name,
                question=final_question,
                chat_history=history_pairs,
                plant_filter=plant_choice,
                k=top_k,
            )
            placeholder.markdown(result.answer)
            if result.sources and show_sources:
                with st.expander(f"📄 Sources ({len(result.sources)} passages)"):
                    for s in result.sources:
                        st.markdown(
                            f'<div class="source-box"><b>{s.plant_name}</b> · Page {s.page}<br>{s.text[:400]}...</div>',
                            unsafe_allow_html=True,
                        )
            st.session_state.messages.append(
                {"role": "assistant", "content": result.answer, "sources": result.sources}
            )
        except Exception as e:
            err_msg = f"❌ Something went wrong calling Groq: {e}"
            placeholder.markdown(err_msg)
            st.session_state.messages.append({"role": "assistant", "content": err_msg, "sources": []})

# ----------------------------------------------------------------------------------
# Empty state
# ----------------------------------------------------------------------------------

if not st.session_state.messages:
    st.info(
        "👋 Ask me anything about the Wembiyagoda or Batathota plant SOPs — e.g. "
        "*\"What's the MIV opening procedure?\"* or *\"Who approves a plant shutdown?\"*",
        icon="💬",
    )