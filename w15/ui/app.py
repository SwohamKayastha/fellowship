"""
Streamlit frontend for the AI Assistant.
ChatGPT/Claude-style minimal UI. Connects to FastAPI backend at API_URL.
"""

import json
import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
TIMEOUT = 360

st.set_page_config(
    page_title="AI Assistant",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — ChatGPT/Claude-style
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
    /* System font stack */
    html, body, [class*="css"], .stMarkdown, .stChatMessage {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Inter",
                     Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Constrain main content — center chat like ChatGPT */
    .main .block-container {
        max-width: 780px;
        padding-top: 2.5rem;
        padding-bottom: 8rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }

    /* Chat message styling — minimal, no heavy boxes */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 0.75rem 0 !important;
        margin: 0 !important;
    }

    /* Chat input — sticky bottom, rounded */
    [data-testid="stChatInput"] {
        max-width: 780px;
        margin: 0 auto;
    }
    [data-testid="stChatInput"] textarea {
        border-radius: 14px !important;
        border: 1px solid rgba(120, 120, 120, 0.25) !important;
        padding: 14px 18px !important;
        font-size: 15px !important;
        line-height: 1.5 !important;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer, [data-testid="stToolbar"] {visibility: hidden;}

    /* Sidebar polish */
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(120, 120, 120, 0.15);
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* Sidebar section labels */
    .sidebar-label {
        text-transform: uppercase;
        font-size: 0.7rem;
        letter-spacing: 0.08em;
        color: rgba(120, 120, 120, 0.85);
        margin-bottom: 0.4rem;
        margin-top: 0.8rem;
        font-weight: 600;
    }

    /* Heading */
    h1 {
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.25rem !important;
    }
    h2, h3 {
        font-weight: 600 !important;
    }

    /* Message paragraphs — better line height */
    [data-testid="stChatMessage"] p {
        line-height: 1.65 !important;
        font-size: 15px !important;
        margin-bottom: 0.6rem !important;
    }

    /* Metadata captions — subtle */
    [data-testid="stCaptionContainer"] {
        color: rgba(140, 140, 140, 0.9) !important;
        font-size: 0.72rem !important;
        margin-top: 0.25rem !important;
    }

    /* Example prompt buttons */
    .stButton > button {
        border-radius: 12px !important;
        border: 1px solid rgba(120, 120, 120, 0.2) !important;
        background: transparent !important;
        text-align: left !important;
        font-weight: 400 !important;
        padding: 12px 16px !important;
        transition: all 0.15s !important;
    }
    .stButton > button:hover {
        background: rgba(120, 120, 120, 0.06) !important;
        border-color: rgba(120, 120, 120, 0.35) !important;
    }

    /* Expander polish */
    [data-testid="stExpander"] {
        border: 1px solid rgba(120, 120, 120, 0.15) !important;
        border-radius: 10px !important;
        margin-top: 0.5rem !important;
    }
    [data-testid="stExpander"] summary {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }

    /* Code blocks */
    code {
        background: rgba(120, 120, 120, 0.12) !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
        font-size: 0.85em !important;
    }

    /* Status dot */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
        vertical-align: middle;
    }
    .status-ok    { background: #22c55e; box-shadow: 0 0 6px rgba(34, 197, 94, 0.5); }
    .status-fail  { background: #ef4444; }

    /* Empty-state heading */
    .empty-hero {
        text-align: center;
        margin-top: 4rem;
        margin-bottom: 2rem;
    }
    .empty-hero h2 {
        font-size: 1.8rem !important;
        font-weight: 600;
        color: rgba(140, 140, 140, 0.9);
        margin-bottom: 0.5rem;
    }
    .empty-hero p {
        color: rgba(140, 140, 140, 0.7);
        font-size: 0.95rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Backend health probe
# ---------------------------------------------------------------------------

@st.cache_data(ttl=15)
def probe_health() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ◈ AI Assistant")
    st.caption("RAG · Tools · Fallback")

    st.markdown("")

    if st.button("＋ New chat", use_container_width=True, key="new_chat_btn"):
        st.session_state.messages = []
        st.rerun()

    # ── Settings ────────────────────────────────────────────────────────
    with st.expander("Settings", expanded=True):
        use_rag = st.toggle(
            "Use knowledge base",
            value=True,
            help="Retrieve relevant chunks from ChromaDB before answering.",
        )
        structured = st.toggle(
            "JSON response",
            value=False,
            help="Force the model to reply with valid JSON.",
        )
        temperature = st.slider(
            "Temperature", 0.0, 1.0, 0.1, 0.05,
            help="Lower = deterministic. Higher = creative.",
        )
        top_p = st.slider("Top-P", 0.0, 1.0, 0.9, 0.05)

    # ── Knowledge Base ──────────────────────────────────────────────────
    with st.expander("Knowledge base", expanded=False):
        ingest_text = st.text_area(
            "Paste text",
            height=120,
            placeholder="Paste any document…",
            label_visibility="collapsed",
        )
        source_label = st.text_input(
            "Source",
            placeholder="Source label (optional)",
            label_visibility="collapsed",
        )

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Add text", use_container_width=True, disabled=not ingest_text.strip()):
                with st.spinner("Ingesting…"):
                    try:
                        r = requests.post(
                            f"{API_URL}/ingest",
                            json={
                                "text": ingest_text,
                                "metadata": {"source": source_label or "manual"},
                            },
                            timeout=30,
                        )
                        r.raise_for_status()
                        st.success(f"✓ {r.json()['chunks_added']} chunks added")
                    except Exception as exc:
                        st.error(f"Failed: {exc}")

        uploaded = st.file_uploader(
            "Upload .txt",
            type=["txt"],
            label_visibility="collapsed",
        )
        with col_b:
            if st.button("Add file", use_container_width=True, disabled=uploaded is None):
                with st.spinner("Ingesting…"):
                    try:
                        text = uploaded.read().decode("utf-8", errors="ignore")
                        r = requests.post(
                            f"{API_URL}/ingest",
                            json={"text": text, "metadata": {"source": uploaded.name}},
                            timeout=30,
                        )
                        r.raise_for_status()
                        st.success(f"✓ {r.json()['chunks_added']} chunks added")
                    except Exception as exc:
                        st.error(f"Failed: {exc}")

    # ── Status footer ───────────────────────────────────────────────────
    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    health = probe_health()
    if health:
        st.markdown(
            f"<span class='status-dot status-ok'></span>"
            f"<span style='font-size:0.75rem; color:rgba(140,140,140,0.9);'>Backend online</span>",
            unsafe_allow_html=True,
        )
        st.caption(f"Primary  ·  `{health['primary_model']}`")
        st.caption(f"Fallback ·  `{health['fallback_model']}`")
    else:
        st.markdown(
            "<span class='status-dot status-fail'></span>"
            "<span style='font-size:0.75rem; color:rgba(140,140,140,0.9);'>Backend unreachable</span>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# Empty state — example prompts
if not st.session_state.messages:
    st.markdown(
        """
        <div class='empty-hero'>
            <h2>How can I help you today?</h2>
            <p>Ask a question, do math, or query the knowledge base.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    examples = [
        ("Explain retrieval-augmented generation", "Explain how RAG works in this app."),
        ("Do a calculation", "What is sqrt(144) + 2^10?"),
        ("Get the current time", "What is the current date and time?"),
        ("Summarize the knowledge base", "What documents are in the knowledge base?"),
    ]

    cols = st.columns(2)
    for i, (label, prompt_text) in enumerate(examples):
        with cols[i % 2]:
            if st.button(label, use_container_width=True, key=f"ex_{i}"):
                st.session_state.pending_prompt = prompt_text
                st.rerun()

# Render existing conversation
for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        # Restore metadata for assistant messages
        if msg["role"] == "assistant" and "meta" in msg:
            m = msg["meta"]
            tags = [f"`{m['model_used']}`"]
            if m.get("cached"):
                tags.append("⚡ cached")
            st.caption("  ·  ".join(tags))
            if m.get("sources"):
                with st.expander(f"Sources · {len(m['sources'])}"):
                    for i, src in enumerate(m["sources"], 1):
                        preview = src["content"][:300]
                        if len(src["content"]) > 300:
                            preview += "…"
                        st.markdown(f"**{i}.** {preview}")
                        if src.get("metadata"):
                            st.caption(str(src["metadata"]))
            if m.get("tool_calls"):
                with st.expander(f"Tool calls · {len(m['tool_calls'])}"):
                    st.json(m["tool_calls"])


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

prompt = st.session_state.pop("pending_prompt", None) or st.chat_input("Message AI Assistant…")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking…"):
            endpoint = "/chat/json" if structured else "/chat"
            api_messages = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]

            try:
                resp = requests.post(
                    f"{API_URL}{endpoint}",
                    json={
                        "messages": api_messages,
                        "use_rag": use_rag,
                        "structured_output": structured,
                        "temperature": temperature,
                        "top_p": top_p,
                    },
                    timeout=TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.exceptions.Timeout:
                st.error("Request timed out.")
                st.stop()
            except requests.exceptions.HTTPError as exc:
                try:
                    detail = exc.response.json().get("detail", str(exc))
                except Exception:
                    detail = str(exc)
                st.error(f"API error · {detail}")
                st.stop()
            except Exception as exc:
                st.error(f"Error · {exc}")
                st.stop()

        response_text = data["response"]

        # Render answer
        if structured:
            try:
                st.json(json.loads(response_text))
            except Exception:
                st.markdown(response_text)
        else:
            st.markdown(response_text)

        # Metadata row
        tags = [f"`{data['model_used']}`"]
        if data.get("cached"):
            tags.append("⚡ cached")
        st.caption("  ·  ".join(tags))

        # Sources
        if data.get("sources"):
            with st.expander(f"Sources · {len(data['sources'])}"):
                for i, src in enumerate(data["sources"], 1):
                    preview = src["content"][:300]
                    if len(src["content"]) > 300:
                        preview += "…"
                    st.markdown(f"**{i}.** {preview}")
                    if src.get("metadata"):
                        st.caption(str(src["metadata"]))

        # Tool calls
        if data.get("tool_calls"):
            with st.expander(f"Tool calls · {len(data['tool_calls'])}"):
                st.json(data["tool_calls"])

        # Persist message + metadata for re-render
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "meta": {
                "model_used": data["model_used"],
                "cached": data.get("cached", False),
                "sources": data.get("sources", []),
                "tool_calls": data.get("tool_calls", []),
            },
        })
