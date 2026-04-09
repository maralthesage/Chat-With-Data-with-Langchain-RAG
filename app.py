
import streamlit as st
from rag_engine.loader import load_and_prepare_csv
from rag_engine.analyzer import OllamaCsvRAG
from config import rechnung_path

CACHE_VERSION = "trace-v2"

# ✅ Title
st.title("📊 CSV RAG Chatbot (Ollama-based)")

# ✅ Load your CSV ONCE at app startup
@st.cache_resource
def load_rag(cache_version: str):
    df = load_and_prepare_csv(rechnung_path)
    rag = OllamaCsvRAG(df)
    return rag

rag = load_rag(CACHE_VERSION)

st.success("RAG engine initialized and CSV loaded.")


@st.cache_data(show_spinner=False)
def ask_cached(question: str, cache_version: str):
    return ask_with_trace_compat(question)


def ask_with_trace_compat(question: str, progress_callback=None):
    if hasattr(rag, "ask_with_trace"):
        return rag.ask_with_trace(question, progress_callback=progress_callback)

    answer = rag.ask(question, progress_callback=progress_callback)
    if hasattr(rag, "get_last_trace"):
        trace = rag.get_last_trace()
    else:
        trace = {
            "route": "legacy",
            "schema_context": "",
            "literal_candidates": [],
            "hybrid_candidates": [],
            "deterministic_match": {},
            "used_fallback": False,
            "llm_output": "",
            "code": "",
            "result_preview": "",
        }
    return answer, trace

# 📝 Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ✅ Chat input
question = st.chat_input("Ask something about the CSV...")

if question:
    # Display user message
    with st.chat_message("user"):
        st.write(question)

    # Get answer from RAG
    # Display assistant message
    with st.chat_message("assistant"):
        status_box = st.status("Working on your question...", expanded=True)
        status_placeholder = st.empty()
        status_messages = []

        def update_status(message: str):
            status_messages.append(message)
            status_placeholder.write("\n".join(f"- {item}" for item in status_messages))

        if question.strip() in getattr(rag, "_answer_cache", {}):
            update_status("Using cached answer.")
            answer, trace = ask_cached(question, CACHE_VERSION)
        else:
            update_status("Preparing analysis.")
            answer, trace = ask_with_trace_compat(question, progress_callback=update_status)

        status_box.update(label="Answer ready", state="complete", expanded=False)
        st.write(answer)
        with st.expander("Analysis details"):
            st.write(f"Route: `{trace.get('route', '')}`")
            if trace.get("deterministic_match"):
                st.write("Deterministic match:")
                st.json(trace["deterministic_match"])
            if trace.get("literal_candidates"):
                st.write("Literal candidates:")
                st.json(trace["literal_candidates"])
            if trace.get("hybrid_candidates"):
                st.write("Hybrid product candidates:")
                st.json(trace["hybrid_candidates"])
            if trace.get("schema_context"):
                st.write("Schema context:")
                st.code(trace["schema_context"], language="text")
            if trace.get("code"):
                st.write("Executed code:")
                st.code(trace["code"], language="python")
            if trace.get("llm_output"):
                st.write("Raw model output:")
                st.code(trace["llm_output"], language="text")
            if trace.get("result_preview"):
                st.write("Result preview:")
                st.code(trace["result_preview"], language="text")

    # Streamlit session
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": answer})
