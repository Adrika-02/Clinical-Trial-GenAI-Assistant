"""Page 2 — Chat with Trial Data (Interaction 1: Chatbot). Routes natural-
language questions to one of the three LangChain agents and renders the
answer with an auto-generated chart and the SQL used, for transparency."""
import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_DIR))

from utils import APP_TITLE, page_header, render_chart

st.set_page_config(page_title=f"Chat — {APP_TITLE}", page_icon="💬", layout="wide")
page_header("💬 Chat with Trial Data", "Ask any question in plain English — no SQL or coding required.")

SUGGESTED_QUESTIONS = [
    "Which patients had the best outcomes?",
    "Compare AE rates between arms",
    "What drove HbA1c improvement?",
    "Show me dropout patterns",
]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def _handle_question(question: str):
    from src.agents.router import route_question

    with st.spinner("Routing to the right agent and querying live data..."):
        try:
            result = route_question(question)
        except Exception as e:
            result = {"agent": "error", "answer": f"Something went wrong calling the LLM: {e}"}
    st.session_state.chat_history.append({"role": "user", "content": question})
    st.session_state.chat_history.append({"role": "assistant", "result": result})


st.write("**Try a suggested question:**")
cols = st.columns(len(SUGGESTED_QUESTIONS))
for col, q in zip(cols, SUGGESTED_QUESTIONS):
    if col.button(q, use_container_width=True):
        _handle_question(q)

st.divider()

AGENT_LABELS = {
    "data_analysis": "📊 Data Analysis Agent",
    "clinical_notes": "📝 Clinical Notes Agent",
    "insight": "💡 Insight Generation Agent",
    "error": "⚠️ Error",
}

for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        result = msg["result"]
        with st.chat_message("assistant"):
            st.caption(AGENT_LABELS.get(result.get("agent"), "Assistant"))
            st.write(result["answer"])

            if result.get("agent") == "data_analysis" and result.get("sql"):
                with st.expander("Show SQL query used"):
                    st.code(result["sql"], language="sql")
                if result.get("dataframe") is not None:
                    render_chart(result["dataframe"], result.get("chart_spec"))

            if result.get("agent") == "clinical_notes" and result.get("patient_ids_mentioned"):
                st.caption(f"Patients referenced: {', '.join(str(p) for p in result['patient_ids_mentioned'])}")

question = st.chat_input("Ask a question about the trial data...")
if question:
    _handle_question(question)
    st.rerun()
