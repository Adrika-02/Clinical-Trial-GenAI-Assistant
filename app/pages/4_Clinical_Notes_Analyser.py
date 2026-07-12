"""Page 4 — Clinical Notes Analyser (Interaction 2 continued).
Mode A: paste a note -> AE classification + NER + SHAP explanation + Claude
recommended action. Mode B: search existing notes by patient/week/keyword.
"""
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_DIR))

# NLP/SHAP (spaCy, scikit-learn, SHAP) are imported before pandas/utils are
# touched: on this dev machine's mixed conda-forge/pip toolchain, loading
# pandas/sqlite3 first and spaCy's native extensions second reliably
# segfaults the interpreter (no Python traceback, pure native crash) —
# almost certainly a local OpenMP/BLAS runtime clash rather than an app bug,
# but importing NLP/SHAP first avoids it entirely.
from src.nlp.analyze_note import analyze_note
from src.explainability.shap_explain import AEShapExplainer
from src.agents.bedrock_llm import get_llm

import pandas as pd
import plotly.express as px
import streamlit as st

from utils import APP_TITLE, page_header, run_query

st.set_page_config(page_title=f"Notes Analyser — {APP_TITLE}", page_icon="📝", layout="wide")
page_header("📝 Clinical Notes Analyser")


@st.cache_resource
def _get_shap_explainer():
    return AEShapExplainer()


mode = st.radio("Mode", ["Analyse a new note", "Search existing notes"], horizontal=True)

if mode == "Analyse a new note":
    default_note = "Moderate dizziness reported. Dose reduced from 20mg to 10mg. Follow-up scheduled in 2 weeks."
    note_text = st.text_area("Paste a clinical note", value=default_note, height=120)

    if st.button("Analyse note", type="primary"):
        with st.spinner("Running NLP pipeline..."):
            result = analyze_note(note_text)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Classification")
            severity_color = {"No AE": "green", "Mild AE": "orange", "Severe AE": "red"}.get(result["prediction"], "gray")
            st.markdown(f"**Predicted class:** :{severity_color}[{result['prediction']}]")
            st.metric("Confidence", f"{result['confidence']*100:.1f}%")
            st.write("Class probabilities:")
            probs_df = pd.DataFrame(
                {"class": list(result["class_probabilities"].keys()), "probability": list(result["class_probabilities"].values())}
            )
            fig_probs = px.bar(probs_df, x="class", y="probability", text_auto=".1%")
            fig_probs.update_layout(height=280, yaxis_range=[0, 1])
            st.plotly_chart(fig_probs, use_container_width=True)

        with col2:
            st.subheader("Extracted Medical Entities")
            entities = result["entities"]
            st.write(f"**Symptoms:** {', '.join(entities['symptoms']) or '—'}")
            st.write(f"**Severity descriptors:** {', '.join(entities['severity_descriptors']) or '—'}")
            st.write(f"**Clinical actions:** {', '.join(entities['clinical_actions']) or '—'}")
            st.write(f"**Lab values:** {entities['lab_values'] or '—'}")
            st.write(f"**Blood pressure:** {entities['blood_pressure'] or '—'}")
            st.write(f"**Dosages (mg):** {entities['dosages_mg'] or '—'}")
            st.metric("Severity score (0-3)", result["severity_score_0_3"])

        st.divider()
        st.subheader("SHAP Explanation")
        try:
            with st.spinner("Computing SHAP explanation..."):
                explainer = _get_shap_explainer()
                narrative = explainer.business_narrative(result["clean_text"], class_label=result["prediction"])
                st.write(narrative)

                local = explainer.local_explanation(result["clean_text"], class_label=result["prediction"])
                contributions = local["contributions"][:12]
                if contributions:
                    contrib_df = pd.DataFrame(contributions).sort_values("shap_value")
                    contrib_df["direction"] = contrib_df["shap_value"].apply(
                        lambda v: f"Pushes toward {result['prediction']}" if v > 0 else "Pushes away"
                    )
                    fig = px.bar(
                        contrib_df, x="shap_value", y="feature", orientation="h", color="direction",
                        color_discrete_map={f"Pushes toward {result['prediction']}": "#C44E52", "Pushes away": "#4C72B0"},
                        title=f"Word-level SHAP contributions — predicted {result['prediction']}",
                    )
                    fig.update_layout(height=380, yaxis_title="", xaxis_title="SHAP value (impact on prediction)")
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"SHAP explanation unavailable: {e}")

        st.divider()
        st.subheader("Recommended Action")
        with st.spinner("Generating recommendation..."):
            prompt = (
                f"A clinical trial note was classified as '{result['prediction']}' "
                f"(confidence {result['confidence']*100:.0f}%). "
                f"Extracted entities: symptoms={entities['symptoms']}, "
                f"severity descriptors={entities['severity_descriptors']}, "
                f"clinical actions already noted={entities['clinical_actions']}.\n"
                f"Note text: \"{note_text}\"\n\n"
                f"As a clinical safety reviewer, give a single concise recommended next action "
                f"for the site investigator (1-2 sentences)."
            )
            recommendation = get_llm().invoke(prompt).content
        st.write(recommendation)

else:
    st.subheader("Search Existing Notes")
    c1, c2, c3 = st.columns(3)
    patient_id = c1.text_input("Patient ID (optional)")
    week = c2.selectbox("Visit week", ["Any", 0, 2, 4, 8, 12, 24])
    keyword = c3.text_input("Keyword (optional)")

    query = "SELECT patient_id, visit_week, note_date, note_text, true_ae_class FROM clinical_notes WHERE 1=1"
    params = []
    if patient_id.strip().isdigit():
        query += " AND patient_id = ?"
        params.append(int(patient_id.strip()))
    if week != "Any":
        query += " AND visit_week = ?"
        params.append(week)
    if keyword.strip():
        query += " AND note_text LIKE ?"
        params.append(f"%{keyword.strip()}%")
    query += " LIMIT 100"

    results = run_query(query, tuple(params))
    st.caption(f"{len(results)} notes found (showing up to 100)")
    st.dataframe(results, use_container_width=True, hide_index=True)
