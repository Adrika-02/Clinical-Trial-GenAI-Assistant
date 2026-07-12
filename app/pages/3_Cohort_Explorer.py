"""Page 3 — Cohort Explorer (Interaction 2: Form-based). Sidebar filters
drive an instant cohort analysis: size, lab trends, AE profile, statistical
comparison against the full trial population, and an auto-generated
AI cohort summary (Groq-backed by default; swappable to Claude via Bedrock/Anthropic — see src/agents/bedrock_llm.py)."""
import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_DIR))

from utils import APP_TITLE, load_patients, load_visits, load_adverse_events, page_header
from src.stats.statistical_tests import welch_t_test, confidence_interval

st.set_page_config(page_title=f"Cohort Explorer — {APP_TITLE}", page_icon="🔎", layout="wide")
page_header("🔎 Cohort Explorer", "Filter patients by criteria for an instant, statistically-grounded cohort analysis.")

patients = load_patients()
visits = load_visits()
ae = load_adverse_events()

severity_rank = {"None": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
ae_rank = ae.assign(severity_rank=ae["severity"].map(severity_rank))
max_severity = ae_rank.groupby("patient_id")["severity_rank"].max().rename("max_ae_severity_rank")
patients = patients.set_index("patient_id").join(max_severity).reset_index()
patients["max_ae_severity_rank"] = patients["max_ae_severity_rank"].fillna(0)
rank_to_label = {0: "None", 1: "Mild", 2: "Moderate", 3: "Severe"}
patients["worst_ae_severity"] = patients["max_ae_severity_rank"].map(rank_to_label)

# --- Sidebar filters ---
st.sidebar.header("Cohort Filters")
age_min, age_max = int(patients["age"].min()), int(patients["age"].max())
age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))

genders = st.sidebar.multiselect("Gender", options=sorted(patients["gender"].unique()), default=list(patients["gender"].unique()))
arms = st.sidebar.multiselect("Treatment arm", options=sorted(patients["treatment_arm"].unique()), default=list(patients["treatment_arm"].unique()))
severities = st.sidebar.multiselect(
    "Worst AE severity", options=["None", "Mild", "Moderate", "Severe"],
    default=["None", "Mild", "Moderate", "Severe"],
)

score_min, score_max = float(patients["primary_outcome_score"].min()), float(patients["primary_outcome_score"].max())
score_range = st.sidebar.slider("Primary outcome score range", score_min, score_max, (score_min, score_max))

cohort = patients[
    (patients["age"].between(*age_range))
    & (patients["gender"].isin(genders))
    & (patients["treatment_arm"].isin(arms))
    & (patients["worst_ae_severity"].isin(severities))
    & (patients["primary_outcome_score"].between(*score_range))
]

st.metric("Cohort size", f"{len(cohort)} / {len(patients)} patients ({len(cohort)/len(patients)*100:.1f}%)")

if len(cohort) == 0:
    st.warning("No patients match the selected filters.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Lab Value Trends (Cohort)")
    cohort_visits = visits[visits["patient_id"].isin(cohort["patient_id"])]
    trend = cohort_visits.groupby("visit_week")[["hba1c", "systolic_bp", "diastolic_bp", "ldl_cholesterol"]].mean().reset_index()
    metric = st.selectbox("Lab value", ["hba1c", "systolic_bp", "diastolic_bp", "ldl_cholesterol"])
    fig = px.line(trend, x="visit_week", y=metric, markers=True, title=f"Mean {metric} over time (cohort)")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("AE Profile (Cohort)")
    ae_dist = cohort["worst_ae_severity"].value_counts().reindex(["None", "Mild", "Moderate", "Severe"]).fillna(0).reset_index()
    ae_dist.columns = ["severity", "count"]
    fig2 = px.bar(ae_dist, x="severity", y="count", color="severity", title="Worst AE severity distribution")
    st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.subheader("Statistical Comparison vs. Full Population")
if len(cohort) >= 2 and len(cohort) < len(patients):
    result = welch_t_test(
        cohort["primary_outcome_score"], patients["primary_outcome_score"],
        label_a="Cohort", label_b="Full population", metric_name="primary outcome score",
    )
    st.write(result.plain_english)
    ci = confidence_interval(cohort["primary_outcome_score"], metric_name="cohort primary outcome score")
    st.caption(ci["plain_english"])
else:
    st.info("Adjust filters to select a strict subset of the population for a comparison.")

st.divider()
st.subheader("AI Cohort Summary")
if st.button("Generate AI summary of this cohort"):
    from src.agents.bedrock_llm import get_llm

    with st.spinner("Generating summary..."):
        prompt = (
            f"Summarize this clinical trial patient cohort for a medical researcher in 3-4 sentences.\n"
            f"Cohort size: {len(cohort)} of {len(patients)} total patients.\n"
            f"Age range filter: {age_range}. Genders: {genders}. Treatment arms: {arms}. "
            f"Worst AE severities included: {severities}.\n"
            f"Mean primary outcome score: {cohort['primary_outcome_score'].mean():.1f} "
            f"(full population: {patients['primary_outcome_score'].mean():.1f}).\n"
            f"AE severity distribution in cohort: {cohort['worst_ae_severity'].value_counts().to_dict()}.\n"
            f"Dropout rate in cohort: {cohort['dropout_flag'].mean()*100:.1f}%."
        )
        summary = get_llm().invoke(prompt).content
    st.write(summary)

st.divider()
st.download_button(
    "Export filtered cohort to CSV",
    data=cohort.to_csv(index=False),
    file_name="cohort_export.csv",
    mime="text/csv",
)
