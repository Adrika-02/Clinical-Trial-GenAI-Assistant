"""Page 1 — Trial Overview. Auto-loads KPIs, primary endpoint comparison
with statistical significance, and demographic breakdowns."""
import json
import sys
from datetime import datetime
from pathlib import Path

import plotly.express as px
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_DIR))

from utils import APP_TITLE, load_patients, load_visits, load_adverse_events, inject_theme_css, hero_banner, kpi_card

st.set_page_config(page_title=APP_TITLE, page_icon="🧪", layout="wide")
inject_theme_css()

STATS_PATH = PROJECT_ROOT / "data" / "processed" / "stats_results.json"


def _load_stats() -> dict:
    if not STATS_PATH.exists():
        return {}
    with open(STATS_PATH) as f:
        return json.load(f)


hero_banner(
    "🧪 Trial Overview",
    "Drug X vs. Placebo — Phase III Efficacy &amp; Safety Trial, analyzed end-to-end with real statistics, ML, and GenAI agents.",
    chips=["Phase III", "500 Patients", "GenAI-Powered"],
)

patients = load_patients()
visits = load_visits()
ae = load_adverse_events()
stats = _load_stats()

# --- KPI cards ---
n_total = len(patients)
completion_rate = (patients["dropout_flag"] == 0).mean() * 100
week24 = visits[visits["visit_week"] == 24].merge(patients[["patient_id", "treatment_arm"]], on="patient_id")
hba1c_control_rate = (week24["hba1c"] < 7.0).mean() * 100 if len(week24) else 0.0

ae_with_arm = ae.merge(patients[["patient_id", "treatment_arm"]], on="patient_id")
ae_with_arm["ae_occurred"] = ae_with_arm["ae_code"] != "None"
ae_ever = ae_with_arm.groupby("patient_id").agg(
    treatment_arm=("treatment_arm", "first"), ae_occurred=("ae_occurred", "max")
).reset_index()
overall_ae_rate = ae_ever["ae_occurred"].mean() * 100

st.markdown(
    '<div class="kpi-row">'
    + kpi_card("👥", "Total Patients", f"{n_total}")
    + kpi_card("✅", "Trial Completion Rate", f"{completion_rate:.1f}%")
    + kpi_card("🎯", "Patients with HbA1c &lt; 7% (Wk 24)", f"{hba1c_control_rate:.1f}%")
    + kpi_card("⚠️", "Overall Adverse Event Rate", f"{overall_ae_rate:.1f}%")
    + "</div>",
    unsafe_allow_html=True,
)

st.divider()

# --- Primary endpoint comparison ---
left, right = st.columns([2, 1])
with left:
    st.subheader("Primary Outcome Score: Drug X vs. Placebo")
    arm_means = patients.groupby("treatment_arm")["primary_outcome_score"].mean().reset_index()
    fig = px.bar(
        arm_means, x="treatment_arm", y="primary_outcome_score", color="treatment_arm",
        color_discrete_map={"Drug X": "#4C72B0", "Placebo": "#DD8452"},
        labels={"primary_outcome_score": "Mean Primary Outcome Score (0-100)", "treatment_arm": ""},
        text_auto=".1f",
    )
    fig.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Statistical Significance")
    ttest = stats.get("primary_endpoint_ttest", {})
    if ttest:
        st.metric("p-value", "< 0.001" if ttest["p_value"] < 0.001 else f"{ttest['p_value']:.4f}")
        st.metric("Cohen's d (effect size)", f"{ttest['cohens_d']:.2f} ({ttest['effect_size_interpretation']})")
        st.metric("Welch's t-statistic", f"{ttest['statistic']:.2f}")
        st.caption(ttest["plain_english"])
    else:
        st.info("Run `python -m src.stats.run_analysis` to populate statistics.")

st.divider()

# --- Demographics ---
st.subheader("Patient Demographics")
d1, d2, d3 = st.columns(3)

with d1:
    fig_age = px.histogram(patients, x="age", nbins=20, title="Age Distribution", color_discrete_sequence=["#4C72B0"])
    fig_age.update_layout(height=320)
    st.plotly_chart(fig_age, use_container_width=True)

with d2:
    gender_counts = patients["gender"].value_counts().reset_index()
    gender_counts.columns = ["gender", "count"]
    fig_gender = px.pie(gender_counts, names="gender", values="count", title="Gender Split", hole=0.4)
    fig_gender.update_layout(height=320)
    st.plotly_chart(fig_gender, use_container_width=True)

with d3:
    arm_counts = patients["treatment_arm"].value_counts().reset_index()
    arm_counts.columns = ["treatment_arm", "count"]
    fig_arm = px.pie(
        arm_counts, names="treatment_arm", values="count", title="Treatment Arm Split", hole=0.4,
        color="treatment_arm", color_discrete_map={"Drug X": "#4C72B0", "Placebo": "#DD8452"},
    )
    fig_arm.update_layout(height=320)
    st.plotly_chart(fig_arm, use_container_width=True)

st.divider()
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
