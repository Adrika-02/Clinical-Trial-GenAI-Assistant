"""Page 6 — Executive Report (Download). Compiles the full ReportLab PDF
on demand and offers it for download, timing the generation for the
"time to generate" business-impact metric."""
import sys
import time
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_DIR))

from utils import APP_TITLE, page_header

st.set_page_config(page_title=f"Executive Report — {APP_TITLE}", page_icon="📄", layout="wide")
page_header("📄 Executive Report", "Auto-compiled PDF: overview, endpoint statistics, safety profile, clusters, SHAP, and Claude-generated insights.")

st.write(
    "This report combines every module in the pipeline — the statistical analysis, K-Means "
    "patient clustering, the NLP adverse-event classifier and its SHAP explainability, and a "
    "live Claude call (Agent 3) that generates the executive insights and recommendations "
    "grounded in the real numbers above it."
)

include_llm = st.checkbox("Include live Claude-generated insights (adds ~10-20s)", value=True)

if st.button("Generate Executive Report", type="primary"):
    from src.reports.pdf_generator import generate_executive_pdf, DEFAULT_OUTPUT_PATH

    start = time.time()
    with st.spinner("Compiling report..."):
        path = generate_executive_pdf(DEFAULT_OUTPUT_PATH, include_llm_insights=include_llm)
    elapsed = time.time() - start

    st.success(f"Report generated in {elapsed:.1f} seconds.")
    st.caption("Business impact: an equivalent manual compilation of these statistics, cluster profiles, and clinical insights typically takes a biostatistician / medical writer 2-3 days.")

    with open(path, "rb") as f:
        st.download_button(
            "Download Executive Report (PDF)", data=f.read(),
            file_name="clinical_trial_executive_report.pdf", mime="application/pdf",
        )
