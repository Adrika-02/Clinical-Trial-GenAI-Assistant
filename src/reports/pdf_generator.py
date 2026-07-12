"""Compiles the executive PDF report: trial overview, primary/secondary
endpoint statistics, safety profile, patient cluster profiles, AE
classifier + SHAP explainability summary, and Claude-generated executive
insights and recommendations — all sourced from real pipeline artifacts,
with the insights/recommendations coming from a live LLM call, not
hardcoded text.
"""
import json
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.db.database import DEFAULT_DB_PATH

STATS_PATH = PROJECT_ROOT / "data" / "processed" / "stats_results.json"
CLUSTERING_PATH = PROJECT_ROOT / "data" / "processed" / "clustering_results.json"
AE_METRICS_PATH = PROJECT_ROOT / "models" / "saved" / "ae_classifier_metrics.json"
SHAP_SUMMARY_PATH = PROJECT_ROOT / "models" / "shap_plots" / "shap_summary.json"
SCREENSHOTS_DIR = PROJECT_ROOT / "docs" / "screenshots"

DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "executive_report.pdf"


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _get_demographics() -> dict:
    conn = sqlite3.connect(DEFAULT_DB_PATH)
    patients = pd.read_sql("SELECT * FROM patients", conn)
    conn.close()

    return {
        "n_total": len(patients),
        "n_drug_x": int((patients["treatment_arm"] == "Drug X").sum()),
        "n_placebo": int((patients["treatment_arm"] == "Placebo").sum()),
        "mean_age": round(patients["age"].mean(), 1),
        "age_range": (int(patients["age"].min()), int(patients["age"].max())),
        "pct_female": round((patients["gender"] == "Female").mean() * 100, 1),
        "completion_rate": round((patients["dropout_flag"] == 0).mean() * 100, 1),
        "dropout_rate": round((patients["dropout_flag"] == 1).mean() * 100, 1),
    }


def _generate_insights_and_recommendations() -> dict:
    """Real LLM call (via the same agent tools as Agent 3) asking for exactly
    5 numbered executive insights and 3 numbered recommendations."""
    from src.agents.insight_agent import _get_agent

    agent = _get_agent()
    prompt = (
        "Using the available tools, gather the real trial statistics, cluster profiles, "
        "and AE classifier metrics. Then respond in exactly this format:\n\n"
        "INSIGHTS:\n1. ...\n2. ...\n3. ...\n4. ...\n5. ...\n\n"
        "RECOMMENDATIONS:\n1. ...\n2. ...\n3. ...\n\n"
        "Each insight and recommendation must be a single sentence grounded in an actual "
        "number from a tool result. No preamble, no other text."
    )
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    text = result["messages"][-1].content

    insights_match = re.search(r"INSIGHTS:(.*?)RECOMMENDATIONS:", text, re.DOTALL)
    recs_match = re.search(r"RECOMMENDATIONS:(.*)", text, re.DOTALL)

    def _parse_numbered(block: str) -> list:
        if not block:
            return []
        items = re.findall(r"^\s*\d+\.\s*(.+)$", block, re.MULTILINE)
        return [i.strip() for i in items if i.strip()]

    return {
        "insights": _parse_numbered(insights_match.group(1) if insights_match else ""),
        "recommendations": _parse_numbered(recs_match.group(1) if recs_match else ""),
        "raw_text": text,
    }


def generate_executive_pdf(output_path: Path = DEFAULT_OUTPUT_PATH, include_llm_insights: bool = True) -> Path:
    demo = _get_demographics()
    stats = _load_json(STATS_PATH)
    clustering = _load_json(CLUSTERING_PATH)
    ae_metrics = _load_json(AE_METRICS_PATH)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleCustom", parent=styles["Title"], fontSize=20)
    h2 = ParagraphStyle("H2Custom", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6)
    body = styles["BodyText"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path), pagesize=letter,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
    )
    elements = []

    # --- Title ---
    elements.append(Paragraph("Clinical Trial Executive Report", title_style))
    elements.append(Paragraph("Drug X vs. Placebo — Phase III Efficacy & Safety Analysis", styles["Heading3"]))
    elements.append(Paragraph(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", body))
    elements.append(Spacer(1, 0.2 * inch))

    # --- Section 1: Trial Overview & Demographics ---
    elements.append(Paragraph("1. Trial Overview & Demographics", h2))
    demo_table_data = [
        ["Metric", "Value"],
        ["Total patients enrolled", str(demo["n_total"])],
        ["Drug X arm", str(demo["n_drug_x"])],
        ["Placebo arm", str(demo["n_placebo"])],
        ["Mean age (range)", f"{demo['mean_age']} ({demo['age_range'][0]}-{demo['age_range'][1]})"],
        ["Female patients", f"{demo['pct_female']}%"],
        ["Trial completion rate", f"{demo['completion_rate']}%"],
        ["Dropout rate", f"{demo['dropout_rate']}%"],
    ]
    elements.append(_make_table(demo_table_data))

    # --- Section 2: Primary & Secondary Endpoint Results ---
    elements.append(Paragraph("2. Primary & Secondary Endpoint Results", h2))
    if "primary_endpoint_ttest" in stats:
        elements.append(Paragraph(stats["primary_endpoint_ttest"]["plain_english"], body))
    if "hba1c_control_chisq" in stats:
        elements.append(Spacer(1, 0.1 * inch))
        elements.append(Paragraph(stats["hba1c_control_chisq"]["plain_english"], body))

    # --- Section 3: Safety Profile ---
    elements.append(Paragraph("3. Safety Profile", h2))
    if "ae_occurrence_chisq" in stats:
        elements.append(Paragraph(stats["ae_occurrence_chisq"]["plain_english"], body))
    if "ae_severity_class_chisq" in stats:
        elements.append(Spacer(1, 0.1 * inch))
        elements.append(Paragraph(stats["ae_severity_class_chisq"]["plain_english"], body))
    if "kruskal_hba1c_by_ae_class" in stats:
        elements.append(Spacer(1, 0.1 * inch))
        elements.append(Paragraph(stats["kruskal_hba1c_by_ae_class"]["plain_english"], body))

    # --- Section 4: Patient Cluster Profiles ---
    elements.append(PageBreak())
    elements.append(Paragraph("4. Patient Cluster Profiles (K-Means Segmentation)", h2))
    if clustering.get("cluster_profiles"):
        elements.append(Paragraph(
            f"K-Means with k={clustering.get('chosen_k')} clusters "
            f"(silhouette score {clustering.get('silhouette_score')}).", body,
        ))
        elements.append(Spacer(1, 0.1 * inch))
        rows = [["Cluster", "N", "Label", "Outcome\nScore", "Any-AE\nRate", "Severe-AE\nRate", "Dropout\nRate"]]
        for c in clustering["cluster_profiles"]:
            rows.append([
                str(c["cluster"]), str(c["n_patients"]), c.get("business_label", ""),
                str(c["mean_outcome_score"]), f"{c['any_ae_rate']*100:.1f}%",
                f"{c['severe_ae_rate']*100:.1f}%", f"{c['dropout_rate']*100:.1f}%",
            ])
        elements.append(_make_table(rows, col_widths=[0.5, 0.4, 1.7, 0.9, 0.85, 0.95, 0.85]))
        pca_path = SCREENSHOTS_DIR / "clustering_pca_scatter.png"
        if pca_path.exists():
            elements.append(Spacer(1, 0.15 * inch))
            elements.append(Image(str(pca_path), width=5.5 * inch, height=4.6 * inch))

    # --- Section 5: AE Classifier & Explainability ---
    elements.append(PageBreak())
    elements.append(Paragraph("5. NLP Adverse-Event Classifier & Explainability", h2))
    if ae_metrics:
        rows = [
            ["Metric", "Value"],
            ["Accuracy", f"{ae_metrics.get('accuracy', 0)*100:.1f}%"],
            ["Precision (macro)", f"{ae_metrics.get('precision_macro', 0)*100:.1f}%"],
            ["Recall (macro)", f"{ae_metrics.get('recall_macro', 0)*100:.1f}%"],
            ["F1 (macro)", f"{ae_metrics.get('f1_macro', 0):.3f}"],
        ]
        elements.append(_make_table(rows))
        bi = ae_metrics.get("business_impact", {})
        if bi:
            elements.append(Spacer(1, 0.1 * inch))
            elements.append(Paragraph(
                f"On the held-out test set, the NLP classifier detected {bi.get('nlp_detected_ae_count')} "
                f"adverse-event notes vs. {bi.get('manual_coded_ae_count')} caught by manual CRF coding "
                f"({bi.get('uplift_pct')}% uplift), and recovered {bi.get('recovery_rate_pct_of_missed')}% "
                f"of the {bi.get('manually_missed_ae_count')} adverse events manual coding missed entirely.", body,
            ))
    shap_img = SCREENSHOTS_DIR / "shap_global_summary_severe_ae.png"
    if shap_img.exists():
        elements.append(Spacer(1, 0.15 * inch))
        elements.append(Paragraph("SHAP global feature importance — Severe AE classification:", body))
        elements.append(Image(str(shap_img), width=5.0 * inch, height=3.6 * inch))

    # --- Section 6 & 7: Claude-generated insights and recommendations ---
    elements.append(PageBreak())
    elements.append(Paragraph("6. Top Clinical Insights", h2))
    elements.append(Paragraph(
        "<i>The following insights and recommendations are generated by Agent 3 (Insight "
        "Generation Agent), a live LLM call grounded in the statistics, clustering, and "
        "classifier tool results above.</i>", body,
    ))
    elements.append(Spacer(1, 0.1 * inch))

    if include_llm_insights:
        try:
            generated = _generate_insights_and_recommendations()
        except Exception as e:
            generated = {"insights": [], "recommendations": [], "raw_text": f"(LLM call failed: {e})"}

        if generated["insights"]:
            for i, insight in enumerate(generated["insights"], 1):
                elements.append(Paragraph(f"{i}. {insight}", body))
        else:
            elements.append(Paragraph(generated["raw_text"], body))

        elements.append(Paragraph("7. Recommendations for Next Steps", h2))
        if generated["recommendations"]:
            for i, rec in enumerate(generated["recommendations"], 1):
                elements.append(Paragraph(f"{i}. {rec}", body))

    doc.build(elements)
    return output_path


def _make_table(data, col_widths=None) -> Table:
    if col_widths:
        col_widths = [w * inch for w in col_widths]
    table = Table(data, colWidths=col_widths)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4C72B0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F0F5")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


if __name__ == "__main__":
    path = generate_executive_pdf()
    print(f"Executive report generated: {path}")
