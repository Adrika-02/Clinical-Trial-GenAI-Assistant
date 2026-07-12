"""Shared data-access and styling helpers for the Streamlit dashboard."""
import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# On Streamlit Community Cloud there is no .env file — secrets are supplied via
# st.secrets instead. Mirror them into os.environ so src/agents/bedrock_llm.py
# and every other os.environ.get()/os.environ[...] call works unchanged locally
# and in the cloud.
try:
    for _key, _value in st.secrets.items():
        os.environ.setdefault(_key, str(_value))
except (FileNotFoundError, KeyError):
    pass

from src.db.database import DEFAULT_DB_PATH

APP_TITLE = "Clinical Trial Intelligence Assistant"


@st.cache_resource
def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DEFAULT_DB_PATH, check_same_thread=False)


def run_query(sql: str, params: tuple = ()) -> pd.DataFrame:
    conn = get_connection()
    return pd.read_sql(sql, conn, params=params)


@st.cache_data(ttl=60)
def load_patients() -> pd.DataFrame:
    return run_query("SELECT * FROM patients")


@st.cache_data(ttl=60)
def load_visits() -> pd.DataFrame:
    return run_query("SELECT * FROM visits")


@st.cache_data(ttl=60)
def load_adverse_events() -> pd.DataFrame:
    return run_query("SELECT * FROM adverse_events")


@st.cache_data(ttl=60)
def load_clinical_notes() -> pd.DataFrame:
    return run_query("SELECT * FROM clinical_notes")


@st.cache_data(ttl=60)
def load_clusters() -> pd.DataFrame:
    try:
        return run_query("SELECT * FROM patient_clusters")
    except Exception:
        return pd.DataFrame(columns=["patient_id", "cluster", "cluster_label"])


def clear_all_caches():
    load_patients.clear()
    load_visits.clear()
    load_adverse_events.clear()
    load_clinical_notes.clear()
    load_clusters.clear()


_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }

h1, h2, h3 { color: #0F172A; font-weight: 700; letter-spacing: -0.01em; }

.ctia-page-header { display: flex; align-items: center; gap: 0.9rem; margin-bottom: 0.3rem; }
.ctia-icon-badge {
    display: flex; align-items: center; justify-content: center;
    width: 52px; height: 52px; min-width: 52px; border-radius: 14px;
    background: linear-gradient(135deg, #2563EB 0%, #0EA5E9 100%);
    font-size: 1.6rem; box-shadow: 0 4px 10px rgba(37, 99, 235, 0.25);
}
.ctia-page-title { font-size: 2rem; font-weight: 800; color: #0F172A; line-height: 1.15; }
.ctia-page-subtitle { color: #64748B; font-size: 0.98rem; margin-top: 0.15rem; }
.ctia-header-rule {
    border: none; height: 1px;
    background: linear-gradient(90deg, #2563EB 0%, #E2E8F0 40%, transparent 100%);
    margin: 1rem 0 1.4rem 0;
}

.hero-banner {
    background: linear-gradient(135deg, #1E40AF 0%, #2563EB 55%, #0EA5E9 100%);
    border-radius: 18px; padding: 2.1rem 2.5rem; color: #FFFFFF; margin-bottom: 1.6rem;
    box-shadow: 0 8px 24px rgba(37, 99, 235, 0.25);
}
.hero-banner .hero-title { font-size: 2.1rem; font-weight: 800; color: #FFFFFF; margin: 0 0 0.4rem 0; }
.hero-banner .hero-subtitle { color: rgba(255,255,255,0.92); font-size: 1.05rem; margin: 0 0 1rem 0; }
.hero-chip {
    display: inline-block; background: rgba(255,255,255,0.16);
    border: 1px solid rgba(255,255,255,0.38); border-radius: 999px;
    padding: 0.3rem 0.95rem; font-size: 0.82rem; font-weight: 600;
    margin-right: 0.5rem; color: #FFFFFF;
}

.kpi-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 0.5rem; }
.kpi-card {
    background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid #2563EB;
    border-radius: 12px; padding: 1rem 1.2rem; box-shadow: 0 1px 4px rgba(15, 23, 42, 0.06);
    flex: 1 1 200px; min-width: 200px;
}
.kpi-card .kpi-icon { font-size: 1.3rem; margin-bottom: 0.35rem; }
.kpi-card .kpi-label { color: #64748B; font-size: 0.85rem; font-weight: 500; margin-bottom: 0.15rem; }
.kpi-card .kpi-value { color: #0F172A; font-size: 1.65rem; font-weight: 800; }

section[data-testid="stSidebar"] { border-right: 1px solid #E2E8F0; }
</style>
"""


def inject_theme_css():
    """Inject shared fonts/colors/component CSS. Call once per page, right
    after st.set_page_config()."""
    st.markdown(_THEME_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = ""):
    icon, _, title_text = title.partition(" ")
    subtitle_html = f'<div class="ctia-page-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="ctia-page-header">
            <div class="ctia-icon-badge">{icon}</div>
            <div>
                <div class="ctia-page-title">{title_text}</div>
                {subtitle_html}
            </div>
        </div>
        <hr class="ctia-header-rule"/>
        """,
        unsafe_allow_html=True,
    )


def hero_banner(title: str, subtitle: str, chips: list = None):
    chips_html = "".join(f'<span class="hero-chip">{c}</span>' for c in (chips or []))
    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="hero-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
            {chips_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(icon: str, label: str, value: str) -> str:
    """Return the HTML for one KPI card. Combine several inside a
    '<div class="kpi-row">...</div>' wrapper passed to st.markdown."""
    return (
        f'<div class="kpi-card"><div class="kpi-icon">{icon}</div>'
        f'<div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>'
    )


def render_chart(df: pd.DataFrame, spec: dict, key: str = None):
    """Render a dataframe using the chart spec produced by chart_helper.suggest_chart."""
    import plotly.express as px

    if df is None or spec is None or spec.get("type") == "table" or df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True, key=key)
        return

    chart_type = spec["type"]
    if chart_type == "line":
        fig = px.line(df, x=spec["x"], y=spec["y"], color=spec.get("color"), markers=True)
    elif chart_type == "bar":
        fig = px.bar(df, x=spec["x"], y=spec["y"], text_auto=".2f")
    elif chart_type == "pie":
        fig = px.pie(df, names=spec["names"], values=spec["values"], hole=0.4)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True, key=key)
        return

    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True, key=key)
