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


def page_header(title: str, subtitle: str = ""):
    st.title(title)
    if subtitle:
        st.caption(subtitle)


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
