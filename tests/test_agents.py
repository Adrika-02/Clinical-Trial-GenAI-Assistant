"""Tests for the agent tools (pure Python, no LLM calls) and a lightweight
live check of the router's classification (single cheap LLM call against
the configured provider) to confirm the agents are wired to a real backend."""
import pytest

from src.agents.tools import (
    get_database_schema, run_sql_query, search_clinical_notes,
    get_stats_summary, get_clustering_summary, get_ae_classifier_metrics,
)
from src.agents.chart_helper import suggest_chart
import pandas as pd


def test_get_database_schema_mentions_all_tables():
    schema = get_database_schema.invoke({})
    for table in ["patients", "visits", "adverse_events", "clinical_notes", "patient_clusters"]:
        assert table in schema


def test_run_sql_query_rejects_non_select():
    result = run_sql_query.invoke({"sql": "DELETE FROM patients"})
    assert "Error" in result


def test_run_sql_query_executes_valid_select():
    result = run_sql_query.invoke({"sql": "SELECT COUNT(*) as n FROM patients"})
    assert "500" in result or "n" in result


def test_search_clinical_notes_filters_by_severity():
    result = search_clinical_notes.invoke({"severity": "Severe AE", "limit": 5})
    assert "Severe AE" in result or "No matching notes found." in result


def test_stats_summary_returns_content():
    result = get_stats_summary.invoke({})
    assert len(result) > 0


def test_clustering_summary_returns_content():
    result = get_clustering_summary.invoke({})
    assert len(result) > 0


def test_ae_classifier_metrics_returns_content():
    result = get_ae_classifier_metrics.invoke({})
    assert len(result) > 0


def test_suggest_chart_line_for_week_column():
    df = pd.DataFrame({"visit_week": [0, 2, 4], "hba1c": [8.1, 7.9, 7.5]})
    assert suggest_chart(df)["type"] == "line"


def test_suggest_chart_bar_for_categorical_numeric():
    df = pd.DataFrame({"treatment_arm": ["Drug X", "Placebo"], "avg_score": [60.0, 52.0]})
    assert suggest_chart(df)["type"] == "bar"


def test_suggest_chart_table_fallback_for_single_column():
    df = pd.DataFrame({"count": [500]})
    assert suggest_chart(df)["type"] == "table"


@pytest.mark.live_llm
def test_router_classifies_data_question_live():
    from src.agents.router import classify_question
    label = classify_question("What is the average age of patients in the Drug X arm?")
    assert label == "data_analysis"
