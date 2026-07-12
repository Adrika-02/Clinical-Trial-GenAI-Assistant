"""Deterministic chart-type selection from a SQL result shape. Kept separate
from the LLM's job (writing/executing SQL) since chart selection from a
known dataframe shape is a solved, cheap problem that doesn't need a model
call — more reliable and instant for the dashboard.
"""
import sqlite3

import pandas as pd

from src.db.database import DEFAULT_DB_PATH


def fetch_dataframe(sql: str) -> pd.DataFrame:
    conn = sqlite3.connect(DEFAULT_DB_PATH)
    try:
        return pd.read_sql(sql, conn)
    finally:
        conn.close()


def suggest_chart(df: pd.DataFrame) -> dict:
    """Return {"type": ..., "x": ..., "y": ...} or {"type": "table"}."""
    if df.empty or len(df.columns) < 2:
        return {"type": "table"}

    cols = list(df.columns)
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    non_numeric_cols = [c for c in cols if c not in numeric_cols]

    week_cols = [c for c in cols if "week" in c.lower()]
    if week_cols and numeric_cols:
        y = next((c for c in numeric_cols if c not in week_cols), numeric_cols[0])
        color = non_numeric_cols[0] if non_numeric_cols else None
        return {"type": "line", "x": week_cols[0], "y": y, "color": color}

    if len(non_numeric_cols) == 1 and len(numeric_cols) == 1 and len(df) <= 12:
        return {"type": "bar", "x": non_numeric_cols[0], "y": numeric_cols[0]}

    if len(non_numeric_cols) == 1 and len(numeric_cols) == 1 and len(df) <= 6:
        return {"type": "pie", "names": non_numeric_cols[0], "values": numeric_cols[0]}

    return {"type": "table"}
