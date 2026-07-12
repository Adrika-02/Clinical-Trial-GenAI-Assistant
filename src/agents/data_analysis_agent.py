"""Agent 1 — Data Analysis Agent: converts a natural-language question into
SQL, executes it against SQLite, and returns the answer plus the exact SQL
used (for transparency) plus an auto-selected chart spec.
"""
import sys
from pathlib import Path

from langchain.agents import create_agent

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.bedrock_llm import get_llm
from src.agents.tools import get_database_schema, run_sql_query
from src.agents.chart_helper import fetch_dataframe, suggest_chart

SYSTEM_PROMPT = """You are a clinical trial data analyst with SQL access to a SQLite database.

Always call get_database_schema first if you haven't already in this conversation, then
write exactly one SELECT statement and call run_sql_query to answer the user's question.
Never guess table or column names — always check the schema.
If a query fails, read the error and correct the SQL, then try again.
Give a concise final answer citing the actual numbers returned by the query.
Do not fabricate numbers that did not come from a tool result."""

_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        _agent = create_agent(get_llm(), tools=[get_database_schema, run_sql_query], system_prompt=SYSTEM_PROMPT)
    return _agent


def _extract_last_sql(messages) -> str | None:
    last_sql = None
    for m in messages:
        for call in getattr(m, "tool_calls", None) or []:
            if call["name"] == "run_sql_query":
                last_sql = call["args"].get("sql")
    return last_sql


def run_data_analysis_agent(question: str) -> dict:
    agent = _get_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    messages = result["messages"]

    answer = messages[-1].content
    sql = _extract_last_sql(messages)

    chart_spec = None
    df = None
    if sql:
        try:
            df = fetch_dataframe(sql)
            chart_spec = suggest_chart(df)
        except Exception:
            chart_spec = {"type": "table"}

    return {
        "question": question,
        "answer": answer,
        "sql": sql,
        "dataframe": df,
        "chart_spec": chart_spec,
    }


if __name__ == "__main__":
    result = run_data_analysis_agent("What is the average HbA1c reduction from baseline to Week 24 in the treatment arm vs placebo?")
    print("SQL:", result["sql"])
    print("Answer:", result["answer"])
    print("Chart spec:", result["chart_spec"])
