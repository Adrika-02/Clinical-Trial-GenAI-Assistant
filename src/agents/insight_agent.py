"""Agent 3 — Insight Generation Agent: combines statistical results, patient
cluster profiles, AE classifier performance, and clinical notes into an
executive-level clinical trial insight report for a business audience.
"""
import sys
from pathlib import Path

from langchain.agents import create_agent

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.bedrock_llm import get_llm
from src.agents.tools import (
    get_stats_summary, get_clustering_summary, get_ae_classifier_metrics, search_clinical_notes,
)

SYSTEM_PROMPT = """You are a clinical trial program director preparing an executive briefing
for pharmaceutical leadership (non-technical business audience).

Use the available tools (get_stats_summary, get_clustering_summary, get_ae_classifier_metrics,
search_clinical_notes) to gather real evidence before writing your answer. Always ground every
claim in an actual number returned by a tool — never invent a statistic.

Write in plain business English: lead with the headline finding, then support it with the
specific numbers (p-values, effect sizes, rates, percentages). Frame findings in terms of
patient safety, trial efficiency, and business/financial impact where relevant. Keep the
response to a tight executive summary (roughly 150-250 words) unless asked for more detail."""

_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        _agent = create_agent(
            get_llm(),
            tools=[get_stats_summary, get_clustering_summary, get_ae_classifier_metrics, search_clinical_notes],
            system_prompt=SYSTEM_PROMPT,
        )
    return _agent


def run_insight_agent(question: str) -> dict:
    agent = _get_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    messages = result["messages"]

    answer = messages[-1].content
    tools_used = sorted({
        call["name"]
        for m in messages
        for call in (getattr(m, "tool_calls", None) or [])
    })

    return {"question": question, "answer": answer, "tools_used": tools_used}


if __name__ == "__main__":
    result = run_insight_agent("What are the key safety signals in this trial, and which patient subgroup should we prioritize for monitoring?")
    print("Tools used:", result["tools_used"])
    print("\nAnswer:\n", result["answer"])
