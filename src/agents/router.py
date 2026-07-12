"""Routes a natural-language question to one of the three agents. Uses a
single fast LLM classification call (no tools) rather than keyword
heuristics, since question phrasing varies too much for reliable keyword
matching — this is the entry point the Streamlit chat page calls.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.bedrock_llm import get_llm
from src.agents.data_analysis_agent import run_data_analysis_agent
from src.agents.clinical_notes_agent import run_clinical_notes_agent
from src.agents.insight_agent import run_insight_agent

ROUTER_PROMPT = """Classify the user's clinical trial question into exactly one category.
Reply with only the single category word, nothing else.

- data_analysis: questions answerable by aggregating/querying structured trial data
  (counts, averages, rates, comparisons between arms, trends over visits).
  Example: "What is the average HbA1c reduction in the treatment arm?"
- clinical_notes: questions about what investigators wrote in free-text notes,
  or that ask to search/summarize/find notes about specific events or patients.
  Example: "Summarise all severe adverse events in Week 4."
- insight: broad/strategic questions asking for an executive summary, key
  findings, safety signals, or recommendations combining multiple sources.
  Example: "What are the key safety signals in this trial?"

Question: {question}
Category:"""

AGENT_DISPATCH = {
    "data_analysis": run_data_analysis_agent,
    "clinical_notes": run_clinical_notes_agent,
    "insight": run_insight_agent,
}


def classify_question(question: str) -> str:
    llm = get_llm()
    response = llm.invoke(ROUTER_PROMPT.format(question=question))
    label = response.content.strip().lower()
    for valid in AGENT_DISPATCH:
        if valid in label:
            return valid
    return "insight"  # safe default: broadest agent


def route_question(question: str) -> dict:
    agent_name = classify_question(question)
    result = AGENT_DISPATCH[agent_name](question)
    result["agent"] = agent_name
    return result


if __name__ == "__main__":
    for q in [
        "What is the average HbA1c reduction in the treatment arm?",
        "Summarise all severe adverse events in Week 4",
        "What are the key safety signals in this trial?",
    ]:
        agent_name = classify_question(q)
        print(f"{q!r} -> {agent_name}")
