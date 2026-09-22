"""Agent 2 — Clinical Notes Agent: searches unstructured clinical notes and
synthesizes a plain-English summary (e.g. of adverse events in a given week),
including a list of at-risk patient IDs pulled from the retrieved notes.
"""
import re
import sys
from pathlib import Path

from langchain.agents import create_agent

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.bedrock_llm import get_llm
from src.agents.tools import search_clinical_notes, search_clinical_notes_semantic

SYSTEM_PROMPT = """You are a clinical safety reviewer summarizing investigator notes from a
clinical trial. You have two tools:
- search_clinical_notes: filter notes by keyword, severity class, and/or visit week.
- search_clinical_notes_semantic: semantic search for a natural-language question or
  description, useful when there's no exact keyword to filter on.
Use whichever tool (or both) best retrieves notes relevant to the user's question.
Severity classes are exactly: 'No AE', 'Mild AE', 'Severe AE'.

Then write a concise plain-English summary of what the notes show, and list the specific
patient IDs mentioned in the retrieved notes that a clinician should review, citing each
patient's visit_week alongside their patient_id, with a one-phrase reason each. Only
reference patients and details that actually appeared in the tool results — never invent
patient IDs, visit weeks, or findings."""

_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        _agent = create_agent(
            get_llm(),
            tools=[search_clinical_notes, search_clinical_notes_semantic],
            system_prompt=SYSTEM_PROMPT,
        )
    return _agent


def _extract_patient_ids(text: str) -> list:
    return sorted({int(m) for m in re.findall(r"patient\s*(\d+)", text, flags=re.IGNORECASE)})


def run_clinical_notes_agent(question: str) -> dict:
    agent = _get_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    messages = result["messages"]

    answer = messages[-1].content
    searched_notes = []
    for m in messages:
        if getattr(m, "name", None) == "search_clinical_notes" or type(m).__name__ == "ToolMessage":
            if isinstance(m.content, str) and m.content.startswith("[patient"):
                searched_notes.append(m.content)

    return {
        "question": question,
        "answer": answer,
        "notes_reviewed": "\n\n".join(searched_notes),
        "patient_ids_mentioned": _extract_patient_ids("\n".join(searched_notes) + answer),
    }


if __name__ == "__main__":
    result = run_clinical_notes_agent("Summarise all severe adverse events in Week 4")
    print("Answer:", result["answer"])
    print("Patients mentioned:", result["patient_ids_mentioned"])
