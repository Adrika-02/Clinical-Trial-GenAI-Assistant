"""Build a fixed retrieval-eval set: sample N clinical notes with a fixed
seed, and have the LLM write exactly one question each note alone answers.
Used by evaluate_retrieval.py to check whether semantic search over the
chunked, embedded notes retrieves the right source note back.
"""
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.bedrock_llm import get_llm
from src.rag.ingest import load_documents

DEFAULT_OUT_PATH = PROJECT_ROOT / "data" / "eval" / "retrieval_eval.jsonl"
DEFAULT_SEED = 42
DEFAULT_SAMPLE_SIZE = 100

QUESTION_PROMPT = """Below is a clinical trial investigator note. Write exactly ONE
question that this note answers. The question must be answerable using only the
information in this note, and specific enough that it would not be equally well
answered by an unrelated note. Return only the question text, with no numbering,
quotes, or preamble.

Note:
{note_text}"""


def sample_notes(sample_size: int = DEFAULT_SAMPLE_SIZE, seed: int = DEFAULT_SEED) -> list:
    documents = load_documents()
    return random.Random(seed).sample(documents, sample_size)


def build_eval_set(
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
    out_path: Path = DEFAULT_OUT_PATH,
) -> list:
    sampled = sample_notes(sample_size, seed)
    llm = get_llm()

    records = []
    for doc in sampled:
        question = llm.invoke(QUESTION_PROMPT.format(note_text=doc["text"])).content.strip()
        records.append({
            "doc_id": doc["doc_id"],
            "patient_id": doc["metadata"]["patient_id"],
            "visit_week": doc["metadata"]["visit_week"],
            "note_date": doc["metadata"]["note_date"],
            "question": question,
        })

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    return records


if __name__ == "__main__":
    records = build_eval_set()
    print(f"Wrote {len(records)} eval questions to {DEFAULT_OUT_PATH}")
