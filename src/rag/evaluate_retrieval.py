"""Measure ChromaDB vs FAISS retrieval quality (recall@1, recall@5, MRR) and
query latency (p50/p95) against the fixed eval set in
data/eval/retrieval_eval.jsonl. A hit counts as correct when the retrieved
chunk's doc_id matches the eval question's source note.
"""
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.retrieve import faiss_search, semantic_search

EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "retrieval_eval.jsonl"
RESULTS_PATH = PROJECT_ROOT / "results" / "retrieval_eval.json"
K = 5


def load_eval_set(path: Path = EVAL_PATH) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _percentile(values: list, pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(round(pct * (len(s) - 1))))
    return s[idx]


def _rank_of_correct(hits: list, doc_id: str):
    for rank, hit in enumerate(hits, start=1):
        if hit["doc_id"] == doc_id:
            return rank
    return None


def evaluate(search_fn, eval_records: list, k: int = K) -> dict:
    ranks, latencies = [], []
    for record in eval_records:
        start = time.perf_counter()
        hits = search_fn(record["question"], k)
        latencies.append(time.perf_counter() - start)
        ranks.append(_rank_of_correct(hits, record["doc_id"]))

    n = len(ranks)
    recall_at_1 = sum(1 for r in ranks if r == 1) / n
    recall_at_5 = sum(1 for r in ranks if r is not None and r <= 5) / n
    mrr = sum((1 / r) if r else 0 for r in ranks) / n

    return {
        "recall_at_1": round(recall_at_1, 4),
        "recall_at_5": round(recall_at_5, 4),
        "mrr": round(mrr, 4),
        "p50_latency_ms": round(_percentile(latencies, 0.5) * 1000, 2),
        "p95_latency_ms": round(_percentile(latencies, 0.95) * 1000, 2),
        "n_queries": n,
    }


def main() -> dict:
    eval_records = load_eval_set()
    results = {
        "k": K,
        "eval_set_size": len(eval_records),
        "chromadb": evaluate(semantic_search, eval_records),
        "faiss": evaluate(faiss_search, eval_records),
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
