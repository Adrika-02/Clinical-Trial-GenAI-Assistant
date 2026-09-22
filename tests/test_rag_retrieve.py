"""Tests for semantic retrieval (src/rag/retrieve.py) against the persisted
ChromaDB/FAISS indexes, and for the pure-Python scoring helpers used by the
retrieval eval script (src/rag/evaluate_retrieval.py, src/rag/build_eval_set.py).
No LLM calls — these only need the local embedding model and vector indexes.
"""
import pytest

from src.rag.retrieve import faiss_search, semantic_search
from src.rag.build_eval_set import sample_notes
from src.rag.evaluate_retrieval import _percentile, _rank_of_correct, evaluate

REQUIRED_FIELDS = {"chunk_id", "doc_id", "text", "patient_id", "visit_week", "note_date", "score"}


def test_semantic_search_returns_k_hits_with_citation_fields():
    hits = semantic_search("elevated blood pressure", k=5)
    assert len(hits) == 5
    for hit in hits:
        assert REQUIRED_FIELDS.issubset(hit.keys())
        assert hit["patient_id"] is not None
        assert hit["visit_week"] is not None


def test_faiss_search_returns_k_hits_with_citation_fields():
    hits = faiss_search("elevated blood pressure", k=5)
    assert len(hits) == 5
    for hit in hits:
        assert REQUIRED_FIELDS.issubset(hit.keys())
        assert hit["patient_id"] is not None
        assert hit["visit_week"] is not None


def test_semantic_search_retrieves_source_note_verbatim():
    from src.rag.ingest import load_documents

    doc = load_documents()[0]
    hits = semantic_search(doc["text"], k=1)
    assert hits[0]["doc_id"] == doc["doc_id"]


def test_faiss_search_retrieves_source_note_verbatim():
    from src.rag.ingest import load_documents

    doc = load_documents()[0]
    hits = faiss_search(doc["text"], k=1)
    assert hits[0]["doc_id"] == doc["doc_id"]


def test_sample_notes_is_deterministic_for_a_fixed_seed():
    first = sample_notes(sample_size=10, seed=42)
    second = sample_notes(sample_size=10, seed=42)
    assert [d["doc_id"] for d in first] == [d["doc_id"] for d in second]


def test_sample_notes_differs_across_seeds():
    a = sample_notes(sample_size=10, seed=1)
    b = sample_notes(sample_size=10, seed=2)
    assert [d["doc_id"] for d in a] != [d["doc_id"] for d in b]


def test_rank_of_correct_finds_matching_doc_id():
    hits = [{"doc_id": "note_1"}, {"doc_id": "note_2"}, {"doc_id": "note_3"}]
    assert _rank_of_correct(hits, "note_2") == 2


def test_rank_of_correct_returns_none_when_absent():
    hits = [{"doc_id": "note_1"}, {"doc_id": "note_2"}]
    assert _rank_of_correct(hits, "note_99") is None


def test_percentile_of_empty_list_is_zero():
    assert _percentile([], 0.5) == 0.0


def test_percentile_p50_of_sorted_values():
    assert _percentile([1, 2, 3, 4, 5], 0.5) == 3


def test_evaluate_computes_recall_and_mrr_from_a_stub_search_fn():
    eval_records = [
        {"question": "q1", "doc_id": "a"},
        {"question": "q2", "doc_id": "b"},
        {"question": "q3", "doc_id": "z"},
    ]

    def stub_search(query, k):
        # "a" is always found at rank 1, "b" at rank 3, "z" never found.
        return [{"doc_id": "a"}, {"doc_id": "x"}, {"doc_id": "b"}, {"doc_id": "y"}, {"doc_id": "w"}]

    result = evaluate(stub_search, eval_records, k=5)
    assert result["n_queries"] == 3
    assert result["recall_at_1"] == pytest.approx(1 / 3, abs=1e-4)
    assert result["recall_at_5"] == pytest.approx(2 / 3, abs=1e-4)
    assert result["mrr"] == pytest.approx((1 / 1 + 1 / 3 + 0) / 3, abs=1e-4)
