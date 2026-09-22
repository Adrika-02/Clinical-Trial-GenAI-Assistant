"""Semantic search over the RAG pipeline's clinical-notes embeddings, via
ChromaDB (semantic_search) or the standalone FAISS index built by
build_faiss_index.py from the same embeddings (faiss_search). Both return
the retrieved chunk text plus patient_id, visit_week, and note_date so
results can be traced back to and cited from a source note.
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
import faiss
from sentence_transformers import SentenceTransformer

from src.rag.ingest import DEFAULT_COLLECTION, DEFAULT_EMBEDDING_MODEL, DEFAULT_PERSIST_DIR

DEFAULT_FAISS_DIR = PROJECT_ROOT / "data" / "faiss_index"

_models = {}
_collections = {}
_faiss_indexes = {}


def _get_model(embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    if embedding_model not in _models:
        _models[embedding_model] = SentenceTransformer(embedding_model)
    return _models[embedding_model]


def _get_collection(persist_dir: Path = DEFAULT_PERSIST_DIR, collection_name: str = DEFAULT_COLLECTION):
    key = (str(persist_dir), collection_name)
    if key not in _collections:
        client = chromadb.PersistentClient(path=str(persist_dir))
        _collections[key] = client.get_collection(collection_name)
    return _collections[key]


def _get_faiss_index(faiss_dir: Path = DEFAULT_FAISS_DIR):
    key = str(faiss_dir)
    if key not in _faiss_indexes:
        index = faiss.read_index(str(Path(faiss_dir) / "index.faiss"))
        with open(Path(faiss_dir) / "metadata.json") as f:
            metadata = json.load(f)
        _faiss_indexes[key] = (index, metadata)
    return _faiss_indexes[key]


def _hit(chunk_id, text, meta, score):
    return {
        "chunk_id": chunk_id,
        "doc_id": meta.get("doc_id"),
        "text": text,
        "patient_id": meta.get("patient_id"),
        "visit_week": meta.get("visit_week"),
        "note_date": meta.get("note_date"),
        "score": score,
    }


def semantic_search(
    query: str,
    k: int = 5,
    persist_dir: Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> list:
    """Embed `query` and return the top-k chunks from the ChromaDB collection,
    each with its patient_id, visit_week, and note_date for citation."""
    model = _get_model(embedding_model)
    collection = _get_collection(persist_dir, collection_name)
    query_embedding = model.encode([query], convert_to_numpy=True)[0].tolist()

    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    return [
        _hit(chunk_id, text, meta, distance)
        for chunk_id, text, meta, distance in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]


def faiss_search(
    query: str,
    k: int = 5,
    faiss_dir: Path = DEFAULT_FAISS_DIR,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> list:
    """Embed `query` and return the top-k chunks from the standalone FAISS
    IndexFlatIP built by build_faiss_index.py, with the same citation
    fields as semantic_search."""
    model = _get_model(embedding_model)
    index, metadata = _get_faiss_index(faiss_dir)

    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, k)
    hits = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        entry = metadata[idx]
        hits.append(_hit(entry["chunk_id"], entry["text"], entry, float(score)))
    return hits


if __name__ == "__main__":
    for h in semantic_search("severe hypoglycemia after dose increase", k=3):
        print(h["patient_id"], h["visit_week"], h["note_date"], h["text"][:80])
