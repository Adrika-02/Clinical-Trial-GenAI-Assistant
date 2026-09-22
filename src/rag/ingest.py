"""Chunk clinical notes, embed with sentence-transformers, and persist to
ChromaDB. Every chunk carries doc_id, chunk_index, and source metadata
(patient_id, visit_week, note_date, true_ae_class) so retrieval results can
be traced back to the exact note and cited.
"""
import json
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
from sentence_transformers import SentenceTransformer

from src.db.database import DEFAULT_DB_PATH, get_connection
from src.rag.chunking import chunk_text, count_tokens

DEFAULT_PERSIST_DIR = PROJECT_ROOT / "data" / "chroma_db"
DEFAULT_COLLECTION = "clinical_notes"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RESULTS_PATH = PROJECT_ROOT / "results" / "ingest_stats.json"


def load_documents(db_path: Path = DEFAULT_DB_PATH) -> list:
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            "SELECT note_id, patient_id, visit_week, note_date, note_text, true_ae_class "
            "FROM clinical_notes ORDER BY note_id"
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    return [
        {
            "doc_id": f"note_{note_id}",
            "text": note_text,
            "metadata": {
                "patient_id": patient_id,
                "visit_week": visit_week,
                "note_date": note_date,
                "true_ae_class": true_ae_class,
                "source": "clinical_notes",
            },
        }
        for note_id, patient_id, visit_week, note_date, note_text, true_ae_class in rows
    ]


def _dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _percentile(values: list, pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(round(pct * (len(s) - 1))))
    return s[idx]


def ingest(
    chunk_size: int = 200,
    overlap: int = 40,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    persist_dir: Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    db_path: Path = DEFAULT_DB_PATH,
    reset: bool = True,
    batch_size: int = 256,
) -> dict:
    documents = load_documents(db_path)

    chunk_ids, chunk_texts, chunk_metadatas = [], [], []
    for doc in documents:
        pieces = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
        for idx, piece in enumerate(pieces):
            chunk_ids.append(f"{doc['doc_id']}_chunk{idx}")
            chunk_texts.append(piece)
            chunk_metadatas.append({**doc["metadata"], "doc_id": doc["doc_id"], "chunk_index": idx})

    if reset and persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(embedding_model)

    embed_start = time.perf_counter()
    embeddings = model.encode(
        chunk_texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True
    )
    embed_wall_clock_s = time.perf_counter() - embed_start

    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(collection_name, metadata={"embedding_model": embedding_model})

    for start in range(0, len(chunk_ids), batch_size):
        end = start + batch_size
        collection.add(
            ids=chunk_ids[start:end],
            embeddings=embeddings[start:end].tolist(),
            documents=chunk_texts[start:end],
            metadatas=chunk_metadatas[start:end],
        )

    token_lengths = [count_tokens(t) for t in chunk_texts]
    stats = {
        "config": {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "embedding_model": embedding_model,
            "collection_name": collection_name,
        },
        "total_documents": len(documents),
        "total_chunks": len(chunk_ids),
        "mean_chunk_tokens": round(sum(token_lengths) / len(token_lengths), 2) if token_lengths else 0,
        "p95_chunk_tokens": _percentile(token_lengths, 0.95),
        "max_chunk_tokens": max(token_lengths) if token_lengths else 0,
        "embedding_wall_clock_seconds": round(embed_wall_clock_s, 3),
        "chunks_per_second": round(len(chunk_ids) / embed_wall_clock_s, 1) if embed_wall_clock_s > 0 else None,
        "index_size_bytes": _dir_size_bytes(persist_dir),
        "index_size_mb": round(_dir_size_bytes(persist_dir) / (1024 * 1024), 2),
    }
    return stats


if __name__ == "__main__":
    stats = ingest()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
