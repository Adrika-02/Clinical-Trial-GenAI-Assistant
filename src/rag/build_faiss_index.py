"""Build a FAISS IndexFlatIP over the embeddings already stored in the
ChromaDB collection (L2-normalized so inner product == cosine similarity)
and persist it alongside its id-ordered metadata, so faiss_search() can
serve standalone semantic search without touching ChromaDB.
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
import faiss
import numpy as np

from src.rag.ingest import DEFAULT_COLLECTION, DEFAULT_PERSIST_DIR

DEFAULT_FAISS_DIR = PROJECT_ROOT / "data" / "faiss_index"


def build_faiss_index(
    persist_dir: Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    faiss_dir: Path = DEFAULT_FAISS_DIR,
) -> dict:
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(collection_name)

    data = collection.get(include=["embeddings", "documents", "metadatas"])
    ids = data["ids"]
    embeddings = np.array(data["embeddings"], dtype="float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss_dir = Path(faiss_dir)
    faiss_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_dir / "index.faiss"))

    metadata = [
        {
            "chunk_id": chunk_id,
            "doc_id": meta.get("doc_id"),
            "text": text,
            "patient_id": meta.get("patient_id"),
            "visit_week": meta.get("visit_week"),
            "note_date": meta.get("note_date"),
        }
        for chunk_id, text, meta in zip(ids, data["documents"], data["metadatas"])
    ]
    with open(faiss_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    return {
        "total_vectors": index.ntotal,
        "dim": int(embeddings.shape[1]),
        "faiss_dir": str(faiss_dir),
    }


if __name__ == "__main__":
    stats = build_faiss_index()
    print(json.dumps(stats, indent=2))
