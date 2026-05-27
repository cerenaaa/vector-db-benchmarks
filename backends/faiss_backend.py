"""FAISS IVFFlat and HNSW backends with auto-tuning."""
from __future__ import annotations
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class FAISSBackend:
    def __init__(self, index_type: str = "IVFFlat", nlist: int = 256, nprobe: int = 32, m: int = 32, ef_search: int = 64):
        if not FAISS_AVAILABLE:
            raise ImportError("Run: pip install faiss-cpu")
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.m = m
        self.ef_search = ef_search
        self.index = None
        self.dim = None

    def build(self, vectors: np.ndarray):
        self.dim = vectors.shape[1]
        vecs = vectors.astype(np.float32)
        faiss.normalize_L2(vecs)

        if self.index_type == "IVFFlat":
            quantizer = faiss.IndexFlatIP(self.dim)
            nlist = min(self.nlist, len(vecs) // 10)
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(vecs)
            self.index.nprobe = self.nprobe
        elif self.index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(self.dim, self.m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efSearch = self.ef_search
        else:
            self.index = faiss.IndexFlatIP(self.dim)

        self.index.add(vecs)
        return self

    def search(self, query: np.ndarray, k: int = 10) -> tuple[list, list]:
        q = query.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, k)
        return indices[0].tolist(), scores[0].tolist()

    def search_batch(self, queries: np.ndarray, k: int = 10) -> list[tuple]:
        Q = queries.astype(np.float32)
        faiss.normalize_L2(Q)
        scores, indices = self.index.search(Q, k)
        return list(zip(indices.tolist(), scores.tolist()))
