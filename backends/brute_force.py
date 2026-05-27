"""Exact brute-force retrieval via numpy matrix multiply."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


class BruteForceIndex:
    def __init__(self):
        self.vectors: np.ndarray = None
        self.ids: list = None

    def build(self, vectors: np.ndarray, ids: list = None):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.vectors = vectors / (norms + 1e-9)
        self.ids = ids or list(range(len(vectors)))
        return self

    def search(self, query: np.ndarray, k: int = 10) -> tuple[list, list]:
        q = query / (np.linalg.norm(query) + 1e-9)
        scores = self.vectors @ q
        top_idx = np.argsort(scores)[::-1][:k]
        return [self.ids[i] for i in top_idx], scores[top_idx].tolist()

    def search_batch(self, queries: np.ndarray, k: int = 10) -> list[tuple]:
        return [self.search(q, k) for q in queries]
