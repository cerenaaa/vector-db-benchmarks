"""
Benchmark runner: measures recall, QPS, build time, and memory across backends.
"""
from __future__ import annotations
import time
import tracemalloc
import numpy as np
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    backend: str
    n_vectors: int
    dim: int
    k: int
    recall_at_k: float
    qps: float
    build_time_s: float
    memory_mb: float

    def __str__(self):
        return (f"{self.backend:20s} | recall@{self.k}={self.recall_at_k:.3f} | "
                f"QPS={self.qps:,.0f} | build={self.build_time_s:.2f}s | mem={self.memory_mb:.1f}MB")


def compute_recall(true_ids: list[list], pred_ids: list[list], k: int) -> float:
    hits = sum(len(set(t[:k]) & set(p[:k])) for t, p in zip(true_ids, pred_ids))
    return hits / (len(true_ids) * k)


def run_benchmark(
    backend,
    backend_name: str,
    train_vecs: np.ndarray,
    query_vecs: np.ndarray,
    ground_truth: list[list[int]],
    k: int = 10,
) -> BenchmarkResult:
    # Build
    tracemalloc.start()
    t0 = time.perf_counter()
    backend.build(train_vecs)
    build_time = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Query (measure QPS over all queries)
    t0 = time.perf_counter()
    pred_results = backend.search_batch(query_vecs, k=k)
    query_time = time.perf_counter() - t0
    qps = len(query_vecs) / query_time

    pred_ids = [list(ids) for ids, _ in pred_results]
    recall = compute_recall(ground_truth, pred_ids, k)

    return BenchmarkResult(
        backend=backend_name,
        n_vectors=len(train_vecs),
        dim=train_vecs.shape[1],
        k=k,
        recall_at_k=recall,
        qps=qps,
        build_time_s=build_time,
        memory_mb=peak / 1024 / 1024,
    )
