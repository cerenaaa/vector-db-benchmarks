"""
Run vector DB benchmarks.
Usage: python run_benchmark.py [--n_vectors 10000] [--dim 128] [--k 10]
"""
import argparse
import numpy as np
from backends.brute_force import BruteForceIndex
from benchmark.runner import run_benchmark, compute_recall


def build_ground_truth(train_vecs: np.ndarray, query_vecs: np.ndarray, k: int) -> list[list[int]]:
    bf = BruteForceIndex()
    bf.build(train_vecs)
    results = bf.search_batch(query_vecs, k=k)
    return [list(ids) for ids, _ in results]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_vectors", type=int, default=10_000)
    parser.add_argument("--n_queries", type=int, default=500)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    train = rng.standard_normal((args.n_vectors, args.dim)).astype(np.float32)
    queries = rng.standard_normal((args.n_queries, args.dim)).astype(np.float32)

    print(f"Benchmark: {args.n_vectors:,} vectors | dim={args.dim} | {args.n_queries} queries | k={args.k}")
    print("Computing ground truth...")
    gt = build_ground_truth(train, queries, args.k)

    backends = [("BruteForce-numpy", BruteForceIndex())]
    try:
        from backends.faiss_backend import FAISSBackend
        backends.append(("FAISS-IVFFlat", FAISSBackend("IVFFlat")))
        backends.append(("FAISS-HNSW",    FAISSBackend("HNSW")))
    except ImportError:
        print("FAISS not installed — run `pip install faiss-cpu` to include it")

    print("\nResults:")
    print("-" * 80)
    results = []
    for name, backend in backends:
        r = run_benchmark(backend, name, train, queries, gt, args.k)
        print(r)
        results.append(r)

    best_qps = max(results, key=lambda r: r.qps)
    best_recall = max(results, key=lambda r: r.recall_at_k)
    print("-" * 80)
    print(f"Fastest: {best_qps.backend} ({best_qps.qps:,.0f} QPS)")
    print(f"Most accurate: {best_recall.backend} (recall={best_recall.recall_at_k:.3f})")

if __name__ == "__main__":
    main()
