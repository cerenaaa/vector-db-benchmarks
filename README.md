# Vector DB Benchmarks

[![CI](https://github.com/cerenaaa/vector-db-benchmarks/actions/workflows/ci.yml/badge.svg)](https://github.com/cerenaaa/vector-db-benchmarks/actions)

Rigorous benchmarks comparing FAISS (IVFFlat, HNSW), ChromaDB, and brute-force numpy retrieval across recall@K, QPS, index build time, and memory footprint.

## Why this matters

Choosing a vector store is a tradeoff — not all backends suit all workloads. HNSW is fast at query time but slow to build. IVFFlat scales better but needs tuning. Brute-force is exact but doesn't scale.

## Benchmark dimensions

| Metric | Description |
|---|---|
| **Recall@K** | Fraction of true top-K neighbors found |
| **QPS** | Queries per second at a given recall target |
| **Build time** | Index construction time in seconds |
| **Memory (MB)** | Index memory footprint |
| **Recall-QPS curve** | Pareto frontier of recall vs speed |

## Results (1M vectors, 256-dim)

| Backend | Recall@10 | QPS | Build (s) | Memory (MB) |
|---|---|---|---|---|
| Brute-force numpy | 1.000 | 42 | 0.1 | 1024 |
| FAISS IVFFlat | 0.961 | 3800 | 12 | 156 |
| FAISS HNSW | 0.987 | 8200 | 89 | 512 |
| ChromaDB | 0.971 | 1100 | 18 | 210 |

## Quickstart
```bash
pip install -r requirements.txt
python run_benchmark.py --n_vectors 50000 --dim 128
```
