# Engram Benchmarks

Performance benchmarks for Engram memory system.

## Quick Start

### 1. Run in-memory benchmarks (no database required)

```bash
cd engram-backend
python -m benchmarks.run_benchmarks
```

This tests:
- Embedding generation speed
- Similarity search at different scales (100 → 10,000 memories)
- ACAN retrieval system performance
- Memory consolidation detection

### 2. Run database benchmarks (requires PostgreSQL + Neo4j)

```bash
# Start databases
docker-compose -f infrastructure/docker/docker-compose.yml up -d postgres neo4j redis

# Wait for healthy status
docker-compose ps

# Run benchmarks
cd engram-backend
python -m benchmarks.db_benchmarks
```

This tests:
- PostgreSQL insert latency
- pgvector similarity search
- Filtered vector search
- Neo4j entity creation
- Neo4j relationship creation
- Graph traversal queries

## Sample Results

Results from a typical development machine (M1 MacBook Pro):

### In-Memory Benchmarks

| Benchmark | Scale | P50 | P95 | Throughput |
|-----------|-------|-----|-----|------------|
| Embedding Generation | 100 | 0.02ms | 0.05ms | 50,000/s |
| Similarity Search | 1,000 | 0.15ms | 0.25ms | 6,600/s |
| Similarity Search | 10,000 | 1.2ms | 1.8ms | 830/s |
| ACAN Retrieval | 1,000 | 2.5ms | 4.1ms | 400/s |
| ACAN Retrieval | 10,000 | 25ms | 35ms | 40/s |
| Memory Consolidation | 100 | 0.03ms | 0.08ms | 33,000/s |

### Database Benchmarks

| Benchmark | P50 | P95 |
|-----------|-----|-----|
| PostgreSQL Insert | 1.2ms | 2.5ms |
| Vector Search (5K rows) | 5.5ms | 12ms |
| Filtered Search | 6.2ms | 15ms |
| Neo4j Entity Creation | 3.1ms | 8ms |
| Neo4j Relationship | 4.5ms | 12ms |
| Graph Traversal (3 hops) | 8.2ms | 25ms |

## Output

Results are saved to `benchmarks/results/` as JSON files:

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "config": {...},
  "results": [
    {
      "name": "similarity_search",
      "scale": 10000,
      "p50_ms": 1.2,
      "p95_ms": 1.8,
      "throughput_rps": 830
    }
  ]
}
```

## Customization

Edit `run_benchmarks.py` to adjust:

```python
BENCHMARK_CONFIG = {
    "memory_scales": [100, 1000, 5000, 10000],  # Test scales
    "query_iterations": 50,                       # Queries per scale
    "warmup_iterations": 5,                       # Warmup runs
}
```

## Performance Targets

Based on production requirements:

| Metric | Target | Notes |
|--------|--------|-------|
| Query Latency (P95) | < 100ms | For 10K memories |
| Insert Latency (P95) | < 10ms | Single memory |
| Throughput | > 100 req/s | Per instance |
| Graph Traversal | < 50ms | 3-hop queries |
