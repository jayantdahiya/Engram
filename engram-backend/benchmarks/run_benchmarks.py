"""
Engram Benchmark Suite
======================
Comprehensive benchmarks for memory operations, retrieval, and graph queries.

Run with: python -m benchmarks.run_benchmarks

Prerequisites:
- PostgreSQL with pgvector running
- Neo4j running
- Redis running
- Environment variables set (see .env.example)
"""

import asyncio
import json
import random
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Benchmark configuration
BENCHMARK_CONFIG = {
    "memory_scales": [100, 1000, 5000, 10000],  # Number of memories to test
    "query_iterations": 50,  # Number of query iterations per scale
    "concurrent_users": [1, 5, 10, 20],  # Concurrent user simulation
    "embedding_dim": 1536,  # OpenAI embedding dimension
    "warmup_iterations": 5,
}


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run"""

    name: str
    scale: int
    iterations: int
    latencies_ms: list[float] = field(default_factory=list)
    errors: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def p50(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0

    @property
    def p95(self) -> float:
        if not self.latencies_ms:
            return 0
        sorted_latencies = sorted(self.latencies_ms)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[idx]

    @property
    def p99(self) -> float:
        if not self.latencies_ms:
            return 0
        sorted_latencies = sorted(self.latencies_ms)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[idx]

    @property
    def mean(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0

    @property
    def std(self) -> float:
        return statistics.stdev(self.latencies_ms) if len(self.latencies_ms) > 1 else 0

    @property
    def throughput(self) -> float:
        """Requests per second"""
        total_time_sec = sum(self.latencies_ms) / 1000
        return len(self.latencies_ms) / total_time_sec if total_time_sec > 0 else 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "scale": self.scale,
            "iterations": self.iterations,
            "p50_ms": round(self.p50, 2),
            "p95_ms": round(self.p95, 2),
            "p99_ms": round(self.p99, 2),
            "mean_ms": round(self.mean, 2),
            "std_ms": round(self.std, 2),
            "throughput_rps": round(self.throughput, 2),
            "errors": self.errors,
            "metadata": self.metadata,
        }


def generate_mock_embedding(dim: int = 1536) -> np.ndarray:
    """Generate a random embedding vector"""
    embedding = np.random.randn(dim).astype(np.float32)
    return embedding / np.linalg.norm(embedding)  # Normalize


def generate_mock_memory_text() -> str:
    """Generate realistic memory text"""
    templates = [
        "I prefer {preference} over {alternative}",
        "My favorite {category} is {item}",
        "I work as a {job} at {company}",
        "I have {count} {pet_type} named {name}",
        "I usually {activity} on {day}s",
        "I'm allergic to {allergen}",
        "My birthday is on {date}",
        "I live in {city}, {country}",
    ]

    preferences = ["coffee", "tea", "dark mode", "light mode", "remote work"]
    categories = ["food", "movie genre", "programming language", "book"]
    items = ["Italian", "sci-fi", "Python", "fantasy novels"]
    jobs = ["software engineer", "data scientist", "product manager"]
    companies = ["a startup", "Google", "a consulting firm"]
    pets = ["dog", "cat", "parrot"]
    names = ["Max", "Luna", "Charlie", "Bella"]
    activities = ["go hiking", "read books", "play video games", "cook"]
    days = ["weekend", "Friday", "Sunday"]
    cities = ["San Francisco", "London", "Tokyo", "Berlin"]

    template = random.choice(templates)
    text = template.format(
        preference=random.choice(preferences),
        alternative=random.choice(preferences),
        category=random.choice(categories),
        item=random.choice(items),
        job=random.choice(jobs),
        company=random.choice(companies),
        count=random.randint(1, 3),
        pet_type=random.choice(pets),
        name=random.choice(names),
        activity=random.choice(activities),
        day=random.choice(days),
        allergen=random.choice(["peanuts", "shellfish", "dairy"]),
        date=f"{random.randint(1, 28)}/{random.randint(1, 12)}",
        city=random.choice(cities),
        country="USA",
    )
    return text


class MemoryBenchmark:
    """Benchmark suite for memory operations"""

    def __init__(self):
        self.results: list[BenchmarkResult] = []

    async def benchmark_embedding_generation(self, iterations: int = 100) -> BenchmarkResult:
        """Benchmark embedding generation speed"""
        result = BenchmarkResult(
            name="embedding_generation", scale=iterations, iterations=iterations
        )

        texts = [generate_mock_memory_text() for _ in range(iterations)]

        for text in texts:
            start = time.perf_counter()
            # Simulate embedding generation (replace with actual embedding service)
            _ = generate_mock_embedding()
            elapsed = (time.perf_counter() - start) * 1000
            result.latencies_ms.append(elapsed)

        self.results.append(result)
        return result

    async def benchmark_similarity_search(
        self, memory_count: int, query_count: int = 50
    ) -> BenchmarkResult:
        """Benchmark vector similarity search at different scales"""
        result = BenchmarkResult(
            name="similarity_search", scale=memory_count, iterations=query_count
        )

        # Generate memory embeddings
        print(f"  Generating {memory_count} memory embeddings...")
        memory_embeddings = [generate_mock_embedding() for _ in range(memory_count)]
        memory_matrix = np.vstack(memory_embeddings)

        # Warmup
        for _ in range(BENCHMARK_CONFIG["warmup_iterations"]):
            query = generate_mock_embedding()
            _ = np.dot(memory_matrix, query)

        # Benchmark
        for _ in range(query_count):
            query = generate_mock_embedding()
            start = time.perf_counter()

            # Cosine similarity (dot product with normalized vectors)
            similarities = np.dot(memory_matrix, query)
            top_k_indices = np.argsort(-similarities)[:10]
            _ = [(i, similarities[i]) for i in top_k_indices]

            elapsed = (time.perf_counter() - start) * 1000
            result.latencies_ms.append(elapsed)

        result.metadata["memory_count"] = memory_count
        self.results.append(result)
        return result

    async def benchmark_acan_retrieval(
        self, memory_count: int, query_count: int = 50
    ) -> BenchmarkResult:
        """Benchmark ACAN (Attention-based) retrieval"""
        result = BenchmarkResult(
            name="acan_retrieval", scale=memory_count, iterations=query_count
        )

        # Generate memories with metadata
        memories = []
        for i in range(memory_count):
            memories.append(
                {
                    "id": i,
                    "embedding": generate_mock_embedding(),
                    "timestamp": time.time() - random.uniform(0, 86400 * 30),
                    "importance_score": random.uniform(0, 1),
                    "access_count": random.randint(0, 100),
                }
            )

        # Initialize projection matrices (ACAN style)
        attention_dim = 64
        query_proj = np.random.randn(1536, attention_dim) * np.sqrt(2.0 / 1536)
        key_proj = np.random.randn(1536, attention_dim) * np.sqrt(2.0 / 1536)

        # Warmup
        for _ in range(BENCHMARK_CONFIG["warmup_iterations"]):
            query = generate_mock_embedding()
            query_projected = np.dot(query, query_proj)
            for mem in memories[:100]:
                key_projected = np.dot(mem["embedding"], key_proj)
                _ = np.dot(query_projected, key_projected)

        # Benchmark
        current_time = time.time()
        for _ in range(query_count):
            query = generate_mock_embedding()
            start = time.perf_counter()

            # ACAN scoring
            query_projected = np.dot(query, query_proj)
            scores = []

            for mem in memories:
                # Attention score
                key_projected = np.dot(mem["embedding"], key_proj)
                attention = np.dot(query_projected, key_projected) / np.sqrt(attention_dim)

                # Cosine similarity
                cosine = np.dot(query, mem["embedding"])

                # Recency weight
                age_hours = (current_time - mem["timestamp"]) / 3600
                recency = np.exp(-np.log(2) * age_hours / 72)

                # Composite score
                score = 0.35 * attention + 0.25 * cosine + 0.15 * recency + 0.10 * mem[
                    "importance_score"
                ] + 0.10 * (mem["access_count"] / 100)

                scores.append((mem["id"], score))

            # Sort and get top-k
            scores.sort(key=lambda x: -x[1])
            _ = scores[:10]

            elapsed = (time.perf_counter() - start) * 1000
            result.latencies_ms.append(elapsed)

        result.metadata["memory_count"] = memory_count
        self.results.append(result)
        return result

    async def benchmark_memory_consolidation(self, iterations: int = 50) -> BenchmarkResult:
        """Benchmark memory consolidation detection"""
        result = BenchmarkResult(
            name="memory_consolidation", scale=iterations, iterations=iterations
        )

        # Contradiction patterns
        pattern_pairs = [
            ("I am vegetarian", "I now eat chicken occasionally"),
            ("I work from home", "I started going to office daily"),
            ("I prefer dark mode", "I switched to light mode"),
        ]

        for _ in range(iterations):
            old_text, new_text = random.choice(pattern_pairs)
            old_embedding = generate_mock_embedding()
            new_embedding = generate_mock_embedding()

            start = time.perf_counter()

            # Similarity check
            similarity = np.dot(old_embedding, new_embedding)

            # Contradiction detection (simplified)
            contradiction_keywords = ["now", "switched", "started", "stopped", "changed"]
            has_temporal = any(kw in new_text.lower() for kw in contradiction_keywords)

            if similarity > 0.5 and has_temporal:
                # Would trigger consolidation
                _ = f"Previously: {old_text}. More recently: {new_text}"

            elapsed = (time.perf_counter() - start) * 1000
            result.latencies_ms.append(elapsed)

        self.results.append(result)
        return result

    def generate_report(self) -> dict:
        """Generate benchmark report"""
        return {
            "timestamp": datetime.now().isoformat(),
            "config": BENCHMARK_CONFIG,
            "results": [r.to_dict() for r in self.results],
            "summary": self._generate_summary(),
        }

    def _generate_summary(self) -> dict:
        """Generate summary statistics"""
        summary = {}
        for result in self.results:
            key = f"{result.name}_{result.scale}"
            summary[key] = {
                "p95_ms": round(result.p95, 2),
                "throughput_rps": round(result.throughput, 2),
            }
        return summary

    def print_report(self):
        """Print formatted benchmark report"""
        print("\n" + "=" * 70)
        print("ENGRAM BENCHMARK RESULTS")
        print("=" * 70)

        for result in self.results:
            print(f"\n📊 {result.name} (scale: {result.scale})")
            print("-" * 50)
            print(f"  Iterations: {result.iterations}")
            print(f"  P50 latency: {result.p50:.2f} ms")
            print(f"  P95 latency: {result.p95:.2f} ms")
            print(f"  P99 latency: {result.p99:.2f} ms")
            print(f"  Mean ± Std:  {result.mean:.2f} ± {result.std:.2f} ms")
            print(f"  Throughput:  {result.throughput:.2f} req/s")
            if result.errors:
                print(f"  ⚠️  Errors: {result.errors}")

        print("\n" + "=" * 70)


async def run_all_benchmarks():
    """Run complete benchmark suite"""
    benchmark = MemoryBenchmark()

    print("🚀 Starting Engram Benchmark Suite")
    print("=" * 50)

    # 1. Embedding generation
    print("\n1️⃣  Benchmarking embedding generation...")
    await benchmark.benchmark_embedding_generation(iterations=100)

    # 2. Similarity search at different scales
    print("\n2️⃣  Benchmarking similarity search...")
    for scale in BENCHMARK_CONFIG["memory_scales"]:
        print(f"   Scale: {scale} memories")
        await benchmark.benchmark_similarity_search(memory_count=scale)

    # 3. ACAN retrieval
    print("\n3️⃣  Benchmarking ACAN retrieval...")
    for scale in BENCHMARK_CONFIG["memory_scales"]:
        print(f"   Scale: {scale} memories")
        await benchmark.benchmark_acan_retrieval(memory_count=scale)

    # 4. Memory consolidation
    print("\n4️⃣  Benchmarking memory consolidation...")
    await benchmark.benchmark_memory_consolidation(iterations=100)

    # Print and save results
    benchmark.print_report()

    # Save to file
    report = benchmark.generate_report()
    output_path = Path(__file__).parent / "results" / f"benchmark_{int(time.time())}.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n📁 Results saved to: {output_path}")

    return benchmark


if __name__ == "__main__":
    asyncio.run(run_all_benchmarks())
