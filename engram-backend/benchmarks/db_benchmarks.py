"""
Database-integrated benchmarks for Engram
==========================================
Tests actual PostgreSQL + pgvector and Neo4j performance.

Prerequisites:
- Running PostgreSQL with pgvector
- Running Neo4j
- Environment variables configured

Run with: python -m benchmarks.db_benchmarks
"""

import asyncio
import os
import random
import time
from dataclasses import dataclass, field

import numpy as np

try:
    import asyncpg
    from neo4j import AsyncGraphDatabase

    HAS_DB_DEPS = True
except ImportError:
    HAS_DB_DEPS = False
    print("⚠️  Database dependencies not installed. Install with:")
    print("   pip install asyncpg neo4j")


@dataclass
class DBBenchmarkResult:
    name: str
    scale: int
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def p50(self) -> float:
        return sorted(self.latencies_ms)[len(self.latencies_ms) // 2] if self.latencies_ms else 0

    @property
    def p95(self) -> float:
        if not self.latencies_ms:
            return 0
        return sorted(self.latencies_ms)[int(len(self.latencies_ms) * 0.95)]


def generate_embedding(dim: int = 1536) -> list[float]:
    """Generate normalized random embedding"""
    emb = np.random.randn(dim).astype(np.float32)
    emb = emb / np.linalg.norm(emb)
    return emb.tolist()


class PostgresBenchmark:
    """Benchmark PostgreSQL with pgvector"""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool = None

    async def setup(self):
        """Initialize connection pool and create test table"""
        self.pool = await asyncpg.create_pool(self.database_url, min_size=5, max_size=20)

        async with self.pool.acquire() as conn:
            # Enable pgvector
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Create test table
            await conn.execute(
                """
                DROP TABLE IF EXISTS benchmark_memories;
                CREATE TABLE benchmark_memories (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    embedding vector(1536) NOT NULL,
                    timestamp FLOAT NOT NULL,
                    importance_score FLOAT DEFAULT 0
                );
            """
            )

            # Create index
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS benchmark_memories_embedding_idx 
                ON benchmark_memories 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """
            )

    async def teardown(self):
        """Clean up"""
        if self.pool:
            async with self.pool.acquire() as conn:
                await conn.execute("DROP TABLE IF EXISTS benchmark_memories")
            await self.pool.close()

    async def benchmark_insert(self, count: int) -> DBBenchmarkResult:
        """Benchmark memory insertion"""
        result = DBBenchmarkResult(name="postgres_insert", scale=count)

        async with self.pool.acquire() as conn:
            for i in range(count):
                embedding = generate_embedding()
                text = f"Memory entry number {i} with some additional context"

                start = time.perf_counter()
                await conn.execute(
                    """
                    INSERT INTO benchmark_memories (user_id, text, embedding, timestamp, importance_score)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    "benchmark_user",
                    text,
                    str(embedding),  # pgvector accepts string representation
                    time.time(),
                    random.random(),
                )
                elapsed = (time.perf_counter() - start) * 1000
                result.latencies_ms.append(elapsed)

                if (i + 1) % 1000 == 0:
                    print(f"    Inserted {i + 1}/{count}")

        return result

    async def benchmark_vector_search(self, query_count: int = 50) -> DBBenchmarkResult:
        """Benchmark vector similarity search"""
        result = DBBenchmarkResult(name="postgres_vector_search", scale=query_count)

        async with self.pool.acquire() as conn:
            for _ in range(query_count):
                query_embedding = generate_embedding()

                start = time.perf_counter()
                rows = await conn.fetch(
                    """
                    SELECT id, text, 1 - (embedding <=> $1) as similarity
                    FROM benchmark_memories
                    ORDER BY embedding <=> $1
                    LIMIT 10
                    """,
                    str(query_embedding),
                )
                _ = list(rows)  # Consume results
                elapsed = (time.perf_counter() - start) * 1000
                result.latencies_ms.append(elapsed)

        return result

    async def benchmark_filtered_search(self, query_count: int = 50) -> DBBenchmarkResult:
        """Benchmark filtered vector search"""
        result = DBBenchmarkResult(name="postgres_filtered_search", scale=query_count)

        async with self.pool.acquire() as conn:
            for _ in range(query_count):
                query_embedding = generate_embedding()
                min_importance = random.uniform(0.3, 0.7)

                start = time.perf_counter()
                rows = await conn.fetch(
                    """
                    SELECT id, text, 1 - (embedding <=> $1) as similarity
                    FROM benchmark_memories
                    WHERE importance_score > $2
                    ORDER BY embedding <=> $1
                    LIMIT 10
                    """,
                    str(query_embedding),
                    min_importance,
                )
                _ = list(rows)
                elapsed = (time.perf_counter() - start) * 1000
                result.latencies_ms.append(elapsed)

        return result


class Neo4jBenchmark:
    """Benchmark Neo4j graph operations"""

    def __init__(self, uri: str, user: str, password: str):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None

    async def setup(self):
        """Initialize driver and create test data"""
        self.driver = AsyncGraphDatabase.driver(self.uri, auth=(self.user, self.password))

        async with self.driver.session() as session:
            # Clear existing test data
            await session.run("MATCH (n:BenchmarkEntity) DETACH DELETE n")

    async def teardown(self):
        """Clean up"""
        if self.driver:
            async with self.driver.session() as session:
                await session.run("MATCH (n:BenchmarkEntity) DETACH DELETE n")
            await self.driver.close()

    async def benchmark_entity_creation(self, count: int) -> DBBenchmarkResult:
        """Benchmark entity node creation"""
        result = DBBenchmarkResult(name="neo4j_entity_creation", scale=count)

        async with self.driver.session() as session:
            for i in range(count):
                start = time.perf_counter()
                await session.run(
                    """
                    CREATE (e:BenchmarkEntity {
                        id: $id,
                        name: $name,
                        type: $type,
                        timestamp: $timestamp
                    })
                    """,
                    id=f"entity_{i}",
                    name=f"Entity {i}",
                    type=random.choice(["person", "place", "thing", "concept"]),
                    timestamp=time.time(),
                )
                elapsed = (time.perf_counter() - start) * 1000
                result.latencies_ms.append(elapsed)

                if (i + 1) % 500 == 0:
                    print(f"    Created {i + 1}/{count} entities")

        return result

    async def benchmark_relationship_creation(self, count: int) -> DBBenchmarkResult:
        """Benchmark relationship creation"""
        result = DBBenchmarkResult(name="neo4j_relationship_creation", scale=count)

        relation_types = ["KNOWS", "WORKS_AT", "LIVES_IN", "RELATED_TO", "PREFERS"]

        async with self.driver.session() as session:
            # Get all entity IDs
            records = await session.run(
                "MATCH (e:BenchmarkEntity) RETURN e.id as id LIMIT 1000"
            )
            entity_ids = [record["id"] async for record in records]

            if len(entity_ids) < 2:
                print("    Not enough entities for relationship benchmark")
                return result

            for i in range(min(count, len(entity_ids) * 2)):
                source = random.choice(entity_ids)
                target = random.choice(entity_ids)
                if source == target:
                    continue

                start = time.perf_counter()
                await session.run(
                    f"""
                    MATCH (s:BenchmarkEntity {{id: $source}})
                    MATCH (t:BenchmarkEntity {{id: $target}})
                    CREATE (s)-[r:{random.choice(relation_types)} {{
                        confidence: $confidence,
                        timestamp: $timestamp
                    }}]->(t)
                    """,
                    source=source,
                    target=target,
                    confidence=random.random(),
                    timestamp=time.time(),
                )
                elapsed = (time.perf_counter() - start) * 1000
                result.latencies_ms.append(elapsed)

        return result

    async def benchmark_graph_traversal(self, query_count: int = 50) -> DBBenchmarkResult:
        """Benchmark graph traversal queries"""
        result = DBBenchmarkResult(name="neo4j_graph_traversal", scale=query_count)

        async with self.driver.session() as session:
            # Get entity IDs
            records = await session.run(
                "MATCH (e:BenchmarkEntity) RETURN e.id as id LIMIT 100"
            )
            entity_ids = [record["id"] async for record in records]

            if not entity_ids:
                print("    No entities for traversal benchmark")
                return result

            for _ in range(query_count):
                start_entity = random.choice(entity_ids)

                start = time.perf_counter()
                records = await session.run(
                    """
                    MATCH path = (e:BenchmarkEntity {id: $id})-[*1..3]-(connected)
                    RETURN connected.id, connected.name, length(path) as depth
                    LIMIT 20
                    """,
                    id=start_entity,
                )
                _ = [record async for record in records]
                elapsed = (time.perf_counter() - start) * 1000
                result.latencies_ms.append(elapsed)

        return result


async def run_db_benchmarks():
    """Run database benchmarks"""
    if not HAS_DB_DEPS:
        print("Cannot run DB benchmarks without dependencies")
        return

    # Configuration from environment
    pg_url = os.getenv(
        "DATABASE_URL", "postgresql://engram_user:secure_password@localhost:5432/engram_db"
    )
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "secure_password")

    results = []

    # PostgreSQL benchmarks
    print("\n🐘 PostgreSQL + pgvector Benchmarks")
    print("=" * 50)

    try:
        pg_bench = PostgresBenchmark(pg_url)
        await pg_bench.setup()

        # Insert benchmark
        print("\n📝 Insert benchmark (5000 records)...")
        insert_result = await pg_bench.benchmark_insert(5000)
        results.append(insert_result)
        print(f"   P50: {insert_result.p50:.2f}ms, P95: {insert_result.p95:.2f}ms")

        # Vector search
        print("\n🔍 Vector search benchmark...")
        search_result = await pg_bench.benchmark_vector_search(100)
        results.append(search_result)
        print(f"   P50: {search_result.p50:.2f}ms, P95: {search_result.p95:.2f}ms")

        # Filtered search
        print("\n🔍 Filtered search benchmark...")
        filtered_result = await pg_bench.benchmark_filtered_search(100)
        results.append(filtered_result)
        print(f"   P50: {filtered_result.p50:.2f}ms, P95: {filtered_result.p95:.2f}ms")

        await pg_bench.teardown()

    except Exception as e:
        print(f"   ⚠️  PostgreSQL benchmark failed: {e}")

    # Neo4j benchmarks
    print("\n\n🔷 Neo4j Graph Benchmarks")
    print("=" * 50)

    try:
        neo4j_bench = Neo4jBenchmark(neo4j_uri, neo4j_user, neo4j_password)
        await neo4j_bench.setup()

        # Entity creation
        print("\n📝 Entity creation benchmark (1000 nodes)...")
        entity_result = await neo4j_bench.benchmark_entity_creation(1000)
        results.append(entity_result)
        print(f"   P50: {entity_result.p50:.2f}ms, P95: {entity_result.p95:.2f}ms")

        # Relationship creation
        print("\n🔗 Relationship creation benchmark...")
        rel_result = await neo4j_bench.benchmark_relationship_creation(500)
        results.append(rel_result)
        print(f"   P50: {rel_result.p50:.2f}ms, P95: {rel_result.p95:.2f}ms")

        # Graph traversal
        print("\n🌐 Graph traversal benchmark...")
        traversal_result = await neo4j_bench.benchmark_graph_traversal(100)
        results.append(traversal_result)
        print(f"   P50: {traversal_result.p50:.2f}ms, P95: {traversal_result.p95:.2f}ms")

        await neo4j_bench.teardown()

    except Exception as e:
        print(f"   ⚠️  Neo4j benchmark failed: {e}")

    # Summary
    print("\n\n📊 SUMMARY")
    print("=" * 50)
    for r in results:
        print(f"{r.name:35} | P50: {r.p50:8.2f}ms | P95: {r.p95:8.2f}ms")


if __name__ == "__main__":
    asyncio.run(run_db_benchmarks())
