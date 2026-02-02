"""ACAN (Attention-based Context-Aware Network) retrieval system

Performance Note (Benchmark: Feb 2026, Raspberry Pi 5):
========================================================
Current ACAN retrieval scales linearly O(n) with memory count:
- 100 memories:   ~7ms   (160 req/s)
- 1,000 memories: ~80ms  (16 req/s)
- 10,000 memories: ~620ms (1.6 req/s)

Pure similarity search is ~60x faster at scale.

OPTIMIZATION OPPORTUNITIES:
1. **Two-stage retrieval** (recommended):
   - Stage 1: Fast vector similarity to get top-100 candidates
   - Stage 2: ACAN scoring only on candidates
   - Expected: 10-20ms even at 100K memories

2. **Batch matrix operations**:
   - Replace per-memory loops with batched numpy/torch operations
   - Stack all embeddings into matrix, compute attention in one matmul
   - Expected: 5-10x speedup

3. **Approximate nearest neighbor (ANN)**:
   - Use FAISS, Annoy, or ScaNN for initial filtering
   - Reduces candidate set before expensive scoring

4. **Caching**:
   - Cache projected query vectors
   - Pre-compute and cache memory key projections on insert

5. **GPU acceleration**:
   - Move projection matrices to GPU (torch/cupy)
   - Batch process with CUDA kernels

See benchmarks/README.md for detailed performance data.
"""

import time
from typing import Any

import numpy as np

from core.logging import logger
from services.embedding_service import embedding_service


class ACANRetrievalSystem:
    """Enhanced retrieval system combining ACAN attention with Mem0 deployment signals
    
    Note: Current implementation is O(n) per query. For production use with >1000 memories,
    consider implementing two-stage retrieval (see module docstring for optimization guide).
    """

    def __init__(self, attention_dim: int = 64):
        self.attention_dim = attention_dim
        self.query_projection = self._init_projection_matrix()
        self.key_projection = self._init_projection_matrix()
        self.value_projection = self._init_projection_matrix()

    def _init_projection_matrix(self) -> np.ndarray:
        """Initialize projection matrix with Xavier initialization"""
        return np.random.randn(1536, self.attention_dim) * np.sqrt(2.0 / 1536)

    async def compute_composite_scores(
        self,
        query_embedding: np.ndarray,
        memories: list[dict[str, Any]],
        current_time: float,
        user_context: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Compute composite relevance scores using multiple signals
        
        TODO: Optimize for scale (see module docstring):
        - Batch matrix operations instead of per-memory loops
        - Pre-filter with fast vector search before scoring
        """

        if not memories:
            return np.array([])

        # Extract embeddings and metadata
        # TODO: Stack into matrix for batched operations
        memory_embeddings = [np.array(mem["embedding"]) for mem in memories]
        timestamps = [mem["timestamp"] for mem in memories]
        importance_scores = [mem.get("importance_score", 0.0) for mem in memories]
        access_counts = [mem.get("access_count", 0) for mem in memories]

        # 1. Cross-attention scores
        attention_scores = await self._compute_attention(query_embedding, memory_embeddings)

        # 2. Cosine similarity scores
        # TODO: Vectorize with matrix multiplication: scores = embeddings_matrix @ query
        cosine_scores = np.array(
            [
                await embedding_service.calculate_similarity(query_embedding, mem_emb)
                for mem_emb in memory_embeddings
            ]
        )

        # 3. Recency weights
        recency_weights = np.array([self._recency_weight(ts, current_time) for ts in timestamps])

        # 4. Importance scores (normalized)
        importance_weights = np.array(importance_scores)
        if np.max(importance_weights) > 0:
            importance_weights = importance_weights / np.max(importance_weights)

        # 5. Access frequency weights
        access_weights = np.array(access_counts)
        if np.max(access_weights) > 0:
            access_weights = access_weights / np.max(access_weights)

        # 6. Context-aware weights (if user context provided)
        context_weights = np.ones(len(memories))
        if user_context:
            context_weights = await self._compute_context_weights(memories, user_context)

        # Weighted combination with learned weights
        composite_scores = (
            0.35 * attention_scores  # Cross-attention
            + 0.25 * cosine_scores  # Semantic similarity
            + 0.15 * recency_weights  # Temporal relevance
            + 0.10 * importance_weights  # Memory importance
            + 0.10 * access_weights  # Access frequency
            + 0.05 * context_weights  # Context relevance
        )

        return composite_scores

    async def _compute_attention(
        self, query_embedding: np.ndarray, memory_embeddings: list[np.ndarray]
    ) -> np.ndarray:
        """Compute cross-attention between query and memories
        
        TODO: Vectorize this entire function:
        ```python
        # Batched version (10x faster):
        memory_matrix = np.vstack(memory_embeddings)  # (n, 1536)
        keys = memory_matrix @ self.key_projection    # (n, 64)
        query_proj = query_embedding @ self.query_projection  # (64,)
        scores = keys @ query_proj / np.sqrt(self.attention_dim)  # (n,)
        ```
        """

        # Project query to attention space
        query_projected = np.dot(query_embedding, self.query_projection)

        # TODO: Replace loop with batched matmul (see docstring above)
        attention_scores = []
        for memory_embedding in memory_embeddings:
            # Project memory to key space
            key_projected = np.dot(memory_embedding, self.key_projection)

            # Compute attention score
            score = np.dot(query_projected, key_projected) / np.sqrt(self.attention_dim)
            attention_scores.append(score)

        attention_scores = np.array(attention_scores)

        # Apply softmax normalization
        attention_probs = np.exp(attention_scores - np.max(attention_scores))
        attention_probs = attention_probs / np.sum(attention_probs)

        return attention_probs

    def _recency_weight(
        self, timestamp: float, current_time: float, half_life_hours: float = 72.0
    ) -> float:
        """Calculate recency weight with exponential decay"""
        age_hours = max(0.0, (current_time - timestamp) / 3600.0)
        return np.exp(-np.log(2) * age_hours / half_life_hours)

    async def _compute_context_weights(
        self, memories: list[dict[str, Any]], user_context: dict[str, Any]
    ) -> np.ndarray:
        """Compute context-aware weights based on user context"""

        context_weights = np.ones(len(memories))

        # Extract context features
        current_conversation = user_context.get("conversation_id")
        user_preferences = user_context.get("preferences", {})
        recent_topics = user_context.get("recent_topics", [])

        for i, memory in enumerate(memories):
            weight = 1.0

            # Boost memories from current conversation
            if current_conversation and memory.get("conversation_id") == current_conversation:
                weight *= 1.2

            # Boost memories matching user preferences
            memory_text = memory.get("text", "").lower()
            for preference, _value in user_preferences.items():
                if preference.lower() in memory_text:
                    weight *= 1.1

            # Boost memories related to recent topics
            for topic in recent_topics:
                if topic.lower() in memory_text:
                    weight *= 1.15

            context_weights[i] = weight

        # Normalize weights
        if np.max(context_weights) > 0:
            context_weights = context_weights / np.max(context_weights)

        return context_weights

    async def memory_distillation(
        self,
        memories: list[dict[str, Any]],
        scores: np.ndarray,
        threshold: float = 0.3,
        max_memories: int = 10,
    ) -> list[dict[str, Any]]:
        """Apply memory distillation for noise reduction"""

        if len(memories) == 0:
            return []

        # Filter by threshold
        valid_indices = np.where(scores >= threshold)[0]

        if len(valid_indices) == 0:
            # If no memories meet threshold, take top memories anyway
            valid_indices = np.argsort(-scores)[:max_memories]

        # Sort by scores
        sorted_indices = valid_indices[np.argsort(-scores[valid_indices])]

        # Apply diversity filtering to avoid redundant memories
        distilled_memories = []
        used_embeddings = []

        for idx in sorted_indices:
            if len(distilled_memories) >= max_memories:
                break

            memory = memories[idx]
            memory_embedding = np.array(memory["embedding"])

            # Check for similarity with already selected memories
            is_diverse = True
            for used_emb in used_embeddings:
                similarity = await embedding_service.calculate_similarity(
                    memory_embedding, used_emb
                )
                if similarity > 0.8:  # High similarity threshold
                    is_diverse = False
                    break

            if is_diverse:
                distilled_memories.append(memory)
                used_embeddings.append(memory_embedding)

        logger.debug(f"Memory distillation: {len(memories)} -> {len(distilled_memories)} memories")
        return distilled_memories

    async def retrieve_with_reranking(
        self,
        query_embedding: np.ndarray,
        memories: list[dict[str, Any]],
        top_k: int = 5,
        user_context: dict[str, Any] | None = None,
        apply_distillation: bool = True,
    ) -> tuple[list[dict[str, Any]], np.ndarray]:
        """Retrieve memories with multi-stage ranking"""

        if not memories:
            return [], np.array([])

        current_time = time.time()

        # Stage 1: Compute composite scores
        composite_scores = await self.compute_composite_scores(
            query_embedding, memories, current_time, user_context
        )

        # Stage 2: Initial ranking
        ranked_indices = np.argsort(-composite_scores)
        top_indices = ranked_indices[: min(top_k * 2, len(memories))]  # Get 2x for reranking
        top_memories = [memories[i] for i in top_indices]
        top_scores = composite_scores[top_indices]

        # Stage 3: Apply distillation if requested
        if apply_distillation:
            # Use adaptive threshold based on score distribution
            if len(top_scores) > 1:
                threshold = max(0.2, np.percentile(top_scores, 25))
            else:
                threshold = 0.3

            distilled_memories = await self.memory_distillation(
                top_memories, top_scores, threshold, top_k
            )

            # Recompute scores for distilled memories
            if distilled_memories:
                final_scores = await self.compute_composite_scores(
                    query_embedding, distilled_memories, current_time, user_context
                )
                return distilled_memories, final_scores
            else:
                return [], np.array([])
        else:
            # Return top-k without distillation
            final_memories = top_memories[:top_k]
            final_scores = top_scores[:top_k]
            return final_memories, final_scores

    async def get_retrieval_explanation(
        self,
        query_embedding: np.ndarray,
        retrieved_memories: list[dict[str, Any]],
        scores: np.ndarray,
    ) -> dict[str, Any]:
        """Generate explanation for retrieval results"""

        explanations = []

        for i, (memory, score) in enumerate(zip(retrieved_memories, scores, strict=False)):
            memory_embedding = np.array(memory["embedding"])

            # Calculate individual component scores
            cosine_sim = await embedding_service.calculate_similarity(
                query_embedding, memory_embedding
            )

            recency_weight = self._recency_weight(memory["timestamp"], time.time())

            importance_weight = memory.get("importance_score", 0.0)
            access_weight = memory.get("access_count", 0)

            explanation = {
                "memory_id": memory.get("id"),
                "text_preview": memory.get("text", "")[:100] + "...",
                "overall_score": float(score),
                "component_scores": {
                    "cosine_similarity": float(cosine_sim),
                    "recency_weight": float(recency_weight),
                    "importance_weight": float(importance_weight),
                    "access_weight": float(access_weight),
                },
                "rank": i + 1,
            }

            explanations.append(explanation)

        return {
            "total_retrieved": len(retrieved_memories),
            "explanations": explanations,
            "retrieval_metadata": {
                "attention_dimension": self.attention_dim,
                "distillation_applied": True,
                "timestamp": time.time(),
            },
        }


# Global ACAN retrieval system instance
acan_retrieval = ACANRetrievalSystem()
