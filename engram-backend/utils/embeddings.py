"""Embedding utilities for text processing and similarity calculations"""

import numpy as np

from core.logging import logger


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Normalize embedding vector to unit length"""
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    try:
        # Normalize vectors
        a_norm = normalize_embedding(a)
        b_norm = normalize_embedding(b)

        # Calculate cosine similarity
        similarity = np.dot(a_norm, b_norm)
        return float(similarity)

    except Exception as e:
        logger.error(f"Cosine similarity calculation failed: {e}")
        return 0.0


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Euclidean distance between two vectors"""
    try:
        distance = np.linalg.norm(a - b)
        return float(distance)

    except Exception as e:
        logger.error(f"Euclidean distance calculation failed: {e}")
        return float("inf")


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Manhattan distance between two vectors"""
    try:
        distance = np.sum(np.abs(a - b))
        return float(distance)

    except Exception as e:
        logger.error(f"Manhattan distance calculation failed: {e}")
        return float("inf")


def find_most_similar(
    query_embedding: np.ndarray,
    candidate_embeddings: list[np.ndarray],
    top_k: int = 5,
    similarity_threshold: float = 0.0,
) -> list[tuple[int, float]]:
    """Find most similar embeddings to query"""

    similarities = []

    for i, candidate in enumerate(candidate_embeddings):
        similarity = cosine_similarity(query_embedding, candidate)
        if similarity >= similarity_threshold:
            similarities.append((i, similarity))

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_k]


def calculate_embedding_statistics(embeddings: list[np.ndarray]) -> dict:
    """Calculate statistics for a list of embeddings"""

    if not embeddings:
        return {
            "count": 0,
            "dimension": 0,
            "mean_norm": 0.0,
            "std_norm": 0.0,
            "min_norm": 0.0,
            "max_norm": 0.0,
        }

    # Convert to numpy array
    embeddings_array = np.array(embeddings)

    # Calculate norms
    norms = np.linalg.norm(embeddings_array, axis=1)

    return {
        "count": len(embeddings),
        "dimension": embeddings_array.shape[1] if len(embeddings) > 0 else 0,
        "mean_norm": float(np.mean(norms)),
        "std_norm": float(np.std(norms)),
        "min_norm": float(np.min(norms)),
        "max_norm": float(np.max(norms)),
    }


def batch_cosine_similarity(
    query_embeddings: list[np.ndarray], candidate_embeddings: list[np.ndarray]
) -> np.ndarray:
    """Calculate cosine similarity between batches of embeddings"""

    try:
        # Convert to numpy arrays
        query_array = np.array(query_embeddings)
        candidate_array = np.array(candidate_embeddings)

        # Normalize
        query_norm = query_array / np.linalg.norm(query_array, axis=1, keepdims=True)
        candidate_norm = candidate_array / np.linalg.norm(candidate_array, axis=1, keepdims=True)

        # Calculate similarities
        similarities = np.dot(query_norm, candidate_norm.T)

        return similarities

    except Exception as e:
        logger.error(f"Batch cosine similarity calculation failed: {e}")
        return np.array([])


def reduce_embedding_dimension(
    embeddings: np.ndarray, target_dim: int, method: str = "pca"
) -> np.ndarray:
    """Reduce embedding dimension using various methods"""

    try:
        if method == "pca":
            from sklearn.decomposition import PCA

            pca = PCA(n_components=target_dim)
            return pca.fit_transform(embeddings)

        elif method == "random_projection":
            from sklearn.random_projection import GaussianRandomProjection

            rp = GaussianRandomProjection(n_components=target_dim)
            return rp.fit_transform(embeddings)

        elif method == "truncate":
            return embeddings[:, :target_dim]

        else:
            raise ValueError(f"Unknown reduction method: {method}")

    except Exception as e:
        logger.error(f"Embedding dimension reduction failed: {e}")
        return embeddings


def calculate_embedding_quality_score(embedding: np.ndarray) -> float:
    """Calculate quality score for an embedding"""

    try:
        # Check for NaN or infinite values
        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            return 0.0

        # Calculate norm
        norm = np.linalg.norm(embedding)

        # Check if norm is reasonable (not too small or too large)
        if norm < 0.1 or norm > 10.0:
            return 0.5

        # Check for zero vector
        if norm == 0:
            return 0.0

        # Calculate variance (higher variance is generally better)
        variance = np.var(embedding)

        # Normalize variance score
        variance_score = min(1.0, variance * 10)

        # Combine norm and variance scores
        quality_score = 0.7 * min(1.0, norm) + 0.3 * variance_score

        return float(quality_score)

    except Exception as e:
        logger.error(f"Embedding quality score calculation failed: {e}")
        return 0.0


def validate_embedding(embedding: np.ndarray, expected_dim: int) -> bool:
    """Validate embedding format and content"""

    try:
        # Check if it's a numpy array
        if not isinstance(embedding, np.ndarray):
            return False

        # Check dimension
        if len(embedding.shape) != 1:
            return False

        if embedding.shape[0] != expected_dim:
            return False

        # Check for NaN or infinite values
        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            return False

        # Check if all values are finite
        if not np.all(np.isfinite(embedding)):
            return False

        return True

    except Exception as e:
        logger.error(f"Embedding validation failed: {e}")
        return False
