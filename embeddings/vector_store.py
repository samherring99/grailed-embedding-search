import json
import logging
import os
from typing import List, Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-backed vector store with cosine similarity search.

    Vectors are L2-normalised on insert so that inner-product search
    is equivalent to cosine similarity.  The index and its ID mapping
    can be persisted to / loaded from disk.
    """

    def __init__(self, vector_dimension: int = 512):
        self.dimension = vector_dimension
        self.index = faiss.IndexFlatIP(self.dimension)
        self.id_to_product: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add_vectors(self, vectors: np.ndarray, product_ids: List[str]) -> None:
        """Add normalised vectors and their product IDs to the index."""
        if len(vectors) != len(product_ids):
            raise ValueError("Number of vectors must match number of product IDs")

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        faiss.normalize_L2(vectors)
        self.index.add(vectors)

        for i, pid in enumerate(product_ids):
            self.id_to_product[self.index.ntotal - len(product_ids) + i] = pid

        logger.info("Added %d vectors (total: %d)", len(product_ids), self.index.ntotal)

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """Return the *k* most similar product IDs with their scores."""
        if self.index.ntotal == 0:
            logger.warning("Search called on an empty index")
            return []

        query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)
        faiss.normalize_L2(query_vector)

        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, k)

        results: List[Tuple[str, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx in self.id_to_product:
                results.append((self.id_to_product[idx], float(score)))

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """Save the FAISS index and ID mapping to *directory*."""
        os.makedirs(directory, exist_ok=True)
        index_path = os.path.join(directory, "index.faiss")
        mapping_path = os.path.join(directory, "id_mapping.json")

        faiss.write_index(self.index, index_path)
        with open(mapping_path, "w") as f:
            json.dump({str(k): v for k, v in self.id_to_product.items()}, f)

        logger.info("Saved index (%d vectors) to %s", self.index.ntotal, directory)

    def load(self, directory: str) -> None:
        """Load a previously-saved FAISS index and ID mapping."""
        index_path = os.path.join(directory, "index.faiss")
        mapping_path = os.path.join(directory, "id_mapping.json")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"No index found at {index_path}")

        self.index = faiss.read_index(index_path)
        with open(mapping_path) as f:
            self.id_to_product = {int(k): v for k, v in json.load(f).items()}

        logger.info("Loaded index (%d vectors) from %s", self.index.ntotal, directory)

    @property
    def size(self) -> int:
        """Number of vectors currently in the index."""
        return self.index.ntotal

