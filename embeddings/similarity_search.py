import logging
from typing import Dict, List, Optional

import numpy as np

from .embedding_model import ProductEmbeddingModel
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class SimilaritySearch:
    """High-level facade: fetch listings → embed → index → query.

    Wraps :class:`ProductEmbeddingModel` and :class:`VectorStore` and
    optionally talks to the Grailed API for product fetching.
    """

    def __init__(self, model_name: str = "ViT-B/32"):
        self.embedding_model = ProductEmbeddingModel(model_name=model_name)
        self.vector_store = VectorStore()

        # Lazy Grailed client — only imported/used when needed
        self._grailed_client = None

    @property
    def grailed_client(self):
        """Lazily create the Grailed API client on first access."""
        if self._grailed_client is None:
            from grailed_api.client import Client as GrailedClient
            self._grailed_client = GrailedClient()
        return self._grailed_client

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_products(self, products: List[dict]) -> int:
        """Embed product cover photos and add them to the vector store.

        Uses batch embedding for efficiency.  Returns the number of
        products successfully indexed.
        """
        urls: List[str] = []
        ids: List[str] = []

        for product in products:
            cover = product.get("cover_photo", {})
            url = cover.get("url")
            if url:
                urls.append(url)
                ids.append(str(product["id"]))

        if not urls:
            logger.warning("No indexable products (no cover photos)")
            return 0

        embeddings, valid_indices = self.embedding_model.get_image_embeddings_batch(urls)

        if len(valid_indices) == 0:
            logger.warning("All image downloads failed — nothing indexed")
            return 0

        valid_ids = [ids[i] for i in valid_indices]
        self.vector_store.add_vectors(embeddings, valid_ids)

        logger.info("Indexed %d / %d products", len(valid_ids), len(products))
        return len(valid_ids)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def find_similar_products(
        self,
        query_image: Optional[str] = None,
        query_text: Optional[str] = None,
        k: int = 5,
        fetch_details: bool = True,
    ) -> List[Dict]:
        """Find the *k* most similar products.

        Provide exactly one of *query_image* (URL) or *query_text*.
        If *fetch_details* is True (default), full product info is
        fetched from Grailed for each result; otherwise only the ID
        and similarity score are returned.
        """
        if query_image is None and query_text is None:
            raise ValueError("Must provide either query_image or query_text")

        if query_image is not None:
            query_embedding = self.embedding_model.get_image_embedding(query_image)
        else:
            query_embedding = self.embedding_model.get_text_embedding(query_text)

        if query_embedding is None:
            return []

        matches = self.vector_store.search(query_embedding, k)

        results: List[Dict] = []
        for product_id, similarity in matches:
            entry: Dict = {"product_id": product_id, "similarity_score": similarity}
            if fetch_details:
                try:
                    entry["product"] = self.grailed_client.find_product_by_id(product_id)
                except Exception as e:
                    logger.warning("Could not fetch product %s: %s", product_id, e)
            results.append(entry)

        return results

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save_index(self, directory: str) -> None:
        """Persist the current vector store to disk."""
        self.vector_store.save(directory)

    def load_index(self, directory: str) -> None:
        """Load a previously-saved vector store from disk."""
        self.vector_store.load(directory)
