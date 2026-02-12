import faiss
import numpy as np
from typing import Dict, List, Tuple

class VectorStore:
    def __init__(self, vector_dimension: int = 512):
        self.dimension = vector_dimension
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product similarity
        self.id_to_product = {}
        
    def add_vectors(self, vectors: np.ndarray, product_ids: List[str]):
        if len(vectors) != len(product_ids):
            raise ValueError("Number of vectors must match number of product IDs")
            
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        
        # Update ID mapping
        for i, pid in enumerate(product_ids):
            self.id_to_product[self.index.ntotal - len(product_ids) + i] = pid
            
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        # Normalize query vector
        faiss.normalize_L2(query_vector)
        
        # Perform search
        scores, indices = self.index.search(query_vector, k)
        
        # Return product IDs and similarity scores
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx in self.id_to_product:
                results.append((self.id_to_product[idx], float(score)))
                
        return results
