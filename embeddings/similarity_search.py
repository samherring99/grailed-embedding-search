from typing import List, Optional
import numpy as np
from .embedding_model import ProductEmbeddingModel
from .vector_store import VectorStore
from grailed_api.client import Client as GrailedClient

class SimilaritySearch:
    def __init__(self):
        self.embedding_model = ProductEmbeddingModel()
        self.vector_store = VectorStore()
        self.grailed_client = GrailedClient()
        
    def index_products(self, products: List[dict]):
        """Index product images and their embeddings"""
        embeddings = []
        product_ids = []
        
        for product in products:
            if 'cover_photo' in product and 'url' in product['cover_photo']:
                embedding = self.embedding_model.get_image_embedding(product['cover_photo']['url'])
                if embedding is not None:
                    embeddings.append(embedding[0])  # Remove batch dimension
                    product_ids.append(str(product['id']))
                    
        if embeddings:
            self.vector_store.add_vectors(np.array(embeddings), product_ids)
            
    def find_similar_products(self, 
                            query_image: Optional[str] = None,
                            query_text: Optional[str] = None,
                            k: int = 5):
        """Find similar products by image URL or text description"""
        if query_image is None and query_text is None:
            raise ValueError("Must provide either query_image or query_text")
            
        # Get query embedding
        if query_image is not None:
            query_embedding = self.embedding_model.get_image_embedding(query_image)
        else:
            query_embedding = self.embedding_model.get_text_embedding(query_text)
            
        if query_embedding is None:
            return []
            
        # Search for similar products
        similar_products = self.vector_store.search(query_embedding, k)
        
        # Fetch full product details
        results = []
        for product_id, similarity in similar_products:
            try:
                product = self.grailed_client.find_product_by_id(product_id)
                results.append({
                    'product': product,
                    'similarity_score': similarity
                })
            except Exception as e:
                print(f"Error fetching product {product_id}: {str(e)}")
                
        return results