"""Example: index Grailed listings and search by image or text.

For a more complete interface see ``cli.py``.
"""
import logging

from embeddings import SimilaritySearch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# Initialize
searcher = SimilaritySearch()

# Fetch & index products (uses batch embedding under the hood)
products = searcher.grailed_client.find_products(hits_per_page=200)
searcher.index_products(products)

# Optionally save the index for later reuse
# searcher.save_index("./grailed_index")

# Search by image
results = searcher.find_similar_products(
    query_image="https://upload.wikimedia.org/wikipedia/commons/thumb/7/70/Checkerboard_pattern.svg/480px-Checkerboard_pattern.svg.png"
)

# Or search by text description
# results = searcher.find_similar_products(
#     query_text="psychedelic yet structured high-contrast monochrome"
# )

for r in results:
    product = r["product"]
    print(f"{product['title']}  —  similarity: {r['similarity_score']:.3f}")
    for pic in product.get("photos", [])[:2]:
        print(f"  {pic['url']}")
