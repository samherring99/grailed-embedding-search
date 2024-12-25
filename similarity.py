from embeddings.similarity_search import SimilaritySearch

# Initialize
searcher = SimilaritySearch()

# Get some products to index
products = searcher.grailed_client.find_products(hits_per_page=200)

# Index the products
searcher.index_products(products)

# Find similar products by image
similar_products = searcher.find_similar_products(
    query_image="https://upload.wikimedia.org/wikipedia/commons/thumb/7/70/Checkerboard_pattern.svg/480px-Checkerboard_pattern.svg.png"
)

# Or find similar products by text description
# similar_products = searcher.find_similar_products(
#     query_text="psychedelic yet structured high-contrast monochrome"
# )

for p in similar_products:
    print(p['product']['title'])
    print("Similarity:" + str(p['similarity_score']))
    for pic in p['product']['photos'][:2]:
        print(pic['url'])