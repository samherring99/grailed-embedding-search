# grailed-embedding-search

Semantic similarity search over [Grailed](https://www.grailed.com/) listings using CLIP embeddings and FAISS.

Search by **image URL** or **text description** to find visually similar products across Grailed's marketplace.

## How it works

1. **Fetch listings** from Grailed via the `grailed-api` client
2. **Embed cover photos** using OpenAI's [CLIP](https://github.com/openai/CLIP) (ViT-B/32) — produces 512-dim vectors
3. **Index embeddings** in a FAISS vector store (inner product / cosine similarity)
4. **Query** with an image URL or text description to find the most similar listings

## Setup

```bash
# Clone and install
git clone https://github.com/samherring99/grailed-embedding-search.git
cd grailed-embedding-search
pip install -r requirements.txt
```

## Usage

```python
from embeddings import SimilaritySearch

# Initialize (loads CLIP model)
searcher = SimilaritySearch()

# Fetch and index products from Grailed
products = searcher.grailed_client.find_products(hits_per_page=200)
searcher.index_products(products)

# Search by image
results = searcher.find_similar_products(
    query_image="https://example.com/jacket.jpg"
)

# Or search by text description
results = searcher.find_similar_products(
    query_text="vintage black leather jacket"
)

for r in results:
    print(f"{r['product']['title']} — similarity: {r['similarity_score']:.3f}")
```

## Project Structure

```
embeddings/
├── __init__.py             # Public API exports
├── embedding_model.py      # CLIP-based image & text embedding
├── vector_store.py         # FAISS index wrapper
└── similarity_search.py    # High-level search orchestrator
similarity.py               # Example usage script
```

## TODO

- [ ] Persistent vector store (save/load FAISS index to disk)
- [ ] Embedding cache (avoid re-embedding known products)
- [ ] Rate limiting for Grailed API calls
- [ ] Async embedding pipeline for faster indexing
- [ ] Search result visualization (matplotlib grid)
- [ ] CLI interface

