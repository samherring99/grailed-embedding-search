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
git clone https://github.com/samherring99/grailed-embedding-search.git
cd grailed-embedding-search
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### CLI

```bash
# Index 200 listings
python cli.py index --count 200

# Index and save to disk for later
python cli.py index --count 500 --save ./my_index

# Search by text
python cli.py search --text "vintage black leather jacket" --load ./my_index

# Search by image URL
python cli.py search --image "https://example.com/jacket.jpg" --load ./my_index

# Verbose / debug output
python cli.py -v search --text "patchwork denim" --load ./my_index
```

### Python API

```python
from embeddings import SimilaritySearch

searcher = SimilaritySearch()

# Fetch and index products (uses batch embedding)
products = searcher.grailed_client.find_products(hits_per_page=200)
searcher.index_products(products)

# Save for later reuse
searcher.save_index("./grailed_index")

# Search by image
results = searcher.find_similar_products(
    query_image="https://example.com/jacket.jpg"
)

# Or search by text description
results = searcher.find_similar_products(
    query_text="vintage black leather jacket"
)

# Quick search without fetching full product details
results = searcher.find_similar_products(
    query_text="rick owens geobasket", fetch_details=False
)

for r in results:
    print(f"{r['product_id']} — similarity: {r['similarity_score']:.3f}")
```

### Loading a saved index

```python
searcher = SimilaritySearch()
searcher.load_index("./grailed_index")
results = searcher.find_similar_products(query_text="oversized hoodie")
```

## Project Structure

```
embeddings/
├── __init__.py             # Public API exports
├── embedding_model.py      # CLIP-based image & text embedding (single + batch)
├── vector_store.py         # FAISS index wrapper with save/load
└── similarity_search.py    # High-level search orchestrator
cli.py                      # Command-line interface
similarity.py               # Example usage script
```

## TODO

- [ ] Embedding cache (avoid re-embedding known product URLs)
- [ ] Async/threaded image downloads for faster batch indexing
- [ ] Search result visualization (matplotlib grid of cover photos)
- [ ] Filter by category, designer, price range before search
- [ ] Web UI (Gradio or Streamlit)

