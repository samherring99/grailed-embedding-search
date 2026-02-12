#!/usr/bin/env python3
"""Search the Grailed index for items similar to a given image URL."""
import json
import logging
import sys

from embeddings import SimilaritySearch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def search(image_url: str, k: int = 5):
    searcher = SimilaritySearch()
    searcher.load_index("./grailed_index")

    # Load cached metadata
    with open("./grailed_index/product_metadata.json") as f:
        metadata = json.load(f)

    # Search by image (no need to re-fetch from Grailed API)
    results = searcher.find_similar_products(
        query_image=image_url, k=k, fetch_details=False
    )

    output = []
    for r in results:
        pid = r["product_id"]
        score = r["similarity_score"]
        meta = metadata.get(pid, {})
        output.append({
            "title": meta.get("title", "Unknown"),
            "designer": meta.get("designer", ""),
            "price": meta.get("price", ""),
            "size": meta.get("size", ""),
            "score": round(score, 3),
            "url": meta.get("url", f"https://www.grailed.com/listings/{pid}"),
        })

    return output


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python search_image.py <image_url> [k]")
        sys.exit(1)

    url = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    results = search(url, k)

    for i, r in enumerate(results, 1):
        print(f"\n{i}. {r['title']}")
        print(f"   Designer: {r['designer']} | Price: ${r['price']} | Size: {r['size']}")
        print(f"   Similarity: {r['score']}")
        print(f"   {r['url']}")

