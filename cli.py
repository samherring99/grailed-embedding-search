#!/usr/bin/env python3
"""Command-line interface for grailed-embedding-search.

Examples
--------
  # Index 200 listings and search by text
  python cli.py index --count 200
  python cli.py search --text "vintage black leather jacket"

  # Search by image URL
  python cli.py search --image "https://example.com/jacket.jpg"

  # Save / load a persistent index
  python cli.py index --count 500 --save ./my_index
  python cli.py search --text "patchwork denim" --load ./my_index
"""
import argparse
import logging
import sys

from embeddings import SimilaritySearch


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Semantic similarity search over Grailed listings."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    sub = parser.add_subparsers(dest="command")

    # --- index ---
    idx = sub.add_parser("index", help="Fetch and index Grailed listings")
    idx.add_argument(
        "-n", "--count", type=int, default=200, help="Number of listings to fetch"
    )
    idx.add_argument(
        "--save", metavar="DIR", help="Save the index to this directory after indexing"
    )

    # --- search ---
    srch = sub.add_parser("search", help="Query the index")
    group = srch.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Text query")
    group.add_argument("--image", help="Image URL query")
    srch.add_argument(
        "-k", type=int, default=5, help="Number of results (default: 5)"
    )
    srch.add_argument(
        "--load", metavar="DIR", help="Load a previously saved index"
    )
    srch.add_argument(
        "--no-fetch",
        action="store_true",
        help="Skip fetching full product details (faster)",
    )

    return parser


def cmd_index(args, searcher: SimilaritySearch) -> None:
    print(f"Fetching {args.count} listings from Grailed …")
    products = searcher.grailed_client.find_products(hits_per_page=args.count)
    indexed = searcher.index_products(products)
    print(f"Indexed {indexed} / {len(products)} products.")

    if args.save:
        searcher.save_index(args.save)
        print(f"Index saved to {args.save}/")


def cmd_search(args, searcher: SimilaritySearch) -> None:
    if args.load:
        searcher.load_index(args.load)
        print(f"Loaded index from {args.load}/")

    results = searcher.find_similar_products(
        query_image=args.image,
        query_text=args.text,
        k=args.k,
        fetch_details=not args.no_fetch,
    )

    if not results:
        print("No results found.")
        return

    for i, r in enumerate(results, 1):
        score = r["similarity_score"]
        if "product" in r:
            title = r["product"].get("title", "Unknown")
            print(f"  {i}. {title}  (score: {score:.3f})")
        else:
            print(f"  {i}. product_id={r['product_id']}  (score: {score:.3f})")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    searcher = SimilaritySearch()

    if args.command == "index":
        cmd_index(args, searcher)
    elif args.command == "search":
        cmd_search(args, searcher)


if __name__ == "__main__":
    main()

