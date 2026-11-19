"""Try retrieval + generation from the command line.

Usage:
    python -m app.ask "How many IPv4 addresses are reserved in a subnet?"
    python -m app.ask --retrieve-only "your question"     # skip the LLM call
"""

import sys

from app.generator import generate
from app.retriever import search


def main(argv: list[str]) -> None:
    args = argv[1:]
    retrieve_only = "--retrieve-only" in args
    question = " ".join(a for a in args if a != "--retrieve-only").strip()
    if not question:
        sys.exit('usage: python -m app.ask [--retrieve-only] "your question"')

    chunks = search(question)
    print("Retrieved chunks:", flush=True)
    for c in chunks:
        print(f"  {c['score']:.3f}  {c['filename']} p.{c['page']}: {c['content'][:100]}", flush=True)

    if not retrieve_only:
        print("\nCalling the LLM...", flush=True)
        print("\nAnswer:")
        print(generate(question, chunks))


if __name__ == "__main__":
    main(sys.argv)
