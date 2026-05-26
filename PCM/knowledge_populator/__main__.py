"""
CLI entry point for the knowledge populator.

Usage:
    # Extract from a single text
    python -m knowledge_populator "I just bought a silver Honda Civic" --agent Alice

    # Extract from all conversations in a JSON file → single output
    python -m knowledge_populator -f sample_data/conversations.json -o output/kg.ttl

    # Extract one TTL per conversation
    python -m knowledge_populator -f sample_data/conversations.json --per-conversation -o output/

    # Use a different model
    python -m knowledge_populator -f sample_data/conversations.json --model gpt-4o -o output/kg.ttl
"""

import sys
import argparse
from pathlib import Path

from .extractor import (
    extract_turtle_from_text,
    extract_conversation,
    _create_session,
)
from .utils import (
    load_conversations,
    add_turtle_to_store,
    _load_ontology_schema,
)
from .rdf_store import RDFMemoryStore


def main():
    parser = argparse.ArgumentParser(
        description="Extract a personal knowledge graph from conversational text"
    )
    parser.add_argument(
        "text", nargs="?",
        help="Text to extract (or use -f for a conversations file)",
    )
    parser.add_argument(
        "-f", "--file",
        help="Path to conversations.json file",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output path: a .ttl file, or a directory with --per-conversation",
        default="output/kg.ttl",
    )
    parser.add_argument(
        "-a", "--agent",
        help="Name of the person whose knowledge graph this is",
        default="User",
    )
    parser.add_argument(
        "--model",
        help="LLM model in litellm format",
        default="anthropic/claude-sonnet-4-5-20250929",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Number of user messages per LLM call (default: 1)",
    )
    parser.add_argument(
        "--per-conversation", action="store_true",
        help="Generate one TTL file per conversation (output must be a directory)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max conversations to process (0 = all)",
    )
    parser.add_argument(
        "--no-store", action="store_true",
        help="Print extracted Turtle but don't save to RDF store",
    )

    args = parser.parse_args()

    # --- Single text mode ---
    if args.text and not args.file:
        print("=" * 60)
        print("Extracting knowledge graph...")
        print(f"Model: {args.model}")
        print("=" * 60)

        ttl = extract_turtle_from_text(
            args.text, agent_name=args.agent, model=args.model
        )
        print(f"\n{ttl}")

        if not args.no_store:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            store = RDFMemoryStore(storage_path=args.output, agent_id="extractor")
            _load_ontology_schema(store)
            added = add_turtle_to_store(ttl, store)
            print(f"\nAdded {added} triples → {args.output}")
        return

    # --- File mode ---
    if not args.file:
        print("Error: Provide text or use -f/--file")
        parser.print_help()
        sys.exit(1)

    conversations = load_conversations(args.file)
    if args.limit:
        conversations = conversations[:args.limit]

    print("=" * 60)
    print("Knowledge Graph Extraction")
    print(f"Model:         {args.model}")
    print(f"Agent:         {args.agent}")
    print(f"Conversations: {len(conversations)}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Output:        {args.output}")
    print("=" * 60)

    session = _create_session(args.model)

    if args.per_conversation:
        # One TTL per conversation
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        for conv in conversations:
            conv_id = conv["conversation_id"]
            ttl_path = output_dir / f"{conv_id}.ttl"
            print(f"\n--- {conv_id}: {conv.get('description', '')} ---")

            if ttl_path.exists():
                print(f"  SKIP {conv_id} (already exists)")
                continue

            store = RDFMemoryStore(storage_path=str(ttl_path), agent_id="extractor")
            _load_ontology_schema(store)

            added = extract_conversation(
                conv, store,
                agent_name=args.agent, model=args.model,
                batch_size=args.batch_size, session=session,
            )
            store._save()

            n_msgs = sum(1 for m in conv["messages"] if m["role"] == "user")
            print(f"  {n_msgs} user messages → {added} triples → {ttl_path}")

    else:
        # All conversations → single TTL
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        store = RDFMemoryStore(storage_path=str(output_path), agent_id="extractor")
        _load_ontology_schema(store)

        total_added = 0
        for conv in conversations:
            conv_id = conv["conversation_id"]
            print(f"\n--- {conv_id}: {conv.get('description', '')} ---")

            added = extract_conversation(
                conv, store,
                agent_name=args.agent, model=args.model,
                batch_size=args.batch_size, session=session,
            )
            total_added += added

            n_msgs = sum(1 for m in conv["messages"] if m["role"] == "user")
            print(f"  {n_msgs} user messages → +{added} triples (total: {len(store.graph)})")

        store._save()

        # Summary
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"Conversations: {len(conversations)}")
        print(f"Total triples: {len(store.graph)}")

        stats = store.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

        print(f"\nSaved to: {args.output}")
        print("=" * 60)


if __name__ == "__main__":
    # Suppress litellm/aiohttp "Task was destroyed but it is pending" noise.
    # This is a known issue: litellm's sync API creates aiohttp sessions on an
    # internal event loop; when the process exits, Python's GC destroys the
    # connector before the _wait_for_close coroutine can finish. The warning is
    # cosmetic — all data is already saved and connections are cleaned up by the OS.
    import warnings

    warnings.filterwarnings("ignore", message=r".*coroutine.*was never awaited")
    warnings.filterwarnings("ignore", message=r".*Enable tracemalloc.*")

    main()
