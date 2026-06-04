"""Command-line interface for RedisVL.

Usage:
    rvl index create -s schema.yaml
    rvl index list
    rvl index info -n my_index
    rvl index drop -n my_index
    rvl search -n my_index --text "hello" --top-k 5
    rvl version
"""

import argparse
import json
import sys
from typing import Any

from redisvl import __version__
from redisvl.client import RedisVL
from redisvl.schema import IndexSchema


def _get_client(args: argparse.Namespace) -> RedisVL:
    return RedisVL(url=args.url)


def _print(data: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                for k, v in item.items():
                    print(f"  {k}: {v}")
                print("  ---")
            else:
                print(f"  {item}")
    elif isinstance(data, dict):
        for k, v in data.items():
            print(f"  {k}: {v}")
    else:
        print(data)


def cmd_index_create(args: argparse.Namespace) -> None:
    schema = IndexSchema.from_yaml(args.schema)
    rvl = _get_client(args)
    rvl.create_index(args.name or schema.index_name, schema, overwrite=args.overwrite)
    print(f"Index '{args.name or schema.index_name}' created.")


def cmd_index_list(args: argparse.Namespace) -> None:
    rvl = _get_client(args)
    indexes = rvl.list_indexes()
    if args.json:
        _print(indexes, as_json=True)
    elif indexes:
        print("Indexes:")
        for idx in indexes:
            print(f"  - {idx}")
    else:
        print("No indexes found.")


def cmd_index_info(args: argparse.Namespace) -> None:
    rvl = _get_client(args)
    info = rvl.index_info(args.name)
    _print(info, as_json=args.json)


def cmd_index_drop(args: argparse.Namespace) -> None:
    rvl = _get_client(args)
    rvl.drop_index(args.name, drop_keys=not args.keep_keys)
    print(f"Index '{args.name}' dropped.")


def cmd_search(args: argparse.Namespace) -> None:
    rvl = _get_client(args)

    return_fields = args.return_fields.split(",") if args.return_fields else None

    if args.vector:
        vector = json.loads(args.vector)
        results = rvl.vector_search(
            vector=vector,
            index_name=args.name,
            vector_field=args.vector_field or "embedding",
            top_k=args.top_k,
            filter_expr=args.filter or "*",
            return_fields=return_fields,
        )
    elif args.text:
        results = rvl.text_search(
            text=args.text,
            index_name=args.name,
            text_field=args.text_field or "*",
            top_k=args.top_k,
            return_fields=return_fields,
        )
    elif args.filter:
        results = rvl.filter_search(
            filter_expr=args.filter,
            index_name=args.name,
            top_k=args.top_k,
            return_fields=return_fields,
        )
    elif args.hybrid and args.vector:
        vector = json.loads(args.vector)
        results = rvl.hybrid_search(
            text=args.hybrid,
            text_field=args.text_field or "text",
            vector=vector,
            vector_field=args.vector_field or "embedding",
            index_name=args.name,
            top_k=args.top_k,
            return_fields=return_fields,
        )
    else:
        print("Error: Provide --text, --vector, --filter, or --hybrid", file=sys.stderr)
        sys.exit(1)

    _print(results, as_json=args.json)


def cmd_version(args: argparse.Namespace) -> None:
    print(f"rvl v{__version__}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rvl",
        description="Redis Vector Library CLI -- manage indexes and search.",
    )
    parser.add_argument("--version", action="store_true", help="Show version and exit")
    parser.add_argument("--url", "-u", default="redis://localhost:6379", help="Redis URL")

    sub = parser.add_subparsers(dest="command")

    # --- index subcommands ---
    idx = sub.add_parser("index", help="Index management")
    idx_subs = idx.add_subparsers(dest="index_command")

    idx_create = idx_subs.add_parser("create", help="Create an index from a YAML schema")
    idx_create.add_argument("-s", "--schema", required=True, help="Path to schema YAML")
    idx_create.add_argument("-n", "--name", help="Index name (overrides schema name)")
    idx_create.add_argument("--overwrite", action="store_true", help="Overwrite existing index")

    idx_subs.add_parser("list", help="List all search indexes")

    idx_info = idx_subs.add_parser("info", help="Show index details")
    idx_info.add_argument("-n", "--name", required=True, help="Index name")

    idx_drop = idx_subs.add_parser("drop", help="Drop an index")
    idx_drop.add_argument("-n", "--name", required=True, help="Index name")
    idx_drop.add_argument("--keep-keys", action="store_true", help="Keep the associated keys")

    # --- search subcommand ---
    search = sub.add_parser("search", help="Search an index")
    search.add_argument("-n", "--name", required=True, help="Index name")
    search.add_argument("--text", help="Text search query")
    search.add_argument("--text-field", help="Field for text search")
    search.add_argument("--vector", help="Query vector as JSON array")
    search.add_argument("--vector-field", help="Vector field name")
    search.add_argument("--filter", help="Filter expression")
    search.add_argument("--hybrid", help="Text for hybrid search (use with --vector)")
    search.add_argument("--top-k", type=int, default=10, help="Number of results")
    search.add_argument("--return-fields", help="Comma-separated fields to return")

    # Global json flag
    parser.add_argument("--json", "-j", action="store_true", help="JSON output")

    args = parser.parse_args()

    if args.version:
        cmd_version(args)
        return

    if args.command == "index":
        if args.index_command == "create":
            cmd_index_create(args)
        elif args.index_command == "list":
            cmd_index_list(args)
        elif args.index_command == "info":
            cmd_index_info(args)
        elif args.index_command == "drop":
            cmd_index_drop(args)
        else:
            parser.print_help()
    elif args.command == "search":
        cmd_search(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
