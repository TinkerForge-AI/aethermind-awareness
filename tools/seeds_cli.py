from __future__ import annotations
import argparse, json, sys
from pathlib import Path

DEFAULT_SEEDS = "chunks/seeds.jsonl"

def cmd_info(args):
    # placeholder: implement your actual logic
    print(f"[INFO] Path = {args.path}")
    # TODO: open JSONL, show counts, etc.

def cmd_validate(args):
    print(f"[VALIDATE] Path = {args.path}")
    # TODO: schema + sanity checks

def cmd_join_check(args):
    print(f"[JOIN-CHECK] Path = {args.path}")
    # TODO: join with interpretation DB

def main():
    # Common parser with shared args for all subcommands
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--path",
        default=DEFAULT_SEEDS,
        help=f"path to seeds.jsonl (default: {DEFAULT_SEEDS})"
    )

    ap = argparse.ArgumentParser(
        description="Seeds JSONL tools (contract: perception → interpretation)"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Each subparser inherits the `common` parser so `--path` works everywhere
    p1 = sub.add_parser("info", parents=[common], help="show counts + sample")
    p1.set_defaults(fn=cmd_info)

    p2 = sub.add_parser("validate", parents=[common], help="schema/sanity check")
    p2.set_defaults(fn=cmd_validate)

    p3 = sub.add_parser("join-check", parents=[common], help="overlap join with interpretation DB")
    p3.set_defaults(fn=cmd_join_check)

    args = ap.parse_args()
    args.fn(args)

if __name__ == "__main__":
    main()
