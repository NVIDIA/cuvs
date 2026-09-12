#!/usr/bin/env python3
"""
Poll a Milvus collection entity count until it matches an expected row count.

Usage:
  python wait_rows.py --host 10.0.0.5 --collection my_col --rows 1000000
"""

from __future__ import annotations

import argparse
import sys
import time

from pymilvus import Collection, connections, utility
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wait until Milvus collection has expected row count.")
    p.add_argument("-a","--host", required=True, help="Milvus IP/hostname (e.g. 10.0.0.5)")
    p.add_argument("--port", type=int, default=19530, help="Milvus port (default: 19530)")
    p.add_argument("-c", "--collection", required=True, help="Collection name")
    p.add_argument("-r", "--rows", type=int, required=True, help="Expected number of rows (entities)")
    p.add_argument("--interval", type=float, default=5.0, help="Polling interval in seconds (default: 5)")
    p.add_argument("--alias", default="default", help="pymilvus connection alias (default: default)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.rows < 0:
        print("ERROR: --rows must be >= 0", file=sys.stderr)
        return 2
    if args.interval <= 0:
        print("ERROR: --interval must be > 0", file=sys.stderr)
        return 2

    connections.connect(alias=args.alias, host=args.host, port=args.port)

    if not utility.has_collection(args.collection, using=args.alias):
        print(f"ERROR: collection '{args.collection}' not found", file=sys.stderr)
        return 2

    coll = Collection(name=args.collection, using=args.alias)
    target = args.rows

    with tqdm(
        total=target,
        desc=f"{args.collection} rows",
        unit="rows",
        dynamic_ncols=True,
        leave=True,
    ) as bar:
        last = 0
        while True:
            try:
                current = coll.num_entities
            except Exception as e:
                tqdm.write(f"ERROR: failed to read num_entities: {e}")
                time.sleep(args.interval)
                continue

            # Update tqdm safely (can jump forward)
            if current > bar.total:
                tqdm.write(
                    f"ERROR: collection row count exceeded target ({current} > {target})"
                )
                return 3

            bar.n = current
            bar.refresh()

            if current == target:
                return 0

            # avoid tight loop even if value is unchanged
            if current == last:
                time.sleep(args.interval)
            last = current


if __name__ == "__main__":
    raise SystemExit(main())
