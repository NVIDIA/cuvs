#!/usr/bin/env python3
"""
Wait until a Milvus collection is fully indexed (no pending rows), then load it,
then wait until it is fully loaded. Polling interval: 5 seconds.

Usage:
  python wait_index_and_load.py --host 10.0.0.5 --collection my_col
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Dict, Optional

from pymilvus import Collection, connections, utility


POLL_SEC = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wait for Milvus collection to be indexed and loaded.")
    p.add_argument("-a", "--host", required=True, help="Milvus IP/hostname (e.g. 10.0.0.5)")
    p.add_argument("--port", type=int, default=19530, help="Milvus port (default: 19530)")
    p.add_argument("-c", "--collection", required=True, help="Collection name")
    p.add_argument("--alias", default="default", help="pymilvus connection alias (default: default)")
    return p.parse_args()


def _call_first_existing(func_names: list[str], *args, **kwargs) -> Any:
    for name in func_names:
        fn = getattr(utility, name, None)
        if callable(fn):
            return fn(*args, **kwargs)
    raise AttributeError(f"None of these utility functions exist: {func_names}")


def get_index_progress(collection: str, using: str) -> Dict[str, Any]:
    # Prefer the function that returns exactly what you showed earlier.
    # Fallbacks included for minor API name changes across versions.
    return _call_first_existing(
        ["index_building_progress", "get_index_build_progress", "index_build_progress"],
        collection,
        using=using,
    )


def is_fully_indexed(progress: Dict[str, Any]) -> bool:
    total = int(progress.get("total_rows", -1))
    indexed = int(progress.get("indexed_rows", -1))
    pending = int(progress.get("pending_index_rows", -1))
    state = str(progress.get("state", "")).lower()

    # Conservative definition: no pending AND indexed==total (when total available).
    if pending != 0:
        return False
    if total >= 0 and indexed >= 0 and indexed != total:
        return False

    # If state is present, require it to be a "done" state, but don’t fail if empty/unknown.
    if state and state not in {"finished", "completed", "done"}:
        return False
    return True


def get_load_progress(collection: str, using: str) -> Optional[int]:
    """
    Returns 0..100 if available, else None.
    """
    fn = getattr(utility, "loading_progress", None)
    if callable(fn):
        info = fn(collection, using=using)
        # common shape: {"loading_progress": 100}
        lp = info.get("loading_progress")
        if lp is not None:
            if lp.endswith('%'):
                return int(lp[:-1])
            return int(lp)

    # Fallback: check load state if progress isn't available.
    fn = getattr(utility, "load_state", None) or getattr(utility, "get_load_state", None)
    if callable(fn):
        state = fn(collection, using=using)
        # In some versions state may be an enum/string-like; treat "Loaded" as 100.
        if str(state).lower().endswith("loaded") or str(state).lower() == "loaded":
            return 100

    return None


def main() -> int:
    args = parse_args()

    connections.connect(alias=args.alias, host=args.host, port=args.port)

    if not utility.has_collection(args.collection, using=args.alias):
        print(f"ERROR: collection '{args.collection}' not found", file=sys.stderr)
        return 2

    coll = Collection(args.collection, using=args.alias)
    start = time.time()
    # 1) Wait for indexing to finish (no pending rows)
    while True:
        try:
            prog = get_index_progress(args.collection, using=args.alias)
        except Exception as e:
            print(f"ERROR: failed to read index progress: {e}", file=sys.stderr)
            time.sleep(POLL_SEC)
            continue

        total = prog.get("total_rows")
        indexed = prog.get("indexed_rows")
        pending = prog.get("pending_index_rows")
        state = prog.get("state")
        print(f"[index] {round(time.time() - start)} - total={total} indexed={indexed} pending={pending} state={state}", end="\r", flush=True)

        if is_fully_indexed(prog):
            break

        time.sleep(POLL_SEC)

    # 2) Load collection
    try:
        coll.load()
        print("[load] load() triggered")
    except Exception as e:
        print(f"ERROR: failed to trigger load(): {e}", file=sys.stderr)
        return 3

    # 3) Wait until fully loaded
    while True:
        try:
            lp = get_load_progress(args.collection, using=args.alias)
        except Exception as e:
            print(f"ERROR: failed to read load progress/state: {e}", file=sys.stderr)
            time.sleep(POLL_SEC)
            continue

        if lp is None:
            # If API doesn't expose anything usable, at least keep the loop alive and visible.
            print("[load] progress=unknown (API not available)")
        else:
            print(f"[load] progress={lp}%")
            if lp >= 100:
                break

        time.sleep(POLL_SEC)

    print("DONE: collection is fully indexed and fully loaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
