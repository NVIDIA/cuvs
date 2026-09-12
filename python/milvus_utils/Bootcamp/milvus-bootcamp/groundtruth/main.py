#!/usr/bin/env python3
"""
Streaming multi-GPU exact groundtruth (query-partitioned, queue-based).

"""

import argparse
import os
import shutil
import struct
import sys
import threading
from pathlib import Path
from queue import Queue

import numpy as np
import torch
from tqdm import tqdm


def _die(msg: str) -> None:
    print(f"ERROR: {msg}")
    sys.exit(1)


def _read_fbin_header(path: str) -> tuple[int, int]:
    try:
        size = os.path.getsize(path)
    except OSError as e:
        _die(f"Cannot stat file: {path}. Details: {e}")

    if size < 8:
        _die(f"FBIN file too small to contain header: {path}")

    try:
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                _die(f"Failed to read FBIN header from: {path}")
            n, d = struct.unpack("<ii", raw)
    except Exception as e:
        _die(f"Invalid FBIN header in {path}. Details: {e}")

    if n <= 0 or d <= 0:
        _die(f"Invalid FBIN header in {path}: N={n}, dim={d}")

    expected = 8 + n * d * 4
    if size != expected:
        _die(
            f"FBIN size mismatch in {path}\n"
            f"       Expected: {expected} bytes\n"
            f"       Actual:   {size} bytes"
        )

    return n, d


def _parse_fbin_paths(path_arg: str) -> list[str]:
    paths = [p.strip() for p in path_arg.split(",")]

    if not paths or any(not p for p in paths):
        _die("Empty path in comma-separated list")

    for p in paths:
        if not os.path.exists(p):
            _die(f"Path does not exist: {p}")
        if not os.path.isfile(p):
            _die(f"Path is not a file: {p}")
        if not p.lower().endswith(".fbin"):
            _die(f"Expected .fbin file, got: {p}")

    return paths


def preflight_validate_fbin_paths(paths: list[str]) -> tuple[int, int]:
    """
    Preflight validation: ensure all FBIN files share the same dimension.
    Reads headers only.

    Returns:
        (dim, total_vectors)
    """

    dim = None
    total_vectors = 0

    for p in paths:
        n, d = _read_fbin_header(p)

        if dim is None:
            dim = d
        elif d != dim:
            _die(
                "Dimension mismatch across FBIN files:\n"
                f"       Expected dim={dim}\n"
                f"       File {p}: dim={d}"
            )

        total_vectors += n

    if total_vectors <= 0:
        _die("Total number of vectors across FBIN files is zero")

    return dim, total_vectors

# ------------------------------------------------------------
# IO helpers
# ------------------------------------------------------------

def read_fvecs(path: str) -> np.ndarray:
    vecs = []
    with open(path, "rb") as f:
        while True:
            dim_b = f.read(4)
            if not dim_b:
                break
            (dim,) = struct.unpack("<i", dim_b)
            data = f.read(4 * dim)
            vecs.append(np.frombuffer(data, dtype=np.float32))
    return np.vstack(vecs).astype(np.float32, copy=False)


def read_fbin(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        n, d = struct.unpack("<ii", f.read(8))
        data = np.frombuffer(f.read(n * d * 4), dtype=np.float32)
    return data.reshape(n, d).astype(np.float32, copy=False)


def stream_fbin_batches(fbin_paths: list[str], batch_size: int, dim: int):
    if batch_size <= 0:
        _die(f"batch_size must be > 0, got: {batch_size}")

    shard_meta = []

    for p in fbin_paths:
        n, _ = _read_fbin_header(p)
        shard_meta.append((p, n))

    # Streaming: file by file, preserving order
    global_offset = 0

    for path, n_vectors in shard_meta:
        with open(path, "rb") as f:
            _ = f.read(8)
            bstart = 0

            while bstart < n_vectors:
                to_read = min(batch_size, n_vectors - bstart)
                raw = f.read(to_read * dim * 4)

                if len(raw) != to_read * dim * 4:
                    _die(f"Truncated data block while reading file {path}")

                batch = np.frombuffer(raw, dtype=np.float32).reshape(to_read, dim)
                yield global_offset + bstart, np.ascontiguousarray(batch)

                bstart += to_read

        global_offset += n_vectors


# ------------------------------------------------------------
# Worker
# ------------------------------------------------------------

def worker_thread(
    gpu_id: int,
    query_indices: np.ndarray,
    queries_np: np.ndarray,
    work_queue: Queue,
    k: int,
    metric: str,
    results_container: list,
):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    local_queries = queries_np[query_indices]
    n_local = local_queries.shape[0]

    if n_local == 0:
        results_container[gpu_id] = (
            np.zeros((0, k), np.float32),
            np.zeros((0, k), np.int32),
        )
        return

    queries_dev = torch.from_numpy(
        np.ascontiguousarray(local_queries)
    ).float().to(device, non_blocking=True)

    # global per-query top-k
    topk_vals = torch.full(
        (n_local, k), float("inf"),
        device=device, dtype=torch.float32
    )
    topk_idx = torch.full(
        (n_local, k), -1,
        device=device, dtype=torch.int64
    )

    with torch.no_grad():
        while True:
            item = work_queue.get()
            if item is None:
                break

            bstart, batch_np = item
            M = batch_np.shape[0]
            batch_dev = torch.from_numpy(batch_np.copy()).float().to(device, non_blocking=True)

            # ---- distance ----
            if metric == "l2":
                q2 = (queries_dev * queries_dev).sum(dim=1, keepdim=True)  # (n_local, 1)
                b2 = (batch_dev * batch_dev).sum(dim=1).unsqueeze(0)  # (1, M)
                qp = queries_dev @ batch_dev.t()  # (n_local, M)
                dists = q2 + b2 - 2 * qp  # squared L2
            elif metric == "ip":
                dists = -(queries_dev @ batch_dev.t())
            else:
                raise ValueError("Unsupported metric")

            # ---- global indices ----
            cand_idx = (
                torch.arange(bstart, bstart + M, device=device, dtype=torch.int64)
                .unsqueeze(0)
                .expand(n_local, -1)
            )

            # ---- merge ----
            concat_vals = torch.cat([topk_vals, dists], dim=1)
            concat_idx = torch.cat([topk_idx, cand_idx], dim=1)


            vals_k, inds_k = torch.topk(
                concat_vals, k, largest=False, sorted=True
            )

            topk_vals = vals_k
            topk_idx = torch.gather(concat_idx, 1, inds_k)

    results_container[gpu_id] = (
        topk_vals.cpu().numpy(),
        topk_idx.cpu().numpy(),
    )


def preflight_validate_output(
    out_prefix: str,
    out_format: str,
) -> None:
    """
    Fail-fast validation of output path before long GT computation.

    Validates:
    - parent directory exists
    - directory is writable
    - required dependencies for format exist
    - (best-effort) disk space availability

    Raises:
        RuntimeError on any fatal issue.
    """

    # Resolve output directory
    out_prefix = Path(out_prefix)
    out_dir = out_prefix.parent if out_prefix.parent != Path("") else Path(".")

    # 1. Directory existence
    if not out_dir.exists():
        raise RuntimeError(f"Output directory does not exist: {out_dir}")

    if not out_dir.is_dir():
        raise RuntimeError(f"Output path is not a directory: {out_dir}")

    # 2. Write permission check (fail fast)
    test_file = out_dir / ".gt_write_test.tmp"
    try:
        with open(test_file, "wb") as f:
            f.write(b"\0")
        test_file.unlink()
    except Exception as e:
        raise RuntimeError(
            f"Output directory is not writable: {out_dir}. "
            f"Reason: {e}"
        )

    # 3. Format-specific dependency checks
    if out_format == "parquet":
        try:
            import pyarrow  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "Parquet output requested but pyarrow is not available. "
                f"Reason: {e}"
            )

    # 4. Best-effort disk space check (warning only)
    try:
        usage = shutil.disk_usage(out_dir)
        free_gb = usage.free / (1024 ** 3)
        if free_gb < 1.0:
            print(
                f"WARNING: low free disk space in {out_dir}: "
                f"{free_gb:.2f} GB available"
            )
    except Exception:
        # Disk usage check is best-effort only
        pass


def write_gt_ibin(
    out_prefix: str,
    neighbors: np.ndarray,
    distances: np.ndarray,
    nq: int,
    k: int,
) -> str:
    """
    Write ground-truth in legacy IBIN format.

    neighbors: int64 array of shape (nq, k)
    distances: float32 array of shape (nq, k)
    """
    path = out_prefix + ".ibin"

    # --- safety check ---
    max_i32 = np.iinfo(np.int32).max
    if neighbors.max() > max_i32:
        raise ValueError(
            "IBIN format does not support neighbor_id > int32. "
            f"Found max={neighbors.max()}."
        )

    with open(path, "wb") as f:
        f.write(struct.pack("<i", nq))
        f.write(struct.pack("<i", k))
        neighbors.astype(np.int32).tofile(f)
        distances.tofile(f)
    return path


def write_gt_parquet(
    out_prefix: str,
    neighbors: np.ndarray,
    distances: np.ndarray,
    nq: int,
    k: int,
    dim: int,
) -> str:
    """
    Write ground-truth in flat Parquet format.

    neighbors: int64 array of shape (nq, k)
    distances: float32 array of shape (nq, k)

    Output rows: nq * k
    Ordering: query-major, contiguous blocks of k rows per query
    """

    import pyarrow as pa
    import pyarrow.parquet as pq

    path = out_prefix + ".parquet"

    neighbors_list = [row.tolist() for row in neighbors]
    distances_list = [row.tolist() for row in distances]

    table = pa.table({
        "neighbors_id": pa.array(neighbors_list, type=pa.list_(pa.int64())),
        "distances": pa.array(distances_list, type=pa.list_(pa.float32())),
    })

    # File-level metadata (keys/values must be bytes)
    metadata = {
        b"K": str(k).encode(),
        b"Q": str(nq).encode(),
        b"D": str(dim).encode(),
    }

    schema = table.schema.with_metadata(metadata)
    table = table.cast(schema)

    pq.write_table(table, path)
    return path



# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--out_prefix", default="groundtruth")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=100000)
    parser.add_argument("--metric", choices=["l2", "ip"], default="l2")
    parser.add_argument(
        "--out_format",
        choices=["ibin", "parquet"],
        default="ibin",
        help="Output format for ground truth"
    )
    args = parser.parse_args()

    # Load queries
    ext = os.path.splitext(args.query)[1].lower()
    if ext == ".fvecs":
        queries = read_fvecs(args.query)
    elif ext == ".fbin":
        queries = read_fbin(args.query)
    else:
        raise ValueError("Query must be .fvecs or .fbin")

    nq, dim = queries.shape
    print(f"Loaded queries: nq={nq}, dim={dim}")

    preflight_validate_output(out_prefix=args.out_prefix, out_format=args.out_format)

    ngpu = torch.cuda.device_count()
    if ngpu == 0:
        raise RuntimeError("No CUDA GPUs found")

    print(f"Using {ngpu} GPUs, metric={args.metric}, k={args.k}")

    query_splits = np.array_split(np.arange(nq), ngpu)

    paths = _parse_fbin_paths(args.base) 
    base_dim, total_base = preflight_validate_fbin_paths(paths)
    if base_dim != dim:
        raise ValueError("Base/query dimension mismatch")

    n_batches = (total_base + args.batch_size - 1) // args.batch_size

    queues = [Queue(maxsize=2) for _ in range(ngpu)]
    results_container = [None] * ngpu

    workers = []
    for gid in range(ngpu):
        t = threading.Thread(
            target=worker_thread,
            args=(
                gid,
                query_splits[gid],
                queries,
                queues[gid],
                args.k,
                args.metric,
                results_container,
            ),
            daemon=True,
        )
        t.start()
        workers.append(t)

    pbar = tqdm(total=n_batches, desc="Processed batches")
    for bstart, batch_np in stream_fbin_batches(paths, args.batch_size, dim=base_dim):
        for q in queues:
            q.put((bstart, batch_np))
        pbar.update(1)
    pbar.close()

    for q in queues:
        q.put(None)
    for t in workers:
        t.join()

    final_idx = np.empty((nq, args.k), dtype=np.int64)
    final_dist = np.empty((nq, args.k), dtype=np.float32)

    for gid, ids in enumerate(query_splits):
        vals, idxs = results_container[gid]
        final_idx[ids] = idxs
        final_dist[ids] = vals

    if args.out_format == "ibin":
        out_path = write_gt_ibin(
            args.out_prefix,
            final_idx,
            final_dist,
            nq=nq,
            k=args.k,
        )
    else:
        out_path = write_gt_parquet(
            args.out_prefix,
            final_idx,
            final_dist,
            nq=nq,
            k=args.k,
            dim=dim,
        )

    print(f"Saved groundtruth to {out_path}")


if __name__ == "__main__":
    main()
