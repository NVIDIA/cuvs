#!/usr/bin/env python3
"""Estimate total on-disk storage for a Knowhere DiskANN / AISAQ index.

Formulas match the layout in thirdparty/DiskANN (aux_utils.cpp, partition_and_pq.cpp,
aisaq_pq_reader.cpp) for the common case: float32, L2, disk_pq_dims=0, no inline PQ.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


SECTOR_LEN = 4096
MAX_PQ_CHUNKS = 512
MAX_SAMPLE_POINTS = 100_000
DEFAULT_MAX_DEGREE = 64
DEFAULT_PQ_RATIO = 0.125
DEFAULT_NUM_ENTRY_POINTS = 100
DEFAULT_REARRANGE = True


def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def pq_bytes_per_vector(
    n_rows: int,
    n_dim: int,
    pq_code_budget_gb: float | None = None,
    pq_code_budget_ratio: float = DEFAULT_PQ_RATIO,
    elem_size: int = 4,
) -> int:
    """Number of uint8 PQ chunks stored per vector (C)."""
    if pq_code_budget_gb is None:
        pq_budget_bytes = n_rows * n_dim * elem_size * pq_code_budget_ratio
    else:
        pq_budget_bytes = pq_code_budget_gb * (1024**3)

    chunks = max(1, int(math.floor(pq_budget_bytes / n_rows)))
    return min(chunks, n_dim, MAX_PQ_CHUNKS)


def disk_node_len(
    n_dim: int,
    max_degree: int = DEFAULT_MAX_DEGREE,
    elem_size: int = 4,
    disk_pq_dims: int = 0,
    pq_bytes: int = 0,
    inline_pq_vectors: int = 0,
) -> int:
    """Max bytes per graph node on disk (L)."""
    if disk_pq_dims > 0:
        on_disk_dims = min(disk_pq_dims, n_dim)
        vector_bytes = on_disk_dims * 1  # uint8 PQ chunks on disk
    else:
        vector_bytes = n_dim * elem_size

    base = (max_degree + 1) * 4 + vector_bytes
    return base + inline_pq_vectors * pq_bytes


def disk_index_bytes(
    n_rows: int,
    n_dim: int,
    max_degree: int = DEFAULT_MAX_DEGREE,
    elem_size: int = 4,
    disk_pq_dims: int = 0,
    pq_bytes: int = 0,
    inline_pq_vectors: int = 0,
    sector_len: int = SECTOR_LEN,
) -> tuple[int, int, int]:
    """Return (file_bytes, n_sectors, max_node_len) for *_disk.index."""
    node_len = disk_node_len(
        n_dim, max_degree, elem_size, disk_pq_dims, pq_bytes, inline_pq_vectors
    )

    if node_len <= sector_len:
        nnodes_per_sector = sector_len // node_len
        n_sectors = ceil_div(n_rows, nnodes_per_sector)
    else:
        n_sectors = n_rows * ceil_div(node_len, sector_len)

    file_bytes = (n_sectors + 1) * sector_len
    return file_bytes, n_sectors, node_len


def pq_compressed_bytes(
    n_rows: int,
    pq_bytes: int,
    rearrange: bool = DEFAULT_REARRANGE,
    sector_len: int = SECTOR_LEN,
) -> int:
    """PQ vector file size on disk."""
    if rearrange:
        vectors_per_page = sector_len // pq_bytes
        n_pages = ceil_div(n_rows, vectors_per_page)
        return sector_len + n_pages * sector_len
    return 8 + pq_bytes * n_rows


def pq_pivots_bundle_bytes(n_dim: int, pq_bytes: int) -> int:
    """*_pq_pivots.bin plus centroid / permutation / chunk-offset sidecars."""
  # 256 x D float pivots, D float centroid, D uint32 permutation, (C+1) uint32 offsets
    return (
        8 + 256 * n_dim * 4
        + 8 + n_dim * 4
        + 8 + n_dim * 4
        + 8 + (pq_bytes + 1) * 4
    )


def rearrange_map_bytes(n_rows: int) -> int:
    return 8 + 4 * n_rows


def entry_points_bytes(num_entry_points: int) -> int:
    return 8 + 4 * num_entry_points


def sample_data_bytes(n_rows: int, n_dim: int, elem_size: int = 4) -> int:
    n_sample = min(MAX_SAMPLE_POINTS, ceil_div(n_rows, 10))
    return 8 + n_sample * n_dim * elem_size


@dataclass
class StorageBreakdown:
    disk_index: int
    pq_compressed: int
    pq_pivots: int
    rearrange_map: int
    entry_points: int
    sample_data: int
    pq_bytes_per_vector: int
    max_node_len: int
    n_sectors: int

    @property
    def total(self) -> int:
        return (
            self.disk_index
            + self.pq_compressed
            + self.pq_pivots
            + self.rearrange_map
            + self.entry_points
            + self.sample_data
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "pq_bytes_per_vector": self.pq_bytes_per_vector,
            "max_node_len": self.max_node_len,
            "n_sectors": self.n_sectors,
            "disk_index": self.disk_index,
            "pq_compressed": self.pq_compressed,
            "pq_pivots": self.pq_pivots,
            "rearrange_map": self.rearrange_map,
            "entry_points": self.entry_points,
            "sample_data": self.sample_data,
            "total": self.total,
        }


def estimate_storage(
    n_rows: int,
    n_dim: int,
    max_degree: int = DEFAULT_MAX_DEGREE,
    pq_code_budget_gb: float | None = None,
    pq_code_budget_ratio: float = DEFAULT_PQ_RATIO,
    num_entry_points: int = DEFAULT_NUM_ENTRY_POINTS,
    rearrange: bool = DEFAULT_REARRANGE,
    disk_pq_dims: int = 0,
    inline_pq_vectors: int = 0,
    include_sample: bool = True,
    include_entry_points: bool = True,
    include_rearrange_map: bool = True,
    elem_size: int = 4,
) -> StorageBreakdown:
    pq_bytes = pq_bytes_per_vector(
        n_rows, n_dim, pq_code_budget_gb, pq_code_budget_ratio, elem_size
    )
    disk_bytes, n_sectors, node_len = disk_index_bytes(
        n_rows,
        n_dim,
        max_degree,
        elem_size,
        disk_pq_dims,
        pq_bytes,
        inline_pq_vectors,
    )

    return StorageBreakdown(
        disk_index=disk_bytes,
        pq_compressed=pq_compressed_bytes(n_rows, pq_bytes, rearrange),
        pq_pivots=pq_pivots_bundle_bytes(n_dim, pq_bytes),
        rearrange_map=rearrange_map_bytes(n_rows) if include_rearrange_map and rearrange else 0,
        entry_points=entry_points_bytes(num_entry_points) if include_entry_points else 0,
        sample_data=sample_data_bytes(n_rows, n_dim, elem_size) if include_sample else 0,
        pq_bytes_per_vector=pq_bytes,
        max_node_len=node_len,
        n_sectors=n_sectors,
    )


def fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n:,} B"
    units = ("KiB", "MiB", "GiB", "TiB")
    value = float(n)
    for unit in units:
        value /= 1024
        if value < 1024 or unit == units[-1]:
            return f"{value:.3f} {unit}"
    return f"{value:.3f} TiB"


def print_breakdown(label: str, breakdown: StorageBreakdown) -> None:
    print(f"\n{label}")
    print("=" * len(label))
    d = breakdown.as_dict()
    for key in (
        "pq_bytes_per_vector",
        "max_node_len",
        "n_sectors",
        "disk_index",
        "pq_compressed",
        "pq_pivots",
        "rearrange_map",
        "entry_points",
        "sample_data",
        "total",
    ):
        if key in ("pq_bytes_per_vector", "max_node_len", "n_sectors"):
            print(f"  {key:22s} {d[key]:>15,}")
        else:
            print(f"  {key:22s} {d[key]:>15,}  ({fmt_bytes(d[key])})")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Estimate total on-disk storage for DiskANN / AISAQ indexes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("n_rows", type=int, nargs="?", help="Number of vectors (rows)")
    p.add_argument("n_dim", type=int, nargs="?", help="Vector dimension")
    p.add_argument("-R", "--max-degree", type=int, default=DEFAULT_MAX_DEGREE, help="Graph max degree")
    p.add_argument(
        "--pq-ratio",
        type=float,
        default=DEFAULT_PQ_RATIO,
        help="PQ code budget as fraction of raw vector bytes (pq_code_budget_gb_ratio)",
    )
    p.add_argument(
        "--pq-budget-gb",
        type=float,
        default=None,
        help="Explicit PQ code budget in GiB (overrides --pq-ratio)",
    )
    p.add_argument(
        "-E",
        "--num-entry-points",
        type=int,
        default=DEFAULT_NUM_ENTRY_POINTS,
        help="Number of AISAQ entry points",
    )
    p.add_argument(
        "--no-rearrange",
        action="store_true",
        help="Use plain *_pq_compressed.bin instead of rearranged PQ file",
    )
    p.add_argument(
        "--disk-pq-dims",
        type=int,
        default=0,
        help="Compressed dims on disk (0 = store full vectors in graph)",
    )
    p.add_argument("--inline-pq", type=int, default=0, help="Inline PQ vectors per node")
    p.add_argument("--no-sample", action="store_true", help="Exclude warmup sample file")
    p.add_argument("--no-entry-points", action="store_true", help="Exclude entry points file")
    p.add_argument("--examples", action="store_true", help="Print SIFT-1M-128 and GIST-1M-960 examples")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.examples or (args.n_rows is None) != (args.n_dim is None):
        if args.n_rows is not None:
            raise SystemExit("Provide both n_rows and n_dim, or use --examples")
        bench = estimate_storage(
            1_000_000,
            128,
            num_entry_points=1000,
            pq_code_budget_ratio=DEFAULT_PQ_RATIO,
        )
        gist = estimate_storage(
            1_000_000,
            960,
            num_entry_points=1000,
            pq_code_budget_ratio=DEFAULT_PQ_RATIO,
        )
        print_breakdown("SIFT-1M-128 (AISAQ benchmark defaults, E=1000)", bench)
        print_breakdown("GIST-1M-960 (AISAQ benchmark defaults, E=1000)", gist)
        return

    if args.n_rows is None or args.n_dim is None:
        raise SystemExit("n_rows and n_dim are required (or pass --examples)")

    breakdown = estimate_storage(
        args.n_rows,
        args.n_dim,
        max_degree=args.max_degree,
        pq_code_budget_gb=args.pq_budget_gb,
        pq_code_budget_ratio=args.pq_ratio,
        num_entry_points=args.num_entry_points,
        rearrange=not args.no_rearrange,
        disk_pq_dims=args.disk_pq_dims,
        inline_pq_vectors=args.inline_pq,
        include_sample=not args.no_sample,
        include_entry_points=not args.no_entry_points,
        include_rearrange_map=not args.no_rearrange,
    )
    print_breakdown(f"N={args.n_rows:,}, D={args.n_dim}", breakdown)


if __name__ == "__main__":
    main()
