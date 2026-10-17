#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Run CUVS_CLUSTER_ASSIGNMENT_BENCH and plot brute force vs CAGRA cluster-assignment
time per number-of-clusters (K), to visualize the crossover.

Usage:
  # Single panel: pick one N (or the widest-coverage one) from the results.
  python plot_cluster_assignment_bench.py \
      --binary /path/to/CUVS_CLUSTER_ASSIGNMENT_BENCH \
      --out crossover.png [--n 2000000]

  # Side by side: ratio-controlled sweep (N = ratio * K) next to a fixed-N sweep.
  python plot_cluster_assignment_bench.py \
      --binary /path/to/CUVS_CLUSTER_ASSIGNMENT_BENCH \
      --out crossover.png --side-by-side --ratio 5 --n 2000000

  # Or plot from an existing JSON run instead of re-running the benchmark:
  python plot_cluster_assignment_bench.py --json results.json --out crossover.png
"""

import argparse
import json
import re
import subprocess
import sys

BENCH_NAME_RE = re.compile(
    r"^BM_ClusterAssignment_(BruteForce|CAGRA)/(\d+)/(\d+)/(\d+)/real_time$"
)

# Categorical slots 1 (blue) and 2 (orange) from the shared dataviz palette.
COLOR_BRUTE_FORCE = "#2a78d6"
COLOR_CAGRA = "#eb6834"
GRID_GRAY = "#d8d7d2"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"


def run_benchmark(binary_path: str, extra_args: list[str]) -> dict:
    cmd = [binary_path, "--benchmark_format=json"] + extra_args
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(proc.stdout)


def parse_flat(payload: dict) -> list[tuple[str, int, int, int, float]]:
    """Return [(method, n_rows, n_clusters, dim, time_ms), ...]."""
    records = []
    for entry in payload["benchmarks"]:
        match = BENCH_NAME_RE.match(entry["name"])
        if not match:
            continue
        method, n_rows, n_clusters, dim = match.groups()
        time_ms = entry["real_time"] * (
            1000 if entry.get("time_unit") == "s" else 1
        )
        records.append(
            (method, int(n_rows), int(n_clusters), int(dim), time_ms)
        )
    return records


def group_by_fixed_n(records: list) -> dict:
    """Return {(n_rows, dim): {"BruteForce": {k: ms}, "CAGRA": {k: ms}}}."""
    series: dict = {}
    for method, n_rows, n_clusters, dim, time_ms in records:
        key = (n_rows, dim)
        series.setdefault(key, {"BruteForce": {}, "CAGRA": {}})
        series[key][method][n_clusters] = time_ms
    return series


def series_by_ratio(records: list, ratio: int, dim: int | None = None) -> dict:
    """Return {"BruteForce": {k: ms}, "CAGRA": {k: ms}} for points where n_rows == ratio * n_clusters."""
    data = {"BruteForce": {}, "CAGRA": {}}
    for method, n_rows, n_clusters, d, time_ms in records:
        if n_rows == ratio * n_clusters and (dim is None or d == dim):
            data[method][n_clusters] = time_ms
    return data


def find_crossover(ks: list[int], bf: dict, cagra: dict):
    """Return the K interval (k_before, k_after) where CAGRA - BruteForce changes sign, or None."""
    common = [k for k in ks if k in bf and k in cagra]
    prev_k, prev_sign = None, None
    for k in common:
        sign = 1 if cagra[k] > bf[k] else -1
        if prev_sign is not None and sign != prev_sign:
            return (prev_k, k)
        prev_k, prev_sign = k, sign
    return None


def pick_fixed_n_group(series: dict, want_n: int | None):
    candidates = {
        k: v
        for k, v in series.items()
        if len(v["BruteForce"]) >= 2 and len(v["CAGRA"]) >= 2
    }
    if not candidates:
        print(
            "No (N, dim) group has >=2 K points for both methods; nothing to plot.",
            file=sys.stderr,
        )
        sys.exit(1)
    if want_n is not None:
        matches = [k for k in candidates if k[0] == want_n]
        if not matches:
            available = sorted(n for n, _ in candidates)
            print(
                f"No group found for N={want_n}. Available N: {available}",
                file=sys.stderr,
            )
            sys.exit(1)
        return matches[0]
    # Default: the group with the widest K coverage (most points to show the trend clearly).
    return max(
        candidates,
        key=lambda k: min(
            len(candidates[k]["BruteForce"]), len(candidates[k]["CAGRA"])
        ),
    )


def draw_panel(ax, data: dict, title: str, xlabel: str) -> tuple | None:
    """Draw one BruteForce-vs-CAGRA panel on `ax`. Returns the crossover interval, if any."""
    ks_bf = sorted(data["BruteForce"])
    ks_cagra = sorted(data["CAGRA"])
    all_k = sorted(set(ks_bf) | set(ks_cagra))
    if len(ks_bf) < 2 or len(ks_cagra) < 2:
        ax.text(
            0.5,
            0.5,
            "not enough data points",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=TEXT_SECONDARY,
        )
        ax.set_title(
            title,
            color=TEXT_PRIMARY,
            fontsize=11,
            loc="left",
            fontweight="bold",
        )
        return None

    ax.set_facecolor(SURFACE)
    ax.plot(
        ks_bf,
        [data["BruteForce"][k] for k in ks_bf],
        color=COLOR_BRUTE_FORCE,
        linewidth=2,
        marker="o",
        markersize=7,
        markerfacecolor=COLOR_BRUTE_FORCE,
        markeredgecolor=SURFACE,
        markeredgewidth=1.2,
        label="Brute force",
        zorder=3,
    )
    ax.plot(
        ks_cagra,
        [data["CAGRA"][k] for k in ks_cagra],
        color=COLOR_CAGRA,
        linewidth=2,
        marker="o",
        markersize=7,
        markerfacecolor=COLOR_CAGRA,
        markeredgecolor=SURFACE,
        markeredgewidth=1.2,
        label="CAGRA (ANN)",
        zorder=3,
    )

    crossover = find_crossover(all_k, data["BruteForce"], data["CAGRA"])
    if crossover is not None:
        k_before, k_after = crossover
        ax.axvspan(
            k_before, k_after, color=TEXT_SECONDARY, alpha=0.08, zorder=1
        )
        ax.annotate(
            f"crossover\nK {k_before:,}-{k_after:,}",
            xy=((k_before * k_after) ** 0.5, data["CAGRA"][k_after]),
            ha="center",
            xytext=(0, 26),
            textcoords="offset points",
            color=TEXT_PRIMARY,
            fontsize=9,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3", fc=SURFACE, ec=GRID_GRAY, lw=0.8
            ),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=9.5)
    ax.set_ylabel(
        "Cluster assignment time (ms, log scale)",
        color=TEXT_SECONDARY,
        fontsize=9.5,
    )
    ax.set_title(
        title, color=TEXT_PRIMARY, fontsize=11, loc="left", fontweight="bold"
    )

    ax.grid(True, which="major", color=GRID_GRAY, linewidth=0.8, zorder=0)
    ax.grid(
        True,
        which="minor",
        color=GRID_GRAY,
        linewidth=0.4,
        alpha=0.5,
        zorder=0,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID_GRAY)

    ax.set_xticks(all_k)
    ax.set_xticklabels(
        [f"{k:,}" for k in all_k],
        rotation=30,
        ha="right",
        color=TEXT_SECONDARY,
        fontsize=8,
    )
    ax.tick_params(axis="y", colors=TEXT_SECONDARY, labelsize=8)
    ax.legend(
        loc="upper left", frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY
    )
    return crossover


def plot_single(series: dict, out_path: str, want_n: int | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows, dim = pick_fixed_n_group(series, want_n)
    data = series[(n_rows, dim)]

    fig, ax = plt.subplots(figsize=(8.4, 5.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    crossover = draw_panel(
        ax,
        data,
        title="Cluster assignment: Brute force vs CAGRA",
        xlabel=f"Number of clusters K  (N = {n_rows:,} vectors, dim = {dim})",
    )

    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    print(f"Saved plot to {out_path} (N={n_rows:,}, dim={dim})")
    if crossover is not None:
        print(
            f"Crossover: CAGRA overtakes Brute force between K={crossover[0]:,} and K={crossover[1]:,}"
        )
    else:
        print("No crossover found in this K range.")


def plot_side_by_side(
    records: list,
    out_path: str,
    ratio: int,
    want_n: int | None,
    dim: int | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ratio_data = series_by_ratio(records, ratio, dim)
    series = group_by_fixed_n(
        [r for r in records if dim is None or r[3] == dim]
    )
    fixed_n_rows, fixed_dim = pick_fixed_n_group(series, want_n)
    fixed_data = series[(fixed_n_rows, fixed_dim)]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14.5, 5.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)

    crossover_ratio = draw_panel(
        ax_left,
        ratio_data,
        title=f"Ratio-controlled: N = {ratio} × K",
        xlabel="Number of clusters K  (points per cluster held constant)",
    )
    crossover_fixed = draw_panel(
        ax_right,
        fixed_data,
        title=f"Fixed N = {fixed_n_rows:,}",
        xlabel=f"Number of clusters K  (N = {fixed_n_rows:,} vectors, dim = {fixed_dim})",
    )

    fig.suptitle(
        "Cluster assignment: Brute force vs CAGRA",
        color=TEXT_PRIMARY,
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, facecolor=SURFACE)
    print(f"Saved side-by-side plot to {out_path}")
    for label, crossover in (
        ("ratio-controlled", crossover_ratio),
        (f"fixed N={fixed_n_rows:,}", crossover_fixed),
    ):
        if crossover is not None:
            print(
                f"  [{label}] crossover: K={crossover[0]:,} - K={crossover[1]:,}"
            )
        else:
            print(f"  [{label}] no crossover found in this K range")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--binary", help="Path to CUVS_CLUSTER_ASSIGNMENT_BENCH"
    )
    parser.add_argument(
        "--json",
        help="Path to an existing --benchmark_format=json result file",
    )
    parser.add_argument(
        "--out",
        default="cluster_assignment_crossover.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Which N (n_rows) to plot for the fixed-N panel/plot, if the results "
        "contain multiple. Defaults to the N with the widest K coverage.",
    )
    parser.add_argument(
        "--side-by-side",
        action="store_true",
        help="Plot the ratio-controlled sweep (N = ratio * K) next to a fixed-N sweep.",
    )
    parser.add_argument(
        "--ratio",
        type=int,
        default=5,
        help="Points-per-cluster ratio for the ratio-controlled panel (default: 5).",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help="Restrict to a specific dim, if ambiguous.",
    )
    parser.add_argument(
        "--benchmark-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args forwarded to the benchmark binary (e.g. --benchmark_filter=...)",
    )
    args = parser.parse_args()

    if not args.binary and not args.json:
        parser.error("one of --binary or --json is required")

    if args.json:
        with open(args.json) as f:
            payload = json.load(f)
    else:
        payload = run_benchmark(args.binary, args.benchmark_args)

    records = parse_flat(payload)
    if args.side_by_side:
        plot_side_by_side(records, args.out, args.ratio, args.n, args.dim)
    else:
        series = group_by_fixed_n(
            records
            if args.dim is None
            else [r for r in records if r[3] == args.dim]
        )
        plot_single(series, args.out, args.n)


if __name__ == "__main__":
    main()
