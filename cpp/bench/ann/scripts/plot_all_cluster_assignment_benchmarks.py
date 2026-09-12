#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Produce all 8 cluster-assignment (brute force vs CAGRA) analysis panels in one figure:

  1. Ratio-controlled sweep: N = 5 * K
  2. Fixed-N sweep: N = 2,000,000, K varies
  3. Fixed-K sweep: K = 65,536, N varies
  4. (N, K) speedup heatmap
  5. Speedup ratio (BruteForce / CAGRA) vs K, for sweeps 1 and 2
  6. CAGRA build-vs-search breakdown vs K (at N = 2,000,000)
  7. Assignment quality (label match %, centroid drift) vs ann_rebuild_interval
  8. Dimension sensitivity, at N = 2,000,000, K = 131,072

Usage:
  python plot_all_cluster_assignment_benchmarks.py \
      --cluster-json cluster_assignment.json \
      --tuning-json fit_ann_tuning.json \
      --out all_panels.png

  # Or run the binaries directly instead of reading saved JSON:
  python plot_all_cluster_assignment_benchmarks.py \
      --cluster-binary /path/to/CUVS_CLUSTER_ASSIGNMENT_BENCH \
      --tuning-binary /path/to/CUVS_FIT_ANN_REBUILD_TUNING_BENCH \
      --out all_panels.png
"""

import argparse
import json
import re
import subprocess
import sys

CLUSTER_NAME_RE = re.compile(
    r"^BM_ClusterAssignment_(BruteForce|CAGRA|CAGRA_SearchOnly)/(\d+)/(\d+)/(\d+)/real_time$"
)
AMORTIZED_NAME_RE = re.compile(
    r"^BM_ClusterAssignment_CAGRA_AmortizedInterval/(\d+)/(\d+)/(\d+)/(\d+)/real_time$"
)
TUNING_NAME_RE = re.compile(
    r"^BM_FitAnnTuning_(Brute|CAGRA)/n_vectors:(\d+)/n_lists:(\d+)/dim:(\d+)"
    r"(?:/ann_rebuild_interval:(\d+))?/real_time$"
)

COLOR_BRUTE_FORCE = "#2a78d6"
COLOR_CAGRA = "#eb6834"
COLOR_CAGRA_SEARCH = "#1baf7a"
GRID_GRAY = "#d8d7d2"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"

CMAP_DIVERGING = None  # set lazily once matplotlib is imported


def run_benchmark(binary_path: str, extra_args: list[str]) -> dict:
    cmd = [binary_path, "--benchmark_format=json"] + extra_args
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(proc.stdout)


def load_json_or_run(
    json_path: str | None, binary_path: str | None, extra_args: list[str]
) -> dict:
    if json_path:
        with open(json_path) as f:
            return json.load(f)
    if binary_path:
        return run_benchmark(binary_path, extra_args)
    return {"benchmarks": []}


def parse_cluster_records(
    payload: dict,
) -> list[tuple[str, int, int, int, float]]:
    records = []
    for entry in payload.get("benchmarks", []):
        match = CLUSTER_NAME_RE.match(entry["name"])
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


def parse_amortized_records(
    payload: dict,
) -> list[tuple[int, int, int, int, float]]:
    """Return [(n_rows, n_clusters, dim, interval, total_ms), ...] for the empirically-measured
    'build once, search `interval` times back to back' benchmark.
    """
    records = []
    for entry in payload.get("benchmarks", []):
        match = AMORTIZED_NAME_RE.match(entry["name"])
        if not match:
            continue
        n_rows, n_clusters, dim, interval = match.groups()
        time_ms = entry["real_time"] * (
            1000 if entry.get("time_unit") == "s" else 1
        )
        records.append(
            (int(n_rows), int(n_clusters), int(dim), int(interval), time_ms)
        )
    return records


def parse_tuning_records(payload: dict) -> list[dict]:
    records = []
    for entry in payload.get("benchmarks", []):
        match = TUNING_NAME_RE.match(entry["name"])
        if not match:
            continue
        method, n_vectors, n_lists, dim, rebuild_interval = match.groups()
        records.append(
            {
                "method": method,
                "n_vectors": int(n_vectors),
                "n_lists": int(n_lists),
                "dim": int(dim),
                "ann_rebuild_interval": int(rebuild_interval)
                if rebuild_interval
                else None,
                "label_match_pct": entry.get("label_match_pct"),
                "centroid_mean_l2_drift": entry.get("centroid_mean_l2_drift"),
                "speedup_vs_brute": entry.get("speedup_vs_brute"),
            }
        )
    return records


def find_crossover(ks: list, bf: dict, other: dict):
    common = [k for k in ks if k in bf and k in other]
    prev_k, prev_sign = None, None
    for k in common:
        sign = 1 if other[k] > bf[k] else -1
        if prev_sign is not None and sign != prev_sign:
            return (prev_k, k)
        prev_k, prev_sign = k, sign
    return None


def style_axes(ax, xlabel, ylabel, title, logx=True, logy=True):
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_facecolor(SURFACE)
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=8.5)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=8.5)
    ax.set_title(
        title, color=TEXT_PRIMARY, fontsize=10, loc="left", fontweight="bold"
    )
    ax.grid(True, which="major", color=GRID_GRAY, linewidth=0.7, zorder=0)
    ax.grid(
        True,
        which="minor",
        color=GRID_GRAY,
        linewidth=0.35,
        alpha=0.5,
        zorder=0,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID_GRAY)
    ax.tick_params(axis="both", colors=TEXT_SECONDARY, labelsize=7.5)


def annotate_crossover(ax, crossover, y_at_after):
    if crossover is None:
        return
    k_before, k_after = crossover
    ax.axvspan(k_before, k_after, color=TEXT_SECONDARY, alpha=0.08, zorder=1)
    ax.annotate(
        f"K {k_before:,}-{k_after:,}",
        xy=((k_before * k_after) ** 0.5, y_at_after),
        ha="center",
        xytext=(0, 18),
        textcoords="offset points",
        color=TEXT_PRIMARY,
        fontsize=7.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec=GRID_GRAY, lw=0.7),
    )


def panel_line_pair(
    ax,
    x_bf,
    y_bf,
    x_other,
    y_other,
    other_label,
    other_color,
    xlabel,
    title,
    x_ticks=None,
):
    ax.plot(
        x_bf,
        y_bf,
        color=COLOR_BRUTE_FORCE,
        linewidth=1.8,
        marker="o",
        markersize=5,
        markerfacecolor=COLOR_BRUTE_FORCE,
        markeredgecolor=SURFACE,
        markeredgewidth=1,
        label="Brute force",
        zorder=3,
    )
    ax.plot(
        x_other,
        y_other,
        color=other_color,
        linewidth=1.8,
        marker="o",
        markersize=5,
        markerfacecolor=other_color,
        markeredgecolor=SURFACE,
        markeredgewidth=1,
        label=other_label,
        zorder=3,
    )
    style_axes(ax, xlabel, "Time (ms, log scale)", title)
    if x_ticks:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(
            [f"{t:,}" for t in x_ticks], rotation=35, ha="right", fontsize=6.5
        )
    ax.legend(
        loc="upper left", frameon=False, fontsize=7.5, labelcolor=TEXT_PRIMARY
    )


def panel1_ratio(ax, records, ratio=5, dim=128):
    data = {"BruteForce": {}, "CAGRA": {}}
    for method, n, k, d, ms in records:
        if method in data and n == ratio * k and d == dim:
            data[method][k] = ms
    ks_bf, ks_c = sorted(data["BruteForce"]), sorted(data["CAGRA"])
    all_k = sorted(set(ks_bf) | set(ks_c))
    panel_line_pair(
        ax,
        ks_bf,
        [data["BruteForce"][k] for k in ks_bf],
        ks_c,
        [data["CAGRA"][k] for k in ks_c],
        "CAGRA (ANN)",
        COLOR_CAGRA,
        "K  (N = 5xK)",
        "1. Ratio-controlled: N = 5xK",
        all_k,
    )
    crossover = find_crossover(all_k, data["BruteForce"], data["CAGRA"])
    if crossover:
        annotate_crossover(ax, crossover, data["CAGRA"][crossover[1]])
    return crossover


def panel2_fixed_n(ax, records, n_fixed=2000000, dim=128):
    data = {"BruteForce": {}, "CAGRA": {}}
    for method, n, k, d, ms in records:
        if method in data and n == n_fixed and d == dim:
            data[method][k] = ms
    ks_bf, ks_c = sorted(data["BruteForce"]), sorted(data["CAGRA"])
    all_k = sorted(set(ks_bf) | set(ks_c))
    panel_line_pair(
        ax,
        ks_bf,
        [data["BruteForce"][k] for k in ks_bf],
        ks_c,
        [data["CAGRA"][k] for k in ks_c],
        "CAGRA (ANN)",
        COLOR_CAGRA,
        f"K  (N = {n_fixed:,})",
        f"2. Fixed N = {n_fixed:,}",
        all_k,
    )
    crossover = find_crossover(all_k, data["BruteForce"], data["CAGRA"])
    if crossover:
        annotate_crossover(ax, crossover, data["CAGRA"][crossover[1]])
    return data, crossover


def panel3_fixed_k(ax, records, k_fixed=65536, dim=128):
    data = {"BruteForce": {}, "CAGRA": {}}
    for method, n, k, d, ms in records:
        if method in data and k == k_fixed and d == dim:
            data[method][n] = ms
    ns_bf, ns_c = sorted(data["BruteForce"]), sorted(data["CAGRA"])
    all_n = sorted(set(ns_bf) | set(ns_c))
    panel_line_pair(
        ax,
        ns_bf,
        [data["BruteForce"][n] for n in ns_bf],
        ns_c,
        [data["CAGRA"][n] for n in ns_c],
        "CAGRA (ANN)",
        COLOR_CAGRA,
        f"N  (K = {k_fixed:,})",
        f"3. Fixed K = {k_fixed:,}, vary N",
        all_n,
    )
    crossover = find_crossover(all_n, data["BruteForce"], data["CAGRA"])
    if crossover:
        annotate_crossover(ax, crossover, data["CAGRA"][crossover[1]])
    return crossover


def panel4_heatmap(ax, records, dim=128):
    import numpy as np

    grid: dict = {}
    for method, n, k, d, ms in records:
        if d != dim or method not in ("BruteForce", "CAGRA"):
            continue
        grid.setdefault((n, k), {})[method] = ms

    points = [
        (n, k)
        for (n, k), v in grid.items()
        if "BruteForce" in v and "CAGRA" in v
    ]
    if not points:
        ax.text(
            0.5,
            0.5,
            "no grid data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=TEXT_SECONDARY,
        )
        ax.set_title(
            "4. (N, K) speedup heatmap",
            color=TEXT_PRIMARY,
            fontsize=10,
            loc="left",
            fontweight="bold",
        )
        return

    ns = sorted({n for n, _ in points})
    ks = sorted({k for _, k in points})
    ratio_matrix = np.full((len(ks), len(ns)), float("nan"))
    for (n, k), v in grid.items():
        if "BruteForce" in v and "CAGRA" in v and n in ns and k in ks:
            ratio_matrix[ks.index(k), ns.index(n)] = (
                v["BruteForce"] / v["CAGRA"]
            )

    import matplotlib.colors as mcolors

    # Diverging: blue (brute force wins, ratio<1) -> gray midpoint (1) -> orange (CAGRA wins, ratio>1)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "bf_cagra_diverging", [COLOR_BRUTE_FORCE, "#e8e7e2", COLOR_CAGRA]
    )
    vmax = max(
        2.0,
        float(np.nanmax(ratio_matrix))
        if not np.all(np.isnan(ratio_matrix))
        else 2.0,
    )
    vmin = min(
        0.5,
        float(np.nanmin(ratio_matrix))
        if not np.all(np.isnan(ratio_matrix))
        else 0.5,
    )
    bound = max(abs(1 - vmin), abs(vmax - 1))
    norm = mcolors.TwoSlopeNorm(vmin=1 - bound, vcenter=1.0, vmax=1 + bound)

    im = ax.imshow(
        ratio_matrix, cmap=cmap, norm=norm, aspect="auto", origin="lower"
    )
    for i in range(len(ks)):
        for j in range(len(ns)):
            v = ratio_matrix[i, j]
            if not np.isnan(v):
                ax.text(
                    j,
                    i,
                    f"{v:.2f}x",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=TEXT_PRIMARY,
                )

    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels(
        [f"{n:,}" for n in ns], rotation=35, ha="right", fontsize=7
    )
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels([f"{k:,}" for k in ks], fontsize=7)
    ax.set_xlabel("N", color=TEXT_SECONDARY, fontsize=8.5)
    ax.set_ylabel("K", color=TEXT_SECONDARY, fontsize=8.5)
    ax.set_title(
        "4. (N, K) speedup heatmap  (BruteForce/CAGRA time, >1 = CAGRA faster)",
        color=TEXT_PRIMARY,
        fontsize=9.5,
        loc="left",
        fontweight="bold",
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig = ax.get_figure()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def panel5_speedup_ratio(ax, ratio_data, fixed_n_data):
    ks1 = sorted(set(ratio_data["BruteForce"]) & set(ratio_data["CAGRA"]))
    ks2 = sorted(set(fixed_n_data["BruteForce"]) & set(fixed_n_data["CAGRA"]))
    ax.plot(
        ks1,
        [ratio_data["BruteForce"][k] / ratio_data["CAGRA"][k] for k in ks1],
        color=COLOR_BRUTE_FORCE,
        linewidth=1.8,
        marker="o",
        markersize=5,
        markerfacecolor=COLOR_BRUTE_FORCE,
        markeredgecolor=SURFACE,
        markeredgewidth=1,
        label="Ratio-controlled (N=5xK)",
        zorder=3,
    )
    ax.plot(
        ks2,
        [
            fixed_n_data["BruteForce"][k] / fixed_n_data["CAGRA"][k]
            for k in ks2
        ],
        color=COLOR_CAGRA,
        linewidth=1.8,
        marker="o",
        markersize=5,
        markerfacecolor=COLOR_CAGRA,
        markeredgecolor=SURFACE,
        markeredgewidth=1,
        label="Fixed N=2,000,000",
        zorder=3,
    )
    ax.axhline(
        1.0, color=TEXT_SECONDARY, linewidth=1, linestyle="--", zorder=2
    )
    style_axes(
        ax,
        "K",
        "Speedup = BruteForce time / CAGRA time",
        "5. Speedup ratio vs K",
        logy=True,
    )
    ax.legend(
        loc="upper left", frameon=False, fontsize=7.5, labelcolor=TEXT_PRIMARY
    )


def panel6_build_vs_search(
    ax, records, amortized_records, n_fixed=2000000, dim=128
):
    """Grouped 100% stacked bar chart: for a representative subset of K (spanning below/at/above
    the crossover), show how the build-vs-search split changes as ann_rebuild_interval amortizes
    the one-time graph build over more search calls.

    total_ms for each (K, interval) bar is an EMPIRICAL measurement from
    BM_ClusterAssignment_CAGRA_AmortizedInterval -- it actually builds once and runs `interval`
    search calls back-to-back inside one timed region, rather than projecting the total from a
    formula. The build/search split within that measured total is then estimated using
    search_only_ms (single measured search call cost, same K) as the per-search unit cost:
    search_component = interval * search_only_ms; build_component = total_ms - search_component.
    """
    search_only_ms: dict = {}
    for method, n, k, d, ms in records:
        if n == n_fixed and d == dim and method == "CAGRA_SearchOnly":
            search_only_ms[k] = ms

    totals: dict = {}
    for n, k, d, interval, ms in amortized_records:
        if n == n_fixed and d == dim:
            totals[(k, interval)] = ms

    preferred_ks = [1000, 65536, 131072, 262144, 500000, 1000000]
    all_ks = sorted({k for k, _ in totals} & set(search_only_ms))
    ks = [k for k in preferred_ks if k in all_ks] or all_ks
    intervals = [1, 2, 5, 10]

    if not ks:
        ax.text(
            0.5,
            0.5,
            "no amortized-interval data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=TEXT_SECONDARY,
        )
        ax.set_title(
            "6. CAGRA build vs. search cost",
            color=TEXT_PRIMARY,
            fontsize=10,
            loc="left",
            fontweight="bold",
        )
        return

    bar_width = 0.19
    group_gap = 0.35
    group_width = bar_width * len(intervals)

    for gi, k in enumerate(ks):
        group_x0 = gi * (group_width + group_gap)
        for ii, interval in enumerate(intervals):
            if (k, interval) not in totals:
                continue
            total_ms = totals[(k, interval)]
            search_component = interval * search_only_ms[k]
            build_pct = max(
                0.0, min(100.0, (total_ms - search_component) / total_ms * 100)
            )
            search_pct = 100 - build_pct
            x = group_x0 + ii * bar_width
            ax.bar(
                x,
                build_pct,
                width=bar_width * 0.92,
                color=COLOR_CAGRA,
                zorder=3,
                label="Graph build" if gi == 0 and ii == 0 else None,
            )
            ax.bar(
                x,
                search_pct,
                width=bar_width * 0.92,
                bottom=build_pct,
                color=COLOR_CAGRA_SEARCH,
                zorder=3,
                label="Search" if gi == 0 and ii == 0 else None,
            )
            if build_pct > 8:
                ax.text(
                    x,
                    build_pct / 2,
                    f"{build_pct:.0f}",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="white",
                )
            ax.text(
                x,
                -6,
                str(interval),
                ha="center",
                va="top",
                fontsize=5.5,
                color=TEXT_SECONDARY,
            )

        ax.text(
            group_x0 + group_width / 2 - bar_width / 2,
            -16,
            f"K={k:,}",
            ha="center",
            va="top",
            fontsize=7,
            color=TEXT_PRIMARY,
            fontweight="bold",
        )

    ax.set_facecolor(SURFACE)
    ax.set_ylim(-22, 108)
    ax.set_xlim(-group_gap, len(ks) * (group_width + group_gap))
    ax.set_xticks([])
    ax.set_xlabel(
        f"ann_rebuild_interval (1, 2, 5, 10), grouped by K  (N = {n_fixed:,}) -- measured",
        color=TEXT_SECONDARY,
        fontsize=8,
        labelpad=22,
    )
    ax.set_ylabel(
        "Share of measured total time (%)", color=TEXT_SECONDARY, fontsize=8.5
    )
    ax.set_title(
        "6. CAGRA time breakdown vs. rebuild interval (measured)",
        color=TEXT_PRIMARY,
        fontsize=10,
        loc="left",
        fontweight="bold",
    )
    ax.grid(True, axis="y", color=GRID_GRAY, linewidth=0.7, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID_GRAY)
    ax.tick_params(axis="y", colors=TEXT_SECONDARY, labelsize=7.5)
    ax.legend(
        loc="upper left", frameon=False, fontsize=7, labelcolor=TEXT_PRIMARY
    )


def panel7_accuracy(ax, tuning_records):
    by_combo: dict = {}
    for r in tuning_records:
        if r["method"] != "CAGRA" or r["ann_rebuild_interval"] is None:
            continue
        key = (r["n_vectors"], r["n_lists"])
        by_combo.setdefault(key, {})[r["ann_rebuild_interval"]] = r

    if not by_combo:
        ax.text(
            0.5,
            0.5,
            "no tuning data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=TEXT_SECONDARY,
        )
        ax.set_title(
            "7. Assignment quality vs rebuild interval",
            color=TEXT_PRIMARY,
            fontsize=10,
            loc="left",
            fontweight="bold",
        )
        return

    (n_vectors, n_lists), by_interval = sorted(by_combo.items())[0]
    intervals = sorted(by_interval)
    label_match = [by_interval[i]["label_match_pct"] for i in intervals]
    ax2 = ax.twinx()

    ax.plot(
        intervals,
        label_match,
        color=COLOR_BRUTE_FORCE,
        linewidth=1.8,
        marker="o",
        markersize=6,
        markerfacecolor=COLOR_BRUTE_FORCE,
        markeredgecolor=SURFACE,
        markeredgewidth=1,
        label="Label match % vs brute force",
        zorder=3,
    )
    ax.set_ylabel("Label match %", color=COLOR_BRUTE_FORCE, fontsize=8.5)
    ax.set_facecolor(SURFACE)
    ax.set_xlabel("ann_rebuild_interval", color=TEXT_SECONDARY, fontsize=8.5)
    ax.set_title(
        f"7. Quality vs rebuild interval  (N={n_vectors:,}, K={n_lists:,})",
        color=TEXT_PRIMARY,
        fontsize=9.5,
        loc="left",
        fontweight="bold",
    )
    ax.grid(True, which="major", color=GRID_GRAY, linewidth=0.7, zorder=0)
    for spine in ("top",):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", colors=TEXT_SECONDARY, labelsize=7.5)

    drift = [by_interval[i]["centroid_mean_l2_drift"] for i in intervals]
    ax2.plot(
        intervals,
        drift,
        color=COLOR_CAGRA,
        linewidth=1.8,
        marker="s",
        markersize=6,
        markerfacecolor=COLOR_CAGRA,
        markeredgecolor=SURFACE,
        markeredgewidth=1,
        label="Centroid mean L2 drift",
        zorder=3,
    )
    ax2.set_ylabel("Centroid mean L2 drift", color=COLOR_CAGRA, fontsize=8.5)
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(axis="y", colors=COLOR_CAGRA, labelsize=7.5)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="lower right",
        frameon=False,
        fontsize=7,
        labelcolor=TEXT_PRIMARY,
    )


def panel8_dim_sensitivity(ax, records, n_fixed=2000000, k_fixed=131072):
    data = {"BruteForce": {}, "CAGRA": {}}
    for method, n, k, d, ms in records:
        if method in data and n == n_fixed and k == k_fixed:
            data[method][d] = ms
    dims_bf, dims_c = sorted(data["BruteForce"]), sorted(data["CAGRA"])
    all_dims = sorted(set(dims_bf) | set(dims_c))
    panel_line_pair(
        ax,
        dims_bf,
        [data["BruteForce"][d] for d in dims_bf],
        dims_c,
        [data["CAGRA"][d] for d in dims_c],
        "CAGRA (ANN)",
        COLOR_CAGRA,
        f"dim  (N={n_fixed:,}, K={k_fixed:,})",
        "8. Dimension sensitivity",
        all_dims,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cluster-json")
    parser.add_argument("--cluster-binary")
    parser.add_argument(
        "--amortized-json",
        help="Optional separate JSON with the "
        "AmortizedInterval sweep, if not already included in --cluster-json",
    )
    parser.add_argument("--tuning-json")
    parser.add_argument("--tuning-binary")
    parser.add_argument("--out", default="all_cluster_assignment_panels.png")
    parser.add_argument("--tuning-filter", default="327680/65536/128")
    args = parser.parse_args()

    if not args.cluster_json and not args.cluster_binary:
        parser.error("one of --cluster-json or --cluster-binary is required")

    cluster_payload = load_json_or_run(
        args.cluster_json, args.cluster_binary, []
    )
    tuning_payload = load_json_or_run(
        args.tuning_json,
        args.tuning_binary,
        ["--benchmark_filter=" + args.tuning_filter],
    )

    if args.amortized_json:
        with open(args.amortized_json) as f:
            amortized_payload = json.load(f)
        cluster_payload.setdefault("benchmarks", []).extend(
            amortized_payload.get("benchmarks", [])
        )

    records = parse_cluster_records(cluster_payload)
    amortized_records = parse_amortized_records(cluster_payload)
    tuning_records = parse_tuning_records(tuning_payload)

    if not records:
        print(
            "No cluster-assignment records parsed; nothing to plot.",
            file=sys.stderr,
        )
        sys.exit(1)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(24, 11), dpi=150)
    fig.patch.set_facecolor(SURFACE)

    panel1_ratio(axes[0][0], records)
    fixed_n_data, _ = panel2_fixed_n(axes[0][1], records)
    panel3_fixed_k(axes[0][2], records)
    panel4_heatmap(axes[0][3], records)

    ratio_data = {"BruteForce": {}, "CAGRA": {}}
    for method, n, k, d, ms in records:
        if method in ratio_data and n == 5 * k and d == 128:
            ratio_data[method][k] = ms
    panel5_speedup_ratio(axes[1][0], ratio_data, fixed_n_data)
    panel6_build_vs_search(axes[1][1], records, amortized_records)
    panel7_accuracy(axes[1][2], tuning_records)
    panel8_dim_sensitivity(axes[1][3], records)

    fig.suptitle(
        "Cluster assignment: Brute force vs CAGRA -- full analysis",
        color=TEXT_PRIMARY,
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, facecolor=SURFACE)
    print(f"Saved 8-panel figure to {args.out}")


if __name__ == "__main__":
    main()
