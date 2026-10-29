#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Produce all 8 cluster-assignment (brute force vs CAGRA) analysis panels in one figure:

  1. Ratio-controlled sweep: N = 5 * K
  2. Fixed-N sweep: N = 2,000,000, K varies
  3. Fixed-K sweep: K = 65,536, N varies
  4. (N, K) speedup heatmap (BruteForce / CAGRA time, log color scale centered on 1x)
  5. Speedup ratio (BruteForce / CAGRA) vs K, for sweeps 1 and 2, with break-even points
  6. CAGRA build-vs-search breakdown vs K (at N = 2,000,000)
  7. Fit quality (mean squared distance from each point to its assigned centroid, ANN fit divided
     by brute-force fit; k-means "inertia" / n_rows) and speed vs ann_rebuild_interval, plus CAGRA 1-NN
     agreement with exact search (CUVS_FIT_ANN_REBUILD_TUNING_BENCH)
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
import math
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
# Panels 5 and 7 plot sweeps / metrics rather than brute force vs CAGRA, so they must not reuse the
# method colors above.
COLOR_SERIES_A = "#7b4fd6"
COLOR_SERIES_B = "#222222"
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
    # Several sweeps register the same (method, N, K, dim); average those repeats instead of
    # letting the last one silently win in the per-panel dicts.
    times: dict = {}
    for entry in payload.get("benchmarks", []):
        match = CLUSTER_NAME_RE.match(entry["name"])
        if not match:
            continue
        method, n_rows, n_clusters, dim = match.groups()
        time_ms = entry["real_time"] * (
            1000 if entry.get("time_unit") == "s" else 1
        )
        key = (method, int(n_rows), int(n_clusters), int(dim))
        times.setdefault(key, []).append(time_ms)
    return [(*key, sum(ts) / len(ts)) for key, ts in times.items()]


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
                "inertia_ratio_vs_brute": entry.get("inertia_ratio_vs_brute"),
                "agreement_itopk64": entry.get(
                    "ann_1nn_agreement_pct_itopk64"
                ),
                "agreement_itopk256": entry.get(
                    "ann_1nn_agreement_pct_itopk256"
                ),
            }
        )
    return records


def find_crossover(xs: list, bf: dict, other: dict):
    """First x where `other` and `bf` times cross, interpolated linearly in log-log space between
    the two measured points that bracket the sign change. None if they never cross.
    """
    common = [x for x in xs if x in bf and x in other]
    for x0, x1 in zip(common, common[1:]):
        r0 = math.log(bf[x0] / other[x0])
        r1 = math.log(bf[x1] / other[x1])
        if r0 * r1 < 0:
            frac = r0 / (r0 - r1)
            return math.exp(
                math.log(x0) + frac * (math.log(x1) - math.log(x0))
            )
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


def annotate_crossover(ax, crossover, xname, xs, bf, other):
    """Mark the interpolated crossover, or say so (with the closest ratio) when there is none."""
    box = dict(boxstyle="round,pad=0.25", fc=SURFACE, ec=GRID_GRAY, lw=0.7)
    if crossover is None:
        x_last = max(x for x in xs if x in bf and x in other)
        ax.text(
            0.98,
            0.04,
            f"no crossover in range\nBF/CAGRA = {bf[x_last] / other[x_last]:.2f}x at {xname}={x_last:,}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color=TEXT_PRIMARY,
            fontsize=7.5,
            fontweight="bold",
            bbox=box,
        )
        return
    ax.axvline(
        crossover, color=TEXT_SECONDARY, linewidth=1, linestyle="--", zorder=2
    )
    ax.annotate(
        f"crossover {xname} ~ {crossover:,.0f}",
        xy=(crossover, 0.04),
        xycoords=("data", "axes fraction"),
        xytext=(5, 0),
        textcoords="offset points",
        ha="left",
        va="bottom",
        color=TEXT_PRIMARY,
        fontsize=7.5,
        fontweight="bold",
        bbox=box,
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
    annotate_crossover(
        ax, crossover, "K", all_k, data["BruteForce"], data["CAGRA"]
    )
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
    annotate_crossover(
        ax, crossover, "K", all_k, data["BruteForce"], data["CAGRA"]
    )
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
    annotate_crossover(
        ax, crossover, "N", all_n, data["BruteForce"], data["CAGRA"]
    )
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
    # Color in log2(ratio) so 2x faster and 2x slower are equally far from the midpoint. Colors
    # saturate at 16x either way; the cell labels still print the exact ratio.
    bound = 4.0
    norm = mcolors.TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    im = ax.imshow(
        np.clip(np.log2(ratio_matrix), -bound, bound),
        cmap=cmap,
        norm=norm,
        aspect="auto",
        origin="lower",
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
        "4. (N, K) speedup heatmap",
        color=TEXT_PRIMARY,
        fontsize=10,
        loc="left",
        fontweight="bold",
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig = ax.get_figure()
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([-4, -2, 0, 2, 4])
    cbar.set_ticklabels(["1/16x", "1/4x", "1x", "4x", "16x"])
    cbar.set_label(
        "BF time / CAGRA time  (>1x: CAGRA faster)",
        color=TEXT_SECONDARY,
        fontsize=8,
    )
    cbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=7)


def panel5_speedup_ratio(ax, ratio_data, fixed_n_data):
    # Blue/orange mean brute force/CAGRA in the other panels; here each curve is a whole sweep
    # (BF time / CAGRA time), so use separate colors and marker shapes. The dashed line is
    # break-even; the diamond on it marks where each sweep's own curve reaches it.
    series = [
        ("N = 5xK sweep", ratio_data, COLOR_SERIES_A, "o", (30, -16)),
        ("N = 2,000,000 sweep", fixed_n_data, COLOR_SERIES_B, "s", (-30, 10)),
    ]
    ax.axhline(
        1.0, color=TEXT_SECONDARY, linewidth=1, linestyle="--", zorder=2
    )
    for label, data, color, marker, text_offset in series:
        ks = sorted(set(data["BruteForce"]) & set(data["CAGRA"]))
        ax.plot(
            ks,
            [data["BruteForce"][k] / data["CAGRA"][k] for k in ks],
            color=color,
            linewidth=1.8,
            marker=marker,
            markersize=5,
            markerfacecolor=color,
            markeredgecolor=SURFACE,
            markeredgewidth=1,
            label=label,
            zorder=3,
        )
        break_even = find_crossover(ks, data["BruteForce"], data["CAGRA"])
        if break_even is not None:
            ax.plot(
                [break_even],
                [1.0],
                marker="D",
                markersize=7,
                markerfacecolor=SURFACE,
                markeredgecolor=color,
                markeredgewidth=1.6,
                zorder=4,
            )
            ax.annotate(
                f"K ~ {break_even:,.0f}",
                xy=(break_even, 1.0),
                xytext=text_offset,
                textcoords="offset points",
                ha="center",
                color=color,
                fontsize=7.5,
                fontweight="bold",
            )
    style_axes(
        ax,
        "K",
        "BF time / CAGRA time  (>1: CAGRA faster)",
        "5. Speedup vs K, per sweep  (dashed = break-even)",
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
    ax.set_ylim(-22, 120)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
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
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=7.5,
        labelcolor=TEXT_PRIMARY,
    )


def panel7_accuracy(ax, tuning_records):
    """Fit quality and speed vs ann_rebuild_interval, for the smallest (N, K) in the tuning JSON.

    Quality is inertia_ratio_vs_brute: the mean squared distance from each point to its assigned
    centroid (k-means inertia / n_rows) for the ANN fit divided by the same quantity for the brute
    fit. Smaller means tighter clusters and 1.0 means equal. Unlike label match or centroid drift it
    does not depend on cluster ids lining up between two independent fits. The 1-NN agreement with
    exact search (fixed centroids, so no staleness) is independent of the rebuild interval and is
    shown as text.
    """
    by_combo: dict = {}
    for r in tuning_records:
        if r["method"] != "CAGRA" or r["ann_rebuild_interval"] is None:
            continue
        key = (r["n_vectors"], r["n_lists"])
        by_combo.setdefault(key, {})[r["ann_rebuild_interval"]] = r

    title = "7. Fit quality and speed vs rebuild interval"
    needed = ("inertia_ratio_vs_brute", "speedup_vs_brute")
    if not by_combo or any(
        r.get(f) is None
        for by in by_combo.values()
        for r in by.values()
        for f in needed
    ):
        ax.text(
            0.5,
            0.5,
            "no tuning data with inertia_ratio_vs_brute\n"
            "(rerun CUVS_FIT_ANN_REBUILD_TUNING_BENCH)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=TEXT_SECONDARY,
            fontsize=8,
        )
        ax.set_title(
            title,
            color=TEXT_PRIMARY,
            fontsize=10,
            loc="left",
            fontweight="bold",
        )
        return

    (n_vectors, n_lists), by_interval = sorted(by_combo.items())[0]
    intervals = sorted(by_interval)
    inertia = [by_interval[i]["inertia_ratio_vs_brute"] for i in intervals]
    speedup = [by_interval[i]["speedup_vs_brute"] for i in intervals]
    ax2 = ax.twinx()

    ax.axhline(
        1.0, color=TEXT_SECONDARY, linewidth=1, linestyle="--", zorder=2
    )
    # The dashed line belongs to the left (distance ratio) axis only; say so, since the speedup
    # line on the right axis crosses it at an unrelated value.
    ax.text(
        0.99,
        1.0,
        "left axis: 1.0 = same as brute force",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="top",
        color=TEXT_SECONDARY,
        fontsize=7,
    )
    ax.plot(
        intervals,
        inertia,
        color=COLOR_SERIES_A,
        linewidth=1.8,
        marker="o",
        markersize=6,
        markerfacecolor=COLOR_SERIES_A,
        markeredgecolor=SURFACE,
        markeredgewidth=1,
        label="Point-to-centroid distance, ANN fit / brute fit (1.0 = same, >1 = looser clusters)",
        zorder=3,
    )
    ax.set_ylabel(
        "Mean sq. distance to assigned centroid\n(ANN fit / brute fit)",
        color=COLOR_SERIES_A,
        fontsize=8.5,
    )
    ax.set_facecolor(SURFACE)
    ax.set_xlabel("ann_rebuild_interval", color=TEXT_SECONDARY, fontsize=8.5)
    ax.set_xticks(intervals)
    ax.set_title(
        f"{title}  (N={n_vectors:,}, K={n_lists:,})",
        color=TEXT_PRIMARY,
        fontsize=9.5,
        loc="left",
        fontweight="bold",
    )
    ax.grid(True, which="major", color=GRID_GRAY, linewidth=0.7, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="x", colors=TEXT_SECONDARY, labelsize=7.5)
    ax.tick_params(axis="y", colors=COLOR_SERIES_A, labelsize=7.5)

    ax2.plot(
        intervals,
        speedup,
        color=COLOR_SERIES_B,
        linewidth=1.8,
        marker="s",
        markersize=6,
        markerfacecolor=COLOR_SERIES_B,
        markeredgecolor=SURFACE,
        markeredgewidth=1,
        label="Fit speedup vs brute (>1 = ANN faster)",
        zorder=3,
    )
    ax2.set_ylabel(
        "Fit speedup (brute time / ANN time)",
        color=COLOR_SERIES_B,
        fontsize=8.5,
    )
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(axis="y", colors=COLOR_SERIES_B, labelsize=7.5)

    # Leave an empty band above the data for the legend so it never sits on a line.
    for axis, values in ((ax, inertia + [1.0]), (ax2, speedup)):
        lo, hi = min(values), max(values)
        pad = max(hi - lo, 1e-3) * 0.08
        axis.set_ylim(lo - pad, hi + pad + 0.45 * (hi - lo + 2 * pad))

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
        frameon=False,
        fontsize=7,
        labelcolor=TEXT_PRIMARY,
    )

    sample = by_interval[intervals[0]]
    if sample.get("agreement_itopk64") is not None:
        ax.text(
            0.98,
            0.04,
            "CAGRA 1-NN == exact, fixed centroids:\n"
            f"  itopk_size 64 (default): {sample['agreement_itopk64']:.1f}%\n"
            f"  itopk_size 256: {sample['agreement_itopk256']:.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color=TEXT_PRIMARY,
            fontsize=7.5,
            bbox=dict(
                boxstyle="round,pad=0.3", fc=SURFACE, ec=GRID_GRAY, lw=0.7
            ),
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
    parser.add_argument(
        "--tuning-filter",
        default="n_vectors:327680/n_lists:65536/dim:128",
        help="--benchmark_filter regex used when running --tuning-binary",
    )
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
