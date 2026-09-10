#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CAGRA Recall/Throughput Benchmark: SINGLE_CTA vs MULTI_CTA with search_width > 1

Measures recall@10 and throughput (QPS) across configurations to evaluate
the impact of the AUTO algorithm selector when search_width > 1.

Configurations tested:
  1. MULTI_CTA (default max_iterations) — reference baseline
  2. SINGLE_CTA (default max_iterations) — shows the recall problem
  3. SINGLE_CTA (floor at 32 base iterations) — alternative fix
  4. AUTO (current code) — shows when the switch happens

For each: sweep batch_size=[1, 64, 256, 512, 1024] x search_width=[1, 4, 8]
"""

import time

import numpy as np
from cuvs.neighbors import brute_force, cagra
from pylibraft.common import device_ndarray


# --- Dataset parameters ---
N_SAMPLES = 100_000
N_QUERIES = 2048
DIM = 128
K = 10
N_WARMUP = 3
N_TIMED = 10

# --- Search configurations ---
BATCH_SIZES = [1, 64, 256, 512, 1024]
SEARCH_WIDTHS = [1, 4, 8]
ITOPK_SIZE = 64
GRAPH_DEGREE = 64


def compute_reachability_iters(dataset_size, graph_degree):
    """Replicate the C++ reachability iteration calculation."""
    iters = 0
    reachable = 1
    while reachable < dataset_size:
        reachable *= max(2, graph_degree // 2)
        iters += 1
    return iters


def compute_max_iterations(
    algo, itopk_size, search_width, dataset_size, graph_degree
):
    """Replicate the C++ max_iterations calculation for each algorithm."""
    reach = compute_reachability_iters(dataset_size, graph_degree)
    if algo == "multi_cta":
        base = 32  # mc_itopk_size / mc_search_width = 32/1
    elif algo == "single_cta_default":
        base = itopk_size // search_width
    elif algo == "single_cta_floor32":
        # Alternative fix: floor base iterations at 32 (match MULTI_CTA)
        base = max(itopk_size // search_width, 32)
    else:
        return 0  # Let cuVS calculate (AUTO mode)
    return base + reach


def compute_recall(results, ground_truth, k):
    """Compute recall@k: fraction of true k-NN found in results."""
    n = results.shape[0]
    recall = 0.0
    for i in range(n):
        gt_set = set(ground_truth[i, :k].tolist())
        res_set = set(results[i, :k].tolist())
        recall += len(gt_set & res_set) / k
    return recall / n


def run_benchmark():
    np.random.seed(42)

    print("Generating dataset...")
    dataset = np.random.random((N_SAMPLES, DIM)).astype(np.float32)
    queries = dataset[:N_QUERIES].copy()

    # Move dataset to device
    dataset_device = device_ndarray(dataset)
    queries_device = device_ndarray(queries)

    # Build brute-force ground truth
    print("Computing brute-force ground truth...")
    bf_index = brute_force.build(dataset_device)
    gt_distances, gt_neighbors = brute_force.search(
        bf_index, queries_device, k=K
    )
    gt_neighbors_host = gt_neighbors.copy_to_host()

    # Build CAGRA index
    print("Building CAGRA index...")
    index_params = cagra.IndexParams(
        intermediate_graph_degree=128,
        graph_degree=GRAPH_DEGREE,
    )
    cagra_index = cagra.build(index_params, dataset_device)
    print(
        f"  Index built: {N_SAMPLES} vectors, dim={DIM}, graph_degree={GRAPH_DEGREE}"
    )

    reach_iters = compute_reachability_iters(N_SAMPLES, GRAPH_DEGREE)
    print(f"  Reachability iterations: {reach_iters}")
    print()

    # Define test configurations
    configs = [
        ("MULTI_CTA", "multi_cta"),
        ("SINGLE_CTA (default)", "single_cta_default"),
        ("SINGLE_CTA (floor@32)", "single_cta_floor32"),
        ("AUTO", "auto"),
    ]

    # Header
    print(
        f"{'Config':<25} {'sw':>3} {'batch':>6} {'max_iter':>9} "
        f"{'recall@10':>10} {'QPS':>10}"
    )
    print("-" * 75)

    results_table = []

    for search_width in SEARCH_WIDTHS:
        for batch_size in BATCH_SIZES:
            batch_queries = queries[:batch_size]
            batch_gt = gt_neighbors_host[:batch_size]

            for config_name, config_key in configs:
                max_iter = compute_max_iterations(
                    config_key,
                    ITOPK_SIZE,
                    search_width,
                    N_SAMPLES,
                    GRAPH_DEGREE,
                )

                # Build SearchParams — algo must be set in constructor
                if config_key == "multi_cta":
                    search_params = cagra.SearchParams(
                        algo="multi_cta",
                        itopk_size=ITOPK_SIZE,
                        search_width=search_width,
                        max_iterations=max_iter,
                    )
                elif config_key.startswith("single_cta"):
                    search_params = cagra.SearchParams(
                        algo="single_cta",
                        itopk_size=ITOPK_SIZE,
                        search_width=search_width,
                        max_iterations=max_iter,
                    )
                else:
                    # AUTO: let cuVS decide algorithm and max_iterations
                    search_params = cagra.SearchParams(
                        algo="auto",
                        itopk_size=ITOPK_SIZE,
                        search_width=search_width,
                    )

                batch_device = device_ndarray(batch_queries)

                # Warmup
                for _ in range(N_WARMUP):
                    _, neighbors = cagra.search(
                        search_params, cagra_index, batch_device, k=K
                    )

                # Timed runs
                elapsed = 0.0
                for _ in range(N_TIMED):
                    start = time.perf_counter()
                    _, neighbors = cagra.search(
                        search_params, cagra_index, batch_device, k=K
                    )
                    elapsed += time.perf_counter() - start

                neighbors_host = neighbors.copy_to_host()
                recall_val = compute_recall(neighbors_host, batch_gt, K)

                avg_time = elapsed / N_TIMED
                qps = batch_size / avg_time if avg_time > 0 else 0

                iter_str = str(max_iter) if max_iter > 0 else "auto"
                print(
                    f"{config_name:<25} {search_width:>3} {batch_size:>6} "
                    f"{iter_str:>9} {recall_val:>10.4f} {qps:>10.0f}"
                )

                results_table.append(
                    {
                        "config": config_name,
                        "search_width": search_width,
                        "batch_size": batch_size,
                        "max_iterations": max_iter,
                        "recall": recall_val,
                        "qps": qps,
                    }
                )

        print()  # Blank line between search_width groups

    # Summary analysis
    print("\n" + "=" * 75)
    print("ANALYSIS: Recall delta at algorithm switch point (batch_size=512)")
    print("=" * 75)
    for sw in SEARCH_WIDTHS:
        multi = [
            r
            for r in results_table
            if r["config"] == "MULTI_CTA"
            and r["search_width"] == sw
            and r["batch_size"] == 512
        ]
        single_def = [
            r
            for r in results_table
            if r["config"] == "SINGLE_CTA (default)"
            and r["search_width"] == sw
            and r["batch_size"] == 512
        ]
        single_fix = [
            r
            for r in results_table
            if r["config"] == "SINGLE_CTA (floor@32)"
            and r["search_width"] == sw
            and r["batch_size"] == 512
        ]
        auto = [
            r
            for r in results_table
            if r["config"] == "AUTO"
            and r["search_width"] == sw
            and r["batch_size"] == 512
        ]

        if multi and single_def and single_fix and auto:
            m = multi[0]
            sd = single_def[0]
            sf = single_fix[0]
            a = auto[0]
            print(f"\n  search_width={sw}:")
            print(
                f"    MULTI_CTA:              recall={m['recall']:.4f}  QPS={m['qps']:.0f}"
            )
            print(
                f"    SINGLE_CTA (default):   recall={sd['recall']:.4f}  QPS={sd['qps']:.0f}  "
                f"(recall delta: {sd['recall'] - m['recall']:+.4f})"
            )
            print(
                f"    SINGLE_CTA (floor@32):  recall={sf['recall']:.4f}  QPS={sf['qps']:.0f}  "
                f"(recall delta: {sf['recall'] - m['recall']:+.4f})"
            )
            print(
                f"    AUTO (current code):    recall={a['recall']:.4f}  QPS={a['qps']:.0f}  "
                f"(recall delta: {a['recall'] - m['recall']:+.4f})"
            )

            if sd["qps"] > 0 and m["qps"] > 0:
                print(
                    f"    Throughput gain (SINGLE_CTA default vs MULTI_CTA): "
                    f"{sd['qps'] / m['qps']:.2f}x"
                )
            if sf["qps"] > 0 and m["qps"] > 0:
                print(
                    f"    Throughput gain (SINGLE_CTA floor@32 vs MULTI_CTA): "
                    f"{sf['qps'] / m['qps']:.2f}x"
                )

    print("\n\nKey question: Does SINGLE_CTA (floor@32) recover recall while")
    print("preserving throughput advantage over MULTI_CTA?")


if __name__ == "__main__":
    run_benchmark()
