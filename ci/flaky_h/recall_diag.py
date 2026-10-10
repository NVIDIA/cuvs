# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# DO NOT MERGE: recall distributions of test_ivf_pq and test_filtered_ivf_flat.
#
# Runs the real test helpers (from the installed cuvs.tests, which the CI
# script overwrites with the PR's test files) many times with different
# global numpy seeds and records the recall that each run asserts on.
import argparse
import time

import numpy as np

import cuvs.tests.ann_utils as ann_utils
import cuvs.tests.test_ivf_flat as t_flat
import cuvs.tests.test_ivf_pq as t_pq

RECALLS = []


def recording(fn):
    def wrapped(*args, **kwargs):
        r = fn(*args, **kwargs)
        RECALLS.append(r)
        return r

    return wrapped


orig = ann_utils.calc_recall
ann_utils.calc_recall = recording(orig)
t_pq.calc_recall = recording(orig)
t_flat.calc_recall = recording(orig)


def summarize(name, rs, fails):
    rs = np.array(rs)
    print(
        f"RECALL_SUMMARY {name}: n={rs.size} mean={rs.mean():.4f} "
        f"std={rs.std():.4f} min={rs.min():.3f} p01={np.quantile(rs, 0.01):.3f} "
        f"max={rs.max():.3f} n_le_0.7={int((rs <= 0.7).sum())} "
        f"asserts_failed={fails}",
        flush=True,
    )


def sweep(name, fn, seeds):
    rs, fails = [], 0
    for s in seeds:
        np.random.seed(s)
        RECALLS.clear()
        try:
            fn()
        except AssertionError:
            fails += 1
        rs.append(RECALLS[-1])
    summarize(name, rs, fails)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--fixed-repeats", type=int, default=50)
    a = ap.parse_args()
    seeds = range(100000, 100000 + a.n)
    t0 = time.time()
    cases = {}
    for metric in ["inner_product", "sqeuclidean", "euclidean"]:
        cases[f"test_ivf_pq[{metric}]"] = (
            lambda m=metric: t_pq.run_ivf_pq_build_search_test(
                dtype=np.float32, inplace=True, metric=m
            )
        )
    for sp in [0.5, 0.7, 1.0]:
        cases[f"test_filtered_ivf_flat[{sp}]"] = (
            lambda sp=sp: ann_utils.run_filtered_search_test(
                t_flat.ivf_flat, sp
            )
        )
    for name, fn in cases.items():
        sweep(name, fn, seeds)
        # Same data every time: only library nondeterminism remains.
        sweep(name + "@fixed-seed", fn, [12345] * a.fixed_repeats)
    print(f"RECALL_DIAG done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
