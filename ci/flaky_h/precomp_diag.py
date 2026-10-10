# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# DO NOT MERGE: diagnostics for test_ivf_pq.py::test_build_precomputed.
#
# Builds the regular and the precomputed index exactly like the test does,
# searches both, and classifies every index mismatch: is it inside a group of
# bitwise-equal distances (a tie) or not? Also checks whether both indexes have
# the same labels/codes and the same in-list storage order.
import argparse
import os
import time

import numpy as np
from pylibraft.common import device_ndarray
from sklearn.preprocessing import normalize

from cuvs.neighbors import ivf_pq


def list_maps(index, n_lists):
    lab, pos, codes = {}, {}, {}
    for c in range(n_lists):
        ids = index.list_indices(c).copy_to_host()
        data = index.list_data(c).copy_to_host()
        for p, i in enumerate(ids):
            lab[int(i)] = c
            pos[int(i)] = p
            codes[int(i)] = data[p].tobytes()
    return lab, pos, codes


def tie_groups(row):
    """Contiguous runs of bitwise-equal values: list of (start, end)."""
    groups, s = [], 0
    for c in range(1, len(row) + 1):
        if c == len(row) or row[c] != row[s]:
            groups.append((s, c))
            s = c
    return groups


def run(
    seed,
    metric,
    cb,
    n_queries,
    n_rows=5000,
    n_cols=32,
    k=10,
    n_lists=50,
    pq_dim=8,
    n_probes=50,
    inspect_lists=False,
):
    rng = np.random.default_rng(seed)
    dataset = rng.random((n_rows, n_cols)).astype(np.float32)
    if metric == "inner_product":
        dataset = normalize(dataset, norm="l2", axis=1)
    dd = device_ndarray(dataset)
    bp = ivf_pq.IndexParams(
        n_lists=n_lists,
        metric=metric,
        kmeans_n_iters=20,
        kmeans_trainset_fraction=1.0,
        pq_bits=8,
        pq_dim=pq_dim,
        codebook_kind=cb,
        force_random_rotation=False,
        add_data_on_build=True,
    )
    reg = ivf_pq.build(bp, dd)
    pbp = ivf_pq.IndexParams(
        n_lists=n_lists,
        metric=metric,
        pq_bits=8,
        pq_dim=pq_dim,
        codebook_kind=cb,
    )
    pre = ivf_pq.build_precomputed(
        pbp,
        reg.dim,
        reg.pq_centers,
        reg.centers_padded,
        reg.centers_rot,
        reg.rotation_matrix,
    )
    pre = ivf_pq.extend(
        pre, dd, device_ndarray(np.arange(n_rows, dtype=np.int64))
    )
    q = device_ndarray(rng.random((n_queries, n_cols)).astype(np.float32))
    sp = ivf_pq.SearchParams(n_probes=n_probes)
    rd, ri = ivf_pq.search(sp, reg, q, k)
    rd, ri = rd.copy_to_host(), ri.copy_to_host()
    pd, pi = ivf_pq.search(sp, pre, q, k)
    pd, pi = pd.copy_to_host(), pi.copy_to_host()
    rd2, ri2 = ivf_pq.search(sp, reg, q, k)
    rd2, ri2 = rd2.copy_to_host(), ri2.copy_to_host()

    # The comparison used by the fixed test (search k + 1, tie-aware).
    xd, xi = ivf_pq.search(sp, reg, q, k + 1)
    xd, xi = xd.copy_to_host(), xi.copy_to_host()
    yd, yi = ivf_pq.search(sp, pre, q, k + 1)
    yd, yi = yd.copy_to_host(), yi.copy_to_host()
    new_check_fail_rows = 0
    new_check_dist_fail = not np.array_equal(xd[:, :k], yd[:, :k])
    for r in range(n_queries):
        dist = xd[r]
        start = 0
        bad = False
        while start < k:
            end = start + 1
            while end <= k and dist[end] == dist[start]:
                end += 1
            if end <= k and not np.array_equal(
                np.sort(xi[r, start:end]), np.sort(yi[r, start:end])
            ):
                bad = True
            start = end
        new_check_fail_rows += bad

    res = dict(
        new_check_fail_rows=new_check_fail_rows,
        new_check_dist_fail=new_check_dist_fail,
        rows_tied=0,
        rows_mismatch=0,
        rows_mismatch_in_tie=0,
        rows_mismatch_not_tie=0,
        rows_boundary_tie_diff=0,
        dist_bitwise_equal=bool((rd == pd).all()),
        dist_maxabs=float(np.abs(rd - pd).max()),
        self_repeat_equal=bool((ri == ri2).all() and (rd == rd2).all()),
        examples=[],
    )
    for r in range(n_queries):
        groups = tie_groups(rd[r])
        tied = any(e - s > 1 for s, e in groups)
        res["rows_tied"] += tied
        if (ri[r] == pi[r]).all():
            continue
        res["rows_mismatch"] += 1
        ok = (rd[r] == pd[r]).all()
        boundary = False
        for s, e in groups:
            if set(ri[r, s:e]) != set(pi[r, s:e]):
                if e == k and e - s > 1:
                    boundary = True
                else:
                    ok = False
            elif e - s == 1 and ri[r, s] != pi[r, s]:
                ok = False
        if ok:
            res["rows_mismatch_in_tie"] += 1
            res["rows_boundary_tie_diff"] += boundary
        else:
            res["rows_mismatch_not_tie"] += 1
        if len(res["examples"]) < 3:
            cols = np.where(ri[r] != pi[r])[0]
            res["examples"].append(
                dict(
                    q=r,
                    cols=cols.tolist(),
                    reg_idx=ri[r, cols].tolist(),
                    pre_idx=pi[r, cols].tolist(),
                    reg_dist=[float.hex(float(x)) for x in rd[r, cols]],
                    pre_dist=[float.hex(float(x)) for x in pd[r, cols]],
                )
            )
    if inspect_lists and res["examples"]:
        rl, rp, rc = list_maps(reg, n_lists)
        pl, pp, pc = list_maps(pre, n_lists)
        res["same_labels"] = all(rl[i] == pl[i] for i in range(n_rows))
        res["same_codes"] = all(rc[i] == pc[i] for i in range(n_rows))
        res["same_in_list_pos_frac"] = (
            sum(rp[i] == pp[i] for i in range(n_rows)) / n_rows
        )
        for ex in res["examples"]:
            ex["lists"] = [rl[int(i)] for i in ex["reg_idx"]]
            ex["reg_pos"] = [rp[int(i)] for i in ex["reg_idx"]]
            ex["pre_pos"] = [pp[int(i)] for i in ex["reg_idx"]]
            ex["identical_codes"] = (
                len({rc[int(i)] for i in ex["reg_idx"]}) == 1
            )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="inner_product")
    ap.add_argument("--codebook", default="cluster")
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--seconds", type=float, default=1e9)
    a = ap.parse_args()
    tot = dict(
        runs=0,
        runs_fail=0,
        rows=0,
        rows_tied=0,
        rows_mismatch=0,
        rows_mismatch_in_tie=0,
        rows_mismatch_not_tie=0,
        rows_boundary_tie_diff=0,
        runs_dist_not_bitwise=0,
        runs_self_repeat_differs=0,
        new_check_fail_rows=0,
        new_check_dist_fail_runs=0,
    )
    t0 = time.time()
    for it in range(a.iters):
        if time.time() - t0 > a.seconds:
            break
        seed = int.from_bytes(os.urandom(4), "little")
        r = run(seed, a.metric, a.codebook, a.n_queries, inspect_lists=True)
        tot["runs"] += 1
        tot["rows"] += a.n_queries
        tot["runs_fail"] += r["rows_mismatch"] > 0
        tot["new_check_fail_rows"] += r["new_check_fail_rows"]
        tot["new_check_dist_fail_runs"] += r["new_check_dist_fail"]
        for key in (
            "rows_tied",
            "rows_mismatch",
            "rows_mismatch_in_tie",
            "rows_mismatch_not_tie",
            "rows_boundary_tie_diff",
        ):
            tot[key] += r[key]
        tot["runs_dist_not_bitwise"] += not r["dist_bitwise_equal"]
        tot["runs_self_repeat_differs"] += not r["self_repeat_equal"]
        if r["rows_mismatch"] or not r["dist_bitwise_equal"]:
            print(f"PRECOMP_DIAG seed={seed} {r}", flush=True)
    print(
        f"PRECOMP_SUMMARY metric={a.metric} codebook={a.codebook} "
        f"n_queries={a.n_queries} {tot}",
        flush=True,
    )


if __name__ == "__main__":
    main()
