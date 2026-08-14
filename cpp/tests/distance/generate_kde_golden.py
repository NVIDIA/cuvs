#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate golden values for KDE gtest from sklearn / scipy."""

import numpy as np
from scipy.spatial.distance import cdist
from math import lgamma
from scipy.special import logsumexp

try:
    from sklearn.neighbors import KernelDensity

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("WARNING: sklearn not available, using manual computation only")


# ============================================================================
# Manual KDE reference (for metrics sklearn doesn't support)
# ============================================================================


def log_vn(d):
    """Log volume of d-dimensional unit ball."""
    return 0.5 * d * np.log(np.pi) - lgamma(0.5 * d + 1)


def log_sn(d):
    """Log surface area of d-dimensional unit sphere."""
    return np.log(2 * np.pi) + log_vn(d - 1)


def norm_factor(kernel, h, d):
    if kernel == "gaussian":
        factor = 0.5 * d * np.log(2 * np.pi)
    elif kernel == "tophat":
        factor = log_vn(d)
    elif kernel == "epanechnikov":
        factor = log_vn(d) + np.log(2.0 / (d + 2))
    elif kernel == "exponential":
        factor = log_sn(d - 1) + lgamma(d)
    elif kernel == "linear":
        factor = log_vn(d) - np.log(d + 1)
    elif kernel == "cosine":
        two_over_pi = 2.0 / np.pi
        two_over_pi_sq = two_over_pi**2
        I_prev = two_over_pi
        I_curr = two_over_pi - two_over_pi_sq
        n = d - 1
        if n == 0:
            factor = np.log(I_prev) + log_sn(d - 1)
        else:
            for j in range(2, n + 1):
                I_next = two_over_pi - j * (j - 1) * two_over_pi_sq * I_prev
                I_prev = I_curr
                I_curr = I_next
            factor = np.log(I_curr) + log_sn(d - 1)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    return factor + d * np.log(h)


def log_kernel_eval(dist, h, kernel):
    """Evaluate log-kernel (element-wise on arrays)."""
    LOWEST = np.finfo(np.float64).min
    if kernel == "gaussian":
        return -(dist**2) / (2 * h**2)
    elif kernel == "tophat":
        return np.where(dist < h, 0.0, LOWEST)
    elif kernel == "epanechnikov":
        z = np.maximum(1 - (dist**2) / (h**2), 1e-30)
        return np.where(dist < h, np.log(z), LOWEST)
    elif kernel == "exponential":
        return -dist / h
    elif kernel == "linear":
        z = np.maximum(1 - dist / h, 1e-30)
        return np.where(dist < h, np.log(z), LOWEST)
    elif kernel == "cosine":
        z = np.maximum(np.cos(0.5 * np.pi * dist / h), 1e-30)
        return np.where(dist < h, np.log(z), LOWEST)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")


def manual_kde(query, train, bandwidth, kernel, dists, weights=None):
    """Compute log-density using precomputed distances."""
    n_train = train.shape[0]
    d = train.shape[1]

    log_k = log_kernel_eval(dists, bandwidth, kernel)

    if weights is not None:
        log_k = log_k + np.log(weights)[np.newaxis, :]
        sw = np.sum(weights)
    else:
        sw = float(n_train)

    log_sum = logsumexp(log_k, axis=1)
    log_norm = np.log(sw) + norm_factor(kernel, bandwidth, d)
    return log_sum - log_norm


def compute_dists_scipy(query, train, metric, metric_arg=None):
    """Compute pairwise distances using scipy."""
    if metric == "euclidean":
        return cdist(query, train, metric="euclidean")
    elif metric == "sqeuclidean":
        return cdist(query, train, metric="sqeuclidean")
    elif metric == "manhattan":
        return cdist(query, train, metric="cityblock")
    elif metric == "chebyshev":
        return cdist(query, train, metric="chebyshev")
    elif metric == "minkowski":
        return cdist(query, train, metric="minkowski", p=metric_arg)
    elif metric == "cosine":
        return cdist(query, train, metric="cosine")
    elif metric == "correlation":
        return cdist(query, train, metric="correlation")
    elif metric == "canberra":
        return cdist(query, train, metric="canberra")
    elif metric == "hellinger":
        # sqrt(max(0, 1 - sum(sqrt(a)*sqrt(b))))
        n_q, n_t = query.shape[0], train.shape[0]
        d = np.zeros((n_q, n_t))
        for i in range(n_q):
            for j in range(n_t):
                val = 1.0 - np.sum(np.sqrt(query[i]) * np.sqrt(train[j]))
                d[i, j] = np.sqrt(max(0.0, val))
        return d
    elif metric == "jensenshannon":
        n_q, n_t = query.shape[0], train.shape[0]
        d = np.zeros((n_q, n_t))
        for i in range(n_q):
            for j in range(n_t):
                a, b = query[i], train[j]
                m = 0.5 * (a + b)
                acc = 0.0
                for f in range(len(a)):
                    logM = np.log(m[f]) if m[f] > 0 else 0.0
                    logA = np.log(a[f]) if a[f] > 0 else 0.0
                    logB = np.log(b[f]) if b[f] > 0 else 0.0
                    acc += (-a[f] * (logM - logA)) + (-b[f] * (logM - logB))
                d[i, j] = np.sqrt(0.5 * acc)
        return d
    elif metric == "hamming":
        return cdist(query, train, metric="hamming")
    elif metric == "kldivergence":
        n_q, n_t = query.shape[0], train.shape[0]
        d = np.zeros((n_q, n_t))
        for i in range(n_q):
            for j in range(n_t):
                a, b = query[i], train[j]
                acc = 0.0
                for f in range(len(a)):
                    if a[f] > 0 and b[f] > 0:
                        acc += a[f] * np.log(a[f] / b[f])
                d[i, j] = acc
        return d
    elif metric == "russellrao":
        n_q, n_t = query.shape[0], train.shape[0]
        dim = query.shape[1]
        d = np.zeros((n_q, n_t))
        for i in range(n_q):
            for j in range(n_t):
                d[i, j] = (dim - np.sum(query[i] * train[j])) / dim
        return d
    else:
        raise ValueError(f"Unknown metric: {metric}")


# ============================================================================
# Data generation
# ============================================================================

np.random.seed(42)

N_QUERY, N_TRAIN, D = 4, 8, 3

# General data in [0.1, 2.0] - works for most metrics
query_gen = np.random.uniform(0.1, 2.0, (N_QUERY, D))
train_gen = np.random.uniform(0.1, 2.0, (N_TRAIN, D))

# Round to 4 decimal places for clean hardcoded values
query_gen = np.round(query_gen, 4)
train_gen = np.round(train_gen, 4)

# Probability data (rows sum to 1) for Hellinger, JS, KL
query_prob = np.abs(np.random.uniform(0.1, 1.0, (N_QUERY, D)))
query_prob = query_prob / query_prob.sum(axis=1, keepdims=True)
train_prob = np.abs(np.random.uniform(0.1, 1.0, (N_TRAIN, D)))
train_prob = train_prob / train_prob.sum(axis=1, keepdims=True)
query_prob = np.round(query_prob, 6)
train_prob = np.round(train_prob, 6)
# Renormalize after rounding
query_prob = query_prob / query_prob.sum(axis=1, keepdims=True)
train_prob = train_prob / train_prob.sum(axis=1, keepdims=True)

# Weights
weights = np.round(np.random.uniform(0.5, 3.0, N_TRAIN), 4)

# High-dimensional dataset for feature/train tiling test (deterministic formula)
N_QUERY_HD, N_TRAIN_HD, D_HD = 4, 100, 128
query_hd = np.zeros((N_QUERY_HD, D_HD))
train_hd = np.zeros((N_TRAIN_HD, D_HD))
for i in range(N_QUERY_HD):
    for j in range(D_HD):
        query_hd[i, j] = 0.1 + ((i * 1337 + j * 7 + 42) % 1000) / 1000.0 * 1.9
for i in range(N_TRAIN_HD):
    for j in range(D_HD):
        train_hd[i, j] = 0.1 + ((i * 1337 + j * 7 + 42) % 1000) / 1000.0 * 1.9

# Large dataset for multi-pass test (deterministic formula)
N_TRAIN_LARGE = 2000
query_large = np.zeros((2, D))
train_large = np.zeros((N_TRAIN_LARGE, D))
for i in range(2):
    for j in range(D):
        query_large[i, j] = (
            0.1 + ((i * 1337 + j * 7 + 42) % 1000) / 1000.0 * 1.9
        )
for i in range(N_TRAIN_LARGE):
    for j in range(D):
        train_large[i, j] = (
            0.1 + ((i * 1337 + j * 7 + 42) % 1000) / 1000.0 * 1.9
        )
weights_large = np.zeros(N_TRAIN_LARGE)
for i in range(N_TRAIN_LARGE):
    weights_large[i] = 0.5 + ((i * 31 + 17) % 1000) / 1000.0 * 2.5


# ============================================================================
# Generate golden values
# ============================================================================


def fmt_array(name, arr, type_str="double"):
    """Format array as C++ initializer."""
    vals = ", ".join(f"{v:.15e}" for v in arr)
    return f"const {type_str} {name}[] = {{{vals}}};"


def fmt_2d_array(name, arr, type_str="double"):
    """Format 2D array as flattened C++ initializer (row-major)."""
    flat = arr.flatten()
    vals = ",\n    ".join(f"{v:.15e}" for v in flat)
    return f"const {type_str} {name}[] = {{\n    {vals}}};"


results = {}


# --- 1. Each kernel with Euclidean metric ---
BW_KERNEL = 4.0  # Large enough for compact-support kernels
kernel_names = [
    "gaussian",
    "tophat",
    "epanechnikov",
    "exponential",
    "linear",
    "cosine",
]

print("// === Kernel tests (Euclidean metric, bandwidth=4.0) ===")
dists_euc = compute_dists_scipy(query_gen, train_gen, "euclidean")
for kname in kernel_names:
    if HAS_SKLEARN:
        kde = KernelDensity(
            bandwidth=BW_KERNEL, kernel=kname, metric="euclidean"
        )
        kde.fit(train_gen)
        expected = kde.score_samples(query_gen)
    else:
        expected = manual_kde(
            query_gen, train_gen, BW_KERNEL, kname, dists_euc
        )

    # Cross-validate: manual should match sklearn
    manual_expected = manual_kde(
        query_gen, train_gen, BW_KERNEL, kname, dists_euc
    )
    if not np.allclose(expected, manual_expected, atol=1e-10):
        print(f"  WARNING: sklearn vs manual mismatch for kernel={kname}")
        print(f"    sklearn: {expected}")
        print(f"    manual:  {manual_expected}")

    results[f"kernel_{kname}"] = expected
    print(f"// kernel={kname}: {expected}")


# --- 2. Each metric with Gaussian kernel ---
BW_METRIC = 1.0

# Metrics that sklearn supports via BallTree
sklearn_metric_map = {
    "L2SqrtUnexpanded": ("euclidean", "euclidean", None),
    "L1": ("manhattan", "manhattan", None),
    "Linf": ("chebyshev", "chebyshev", None),
    "LpUnexpanded": ("minkowski", "minkowski", 3.0),
}

# All metrics with scipy distance names
all_metrics = {
    "L2SqrtUnexpanded": ("euclidean", None),
    "L2Expanded": ("sqeuclidean", None),
    "L1": ("manhattan", None),
    "Linf": ("chebyshev", None),
    "LpUnexpanded": ("minkowski", 3.0),
    "CosineExpanded": ("cosine", None),
    "CorrelationExpanded": ("correlation", None),
    "Canberra": ("canberra", None),
}

# Metrics needing probability data
prob_metrics = {
    "HellingerExpanded": ("hellinger", None),
    "JensenShannon": ("jensenshannon", None),
    "KLDivergence": ("kldivergence", None),
}

# Metrics needing special handling
special_metrics = {
    "HammingUnexpanded": ("hamming", None),
    "RusselRaoExpanded": ("russellrao", None),
}

print("\n// === Metric tests (Gaussian kernel, bandwidth=1.0) ===")

# Standard metrics with general data
for metric_name, (scipy_name, metric_arg) in all_metrics.items():
    dists = compute_dists_scipy(query_gen, train_gen, scipy_name, metric_arg)
    expected = manual_kde(query_gen, train_gen, BW_METRIC, "gaussian", dists)

    # Cross-validate with sklearn where possible
    if HAS_SKLEARN and metric_name in sklearn_metric_map:
        sk_metric, _, sk_arg = sklearn_metric_map[metric_name]
        kwargs = {}
        if sk_arg is not None:
            kwargs["metric_params"] = {"p": sk_arg}
        kde = KernelDensity(
            bandwidth=BW_METRIC, kernel="gaussian", metric=sk_metric, **kwargs
        )
        kde.fit(train_gen)
        sk_expected = kde.score_samples(query_gen)
        if not np.allclose(expected, sk_expected, atol=1e-10):
            print(
                f"  WARNING: sklearn vs manual mismatch for metric={metric_name}"
            )
            print(f"    sklearn: {sk_expected}")
            print(f"    manual:  {expected}")
        expected = sk_expected  # Prefer sklearn values

    results[f"metric_{metric_name}"] = expected
    print(f"// metric={metric_name}: {expected}")

# Probability metrics
for metric_name, (scipy_name, metric_arg) in prob_metrics.items():
    dists = compute_dists_scipy(query_prob, train_prob, scipy_name, metric_arg)
    expected = manual_kde(query_prob, train_prob, BW_METRIC, "gaussian", dists)
    results[f"metric_{metric_name}"] = expected
    print(f"// metric={metric_name} (prob data): {expected}")

# Hamming and RussellRao with general data
for metric_name, (scipy_name, metric_arg) in special_metrics.items():
    dists = compute_dists_scipy(query_gen, train_gen, scipy_name, metric_arg)
    expected = manual_kde(query_gen, train_gen, BW_METRIC, "gaussian", dists)
    results[f"metric_{metric_name}"] = expected
    print(f"// metric={metric_name}: {expected}")


# --- 3. Weighted test ---
print("\n// === Weighted tests ===")
if HAS_SKLEARN:
    kde = KernelDensity(
        bandwidth=BW_METRIC, kernel="gaussian", metric="euclidean"
    )
    kde.fit(train_gen, sample_weight=weights)
    expected_weighted = kde.score_samples(query_gen)
else:
    dists = compute_dists_scipy(query_gen, train_gen, "euclidean")
    expected_weighted = manual_kde(
        query_gen, train_gen, BW_METRIC, "gaussian", dists, weights
    )

# Cross-validate
dists_euc_m = compute_dists_scipy(query_gen, train_gen, "euclidean")
manual_weighted = manual_kde(
    query_gen, train_gen, BW_METRIC, "gaussian", dists_euc_m, weights
)
if HAS_SKLEARN and not np.allclose(
    expected_weighted, manual_weighted, atol=1e-6
):
    print("  WARNING: sklearn vs manual mismatch for weighted")
    print(f"    sklearn: {expected_weighted}")
    print(f"    manual:  {manual_weighted}")

results["weighted_gaussian_euclidean"] = expected_weighted
print(f"// weighted Gaussian+Euclidean: {expected_weighted}")


# --- 4. High-dimensional tiling test ---
print(
    "\n// === High-dimensional tiling test (n_query=4, n_train=100, d=128) ==="
)
dists_hd = compute_dists_scipy(query_hd, train_hd, "euclidean")
expected_hd = manual_kde(query_hd, train_hd, BW_METRIC, "gaussian", dists_hd)
results["highd_gaussian"] = expected_hd
print(f"// high-d Gaussian+Euclidean: {expected_hd}")


# --- 5. Multi-pass test ---
print("\n// === Multi-pass tests (n_query=2, n_train=2000) ===")
BW_LARGE = 1.0
if HAS_SKLEARN:
    kde = KernelDensity(
        bandwidth=BW_LARGE, kernel="gaussian", metric="euclidean"
    )
    kde.fit(train_large)
    expected_mp = kde.score_samples(query_large)
else:
    dists_large = compute_dists_scipy(query_large, train_large, "euclidean")
    expected_mp = manual_kde(
        query_large, train_large, BW_LARGE, "gaussian", dists_large
    )
results["multipass_gaussian_euclidean"] = expected_mp
print(f"// multi-pass Gaussian+Euclidean: {expected_mp}")

# Multi-pass weighted
if HAS_SKLEARN:
    kde = KernelDensity(
        bandwidth=BW_LARGE, kernel="gaussian", metric="euclidean"
    )
    kde.fit(train_large, sample_weight=weights_large)
    expected_mp_w = kde.score_samples(query_large)
else:
    dists_large = compute_dists_scipy(query_large, train_large, "euclidean")
    expected_mp_w = manual_kde(
        query_large,
        train_large,
        BW_LARGE,
        "gaussian",
        dists_large,
        weights_large,
    )
results["multipass_weighted"] = expected_mp_w
print(f"// multi-pass weighted: {expected_mp_w}")


# ============================================================================
# Output C++ code
# ============================================================================

print(
    "\n\n// ============================================================================"
)
print("// C++ golden value arrays (copy into kde.cu test)")
print(
    "// ============================================================================\n"
)

# Input data
print(fmt_2d_array("golden_query", query_gen))
print(fmt_2d_array("golden_train", train_gen))
print(fmt_2d_array("golden_query_prob", query_prob))
print(fmt_2d_array("golden_train_prob", train_prob))
print(fmt_array("golden_weights", weights))

# Kernel test expected values
for kname in kernel_names:
    print(fmt_array(f"expected_kernel_{kname}", results[f"kernel_{kname}"]))

# Metric test expected values
for metric_name in (
    list(all_metrics.keys())
    + list(prob_metrics.keys())
    + list(special_metrics.keys())
):
    safe_name = metric_name.replace("Expanded", "").replace("Unexpanded", "")
    print(
        fmt_array(
            f"expected_metric_{safe_name}", results[f"metric_{metric_name}"]
        )
    )

# Weighted
print(fmt_array("expected_weighted", results["weighted_gaussian_euclidean"]))

# High-d
print(fmt_array("expected_highd_gaussian", results["highd_gaussian"]))

# Multi-pass large data generation formula (for C++)
print("\n// Multi-pass: generate train_large in C++ with:")
print("//   for i in [0, 2000): for j in [0, 3):")
print(
    "//     data[i*3+j] = 0.1 + ((i*1337 + j*7 + 42) % 1000) / 1000.0 * 1.9;"
)
print("//   query_large same formula with i in [0, 2)")
print("//   weights_large: 0.5 + ((i*31 + 17) % 1000) / 1000.0 * 2.5")
print(fmt_array("expected_multipass", results["multipass_gaussian_euclidean"]))
print(fmt_array("expected_multipass_weighted", results["multipass_weighted"]))

# Summary
print(f"\n// Total test cases: {len(results)}")
print(f"// sklearn available: {HAS_SKLEARN}")
