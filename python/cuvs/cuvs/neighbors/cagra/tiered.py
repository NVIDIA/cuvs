# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass


@dataclass(frozen=True)
class TieredGraphParams:
    """Experimental GPU/CPU graph layout and grouping settings.

    ``device_graph_budget_bytes`` covers packed graph bytes only; zero stores
    all edges on the host. It excludes vectors, queues, and build workspaces.
    ``node_per_cacheline`` is in [1, 64] with grouping and [1, 32] without it.
    ``grouping_enabled=False`` selects the rank-only layout.

    ``n_bits=0`` chooses min(24, max(4, align_up_4(ceil(log2(N))))).
    Explicit widths are in [1, 32]. ``n_groups=0`` chooses one group if IDs
    fit, otherwise ceil(N * (1 + balance_tolerance) / 2**n_bits), using the
    resolved width. At most 65,536 groups are supported; each k-means node
    has at most 16 children. ``balance_tolerance`` must be in (0, 1), and
    hard ID capacity still applies.

    ``training_rows=0`` shares a 160,000-row training budget by subtree leaf
    count, bounded by 256 to 10,000 rows per child and actual node size.
    ``assignment_batch_rows=0`` targets a 256 MiB vector buffer; distance,
    label, and host capacity-repair storage are additional.
    ``kmeans_n_iters`` must be positive. ``validate`` checks encoded edges.

    Build-time ``num_seeds=0`` disables medoid generation.
    ``seed_training_rows=0`` selects an automatic sampling limit; ``seed``
    deterministically rotates the uniform k-means training sample.
    """

    device_graph_budget_bytes: int = 0
    node_per_cacheline: int = 2
    grouping_enabled: bool = True
    n_groups: int = 0
    n_bits: int = 0
    balance_tolerance: float = 0.10
    training_rows: int = 0
    assignment_batch_rows: int = 0
    kmeans_n_iters: int = 20
    validate: bool = False
    num_seeds: int = 0
    seed_training_rows: int = 0
    seed: int = 0x9E3779B97F4A7C15


@dataclass(frozen=True)
class TieredSearchParams:
    """Experimental host queue and cross-edge synchronization settings.

    ``num_seeds`` limits stored medoid seeds used per query and cannot exceed
    the index's stored seed count; zero uses random initialization.
    ``sync_window_scale`` scales the deferred-cross-edge synchronization
    window. ``sync_drop_threshold`` is the parent-position drop that forces
    synchronization of a submitted request.

    ``num_queues`` must be positive. ``empty_pause`` counts CPU pause
    instructions after an empty poll; zero disables this pause.
    ``collect_statistics`` enables aggregate queue/poll counters.
    ``keep_pollers_running=True`` keeps workers active between calls and
    consumes CPU while the index is idle. The index retains queue state.
    """

    num_seeds: int = 0
    sync_window_scale: float = 10.0
    sync_drop_threshold: int = 51
    num_queues: int = 1
    empty_pause: int = 64
    collect_statistics: bool = False
    keep_pollers_running: bool = False
