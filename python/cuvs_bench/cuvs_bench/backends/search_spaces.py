#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

# FIXME: Search spaces are currently hardcoded in Python.
# Future: Load tune mode parameter ranges from YAML files (similar to how
# sweep mode loads discrete values from config/algos/*.yaml).

"""
Default search spaces for hyperparameter tuning.

Each algorithm defines valid ranges for its build and search parameters.
Used internally by tune mode to explore the parameter space via Optuna.
"""

from typing import Dict, Any


ALGORITHM_SEARCH_SPACES: Dict[str, Dict[str, Dict[str, Any]]] = {
    # =========================================================================
    # cuVS IVF-Flat
    # =========================================================================
    "cuvs_ivf_flat": {
        "build": {
            "nlist": {"type": "int", "min": 256, "max": 65536, "log": True},
            "ratio": {"type": "int", "min": 1, "max": 10},
            "niter": {"type": "int", "min": 10, "max": 50},
        },
        "search": {
            "nprobe": {"type": "int", "min": 1, "max": "nlist"},
        },
    },
    # =========================================================================
    # cuVS IVF-PQ
    # =========================================================================
    "cuvs_ivf_pq": {
        "build": {
            "nlist": {"type": "int", "min": 256, "max": 65536, "log": True},
            "pq_dim": {"type": "int", "min": 8, "max": 128},
            "pq_bits": {"type": "int", "min": 4, "max": 8},
            "ratio": {"type": "int", "min": 1, "max": 20},
            "niter": {"type": "int", "min": 10, "max": 50},
        },
        "search": {
            "nprobe": {"type": "int", "min": 1, "max": "nlist"},
            "internalDistanceDtype": {
                "type": "categorical",
                "choices": ["float", "half"],
            },
            "smemLutDtype": {
                "type": "categorical",
                "choices": ["float", "fp8", "half"],
            },
            "refine_ratio": {"type": "float", "min": 1.0, "max": 10.0},
        },
    },
    # =========================================================================
    # cuVS CAGRA
    # =========================================================================
    "cuvs_cagra": {
        "build": {
            "graph_degree": {"type": "int", "min": 16, "max": 256},
            "intermediate_graph_degree": {
                "type": "int",
                "min": 32,
                "max": 256,
            },
            "graph_build_algo": {
                "type": "categorical",
                "choices": ["IVF_PQ", "NN_DESCENT"],
            },
        },
        "search": {
            "itopk": {"type": "int", "min": 32, "max": 1024},
            "search_width": {"type": "int", "min": 1, "max": 64},
        },
    },
    # =========================================================================
    # cuVS Brute Force (no tunable params)
    # =========================================================================
    "cuvs_brute_force": {
        "build": {},
        "search": {},
    },
    # =========================================================================
    # HNSW (hnswlib)
    # =========================================================================
    "hnswlib": {
        "build": {
            "M": {"type": "int", "min": 4, "max": 64},
            "efConstruction": {"type": "int", "min": 32, "max": 1024},
        },
        "search": {
            "ef": {"type": "int", "min": 10, "max": 1000},
        },
    },
    # =========================================================================
    # FAISS GPU IVF-Flat
    # =========================================================================
    "faiss_gpu_ivf_flat": {
        "build": {
            "nlist": {"type": "int", "min": 256, "max": 65536, "log": True},
        },
        "search": {
            "nprobe": {"type": "int", "min": 1, "max": "nlist"},
        },
    },
    # =========================================================================
    # FAISS GPU IVF-PQ
    # =========================================================================
    "faiss_gpu_ivf_pq": {
        "build": {
            "nlist": {"type": "int", "min": 256, "max": 65536, "log": True},
            "M": {"type": "int", "min": 8, "max": 128},
            "useFloat16": {"type": "categorical", "choices": [True, False]},
            "usePrecomputed": {
                "type": "categorical",
                "choices": [True, False],
            },
        },
        "search": {
            "nprobe": {"type": "int", "min": 1, "max": "nlist"},
        },
    },
    # =========================================================================
    # FAISS CPU IVF-Flat
    # =========================================================================
    "faiss_cpu_ivf_flat": {
        "build": {
            "nlist": {"type": "int", "min": 256, "max": 65536, "log": True},
        },
        "search": {
            "nprobe": {"type": "int", "min": 1, "max": "nlist"},
        },
    },
    # =========================================================================
    # FAISS CPU IVF-PQ
    # =========================================================================
    "faiss_cpu_ivf_pq": {
        "build": {
            "nlist": {"type": "int", "min": 256, "max": 65536, "log": True},
            "M": {"type": "int", "min": 8, "max": 128},
        },
        "search": {
            "nprobe": {"type": "int", "min": 1, "max": "nlist"},
        },
    },
    # =========================================================================
    # FAISS CPU HNSW
    # =========================================================================
    "faiss_cpu_hnsw_flat": {
        "build": {
            "M": {"type": "int", "min": 4, "max": 64},
            "efConstruction": {"type": "int", "min": 32, "max": 1024},
        },
        "search": {
            "ef": {"type": "int", "min": 10, "max": 1000},
        },
    },
    # =========================================================================
    # OpenSearch HNSW (faiss engine)
    # =========================================================================
    "opensearch_faiss_hnsw": {
        "build": {
            "m": {"type": "int", "min": 4, "max": 64},
        },
        "search": {
            "ef_search": {"type": "int", "min": 10, "max": 1000},
        },
    },
}


def get_search_space(algorithm: str) -> Dict[str, Dict[str, Any]]:
    """
    Get default search space for an algorithm.

    Parameters
    ----------
    algorithm : str
        Algorithm name (e.g., "cuvs_ivf_flat")

    Returns
    -------
    dict
        Search space with "build" and "search" parameter definitions.
        Each parameter has: type, min/max (for numeric) or choices (for categorical)

    Raises
    ------
    ValueError
        If algorithm has no defined search space

    Examples
    --------
    >>> space = get_search_space("cuvs_ivf_flat")
    >>> space["build"]["nlist"]
    {'type': 'int', 'min': 256, 'max': 65536, 'log': True}
    """
    if algorithm not in ALGORITHM_SEARCH_SPACES:
        available = list(ALGORITHM_SEARCH_SPACES.keys())
        raise ValueError(
            f"No search space defined for algorithm '{algorithm}'. "
            f"Available algorithms: {available}"
        )
    return ALGORITHM_SEARCH_SPACES[algorithm]
