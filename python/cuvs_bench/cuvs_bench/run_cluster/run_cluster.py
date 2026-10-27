#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Run the clustering benchmark executables (``*_CLUSTER_BENCH``).

Configuration mirrors the ANN benchmark: ``config/cluster/algorithms.yaml``
maps algorithm names to executables and ``config/cluster/algos/*.yaml`` define
named parameter groups whose ``fit`` lists are expanded into all combinations.
Datasets are taken from the same ``datasets.yaml`` as the ANN benchmark.
"""

import json
import os
import subprocess
import tempfile
from typing import Dict, List, Optional

import numpy as np

from ..backends._utils import dtype_from_filename, expand_param_grid
from ..orchestrator.config_loaders import CppGBenchConfigLoader
from ..run.data_export import convert_json_to_csv_build

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "config", "cluster"
)
DATASET_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "..",
    "config",
    "datasets",
    "datasets.yaml",
)

# The data type names understood by the C++ benchmark's "dtype" key.
_CPP_DTYPES = {
    "float32": "float",
    "float16": "half",
    "uint8": "uint8",
    "int8": "int8",
}


def cluster_name(algo: str, group: str, params: Dict) -> str:
    """Benchmark case name, following the ANN index naming convention."""
    name = algo if group == "base" else f"{algo}_{group}"
    for key, val in params.items():
        name += f".{key}{val}"
    return name


def prepare_conf(
    dataset_conf: dict,
    algo: str,
    group: str,
    group_conf: dict,
    subset_size: Optional[int] = None,
) -> dict:
    """Build the JSON configuration consumed by a ``*_CLUSTER_BENCH`` binary."""
    base_file = dataset_conf["base_file"]
    dtype = np.dtype(dtype_from_filename(base_file)).name
    if dtype not in _CPP_DTYPES:
        raise ValueError(
            f"Dataset '{dataset_conf['name']}' ({base_file}) has data type "
            f"{dtype}; the clustering benchmark supports "
            f"{', '.join(_CPP_DTYPES)} (.fbin, .f16bin, .u8bin, .i8bin)"
        )
    dataset = {
        "name": dataset_conf["name"],
        "base_file": base_file,
        "dtype": _CPP_DTYPES[dtype],
    }
    subset_size = subset_size or dataset_conf.get("subset_size")
    if subset_size:
        dataset["subset_size"] = subset_size
    clustering = [
        {
            "name": cluster_name(algo, group, params),
            "algo": algo,
            "param": params,
        }
        for params in expand_param_grid(group_conf.get("fit", {}))
    ]
    return {"dataset": dataset, "clustering": clustering}


def run_cluster_benchmark(
    dataset: str,
    dataset_path: str,
    dataset_configuration: Optional[str] = None,
    algorithm_configuration: Optional[str] = None,
    algorithms: Optional[str] = None,
    groups: Optional[str] = "base",
    algo_groups: Optional[str] = None,
    subset_size: Optional[int] = None,
    executable_dir: Optional[str] = None,
    force: bool = False,
    dry_run: bool = False,
    config_path: str = CONFIG_PATH,
) -> List[str]:
    """
    Run every selected (algorithm, group) and return the paths of the CSVs.

    Results are written to ``<dataset_path>/<dataset>/result/cluster/`` as
    ``<algo>,<group>.json`` (Google Benchmark output) and ``.csv``.
    Raises ``RuntimeError`` if any benchmark executable fails.
    """
    loader = CppGBenchConfigLoader(config_path, executable_subdir="cluster")
    dataset_conf = loader.get_dataset_configuration(
        dataset,
        loader.load_yaml_file(dataset_configuration or DATASET_CONFIG_PATH),
    )
    algo_groups_found = loader._discover_algo_groups(
        dataset_conf,
        dataset,
        dataset_path,
        algorithm_configuration=algorithm_configuration,
        algorithms=algorithms,
        groups=groups,
        algo_groups=algo_groups,
        executable_dir=executable_dir,
    )

    result_dir = os.path.join(dataset_path, dataset, "result", "cluster")
    out_files, failures = [], []
    for algo, group, group_conf, meta in algo_groups_found:
        out_json = os.path.join(result_dir, f"{meta['file_name'][0]}.json")
        if os.path.exists(out_json) and not force:
            print(f"-- {out_json} exists, skipping (use --force).")
            continue
        conf = prepare_conf(dataset_conf, algo, group, group_conf, subset_size)
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False
        ) as f:
            json.dump(conf, f, indent=2)
            conf_file = f.name
        cmd = [
            meta["executable_path"],
            f"--data_prefix={dataset_path}",
            "--benchmark_counters_tabular=true",
            "--benchmark_out_format=json",
            f"--benchmark_out={out_json}",
            conf_file,
        ]
        try:
            if dry_run:
                print(
                    f"Benchmark command for {algo},{group}:\n{' '.join(cmd)}\n"
                    f"with configuration:\n{json.dumps(conf, indent=2)}"
                )
                continue
            os.makedirs(result_dir, exist_ok=True)
            subprocess.run(cmd, check=True)
            out_files.append(out_json.removesuffix(".json") + ".csv")
        except subprocess.CalledProcessError as e:
            failures.append(f"{algo},{group} (exit code {e.returncode})")
            # A partial output would be skipped as done on the next run.
            if os.path.exists(out_json):
                os.remove(out_json)
        finally:
            os.remove(conf_file)

    if out_files:
        convert_json_to_csv_build(dataset, dataset_path, method="cluster")
    if failures:
        raise RuntimeError(
            "Clustering benchmark failed for: " + ", ".join(failures)
        )
    return out_files
