#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""
ScaNN benchmark backend for cuvs-bench.

The index is built on the GPU and serialized to disk by the cuVS C++
``scann_build`` example, then converted into OSS-ScaNN protobuf assets and
searched on the CPU through OSS ScaNN. Concretely each phase does:

* ``build`` - shell out to ``scann_build`` (GPU build plus serialize), dump the
  base vectors as ``dataset.npy`` (the one artifact cuVS does not serialize),
  then convert the serialized files into protobuf via
  :func:`cuvs_bench.backends._scann_assets.build_scann_assets`.
* ``search`` - load the resulting directory with
  ``scann.scann_ops.py.scann_ops_pybind.load_searcher`` and query it on the CPU.

Requires the ``scann`` Python package for both the asset conversion and search.
"""

import os
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .base import BenchmarkBackend, BuildResult, Dataset, SearchResult
from ._utils import dtype_from_filename
from ..orchestrator.config_loaders import (
    ConfigLoader,
    DatasetConfig,
    BenchmarkConfig,
    IndexConfig,
)


class ScannConfigLoader(ConfigLoader):
    """
    Configuration loader for the ScaNN backend.

    Reads scann-prefixed algorithm YAML files from the standard config
    directory. The shared :class:`ConfigLoader` base handles dataset loading
    and parameter expansion before calling into this loader's ScaNN-specific
    hooks. This built-in loader is registered automatically when
    :mod:`cuvs_bench.orchestrator` is imported.
    """

    def __init__(self, config_path: Optional[Union[str, os.PathLike]] = None):
        """
        Initialize the config loader.

        Parameters
        ----------
        config_path : Optional[Union[str, os.PathLike]]
            Path to the config directory. If None, uses the default path
            bundled with cuvs-bench.
        """
        self.config_path = (
            os.fspath(config_path)
            if config_path is not None
            else os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "../config"
            )
        )

    @property
    def backend_type(self) -> str:
        return "scann"

    def _discover_algo_groups(
        self,
        dataset_conf: dict,
        dataset: str,
        dataset_path: str,
        **kwargs,
    ):
        """Discover ScaNN algorithm groups to benchmark."""
        algo_files = [
            algo_file
            for algo_file in self.gather_algorithm_configs(
                self.config_path, kwargs.get("algorithm_configuration")
            )
            if os.path.basename(algo_file).startswith("scann")
        ]

        algorithms = kwargs.get("algorithms")
        allowed_algos = (
            [a.strip() for a in algorithms.split(",")] if algorithms else None
        )
        allowed_groups = (
            [g.strip() for g in kwargs["groups"].split(",")]
            if kwargs.get("groups")
            else None
        )

        result = []

        for algo_file in algo_files:
            algo_yaml = self.load_yaml_file(algo_file)
            algo_name = algo_yaml.get("name", "")
            if allowed_algos and algo_name not in allowed_algos:
                continue

            groups: Dict[str, Any] = algo_yaml.get("groups", {})
            if allowed_groups:
                groups = {
                    k: v for k, v in groups.items() if k in allowed_groups
                }

            for group_name, group_conf in groups.items():
                result.append((algo_name, group_name, group_conf, {}))

        return result

    def _build_benchmark_configs(
        self,
        dataset_config: DatasetConfig,
        dataset_conf: dict,
        dataset: str,
        dataset_path: str,
        expanded_groups: List[Tuple[str, str, dict, List, List, dict]],
        **kwargs,
    ) -> List[BenchmarkConfig]:
        """
        Build ScaNN benchmark configurations from expanded param combos.

        The base ConfigLoader has already expanded each group's build and
        search parameter grids before calling this hook.
        """
        # Path to the cuVS ``scann_build`` example, forwarded to the backend.
        scann_build_path = kwargs.get("scann_build_path")

        tune_mode = kwargs.get("_tune_mode", False)
        tune_build_params = kwargs.get("_tune_build_params")
        tune_search_params = kwargs.get("_tune_search_params")

        benchmark_configs: List[BenchmarkConfig] = []

        for (
            algo_name,
            group_name,
            _group_conf,
            build_combos,
            search_combos,
            _group_meta,
        ) in expanded_groups:
            if tune_mode and tune_build_params is not None:
                actual_build = [tune_build_params]
                actual_search = (
                    [tune_search_params] if tune_search_params else [{}]
                )
            else:
                actual_build = build_combos
                actual_search = search_combos

            for build_param in actual_build:
                prefix = (
                    algo_name
                    if group_name == "base"
                    else f"{algo_name}_{group_name}"
                )
                parts = [prefix] + [f"{k}{v}" for k, v in build_param.items()]
                index_label = ".".join(parts)

                index_file = os.path.join(
                    dataset_path, dataset, "index", index_label
                )

                index_cfg = IndexConfig(
                    name=index_label,
                    algo=algo_name,
                    build_param=build_param,
                    search_params=actual_search,
                    file=index_file,
                )

                backend_cfg: Dict[str, Any] = {
                    "name": index_label,
                    "group": group_name,
                    "algo": algo_name,
                    "requires_gpu": True,
                }
                if scann_build_path is not None:
                    backend_cfg["scann_build_path"] = scann_build_path

                benchmark_configs.append(
                    BenchmarkConfig(
                        indexes=[index_cfg],
                        backend_config=backend_cfg,
                    )
                )

        return benchmark_configs


# Map the dtype inferred from the dataset file extension to the ``--dtype``
# value understood by the cuVS ``scann_build`` example.
_DTYPE_TO_SCANN_BUILD_FLAG: Dict[Any, str] = {
    np.dtype(np.float32): "float",
    np.dtype(np.float16): "half",
    np.dtype(np.int8): "int8",
    np.dtype(np.uint8): "uint8",
}

# Defaults mirror the cuVS ``scann_build`` example so a sparse build YAML still
# produces a consistent index and a matching ScaNN protobuf config.
_BUILD_DEFAULTS: Dict[str, Any] = {
    "n_leaves": 1000,
    "pq_dim": 2,
    "pq_bits": 4,
    "partitioning_eta": 2.0,
    "soar_lambda": 1.5,
    "n_coarse_clusters": 0,
    "kmeans_n_rows_train": 10000,
    "pq_n_rows_train": 10000,
    "bf16": False,
}

# Conversion-only parameters consumed by build_scann_assets (not by the C++
# build). They feed scann_config.pb and have no effect on the GPU build itself.
_CONVERT_DEFAULTS: Dict[str, Any] = {
    "pq_threshold": 0.2,
    "bf16_threshold": 0.2,
    "n_coarse_clusters_to_search": 1,
}


class ScannBackend(BenchmarkBackend):
    """
    Benchmark backend for OSS ScaNN with a cuVS GPU build.

    The GPU build and serialization are delegated to the cuVS ``scann_build``
    example; the serialized artifacts are converted to OSS-ScaNN protobuf and
    searched on the CPU via ``scann.scann_ops.py.scann_ops_pybind``.

    Requires the ``scann`` Python package.

    Parameters
    ----------
    config : Dict[str, Any]
        Backend configuration produced by :class:`ScannConfigLoader`.
        Recognized keys:

        Required:
        - ``name`` - index label (e.g. ``"scann.n_leaves1000.pq_dim16"``)
        - ``group`` - algorithm configuration group selected from YAML
        - ``algo`` - algorithm name (e.g. ``"scann"``)

        Optional:
        - ``scann_build_path`` - path to the cuVS ``scann_build`` example. If
          omitted, ``scann_build`` is looked up on ``PATH``.
        - ``requires_gpu`` - trigger GPU pre-flight check (default: ``True``)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.__searcher = None
        self.__searcher_dir = None

    @property
    def algo(self) -> str:
        return self.config.get("algo", "scann")

    def _resolve_scann_build(self) -> str:
        """Locate the cuVS ``scann_build`` example binary."""
        path = self.config.get("scann_build_path")
        if path:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"scann_build_path does not exist: {path}"
                )
            return path

        from shutil import which

        found = which("scann_build")
        if found is None:
            raise FileNotFoundError(
                "Could not locate the cuVS `scann_build` example. Pass its "
                "path via the 'scann_build_path' backend-config key or put it "
                "on PATH."
            )
        return found

    def _dtype_flag(self, base_file: str) -> str:
        """Infer the ``scann_build`` ``--dtype`` flag from the dataset file."""
        dtype = np.dtype(dtype_from_filename(base_file))
        if dtype not in _DTYPE_TO_SCANN_BUILD_FLAG:
            raise ValueError(
                f"Unsupported dataset dtype {dtype} for ScaNN build "
                f"(from {base_file})."
            )
        return _DTYPE_TO_SCANN_BUILD_FLAG[dtype]

    def _searcher(self, scann_dir: str):
        """Lazily load (and cache) the OSS-ScaNN searcher for scann_dir."""
        if self.__searcher is not None and self.__searcher_dir == scann_dir:
            return self.__searcher
        try:
            from scann.scann_ops.py.scann_ops_pybind import load_searcher
        except ImportError as e:
            raise ImportError(
                "`scann` is required for the ScaNN backend in cuvs-bench.\n\n"
                "Install it with: `pip install scann`"
            ) from e
        self.__searcher = load_searcher(scann_dir)
        self.__searcher_dir = scann_dir
        return self.__searcher

    def cleanup(self) -> None:
        """Drop the cached searcher."""
        self.__searcher = None
        self.__searcher_dir = None

    def _scann_params(self, build_param: Dict[str, Any]):
        """Build a ScannParams object for the protobuf conversion."""
        from ._scann_assets import ScannParams

        def _get(key):
            if key in _BUILD_DEFAULTS:
                return build_param.get(key, _BUILD_DEFAULTS[key])
            return build_param.get(key, _CONVERT_DEFAULTS[key])

        return ScannParams(
            num_leaves=_get("n_leaves"),
            num_leaves_to_search=_get("n_leaves"),
            partitioning_eta=_get("partitioning_eta"),
            soar_lambda=_get("soar_lambda"),
            partitioning_training_size=_get("kmeans_n_rows_train"),
            pq_dim=_get("pq_dim"),
            pq_training_size=_get("pq_n_rows_train"),
            pq_threshold=_get("pq_threshold"),
            bf16=bool(_get("bf16")),
            bf16_threshold=_get("bf16_threshold"),
            n_coarse_clusters=_get("n_coarse_clusters"),
            n_coarse_clusters_to_search=_get("n_coarse_clusters_to_search"),
        )

    def _scann_build_command(
        self, scann_build: str, base_file: str, scann_dir: str, build_param: dict
    ) -> List[str]:
        """Assemble the ``scann_build`` command line for one index."""

        def _get(key):
            return build_param.get(key, _BUILD_DEFAULTS[key])

        return [
            scann_build,
            "--dataset",
            base_file,
            "--output",
            scann_dir,
            "--dtype",
            self._dtype_flag(base_file),
            "--n_leaves",
            str(_get("n_leaves")),
            "--pq_dim",
            str(_get("pq_dim")),
            "--pq_bits",
            str(_get("pq_bits")),
            "--partitioning_eta",
            str(_get("partitioning_eta")),
            "--soar_lambda",
            str(_get("soar_lambda")),
            "--n_coarse_clusters",
            str(_get("n_coarse_clusters")),
            "--kmeans_n_rows_train",
            str(_get("kmeans_n_rows_train")),
            "--pq_n_rows_train",
            str(_get("pq_n_rows_train")),
            "--bf16",
            "1" if _get("bf16") else "0",
        ]

    def _failed_build_result(
        self, error_message: str, build_params: Optional[Dict[str, Any]] = None
    ) -> BuildResult:
        return BuildResult(
            index_path="",
            build_time_seconds=0.0,
            index_size_bytes=0,
            algorithm=self.algo,
            build_params=build_params or {},
            metadata={"group": self.config["group"]},
            success=False,
            error_message=error_message,
        )

    def _failed_search_result(
        self,
        k: int,
        error_message: str,
        search_params: Optional[List[Dict[str, Any]]] = None,
    ) -> SearchResult:
        return SearchResult(
            neighbors=np.zeros((0, k), dtype=np.int64),
            distances=np.zeros((0, k), dtype=np.float32),
            search_time_ms=0.0,
            queries_per_second=0.0,
            recall=0.0,
            algorithm=self.algo,
            search_params=search_params or [],
            metadata={"group": self.config["group"]},
            success=False,
            error_message=error_message,
        )

    def build(
        self,
        dataset: Dataset,
        indexes: List[IndexConfig],
        force: bool = False,
        dry_run: bool = False,
    ) -> BuildResult:
        """
        Build a ScaNN index: GPU build plus serialize, then protobuf conversion.

        Runs the cuVS ``scann_build`` example to build on the GPU and serialize
        to ``index_cfg.file``, writes the base vectors as ``dataset.npy`` (the
        one asset cuVS does not serialize), then converts the serialized files
        into the OSS-ScaNN protobuf assets. Build time covers all three steps.

        Parameters
        ----------
        dataset : Dataset
            Must have a valid ``base_file`` path (or non-empty
            ``training_vectors``).
        indexes : List[IndexConfig]
            The first element provides the build parameters and index path.
        force : bool
            Rebuild even if the index directory already exists.
        dry_run : bool
            Log what would happen without building anything.

        Returns
        -------
        BuildResult
        """
        skip = self._pre_flight_check()
        if skip:
            return self._failed_build_result(
                f"pre-flight check failed: {skip}"
            )

        if not indexes:
            return self._failed_build_result("No indexes provided")

        index_cfg = indexes[0]
        build_param = index_cfg.build_param
        scann_dir = index_cfg.file

        if dry_run:
            scann_build = self.config.get("scann_build_path", "scann_build")
            cmd = self._scann_build_command(
                scann_build, dataset.base_file or "", scann_dir, build_param
            )
            print(
                f"[dry_run] Would build ScaNN index '{index_cfg.name}':\n"
                f"  {' '.join(cmd)}\n"
                f"  then dump dataset.npy and convert protobuf assets."
            )
            return BuildResult(
                index_path=scann_dir,
                build_time_seconds=0.0,
                index_size_bytes=0,
                algorithm=self.algo,
                build_params=build_param,
                metadata={"group": self.config["group"]},
                success=True,
            )

        if os.path.exists(os.path.join(scann_dir, "scann_assets.pbtxt")):
            if not force:
                return BuildResult(
                    index_path=scann_dir,
                    build_time_seconds=0.0,
                    index_size_bytes=0,
                    algorithm=self.algo,
                    build_params=build_param,
                    metadata={"group": self.config["group"], "skipped": True},
                    success=True,
                )

        if not dataset.base_file:
            return self._failed_build_result(
                "No base_file available for ScaNN build.",
                build_params=build_param,
            )

        try:
            scann_build = self._resolve_scann_build()
        except FileNotFoundError as e:
            return self._failed_build_result(str(e), build_params=build_param)

        os.makedirs(scann_dir, exist_ok=True)
        cmd = self._scann_build_command(
            scann_build, dataset.base_file, scann_dir, build_param
        )

        t0 = time.perf_counter()
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            return self._failed_build_result(
                f"scann_build failed (exit {e.returncode})",
                build_params=build_param,
            )

        # cuVS does not serialize the raw dataset; OSS ScaNN needs it as
        # dataset.npy in the artifact directory.
        np.save(
            os.path.join(scann_dir, "dataset.npy"), dataset.training_vectors
        )

        # Convert the serialized cuVS artifacts into OSS-ScaNN protobuf assets.
        from ._scann_assets import build_scann_assets

        build_scann_assets(scann_dir, self._scann_params(build_param))
        build_time = time.perf_counter() - t0

        index_size_bytes = sum(
            os.path.getsize(os.path.join(scann_dir, f))
            for f in os.listdir(scann_dir)
            if os.path.isfile(os.path.join(scann_dir, f))
        )

        return BuildResult(
            index_path=scann_dir,
            build_time_seconds=build_time,
            index_size_bytes=index_size_bytes,
            algorithm=self.algo,
            build_params=build_param,
            metadata={"group": self.config["group"]},
            success=True,
        )

    def search(
        self,
        dataset: Dataset,
        indexes: List[IndexConfig],
        k: int,
        batch_size: int = 10000,
        mode: str = "latency",
        force: bool = False,
        search_threads: Optional[int] = None,
        dry_run: bool = False,
    ) -> List[SearchResult]:
        """
        Search the ScaNN index on the CPU via OSS ScaNN.

        Loads the converted artifact directory with ``load_searcher`` and runs
        one batched search per search-parameter combination, returning one
        result each so the orchestrator computes recall independently.

        Parameters
        ----------
        dataset : Dataset
            Must have a valid ``query_file`` path (or non-empty
            ``query_vectors``).
        indexes : List[IndexConfig]
            The first element is used; its ``search_params`` list defines the
            ``leaves_to_search`` / ``reorder`` values to sweep.
        k : int
            Number of neighbors to retrieve per query.
        batch_size : int
            Unused; OSS ScaNN batches internally.
        mode : str
            ``"latency"`` or ``"throughput"``; informational for this backend.
        force : bool
            Unused; included for interface compatibility.
        search_threads : Optional[int]
            Unused; OSS ScaNN manages its own threading.
        dry_run : bool
            Log what would happen without searching.

        Returns
        -------
        List[SearchResult]
        """
        skip = self._pre_flight_check()
        if skip:
            return [
                self._failed_search_result(
                    k, f"pre-flight check failed: {skip}"
                )
            ]

        if not indexes:
            return [self._failed_search_result(k, "No indexes provided")]

        index_cfg = indexes[0]
        scann_dir = index_cfg.file
        search_params_list = index_cfg.search_params or [{}]

        if dry_run:
            print(
                f"[dry_run] Would search ScaNN index '{index_cfg.name}' "
                f"with {len(search_params_list)} param set(s) (k={k})"
            )
            return [
                SearchResult(
                    neighbors=np.zeros((0, k), dtype=np.int64),
                    distances=np.zeros((0, k), dtype=np.float32),
                    search_time_ms=0.0,
                    queries_per_second=0.0,
                    recall=0.0,
                    algorithm=self.algo,
                    search_params=[sp],
                    metadata={"group": self.config["group"], "dry_run": True},
                    success=True,
                )
                for sp in search_params_list
            ]

        query_vectors = dataset.query_vectors
        if query_vectors.size == 0:
            return [
                self._failed_search_result(
                    k,
                    "No query vectors available. Provide "
                    "dataset.query_vectors or a valid dataset.query_file path.",
                    search_params=search_params_list,
                )
            ]

        try:
            searcher = self._searcher(scann_dir)
        except (ImportError, FileNotFoundError, RuntimeError) as e:
            return [
                self._failed_search_result(
                    k, str(e), search_params=search_params_list
                )
            ]

        n_queries = query_vectors.shape[0]
        results: List[SearchResult] = []

        for sp in search_params_list:
            leaves_to_search = sp.get("leaves_to_search")
            reorder = sp.get("reorder")

            search_kwargs: Dict[str, Any] = {"final_num_neighbors": k}
            if leaves_to_search is not None:
                search_kwargs["leaves_to_search"] = leaves_to_search
            if reorder is not None:
                search_kwargs["pre_reorder_num_neighbors"] = reorder

            t0 = time.perf_counter()
            neighbors, distances = searcher.search_batched(
                query_vectors, **search_kwargs
            )
            elapsed = time.perf_counter() - t0
            qps = n_queries / elapsed if elapsed > 0 else 0.0

            results.append(
                SearchResult(
                    neighbors=np.asarray(neighbors, dtype=np.int64),
                    distances=np.asarray(distances, dtype=np.float32),
                    search_time_ms=elapsed * 1000.0,
                    queries_per_second=qps,
                    recall=0.0,
                    algorithm=self.algo,
                    search_params=[sp],
                    metadata={
                        "group": self.config["group"],
                        "index_name": index_cfg.name,
                    },
                    success=True,
                )
            )

        return results
