#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

"""
OpenSearch benchmark backend for cuvs-bench supporting faiss and lucene
engines for approximate nearest-neighbor search.

It also supports the remote index build service (OpenSearch 3.0+),
which offloads Faiss HNSW graph construction to a GPU-accelerated external service.
https://docs.opensearch.org/latest/vector-search/remote-index-build/
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .base import BenchmarkBackend, BuildResult, Dataset, SearchResult
from ..orchestrator.config_loaders import (
    ConfigLoader,
    DatasetConfig,
    BenchmarkConfig,
    IndexConfig,
)


class OpenSearchConfigLoader(ConfigLoader):
    """
    Configuration loader for the OpenSearch backend.

    Reads opensearch-prefixed algorithm YAML files from the standard config
    directory. The shared :class:`ConfigLoader` base handles dataset loading
    and parameter expansion before calling into this loader's OpenSearch-
    specific hooks. This built-in loader is registered automatically when
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
        return "opensearch"

    def _discover_algo_groups(
        self,
        dataset_conf: dict,
        dataset: str,
        dataset_path: str,
        **kwargs,
    ):
        """Discover OpenSearch algorithm groups to benchmark."""
        algo_files = [
            algo_file
            for algo_file in self.gather_algorithm_configs(
                self.config_path, kwargs.get("algorithm_configuration")
            )
            if os.path.basename(algo_file).startswith("opensearch")
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
        Build OpenSearch benchmark configurations from expanded param combos.

        The base ConfigLoader has already expanded each group's build and
        search parameter grids before calling this hook.
        """
        host = kwargs.get("host", "localhost")
        port = kwargs.get("port", 9200)

        # Connection and remote-build kwargs forwarded verbatim to backend_config
        _conn_keys = (
            "username",
            "password",
            "use_ssl",
            "verify_certs",
            "build_batch_size",
            # Remote Index Build (OpenSearch 3.0+, faiss engine only)
            "remote_index_build",
            "remote_build_size_min",
            "remote_build_timeout",
        )
        conn_kwargs = {k: kwargs[k] for k in _conn_keys if k in kwargs}

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

                # OpenSearch index names must be lowercase with no dots
                os_index_name = index_label.replace(".", "_").lower()
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

                engine = "lucene"
                if "faiss" in algo_name:
                    engine = "faiss"

                backend_cfg: Dict[str, Any] = {
                    "name": index_label,
                    "host": host,
                    "port": port,
                    "index_name": os_index_name,
                    "engine": engine,
                    "algo": algo_name,
                    "requires_network": True,
                    **conn_kwargs,
                }

                benchmark_configs.append(
                    BenchmarkConfig(
                        indexes=[index_cfg],
                        backend_config=backend_cfg,
                    )
                )

        return benchmark_configs


# Mapping from cuvs-bench distance metric names to OpenSearch space_type
_DISTANCE_TO_SPACE_TYPE: Dict[str, str] = {
    "euclidean": "l2",
    "l2": "l2",
    "inner_product": "innerproduct",
    "innerproduct": "innerproduct",
    "cosine": "cosinesimil",
    "cosinesimil": "cosinesimil",
    "angular": "cosinesimil",
}

_DEFAULT_REMOTE_BUILD_TIMEOUT = 30 * 60
_REMOTE_BUILD_START_TIMEOUT = 30.0

_REMOTE_BUILD_MERGE_OPS = "remote_index_build_current_merge_operations"
_REMOTE_BUILD_FLUSH_OPS = "remote_index_build_current_flush_operations"
_REMOTE_BUILD_REQUEST_SUCCESS_COUNT = "build_request_success_count"
_REMOTE_BUILD_REQUEST_FAILURE_COUNT = "build_request_failure_count"
_REMOTE_BUILD_SUCCESS_COUNT = "index_build_success_count"
_REMOTE_BUILD_FAILURE_COUNT = "index_build_failure_count"

_REMOTE_BUILD_BUILD_STAT_KEYS = (
    _REMOTE_BUILD_MERGE_OPS,
    _REMOTE_BUILD_FLUSH_OPS,
)
_REMOTE_BUILD_CLIENT_STAT_KEYS = (
    _REMOTE_BUILD_REQUEST_SUCCESS_COUNT,
    _REMOTE_BUILD_REQUEST_FAILURE_COUNT,
    _REMOTE_BUILD_SUCCESS_COUNT,
    _REMOTE_BUILD_FAILURE_COUNT,
)
_REMOTE_BUILD_STAT_KEYS = (
    *_REMOTE_BUILD_CLIENT_STAT_KEYS,
    *_REMOTE_BUILD_BUILD_STAT_KEYS,
)


class OpenSearchBackend(BenchmarkBackend):
    """
    Benchmark backend for OpenSearch's k-NN plugin.

    Supports the faiss (HNSW / IVF) and lucene (HNSW) engines. Vectors are
    bulk-indexed as ``knn_vector`` fields and retrieved via the standard
    ``knn`` query type.

    Requires ``opensearch-py`` Python package.

    Parameters
    ----------
    config : Dict[str, Any]
        Backend configuration produced by :class:`OpenSearchConfigLoader`.
        Recognized keys:

        Required:
        - ``name`` – index label (e.g. ``"opensearch_faiss_hnsw.m16.ef_construction100"``)
        - ``index_name`` – OpenSearch index name (lowercase, no dots)
        - ``engine`` – ``"faiss"`` or ``"lucene"``
        - ``algo`` – algorithm name (e.g. ``"opensearch_faiss_hnsw"``)

        Optional:
        - ``host`` – hostname (default: ``"localhost"``)
        - ``port`` – port (default: ``9200``)
        - ``username`` – HTTP basic auth user (default: ``"admin"``)
        - ``password`` – HTTP basic auth password (default: ``"admin"``)
        - ``use_ssl`` – use HTTPS (default: ``False``)
        - ``verify_certs`` – verify SSL certs (default: ``False``)
        - ``build_batch_size`` – vectors per bulk request. If omitted, choose
          a batch size with roughly 1 MiB of raw vector data.
        - ``requires_network`` – trigger network pre-flight check (default: ``True``)
        - ``remote_index_build`` – set ``index.knn.remote_index_build.enabled=true``
          on the index at creation time, opting it into the GPU build path (default: ``False``).
        - ``remote_build_size_min`` – minimum segment size to trigger GPU build, e.g. ``"1kb"``
          (default: OpenSearch's default)
        - ``remote_build_timeout`` – seconds to wait for GPU build (default: ``1800``)

    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.__client = None

    @property
    def algo(self) -> str:
        return self.config.get("algo", "opensearch")

    @property
    def _client(self):
        if self.__client is None:
            try:
                from opensearchpy import OpenSearch
            except ImportError as e:
                raise ImportError(
                    "`opensearch-py` is required for the OpenSearch backend in cuvs-bench.\n\n"
                    "Install it with: `pip install opensearch-py`"
                ) from e

            host = self.config.get("host", "localhost")
            port = self.config.get("port", 9200)
            username = self.config.get("username", "admin")
            password = self.config.get("password", "admin")
            use_ssl = self.config.get("use_ssl", False)
            verify_certs = self.config.get("verify_certs", False)

            self.__client = OpenSearch(
                hosts=[{"host": host, "port": port}],
                http_auth=(username, password) if username else None,
                use_ssl=use_ssl,
                verify_certs=verify_certs,
                timeout=None,
            )
        return self.__client

    def _check_network_available(self) -> bool:
        try:
            self._client.cluster.health(request_timeout=5)
            return True
        except ImportError:
            return False
        except Exception:
            return False

    def initialize(self) -> None:
        """Eagerly open the connection to OpenSearch."""
        _ = self._client

    def cleanup(self) -> None:
        """Close the OpenSearch connection."""
        if self.__client is not None:
            self.__client.close()
            self.__client = None

    def _build_index_mapping(
        self,
        dims: int,
        engine: str,
        space_type: str,
        build_param: Dict[str, Any],
        remote_index_build: bool = False,
        remote_build_size_min: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Construct the OpenSearch index mapping dict for k-NN.

        The mapping enables the k-NN plugin and configures the vector field
        with the chosen engine and method parameters.

        When ``remote_index_build=True``, ``index.knn.remote_index_build.enabled``
        is set to ``true`` in the index settings, opting qualifying segments into
        the GPU build path. If ``remote_build_size_min`` is provided it overrides
        ``index.knn.remote_index_build.size.min``; otherwise OpenSearch's default
        applies. The cluster-level infrastructure is assumed to be pre-configured
        externally.
        """
        m = build_param.get("m", 16)
        ef_construction = build_param.get("ef_construction", 100)

        if engine == "lucene":
            method_name = "hnsw"
            method_params: Dict[str, Any] = {
                "m": m,
                "ef_construction": ef_construction,
            }
        elif engine == "faiss":
            faiss_method = build_param.get("faiss_method", "hnsw")
            method_name = faiss_method
            if faiss_method == "hnsw":
                method_params = {
                    "m": m,
                    "ef_construction": ef_construction,
                }
            elif faiss_method == "ivf":
                method_params = {"nlist": build_param.get("nlist", 4)}
            else:
                raise ValueError(
                    f"Unsupported faiss_method {faiss_method!r}. "
                    "Use 'hnsw' or 'ivf'."
                )
        else:
            raise ValueError(
                f"Unknown OpenSearch k-NN engine {engine!r}. "
                "Use 'faiss' or 'lucene'."
            )

        index_settings = {
            "knn": True,
            "number_of_shards": build_param.get("number_of_shards", 1),
            "number_of_replicas": build_param.get("number_of_replicas", 0),
        }
        if remote_index_build:
            if engine != "faiss":
                raise ValueError(
                    "Remote Index Build only supports the faiss engine "
                    f"(got engine={engine!r}). "
                    "Use algorithms='opensearch_faiss_hnsw'."
                )
            index_settings["knn.remote_index_build.enabled"] = True
            if remote_build_size_min:
                index_settings["knn.remote_index_build.size.min"] = (
                    remote_build_size_min
                )

        return {
            "settings": {"index": index_settings},
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "knn_vector",
                        "dimension": dims,
                        "method": {
                            "name": method_name,
                            "space_type": space_type,
                            "engine": engine,
                            "parameters": method_params,
                        },
                    }
                }
            },
        }

    def _bulk_index(
        self,
        index_name: str,
        vectors: np.ndarray,
        build_batch_size: Optional[int] = None,
    ) -> None:
        """
        Bulk-index vectors into index_name using the bulk API.

        Vectors are stored under the ``"vector"`` field with their integer
        row index as the document ``_id`` so they can be mapped back to
        dataset and ground-truth neighbor IDs.
        """
        if build_batch_size is None:
            vector_size_bytes = vectors[0].nbytes
            build_batch_size = max(1, (1024 * 1024) // vector_size_bytes)

        total = vectors.shape[0]
        indexed = 0
        progress_step = max(total / 10.0, 1.0)
        next_progress = progress_step

        for batch_start in range(0, total, build_batch_size):
            batch_end = min(batch_start + build_batch_size, total)
            body_lines = []

            for doc_id, vec in enumerate(
                vectors[batch_start:batch_end],
                start=batch_start,
            ):
                body_lines.append(json.dumps({"index": {"_id": str(doc_id)}}))
                body_lines.append(json.dumps({"vector": vec.tolist()}))

            response = self._client.bulk(
                index=index_name,
                body="\n".join(body_lines) + "\n",
                request_timeout=120,
                headers={"Content-Type": "application/x-ndjson"},
            )
            if response.get("errors"):
                failures = [
                    item["index"]
                    for item in response.get("items", [])
                    if item.get("index", {}).get("error")
                ]
                first_failure = failures[0] if failures else response
                raise RuntimeError(
                    "Failed to bulk-index "
                    f"{len(failures)} document(s): {first_failure}"
                )

            indexed += batch_end - batch_start
            if indexed >= next_progress or indexed == total:
                print(
                    f"  Indexed {indexed} / {total} vectors "
                    f"(~{100 * indexed // total}%)"
                )
                while next_progress <= indexed:
                    next_progress += progress_step

        print(f"  Indexed all {total} vectors")

    def _get_knn_remote_build_stats(self) -> dict:
        """
        Query the kNN stats API for cluster-wide remote build counters.

        The remote build stats live under a nested structure in the response:
          nodes.<node-id>.remote_vector_index_build_stats.build_stats.*
          nodes.<node-id>.remote_vector_index_build_stats.client_stats.*
        """
        resp = self._client.knn.stats()
        totals = {key: 0 for key in _REMOTE_BUILD_STAT_KEYS}
        for node_stats in resp.get("nodes", {}).values():
            remote = node_stats.get("remote_vector_index_build_stats", {})
            build = remote.get("build_stats", {})
            client = remote.get("client_stats", {})
            for key in _REMOTE_BUILD_BUILD_STAT_KEYS:
                totals[key] += build.get(key, 0)
            for key in _REMOTE_BUILD_CLIENT_STAT_KEYS:
                totals[key] += client.get(key, 0)
        return totals

    def _format_remote_build_stats(self, stats: Dict[str, int]) -> str:
        return ", ".join(
            f"{key}={stats.get(key, 0)}" for key in _REMOTE_BUILD_STAT_KEYS
        )

    def _remote_build_delta(
        self, stats: Dict[str, int], initial_stats: Dict[str, int], key: str
    ) -> int:
        return stats.get(key, 0) - initial_stats.get(key, 0)

    def _flush_index(self, index_name: str) -> None:
        resp = self._client.indices.flush(
            index=index_name, request_timeout=None
        )
        shards = resp.get("_shards", {})
        total = shards.get("total", 0)
        successful = shards.get("successful", 0)
        failed = shards.get("failed", 0)

        if failed or successful != total:
            raise RuntimeError(
                f"Flush did not complete on all shards for {index_name}: {resp}"
            )

    def _resolve_index_name(self, index_cfg: IndexConfig) -> str:
        return self.config.get(
            "index_name", index_cfg.name.replace(".", "_").lower()
        )

    def _failed_build_result(
        self, error_message: str, build_params: Optional[Dict[str, Any]] = None
    ) -> BuildResult:
        return BuildResult(
            index_path="",
            build_time_seconds=0.0,
            index_size_bytes=0,
            algorithm=self.algo,
            build_params=build_params or {},
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
            neighbors=np.empty((0, k), dtype=np.int64),
            distances=np.empty((0, k), dtype=np.float32),
            search_time_ms=0.0,
            queries_per_second=0.0,
            recall=0.0,
            algorithm=self.algo,
            search_params=search_params or [],
            success=False,
            error_message=error_message,
        )

    def _wait_for_remote_build(
        self,
        initial_stats: Dict[str, int],
        timeout: int = _DEFAULT_REMOTE_BUILD_TIMEOUT,
        poll_interval: int = 5,
        no_activity_timeout: float = _REMOTE_BUILD_START_TIMEOUT,
    ) -> None:
        """
        Poll the kNN stats API until all submitted remote GPU builds complete.

        *initial_stats* should be snapshotted from
        ``_get_knn_remote_build_stats()`` immediately before ingestion starts.

        Raises ``RuntimeError`` if the stats report a failed remote build or if
        no remote build activity starts before *no_activity_timeout*. Raises
        ``TimeoutError`` if submitted builds do not complete within *timeout*
        seconds.
        """
        start = time.perf_counter()
        deadline = start + timeout
        no_activity_deadline = start + min(no_activity_timeout, timeout)
        observed_activity = False
        stats: Optional[Dict[str, int]] = None

        while time.perf_counter() < deadline:
            stats = self._get_knn_remote_build_stats()
            request_failure_delta = self._remote_build_delta(
                stats, initial_stats, _REMOTE_BUILD_REQUEST_FAILURE_COUNT
            )
            build_failure_delta = self._remote_build_delta(
                stats, initial_stats, _REMOTE_BUILD_FAILURE_COUNT
            )
            if request_failure_delta > 0 or build_failure_delta > 0:
                raise RuntimeError(
                    "GPU build failed via kNN stats API: "
                    f"{self._format_remote_build_stats(stats)}"
                )

            submitted_delta = self._remote_build_delta(
                stats, initial_stats, _REMOTE_BUILD_REQUEST_SUCCESS_COUNT
            )
            completed_delta = self._remote_build_delta(
                stats, initial_stats, _REMOTE_BUILD_SUCCESS_COUNT
            )
            current_ops = (
                stats[_REMOTE_BUILD_MERGE_OPS] + stats[_REMOTE_BUILD_FLUSH_OPS]
            )
            is_idle = current_ops == 0
            if submitted_delta > 0 or completed_delta > 0 or current_ops > 0:
                observed_activity = True

            if (
                submitted_delta > 0
                and completed_delta >= submitted_delta
                and is_idle
            ):
                return

            if (
                not observed_activity
                and time.perf_counter() >= no_activity_deadline
            ):
                raise RuntimeError(
                    "No remote GPU build was observed via kNN stats API. "
                    "OpenSearch may not have scheduled one because the segment "
                    "size did not meet its default remote build threshold. "
                    "Set remote_build_size_min lower if this benchmark must "
                    "force remote builds. "
                    f"Stats: {self._format_remote_build_stats(stats)}"
                )
            time.sleep(poll_interval)

        stats_msg = (
            self._format_remote_build_stats(stats)
            if stats is not None
            else "unavailable"
        )
        raise TimeoutError(
            f"GPU build not confirmed after {timeout}s: "
            f"last stats: {stats_msg}"
        )

    def build(
        self,
        dataset: Dataset,
        indexes: List[IndexConfig],
        force: bool = False,
        dry_run: bool = False,
    ) -> BuildResult:
        """
        Build an OpenSearch k-NN index from the dataset's training vectors.

        Creates an index with k-NN plugin mapping and bulk-indexes all training
        vectors. If the index already exists and ``force=False`` the build is
        skipped.

        Build time measures ingest and flush. When ``remote_index_build=True``
        it also includes waiting for GPU build confirmation via the kNN stats
        API. The final index refresh runs after build timing is recorded.

        Parameters
        ----------
        dataset : Dataset
            Must have either non-empty ``training_vectors`` or a valid
            ``base_file`` path.
        indexes : List[IndexConfig]
            The first element provides the build parameters.
        force : bool
            Delete and recreate the index if it already exists.
        dry_run : bool
            Log what would happen without making any changes.

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
        index_name = self._resolve_index_name(index_cfg)
        engine = self.config.get("engine", "lucene")
        build_batch_size = self.config.get("build_batch_size")
        remote_index_build = bool(self.config.get("remote_index_build", False))
        remote_build_size_min = self.config.get("remote_build_size_min")

        if dry_run:
            print(
                f"[dry_run] Would build OpenSearch index '{index_name}' "
                f"(engine={engine}, remote_index_build={remote_index_build}, build_param={build_param})"
            )

            return BuildResult(
                index_path=index_name,
                build_time_seconds=0.0,
                index_size_bytes=0,
                algorithm=self.algo,
                build_params=build_param,
                success=True,
            )

        # Handle existing index
        if self._client.indices.exists(index=index_name):
            if force:
                self._client.indices.delete(index=index_name)
            else:
                return BuildResult(
                    index_path=index_name,
                    build_time_seconds=0.0,
                    index_size_bytes=0,
                    algorithm=self.algo,
                    build_params=build_param,
                    success=True,
                )

        # Dataset handles lazy vector loading from base_file when needed.
        base_vectors = dataset.training_vectors

        if base_vectors.size == 0:
            return self._failed_build_result(
                "No training vectors available. Provide dataset.training_vectors "
                "or a valid dataset.base_file path.",
                build_params=build_param,
            )

        dims = base_vectors.shape[1]
        space_type = _DISTANCE_TO_SPACE_TYPE.get(dataset.distance_metric, "l2")

        # Create index
        mapping = self._build_index_mapping(
            dims,
            engine,
            space_type,
            build_param,
            remote_index_build,
            remote_build_size_min,
        )
        self._client.indices.create(index=index_name, body=mapping)

        if remote_index_build:
            remote_timeout = int(
                self.config.get(
                    "remote_build_timeout", _DEFAULT_REMOTE_BUILD_TIMEOUT
                )
            )
            pre_ingest_stats = self._get_knn_remote_build_stats()

        # Bulk index, then flush segments before timing build completion.
        t0 = time.perf_counter()
        self._bulk_index(index_name, base_vectors, build_batch_size)
        self._flush_index(index_name)
        if remote_index_build:
            self._wait_for_remote_build(
                initial_stats=pre_ingest_stats,
                timeout=remote_timeout,
            )
        build_time = time.perf_counter() - t0

        self._client.indices.refresh(index=index_name, request_timeout=120)

        # Index size
        stats = self._client.indices.stats(index=index_name)
        index_size_bytes = int(
            stats["_all"]["total"]["store"]["size_in_bytes"]
        )

        return BuildResult(
            index_path=index_name,
            build_time_seconds=build_time,
            index_size_bytes=index_size_bytes,
            algorithm=self.algo,
            build_params=build_param,
            metadata={
                "engine": engine,
                "space_type": space_type,
                "remote_index_build": remote_index_build,
            },
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
    ) -> SearchResult:
        """
        Search the OpenSearch k-NN index for nearest neighbors.

        Iterates over every search-parameter combination defined in the index
        config, updating the index-level ``ef_search`` setting between runs.
        Metrics (QPS, latency) are collected per parameter set and stored in
        ``SearchResult.metadata["per_search_param_results"]``.

        The *neighbors* and *distances* arrays in the returned result reflect
        the **last** search-parameter combination (highest ef_search by
        convention), while *queries_per_second* is the average across all
        parameter combinations. This backend returns ``recall=0.0``; the
        shared orchestrator path computes recall from the returned neighbors
        and dataset ground truth.

        Parameters
        ----------
        dataset : Dataset
            Must have either non-empty ``query_vectors`` or a valid
            ``query_file`` path.
        indexes : List[IndexConfig]
            The first element is used; its ``search_params`` list defines the
            ef_search values to sweep.
        k : int
            Number of neighbors to retrieve per query.
        batch_size : int
            Number of query vectors to include in each OpenSearch multi-search
            request.
        mode : str
            ``"latency"`` or ``"throughput"``; informational for this backend.
        force : bool
            Unused; included for interface compatibility.
        search_threads : Optional[int]
            Unused; OpenSearch manages its own threading.
        dry_run : bool
            Log what would happen without making any OpenSearch calls.

        Returns
        -------
        SearchResult
        """
        skip = self._pre_flight_check()
        if skip:
            return self._failed_search_result(
                k, f"pre-flight check failed: {skip}"
            )

        if not indexes:
            return self._failed_search_result(k, "No indexes provided")

        index_cfg = indexes[0]
        index_name = self._resolve_index_name(index_cfg)
        engine = self.config.get("engine", "lucene")
        search_params_list = index_cfg.search_params or [{}]

        if dry_run:
            print(
                f"[dry_run] Would search OpenSearch index '{index_name}' "
                f"with {len(search_params_list)} param set(s) "
                f"(k={k}, batch_size={batch_size})"
            )

            return SearchResult(
                neighbors=np.empty((0, k), dtype=np.int64),
                distances=np.empty((0, k), dtype=np.float32),
                search_time_ms=0.0,
                queries_per_second=0.0,
                recall=0.0,
                algorithm=self.algo,
                search_params=search_params_list,
                success=True,
            )

        # Dataset handles lazy loading from query files when needed.
        query_vectors = dataset.query_vectors

        if query_vectors.size == 0:
            return self._failed_search_result(
                k,
                "No query vectors available. Provide dataset.query_vectors "
                "or a valid dataset.query_file path.",
                search_params=search_params_list,
            )

        n_queries = query_vectors.shape[0]
        n_batches = (n_queries + batch_size - 1) // batch_size

        # Run search for each search-parameter combination
        per_param_results: List[Dict[str, Any]] = []
        last_neighbors = np.full((n_queries, k), -1, dtype=np.int64)
        last_distances = np.zeros((n_queries, k), dtype=np.float32)

        for sp in search_params_list:
            ef_search = sp.get("ef_search", 100)

            if engine == "faiss":
                self._client.indices.put_settings(
                    index=index_name,
                    body={"index.knn.algo_param.ef_search": ef_search},
                )

            neighbors = np.full((n_queries, k), -1, dtype=np.int64)
            distances = np.zeros((n_queries, k), dtype=np.float32)

            t0 = time.perf_counter()
            for batch_start in range(0, n_queries, batch_size):
                batch_end = min(batch_start + batch_size, n_queries)
                body: List[Dict[str, Any]] = []
                for q_vec in query_vectors[batch_start:batch_end]:
                    body.append({})
                    body.append(
                        {
                            "size": k,
                            "query": {
                                "knn": {
                                    "vector": {
                                        "vector": q_vec.tolist(),
                                        "k": k,
                                    }
                                }
                            },
                        }
                    )

                resp = self._client.msearch(index=index_name, body=body)
                responses = resp.get("responses", [])
                if len(responses) != batch_end - batch_start:
                    raise RuntimeError(
                        "OpenSearch multi-search returned "
                        f"{len(responses)} responses for "
                        f"{batch_end - batch_start} queries"
                    )

                for batch_offset, search_resp in enumerate(responses):
                    if "error" in search_resp:
                        raise RuntimeError(
                            "OpenSearch multi-search query failed: "
                            f"{search_resp['error']}"
                        )

                    query_index = batch_start + batch_offset
                    hits = search_resp["hits"]["hits"]
                    for j, hit in enumerate(hits[:k]):
                        neighbors[query_index, j] = int(hit["_id"])
                        distances[query_index, j] = float(hit["_score"])

            elapsed = time.perf_counter() - t0
            qps = n_queries / elapsed if elapsed > 0 else 0.0

            per_param_results.append(
                {
                    "search_params": sp,
                    "search_time_ms": elapsed * 1000.0,
                    "queries_per_second": qps,
                    "batch_size": batch_size,
                    "num_batches": n_batches,
                }
            )
            last_neighbors = neighbors
            last_distances = distances

        # Aggregate across all search-param combinations
        avg_qps = float(
            np.mean([r["queries_per_second"] for r in per_param_results])
        )
        total_search_time_ms = float(
            sum(r["search_time_ms"] for r in per_param_results)
        )

        return SearchResult(
            neighbors=last_neighbors,
            distances=last_distances,
            search_time_ms=total_search_time_ms,
            queries_per_second=avg_qps,
            recall=0.0,
            algorithm=self.algo,
            search_params=search_params_list,
            metadata={
                "engine": engine,
                "batch_size": batch_size,
                "num_batches": n_batches,
                "per_search_param_results": per_param_results,
            },
            success=True,
        )
