#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
"""
Elasticsearch GPU HTTP backend for cuvs-bench.

Install with: pip install cuvs-bench[elastic]

Build params (index_options): type, m, ef_construction.
  type: hnsw, int8_hnsw, int4_hnsw, bbq_hnsw (per ES-GPU-API-REFERENCE.md)
  similarity: l2_norm, cosine, max_inner_product (overrides dataset distance)
Index settings: number_of_shards, number_of_replicas, vector_field.
  GPU indexing is configured at the node level (vectors.indexing.use_gpu in elasticsearch.yml).
Search params (knn): num_candidates, vector_field.
"""

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BenchmarkBackend, BuildResult, Dataset, SearchResult
from .registry import register_backend, register_config_loader
from ..orchestrator.config_loaders import (
    BenchmarkConfig,
    ConfigLoader,
    DatasetConfig,
    IndexConfig,
)

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch


def _distance_to_similarity(distance: str) -> str:
    """Map cuvs-bench distance metric to ES dense_vector similarity."""
    m = {
        "euclidean": "l2_norm",
        "inner_product": "max_inner_product",
        "cosine": "cosine",
    }
    return m.get(distance, "l2_norm")


def _supported_elastic_algorithms_message() -> str:
    """Return a readable list of supported elastic algorithm names."""
    return ", ".join(f"'{algo}'" for algo in _SUPPORTED_ALGOS)


def _validate_elastic_index_type(index_type: str) -> str:
    """Validate an Elasticsearch dense_vector index_options type."""
    if index_type not in _SUPPORTED_INDEX_TYPES:
        raise ValueError(
            "Received params for unsupported Elasticsearch index type "
            f"{index_type!r}. Supported index types are: "
            f"{', '.join(_SUPPORTED_INDEX_TYPES)}. "
            "Please check the benchmark configuration."
        )
    return index_type


def _validate_elastic_similarity(similarity: str) -> str:
    """Validate an Elasticsearch dense_vector similarity value."""
    if similarity not in _SUPPORTED_SIMILARITIES:
        raise ValueError(
            "Received unsupported Elasticsearch similarity "
            f"{similarity!r}. Supported similarities are: "
            f"{', '.join(_SUPPORTED_SIMILARITIES)}. "
            "Please check the benchmark configuration or dataset distance metric."
        )
    return similarity


def _validate_elastic_algorithm(algo_name: str) -> str:
    """Validate a cuvs-bench elastic algorithm name."""
    if algo_name not in _SUPPORTED_ALGOS:
        raise ValueError(
            "Received unsupported algorithm "
            f"{algo_name!r} for the Elasticsearch backend. "
            "Supported algorithms are: "
            f"{_supported_elastic_algorithms_message()}."
        )
    return algo_name


# Defaults for index creation when not specified in config
_DEFAULT_INDEX_TYPE = "hnsw"
_DEFAULT_M = 16
_DEFAULT_EF_CONSTRUCTION = 100
_DEFAULT_NUM_SHARDS = 1
_DEFAULT_NUM_REPLICAS = 0

_DEFAULT_VECTOR_FIELD = "embedding"
_DEFAULT_NUM_CANDIDATES = 100

_SUPPORTED_INDEX_TYPES = ("hnsw", "int8_hnsw", "int4_hnsw", "bbq_hnsw")
_SUPPORTED_SIMILARITIES = ("l2_norm", "cosine", "max_inner_product")
_SUPPORTED_ALGOS = tuple(
    f"elastic_{index_type}" for index_type in _SUPPORTED_INDEX_TYPES
)

_BUILD_PARAM_KEYS = (
    "type",
    "m",
    "ef_construction",
    "similarity",
    "number_of_shards",
    "number_of_replicas",
    "vector_field",
)
_SEARCH_PARAM_KEYS = ("num_candidates", "vector_field")


class ElasticBackend(BenchmarkBackend):
    """Elasticsearch GPU backend for vector benchmarking."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._client: Optional["Elasticsearch"] = None

    @property
    def algo(self) -> str:
        """Algorithm name from config (e.g. elastic_hnsw, elastic_int8_hnsw)."""
        index_type = self.config.get("type", _DEFAULT_INDEX_TYPE)
        return f"elastic_{index_type}"

    def _get_client(self) -> "Elasticsearch":
        if self._client is None:
            try:
                from elasticsearch import Elasticsearch
            except ImportError as e:
                raise ImportError(
                    "`elasticsearch` is required for the Elasticsearch backend. "
                    "Install with: pip install cuvs-bench[elastic]"
                ) from e
            host = self.config.get("host", "localhost")
            port = self.config.get("port", 9200)
            scheme = self.config.get("scheme", "http")
            kwargs: Dict[str, Any] = {
                "hosts": [f"{scheme}://{host}:{port}"],
                "request_timeout": 300,  # 5 min for slow bulk ops / remote ES
            }
            basic_auth = self.config.get("basic_auth")
            if basic_auth is not None:
                if (
                    isinstance(basic_auth, (list, tuple))
                    and len(basic_auth) >= 2
                ):
                    kwargs["basic_auth"] = (
                        str(basic_auth[0]),
                        str(basic_auth[1]),
                    )
                elif isinstance(basic_auth, str) and ":" in basic_auth:
                    user, _, passwd = basic_auth.partition(":")
                    kwargs["basic_auth"] = (user, passwd)
            self._client = Elasticsearch(**kwargs)
        return self._client

    def cleanup(self) -> None:
        """Close Elasticsearch client connection."""
        if self._client is not None:
            self._client.close()
            self._client = None

    @property
    def requires_network(self) -> bool:
        """Elasticsearch backend requires network connectivity."""
        return True

    def _check_network_available(self) -> bool:
        """Verify Elasticsearch is reachable."""
        try:
            return self._get_client().ping()
        except Exception:
            return False

    def build(
        self,
        dataset: Dataset,
        indexes: List[IndexConfig],
        force: bool = False,
        dry_run: bool = False,
    ) -> BuildResult:
        """Build ES index from dataset vectors (fbin)."""
        if dry_run:
            return BuildResult(
                index_path="",
                build_time_seconds=0,
                index_size_bytes=0,
                algorithm=self.algo,
                build_params={},
                success=True,
            )

        skip_reason = self._pre_flight_check()
        if skip_reason:
            return BuildResult(
                index_path="",
                build_time_seconds=0.0,
                index_size_bytes=0,
                algorithm=self.algo,
                build_params={},
                success=False,
                error_message=f"pre-flight check failed: {skip_reason}",
            )

        index_name = self.config.get("index_name", "cuvs_bench_vectors")
        idx = indexes[0] if indexes else None
        build_params = dict(idx.build_param or {}) if idx else {}

        try:
            client = self._get_client()
            if client.indices.exists(index=index_name):
                if not force:
                    stats = client.indices.stats(index=index_name)
                    index_size = stats["_all"]["primaries"]["store"][
                        "size_in_bytes"
                    ]
                    return BuildResult(
                        index_path=index_name,
                        build_time_seconds=0,
                        index_size_bytes=index_size,
                        algorithm=self.algo,
                        build_params=build_params,
                        success=True,
                    )
                client.indices.delete(index=index_name)
        except Exception as e:
            return BuildResult(
                index_path="",
                build_time_seconds=0,
                index_size_bytes=0,
                algorithm=self.algo,
                build_params={},
                success=False,
                error_message=str(e),
            )

        vectors = dataset.training_vectors
        if vectors.size == 0:
            return BuildResult(
                index_path="",
                build_time_seconds=0.0,
                index_size_bytes=0,
                algorithm=self.algo,
                build_params={},
                success=False,
                error_message=(
                    "training_vectors are required for Elasticsearch backend "
                    "(directly or via dataset.base_file)"
                ),
            )

        # similarity: from config, or derive from dataset distance
        similarity = build_params.get("similarity") or _distance_to_similarity(
            getattr(dataset, "distance_metric", None) or "euclidean"
        )

        index_type = build_params.get("type", _DEFAULT_INDEX_TYPE)
        try:
            _validate_elastic_index_type(index_type)
            _validate_elastic_similarity(similarity)
        except ValueError as e:
            return BuildResult(
                index_path="",
                build_time_seconds=0.0,
                index_size_bytes=0,
                algorithm=self.algo,
                build_params=build_params,
                success=False,
                error_message=str(e),
            )

        try:
            n_vectors = len(vectors)
            dims = vectors.shape[1]
            client = self._get_client()

            vector_field = build_params.get(
                "vector_field", _DEFAULT_VECTOR_FIELD
            )
            m = build_params.get("m", _DEFAULT_M)
            ef_construction = build_params.get(
                "ef_construction", _DEFAULT_EF_CONSTRUCTION
            )
            num_shards = build_params.get(
                "number_of_shards", _DEFAULT_NUM_SHARDS
            )
            num_replicas = build_params.get(
                "number_of_replicas", _DEFAULT_NUM_REPLICAS
            )
            settings: Dict[str, Any] = {
                "number_of_shards": num_shards,
                "number_of_replicas": num_replicas,
            }
            # Note: GPU indexing is controlled at the node level via
            # vectors.indexing.use_gpu in elasticsearch.yml, not per-index.

            index_options: Dict[str, Any] = {
                "type": index_type,
                "m": m,
                "ef_construction": ef_construction,
            }

            index_config: Dict[str, Any] = {
                "settings": settings,
                "mappings": {
                    "properties": {
                        vector_field: {
                            "type": "dense_vector",
                            "dims": dims,
                            "index": True,
                            "similarity": similarity,
                            "index_options": index_options,
                        },
                    },
                },
            }

            t0 = time.perf_counter()
            client.indices.create(index=index_name, body=index_config)

            from elasticsearch.helpers import bulk

            chunk_size = 1000
            progress_interval = max(
                50, n_vectors // (chunk_size * 20)
            )  # ~20 progress lines
            for i in range(0, n_vectors, chunk_size):
                chunk = vectors[i : i + chunk_size]
                actions = [
                    {
                        "_index": index_name,
                        "_id": str(i + j),
                        vector_field: vec.tolist(),
                    }
                    for j, vec in enumerate(chunk)
                ]
                bulk(client, actions, raise_on_error=True)
                if (
                    progress_interval
                    and (i // chunk_size) % progress_interval == 0
                ):
                    print(
                        f"    Indexed {min(i + chunk_size, n_vectors):,}/{n_vectors:,} vectors"
                    )

            client.indices.refresh(index=index_name)
            build_time = time.perf_counter() - t0

            stats = client.indices.stats(index=index_name)
            index_size = stats["_all"]["primaries"]["store"]["size_in_bytes"]

            return BuildResult(
                index_path=index_name,
                build_time_seconds=build_time,
                index_size_bytes=index_size,
                algorithm=self.algo,
                build_params=build_params,
                success=True,
            )
        except Exception as e:
            return BuildResult(
                index_path="",
                build_time_seconds=0,
                index_size_bytes=0,
                algorithm=self.algo,
                build_params={},
                success=False,
                error_message=str(e),
            )

    def search(
        self,
        dataset: Dataset,
        indexes: List[IndexConfig],
        k: int = 10,
        batch_size: int = 10000,
        mode: str = "latency",
        force: bool = False,
        search_threads: Optional[int] = None,
        dry_run: bool = False,
    ) -> SearchResult:
        """Run kNN search over all search-param combinations and compute recall."""
        if dry_run:
            return SearchResult(
                neighbors=np.empty((0, k), dtype=np.int64),
                distances=np.empty((0, k), dtype=np.float32),
                search_time_ms=0,
                queries_per_second=0,
                recall=0,
                algorithm=self.algo,
                search_params=[],
                success=True,
            )

        skip_reason = self._pre_flight_check()
        if skip_reason:
            return SearchResult(
                neighbors=np.empty((0, k), dtype=np.int64),
                distances=np.empty((0, k), dtype=np.float32),
                search_time_ms=0,
                queries_per_second=0,
                recall=0,
                algorithm=self.algo,
                search_params=[],
                success=False,
                error_message=f"pre-flight check failed: {skip_reason}",
            )

        if not indexes:
            return SearchResult(
                neighbors=np.empty((0, k), dtype=np.int64),
                distances=np.empty((0, k), dtype=np.float32),
                search_time_ms=0,
                queries_per_second=0,
                recall=0,
                algorithm=self.algo,
                search_params=[],
                success=False,
                error_message="No indexes provided",
            )

        query_vectors = dataset.query_vectors
        if query_vectors.size == 0:
            return SearchResult(
                neighbors=np.empty((0, k), dtype=np.int64),
                distances=np.empty((0, k), dtype=np.float32),
                search_time_ms=0,
                queries_per_second=0,
                recall=0,
                algorithm=self.algo,
                search_params=[],
                success=False,
                error_message=(
                    "query_vectors are required for Elasticsearch backend "
                    "(directly or via dataset.query_file)"
                ),
            )

        try:
            n_queries = len(query_vectors)

            groundtruth = dataset.groundtruth_neighbors

            index_name = self.config.get("index_name", "cuvs_bench_vectors")
            index_cfg = indexes[0]
            search_params_list = index_cfg.search_params or [{}]

            per_param_results: List[Dict[str, Any]] = []
            last_neighbors = np.full((n_queries, k), -1, dtype=np.int64)
            last_distances = np.zeros((n_queries, k), dtype=np.float32)

            for sp in search_params_list:
                num_candidates = sp.get(
                    "num_candidates", _DEFAULT_NUM_CANDIDATES
                )
                vector_field = sp.get("vector_field", _DEFAULT_VECTOR_FIELD)

                neighbors = np.full((n_queries, k), -1, dtype=np.int64)
                distances = np.zeros((n_queries, k), dtype=np.float32)
                latencies: List[float] = []

                t0 = time.perf_counter()
                for i, qv in enumerate(query_vectors):
                    body = {
                        "knn": {
                            "field": vector_field,
                            "query_vector": qv.tolist(),
                            "k": k,
                            "num_candidates": num_candidates,
                        }
                    }
                    t_q = time.perf_counter()
                    resp = self._get_client().search(
                        index=index_name, body=body, size=k
                    )
                    latencies.append((time.perf_counter() - t_q) * 1000)
                    hits = resp.get("hits", {}).get("hits", [])
                    for j, hit in enumerate(hits[:k]):
                        neighbors[i, j] = int(hit["_id"])
                        distances[i, j] = float(hit["_score"])

                elapsed_ms = (time.perf_counter() - t0) * 1000
                qps = (
                    n_queries / (elapsed_ms / 1000) if elapsed_ms > 0 else 0.0
                )

                recall = 0.0
                if groundtruth is not None:
                    gt_k = min(k, groundtruth.shape[1])
                    n_correct = sum(
                        len(
                            set(neighbors[i, :k].tolist())
                            & set(groundtruth[i, :gt_k].tolist())
                        )
                        for i in range(n_queries)
                    )
                    recall = (
                        n_correct / (n_queries * gt_k) if gt_k > 0 else 0.0
                    )

                per_param_results.append(
                    {
                        "search_params": sp,
                        "search_time_ms": elapsed_ms,
                        "queries_per_second": qps,
                        "recall": recall,
                        "p50_ms": float(np.percentile(latencies, 50)),
                        "p95_ms": float(np.percentile(latencies, 95)),
                        "p99_ms": float(np.percentile(latencies, 99)),
                    }
                )
                last_neighbors = neighbors
                last_distances = distances

            avg_recall = float(
                np.mean([r["recall"] for r in per_param_results])
            )
            avg_qps = float(
                np.mean([r["queries_per_second"] for r in per_param_results])
            )
            total_ms = float(
                sum(r["search_time_ms"] for r in per_param_results)
            )

            return SearchResult(
                neighbors=last_neighbors,
                distances=last_distances,
                search_time_ms=total_ms,
                queries_per_second=avg_qps,
                recall=avg_recall,
                algorithm=self.algo,
                search_params=search_params_list,
                latency_percentiles={
                    "p50_ms": per_param_results[-1]["p50_ms"],
                    "p95_ms": per_param_results[-1]["p95_ms"],
                    "p99_ms": per_param_results[-1]["p99_ms"],
                },
                metadata={
                    "per_search_param_results": per_param_results,
                    "recall_is_authoritative": True,
                },
                success=True,
            )
        except Exception as e:
            return SearchResult(
                neighbors=np.empty((0, k), dtype=np.int64),
                distances=np.empty((0, k), dtype=np.float32),
                search_time_ms=0,
                queries_per_second=0,
                recall=0,
                algorithm=self.algo,
                search_params=[],
                success=False,
                error_message=str(e),
            )


def _get_cuvs_bench_config_path() -> str:
    """Get cuvs_bench config directory for shared datasets.yaml."""
    import cuvs_bench.orchestrator.config_loaders as _config_loaders

    mod_file = getattr(_config_loaders, "__file__", None)
    if mod_file:
        # config_loaders is at cuvs_bench/orchestrator/config_loaders.py
        # config is at cuvs_bench/config
        pkg_dir = Path(mod_file).resolve().parent.parent
        return str(pkg_dir / "config")
    import cuvs_bench

    if cuvs_bench.__file__:
        return os.path.join(os.path.dirname(cuvs_bench.__file__), "config")
    raise RuntimeError(
        "Cannot determine cuvs_bench config path. "
        "Ensure cuvs_bench is properly installed or on PYTHONPATH."
    )


def _get_elastic_config_path() -> str:
    """Get the config directory for elastic.yaml."""
    return os.path.join(os.path.dirname(__file__), "../config")


class ElasticConfigLoader(ConfigLoader):
    """Config loader for Elasticsearch backend."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or _get_cuvs_bench_config_path()

    @property
    def backend_type(self) -> str:
        return "elastic"

    def _discover_algo_groups(
        self,
        dataset_conf,
        dataset,
        dataset_path,
        **kwargs,
    ):
        """Discover elastic algorithm groups using shared loader semantics."""
        algorithm_configuration = kwargs.get("algorithm_configuration")
        algorithms_arg = kwargs.get("algorithms")
        groups_arg = kwargs.get("groups")
        algo_groups_arg = kwargs.get("algo_groups")

        algos_conf_fs = self.gather_algorithm_configs(
            self.config_path, algorithm_configuration
        )

        elastic_algos = []
        for algo_f in algos_conf_fs:
            try:
                algo_conf = self.load_yaml_file(algo_f)
            except Exception:
                continue
            if not isinstance(algo_conf, dict):
                continue
            algo_name = algo_conf.get("name", "")
            if not algo_name.startswith("elastic_"):
                continue
            elastic_algos.append(algo_conf)

        if not elastic_algos:
            default_grp = {
                "build": {"m": [16], "ef_construction": [100]},
                "search": {"num_candidates": [100]},
            }
            elastic_algos = [
                {"name": "elastic_hnsw", "groups": {"base": default_grp}}
            ]

        allowed_algos = (
            [a.strip() for a in algorithms_arg.split(",") if a.strip()]
            if algorithms_arg
            else None
        )
        allowed_groups = (
            [g.strip() for g in groups_arg.split(",") if g.strip()]
            if groups_arg
            else None
        )
        algo_group_map: Dict[str, set] = {}
        if algo_groups_arg:
            for item in algo_groups_arg.split(","):
                item = item.strip()
                if not item or "." not in item:
                    continue
                algo_name, group_name = item.split(".", 1)
                algo_group_map.setdefault(algo_name, set()).add(group_name)

        # Backward-compatible fallback for older elastic usage where `algorithms`
        # was used as a group selector (e.g. algorithms="test").
        if allowed_algos and not allowed_groups and not algo_group_map:
            known_groups = {
                group_name
                for algo_conf in elastic_algos
                for group_name in algo_conf.get("groups", {})
            }
            if all(name in known_groups for name in allowed_algos):
                allowed_groups = allowed_algos
                allowed_algos = None

        if allowed_groups is None and not algo_group_map:
            allowed_groups = ["base"]

        result = []
        for algo_conf in elastic_algos:
            algo_name = algo_conf["name"]
            if allowed_algos and algo_name not in allowed_algos:
                continue

            groups = dict(algo_conf.get("groups", {}))
            if allowed_groups is not None:
                groups = {
                    group_name: group_conf
                    for group_name, group_conf in groups.items()
                    if group_name in allowed_groups
                }
            if algo_name in algo_group_map:
                groups = {
                    group_name: group_conf
                    for group_name, group_conf in groups.items()
                    if group_name in algo_group_map[algo_name]
                }

            for group_name, group_conf in groups.items():
                result.append((algo_name, group_name, group_conf, {}))

        if not result and allowed_groups:
            raise ValueError(
                f"Could not find elastic groups {allowed_groups} in elastic configs"
            )
        if not result and allowed_algos:
            raise ValueError(
                f"Could not find elastic algorithms {allowed_algos} in elastic configs"
            )

        return result

    def _build_benchmark_configs(
        self,
        dataset_config,
        dataset_conf,
        dataset,
        dataset_path,
        expanded_groups,
        **kwargs,
    ):
        """Build BenchmarkConfigs from shared expanded elastic parameter groups."""
        host = kwargs.get("host", "localhost")
        port = kwargs.get("port", 9200)
        scheme = kwargs.get("scheme", "http")
        basic_auth = kwargs.get("basic_auth")
        username = kwargs.get("username")
        password = kwargs.get("password")
        if basic_auth is None and username and password:
            basic_auth = (username, password)

        tune_mode = kwargs.get("_tune_mode", False)
        tune_build_params = kwargs.get("_tune_build_params")
        tune_search_params = kwargs.get("_tune_search_params")

        benchmark_configs = []
        for (
            algo_name,
            group_name,
            _group_conf,
            build_combos,
            search_combos,
            _group_meta,
        ) in expanded_groups:
            _validate_elastic_algorithm(algo_name)
            if tune_mode and tune_build_params is not None:
                actual_build = [dict(tune_build_params)]
                actual_search = (
                    [dict(tune_search_params)] if tune_search_params else [{}]
                )
            else:
                actual_build = build_combos
                actual_search = search_combos

            for build_param in actual_build:
                build_param = dict(build_param)
                if "type" not in build_param and algo_name.startswith(
                    "elastic_"
                ):
                    build_param["type"] = algo_name.replace("elastic_", "", 1)
                _validate_elastic_index_type(build_param["type"])
                if "similarity" in build_param:
                    _validate_elastic_similarity(build_param["similarity"])

                if tune_mode:
                    label_prefix = f"{algo_name}_tune"
                elif group_name != "base":
                    label_prefix = f"{algo_name}_{group_name}"
                else:
                    label_prefix = algo_name

                name_parts = [
                    f"{k}{v}"
                    for k, v in build_param.items()
                    if k in ("m", "ef_construction")
                ]
                index_label = (
                    "_".join([label_prefix] + name_parts)
                    if name_parts
                    else label_prefix
                )
                es_index_name = index_label.lower().replace(".", "_")

                index_config = IndexConfig(
                    name=index_label,
                    algo=algo_name,
                    build_param=build_param,
                    search_params=[dict(sp) for sp in actual_search],
                    file="",
                )
                benchmark_configs.append(
                    BenchmarkConfig(
                        indexes=[index_config],
                        backend_config={
                            "name": index_label,
                            "host": host,
                            "port": port,
                            "scheme": scheme,
                            "index_name": es_index_name,
                            "basic_auth": basic_auth,
                            **build_param,
                        },
                    )
                )

        return benchmark_configs


def register() -> None:
    """Register Elasticsearch backend and config loader (idempotent)."""
    from cuvs_bench.backends.registry import (
        _CONFIG_LOADER_REGISTRY,
        get_registry,
    )

    reg = get_registry()
    if not reg.is_registered("elastic"):
        register_backend("elastic", ElasticBackend)
    if "elastic" not in _CONFIG_LOADER_REGISTRY:
        register_config_loader("elastic", ElasticConfigLoader)


# ── Convenience API ───────────────────────────────────────────────────────────


def run_build(
    dataset: str = "test-data",
    dataset_path: str = "./datasets",
    host: str = "localhost",
    port: int = 9200,
    algorithms: str = "test",
    force: bool = False,
    **kwargs,
):
    """Build an Elasticsearch vector index. Returns list of BuildResult."""
    register()
    from cuvs_bench.orchestrator import BenchmarkOrchestrator

    orch = BenchmarkOrchestrator(backend_type="elastic")
    return orch.run_benchmark(
        build=True,
        search=False,
        dataset=dataset,
        dataset_path=dataset_path,
        host=host,
        port=port,
        algorithms=algorithms,
        force=force,
        **kwargs,
    )


def run_search(
    dataset: str = "test-data",
    dataset_path: str = "./datasets",
    host: str = "localhost",
    port: int = 9200,
    algorithms: str = "test",
    **kwargs,
):
    """Run kNN search against an existing Elasticsearch index. Returns list of SearchResult."""
    register()
    from cuvs_bench.orchestrator import BenchmarkOrchestrator

    orch = BenchmarkOrchestrator(backend_type="elastic")
    return orch.run_benchmark(
        build=False,
        search=True,
        dataset=dataset,
        dataset_path=dataset_path,
        host=host,
        port=port,
        algorithms=algorithms,
        **kwargs,
    )


def run_benchmark(
    dataset: str = "test-data",
    dataset_path: str = "./datasets",
    host: str = "localhost",
    port: int = 9200,
    algorithms: str = "test",
    build: bool = True,
    search: bool = True,
    force: bool = False,
    **kwargs,
):
    """Run build and/or search. Returns list of results."""
    register()
    from cuvs_bench.orchestrator import BenchmarkOrchestrator

    orch = BenchmarkOrchestrator(backend_type="elastic")
    return orch.run_benchmark(
        build=build,
        search=search,
        dataset=dataset,
        dataset_path=dataset_path,
        host=host,
        port=port,
        algorithms=algorithms,
        force=force,
        **kwargs,
    )
