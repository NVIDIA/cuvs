# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026, JAESOL SHIN.

"""PostgreSQL pgvector and pg_cuvs benchmarks with end-to-end SQL timing.

The server and extensions are managed externally. Each build owns a fresh
schema, loads zero-based corpus IDs, and builds one index before sweeping all
search parameters. cleanup() removes only that schema, including GPU indexes.
"""

import json
import time
import uuid
from pathlib import Path

import numpy as np

from .base import BenchmarkBackend, BuildResult, SearchResult
from ..orchestrator.config_loaders import (
    BenchmarkConfig,
    ConfigLoader,
    IndexConfig,
)

_BUILD_KEYS = {
    "pgvector_hnsw": {"m", "ef_construction"},
    "pgvector_ivfflat": {"lists"},
    "pgcuvs_cagra": {
        "graph_degree",
        "intermediate_graph_degree",
        "build_algo",
    },
    "pgcuvs_hnsw_import": {
        "graph_degree",
        "intermediate_graph_degree",
        "build_algo",
        "mode",
    },
}
_SEARCH_GUC = {
    "pgvector_hnsw": ("ef_search", "hnsw.ef_search"),
    "pgvector_ivfflat": ("probes", "ivfflat.probes"),
    "pgcuvs_cagra": ("candidates", "cuvs.k"),
    "pgcuvs_hnsw_import": ("ef_search", "hnsw.ef_search"),
}


def _positive_int(value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("Index parameters must be positive integers")
    if value <= 0:
        raise ValueError("Index parameters must be positive integers")
    return int(value)


def _build_params(algo, params):
    if algo not in _BUILD_KEYS:
        raise ValueError(f"Unsupported PostgreSQL algorithm: {algo}")
    if set(params) - _BUILD_KEYS[algo]:
        raise ValueError(f"Unsupported build parameters for {algo}")
    validated = {}
    for key, value in params.items():
        if key == "build_algo":
            if value not in ("ivf_pq", "nn_descent", "auto"):
                raise ValueError("Unsupported CAGRA build_algo")
        elif key == "mode":
            if value not in ("nsw", "hnsw", "hnswlib", "hnswlib_file"):
                raise ValueError("Unsupported pg_cuvs HNSW mode")
        else:
            value = _positive_int(value)
        validated[key] = value
    if validated.get("intermediate_graph_degree", 128) < validated.get(
        "graph_degree", 64
    ):
        raise ValueError("intermediate_graph_degree must be >= graph_degree")
    return validated


class PostgresConfigLoader(ConfigLoader):
    """Use standard dataset YAML and Cartesian build/search parameter grids."""

    def __init__(self, config_path=None):
        self.config_path = str(
            config_path or Path(__file__).parent.parent / "config"
        )

    @property
    def backend_type(self):
        return "postgres"

    def _discover_algo_groups(
        self, dataset_conf, dataset, dataset_path, **kwargs
    ):
        algorithms = kwargs.get("algorithms")
        allowed = (
            set(algorithms.split(",")) if algorithms else set(_BUILD_KEYS)
        )
        if allowed - _BUILD_KEYS.keys():
            raise ValueError(f"Unsupported PostgreSQL algorithm: {algorithms}")
        groups = kwargs.get("groups")
        selected_groups = set(groups.split(",")) if groups else None
        algo_groups = kwargs.get("algo_groups")
        explicit_groups = (
            {tuple(item.split(".")) for item in algo_groups.split(",")}
            if algo_groups
            else set()
        )
        if any(len(pair) != 2 or not all(pair) for pair in explicit_groups):
            raise ValueError("Expected algo_groups as algorithm.group pairs")
        result = {}
        for path in self.gather_algorithm_configs(
            self.config_path, kwargs.get("algorithm_configuration")
        ):
            cfg = self.load_yaml_file(path)
            algo = cfg.get("name")
            if algo not in allowed:
                continue
            for group, params in cfg.get("groups", {}).items():
                if (
                    selected_groups is None
                    or group in selected_groups
                    or (algo, group) in explicit_groups
                ):
                    result[algo, group] = (algo, group, params, {})
        if explicit_groups - result.keys():
            raise ValueError("Requested PostgreSQL algo_groups not found")
        if not result:
            raise ValueError("No PostgreSQL algorithm configuration selected")
        return list(result.values())

    def _build_benchmark_configs(
        self,
        dataset_config,
        dataset_conf,
        dataset,
        dataset_path,
        expanded_groups,
        **kwargs,
    ):
        configs = []
        for algo, group, _, builds, searches, _ in expanded_groups:
            for build in builds:
                build = _build_params(algo, build)
                label = ".".join(
                    [algo, group]
                    + [f"{k}{v}" for k, v in sorted(build.items())]
                )
                idx = IndexConfig(
                    name=label,
                    algo=algo,
                    build_param=build,
                    search_params=searches,
                    file=str(Path(dataset_path) / dataset / "index" / label),
                )
                conn_cfg = {
                    key: kwargs[key]
                    for key in ("conninfo", "index_dir", "warmup_queries")
                    if key in kwargs
                }
                configs.append(
                    BenchmarkConfig(
                        indexes=[idx],
                        backend_config={
                            "name": label,
                            "algo": algo,
                            "group": group,
                            **conn_cfg,
                        },
                    )
                )
        return configs


class PostgresBackend(BenchmarkBackend):
    """Run single-client SQL latency sweeps in a backend-owned schema.

    Requires a database with vector installed; pgcuvs_* also need pg_cuvs
    preloaded and its daemon running. Standard libpq PG* variables and service
    files are supported, so credentials need not appear in benchmark YAML.
    """

    def __init__(self, config):
        super().__init__(config)
        self._conn = None
        self._schema = None
        self._index = None
        self._measurement_metadata = {}

    @property
    def algo(self):
        return self.config["algo"]

    def _connect(self):
        if self._conn is not None:
            return self._conn
        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError as exc:
            raise ImportError(
                "PostgreSQL backend needs pip install 'cuvs-bench[postgres]'"
            ) from exc
        try:
            self._conn = psycopg.connect(
                self.config.get("conninfo", ""),
                autocommit=True,
                prepare_threshold=None,
            )
        except Exception:
            # libpq parsing errors can echo malformed credential tokens.
            raise RuntimeError(
                "PostgreSQL connection failed; check libpq connection settings"
            ) from None
        installed = {
            row[0]
            for row in self._conn.execute(
                "SELECT extname FROM pg_extension"
            ).fetchall()
        }
        required = {"vector"}
        if self.algo.startswith("pgcuvs_"):
            required.add("pg_cuvs")
        if required - installed:
            raise RuntimeError(
                f"Install database extensions: {required - installed}"
            )
        register_vector(self._conn)
        if "pg_cuvs" in installed:
            self._conn.execute(
                "SELECT set_config('enable_cuvs', %s, false)",
                ("on" if self.algo == "pgcuvs_cagra" else "off",),
            )
            if self.algo == "pgcuvs_cagra":
                self._conn.execute("SET cuvs.search_mode = 'cagra'")
                self._conn.execute("SET cuvs.cpu_hnsw_fallback = off")
                self._conn.execute("SET cuvs.cagra_batch_wait_us = 0")
                self._measurement_metadata["cagra_batch_wait_us"] = int(
                    self._conn.execute(
                        "SHOW cuvs.cagra_batch_wait_us"
                    ).fetchone()[0]
                )
            if "index_dir" in self.config:
                self._conn.execute(
                    "SELECT set_config('cuvs.index_dir', %s, false)",
                    (self.config["index_dir"],),
                )
        return self._conn

    def _metadata(self, idx):
        return {
            "group": self.config.get("group", "base"),
            "index_name": idx.file,
            "measurement_boundary": "end_to_end_sql",
            "axis": "single_client_latency",
            "build_params": json.dumps(idx.build_param, sort_keys=True),
            **self._measurement_metadata,
        }

    def _create_index(self, name, method, params):
        from psycopg import sql

        statement = sql.SQL(
            "CREATE INDEX {} ON {}.vectors USING {} (embedding vector_l2_ops)"
        ).format(
            sql.Identifier(name),
            sql.Identifier(self._schema),
            sql.Identifier(method),
        )
        if params:
            statement += sql.SQL(" WITH ({})").format(
                sql.SQL(", ").join(
                    sql.SQL("{} = {}").format(
                        sql.Identifier(k), sql.Literal(v)
                    )
                    for k, v in params.items()
                )
            )
        self._conn.execute(statement)

    def build(self, dataset, indexes, force=False, dry_run=False):
        """COPY the corpus, then time only CREATE INDEX (including GPU export)."""
        idx = indexes[0]
        result = BuildResult(
            index_path=idx.file,
            build_time_seconds=0,
            index_size_bytes=0,
            algorithm=self.algo,
            build_params=dict(idx.build_param),
            metadata=self._metadata(idx),
        )
        try:
            if len(indexes) != 1 or idx.algo != self.algo:
                raise ValueError(
                    "Expected one index matching the backend algorithm"
                )
            params = _build_params(self.algo, idx.build_param)
            if dataset.distance_metric not in ("euclidean", "l2"):
                raise ValueError(
                    "PostgreSQL backend currently supports euclidean only"
                )
            if dry_run:
                result.metadata.update(dry_run=True, skipped=True)
                return result
            if self._schema is not None:
                raise RuntimeError(
                    "Use a fresh backend for each build configuration"
                )
            conn = self._connect()
            from psycopg import sql

            vectors = dataset.training_vectors
            if vectors.ndim != 2 or not len(vectors) or not vectors.shape[1]:
                raise ValueError("Expected a nonempty two-dimensional corpus")
            schema = "cuvs_bench_" + uuid.uuid4().hex
            conn.execute(
                sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema))
            )
            self._schema = schema
            conn.execute(
                sql.SQL(
                    "CREATE TABLE {}.vectors (id bigint, embedding vector({}))"
                ).format(sql.Identifier(schema), sql.Literal(vectors.shape[1]))
            )
            with conn.cursor().copy(
                sql.SQL(
                    "COPY {}.vectors (id, embedding) FROM STDIN (FORMAT BINARY)"
                ).format(sql.Identifier(schema))
            ) as copy:
                copy.set_types(["int8", "vector"])
                for i, vector in enumerate(vectors):
                    copy.write_row((i, vector))
            conn.execute(
                sql.SQL("ANALYZE {}.vectors").format(sql.Identifier(schema))
            )
            started = time.perf_counter()
            if self.algo in ("pgvector_hnsw", "pgvector_ivfflat"):
                self._create_index(
                    "ann", self.algo.removeprefix("pgvector_"), params
                )
            else:
                mode = params.pop("mode", "nsw")
                source = (
                    "ann" if self.algo == "pgcuvs_cagra" else "source_graph"
                )
                self._create_index(source, "cagra", params)
                if self.algo == "pgcuvs_hnsw_import":
                    self._create_index(
                        "ann",
                        "pg_cuvs_hnsw",
                        {
                            "source": f"{schema}.source_graph",
                            "mode": mode,
                        },
                    )
            result.build_time_seconds = time.perf_counter() - started
            relation = f"{schema}.ann"
            disk_bytes = int(
                conn.execute(
                    "SELECT pg_relation_size(%s::regclass)", (relation,)
                ).fetchone()[0]
            )
            self._measurement_metadata["index_disk_bytes"] = disk_bytes
            if self.algo == "pgcuvs_cagra":
                row = conn.execute(
                    "SELECT vram_bytes FROM pg_stat_gpu_search "
                    "WHERE index_oid = %s::regclass",
                    (relation,),
                ).fetchone()
                if row is None or row[0] is None:
                    raise RuntimeError(
                        "No resident CAGRA VRAM accounting available"
                    )
                result.index_size_bytes = int(row[0])
                self._measurement_metadata.update(
                    index_size_source="daemon_vram_bytes",
                    daemon_vram_bytes=result.index_size_bytes,
                )
            else:
                result.index_size_bytes = disk_bytes
                started = time.perf_counter()
                blocks = conn.execute(
                    "SELECT pg_prewarm(%s::regclass, 'buffer')", (relation,)
                ).fetchone()[0]
                self._measurement_metadata.update(
                    index_size_source="pg_relation_size",
                    prewarm_method="pg_prewarm_buffer",
                    prewarm_blocks=int(blocks),
                    prewarm_seconds=time.perf_counter() - started,
                )
            self._measurement_metadata["index_size_bytes"] = (
                result.index_size_bytes
            )
            result.metadata.update(self._measurement_metadata)
            self._index = idx
        except Exception as exc:
            result.success = False
            result.error_message = str(exc)
        return result

    def _query(self, vector, k):
        from psycopg import sql

        return sql.SQL(
            "SELECT id, embedding <-> %s AS distance FROM {}.vectors "
            "ORDER BY distance LIMIT %s"
        ).format(sql.Identifier(self._schema)), (vector, k)

    def _check_plan(self, statement, params):
        from psycopg import sql

        plan = self._conn.execute(
            sql.SQL("EXPLAIN (FORMAT JSON) ") + statement, params
        ).fetchone()[0][0]["Plan"]

        def uses_index(node):
            return (
                node.get("Node Type") == "Index Scan"
                and node.get("Index Name") == "ann"
            ) or any(uses_index(child) for child in node.get("Plans", []))

        if not uses_index(plan):
            raise RuntimeError(
                "Expected Index Scan on ann; refusing fallback timing"
            )

    def _gpu_search_count(self):
        row = self._conn.execute(
            "SELECT resident, search_mode, search_count "
            "FROM pg_stat_gpu_search WHERE index_oid = %s::regclass",
            (f"{self._schema}.ann",),
        ).fetchone()
        if row is None or not row[0] or row[1] != "gpu_cagra":
            raise RuntimeError("Expected resident GPU CAGRA execution")
        return int(row[2])

    def search(
        self,
        dataset,
        indexes,
        k,
        batch_size=10000,
        mode="latency",
        force=False,
        search_threads=None,
        dry_run=False,
    ):
        """Return one result per search setting; the orchestrator computes recall."""
        idx = indexes[0]
        results = []
        for search in idx.search_params:
            result = SearchResult(
                neighbors=np.empty((0, 0), dtype=np.int64),
                distances=np.empty((0, 0), dtype=np.float32),
                search_time_ms=0,
                queries_per_second=0,
                recall=0,
                algorithm=self.algo,
                search_params=[dict(search)],
                metadata=self._metadata(idx),
            )
            results.append(result)
            try:
                if mode != "latency" or search_threads not in (None, 1, "1"):
                    raise ValueError(
                        "PostgreSQL supports single-client latency only"
                    )
                key, guc = _SEARCH_GUC[self.algo]
                if set(search) != {key}:
                    raise ValueError(f"Expected search parameter {key}")
                value, k = _positive_int(search[key]), _positive_int(k)
                if key != "probes" and value < k:
                    raise ValueError(f"{key} must be >= k")
                if dry_run:
                    result.metadata.update(dry_run=True, skipped=True)
                    continue
                if self._index != idx:
                    raise RuntimeError("Build this index before searching")
                queries = dataset.query_vectors
                if not len(queries):
                    raise ValueError("Expected at least one query")
                conn = self._conn
                conn.execute("SET enable_seqscan = off")
                conn.execute(
                    "SELECT set_config(%s, %s, false)", (guc, str(value))
                )
                self._check_plan(*self._query(queries[0], k))
                for query in queries[
                    : int(self.config.get("warmup_queries", 10))
                ]:
                    conn.execute(*self._query(query, k)).fetchall()
                gpu_before = (
                    self._gpu_search_count()
                    if self.algo == "pgcuvs_cagra"
                    else None
                )
                ids = np.full((len(queries), k), -1, dtype=np.int64)
                distances = np.full(
                    (len(queries), k), np.inf, dtype=np.float32
                )
                latencies = []
                for i, query in enumerate(queries):
                    statement, bindings = self._query(query, k)
                    started = time.perf_counter()
                    rows = conn.execute(statement, bindings).fetchall()
                    latencies.append(time.perf_counter() - started)
                    for j, (neighbor, distance) in enumerate(rows):
                        ids[i, j], distances[i, j] = neighbor, distance
                self._check_plan(*self._query(queries[0], k))
                if gpu_before is not None:
                    gpu_queries = self._gpu_search_count() - gpu_before
                    if gpu_queries != len(queries):
                        raise RuntimeError(
                            "GPU query count does not match measured queries"
                        )
                    result.metadata.update(
                        observed_search_mode="gpu_cagra",
                        gpu_queries=gpu_queries,
                    )
                total = sum(latencies)
                result.neighbors, result.distances = ids, distances
                result.search_time_ms = total * 1000
                result.queries_per_second = len(queries) / total
                result.latency_percentiles = dict(
                    zip(
                        ("p50_ms", "p95_ms", "p99_ms"),
                        np.percentile(latencies, [50, 95, 99]) * 1000,
                    )
                )
                result.metadata.update(
                    n_queries=len(queries),
                    latency_seconds=total / len(queries),
                )
            except Exception as exc:
                result.success = False
                result.error_message = str(exc)
        return results

    def cleanup(self):
        """Remove only this backend's schema and always close its connection."""
        if self._conn is None:
            return
        try:
            if self._schema is not None:
                from psycopg import sql

                self._conn.execute(
                    sql.SQL("DROP SCHEMA {} CASCADE").format(
                        sql.Identifier(self._schema)
                    )
                )
        finally:
            self._conn.close()
            self._conn = None
            self._schema = None
            self._index = None
            self._measurement_metadata = {}


def register():
    """Register the optional backend without importing database dependencies."""
    from .registry import (
        _CONFIG_LOADER_REGISTRY,
        get_registry,
        register_backend,
        register_config_loader,
    )

    if not get_registry().is_registered("postgres"):
        register_backend("postgres", PostgresBackend)
    if "postgres" not in _CONFIG_LOADER_REGISTRY:
        register_config_loader("postgres", PostgresConfigLoader)
