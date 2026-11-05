# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026, JAESOL SHIN.

"""PostgreSQL backend contracts and opt-in live database tests."""

import os
import csv
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
import yaml

from cuvs_bench.backends.base import Dataset, SearchResult
from cuvs_bench.backends.postgres import (
    PostgresBackend,
    PostgresConfigLoader,
    register,
)
from cuvs_bench.backends.registry import get_backend_class, get_config_loader
from cuvs_bench.orchestrator.config_loaders import BenchmarkConfig, IndexConfig
from cuvs_bench.orchestrator.orchestrator import BenchmarkOrchestrator


@pytest.fixture
def config_dir(tmp_path):
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / "datasets.yaml").write_text(
        yaml.safe_dump([{"name": "small", "dims": 4}])
    )
    (tmp_path / "algos").mkdir()
    (tmp_path / "algos" / "pgvector_hnsw.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "pgvector_hnsw",
                "groups": {
                    "test": {
                        "build": {"m": [8, 16], "ef_construction": [64]},
                        "search": {"ef_search": [10, 40]},
                    }
                },
            }
        )
    )
    return tmp_path


def dataset():
    rng = np.random.default_rng(7)
    base = rng.normal(size=(128, 4)).astype(np.float32)
    queries = base[:5].copy()
    dists = ((queries[:, None] - base[None]) ** 2).sum(axis=2)
    return Dataset(
        "small",
        training_vectors=base,
        query_vectors=queries,
        groundtruth_neighbors=np.argsort(dists, axis=1)[:, :5],
    )


def index(algo="pgvector_hnsw", build=None, search=None):
    return IndexConfig(
        name=algo,
        algo=algo,
        build_param=build or {"m": 8, "ef_construction": 64},
        search_params=search or [{"ef_search": 10}, {"ef_search": 40}],
        file="unused-index-label",
    )


def test_registration_is_idempotent():
    register()
    register()
    assert get_backend_class("postgres") is PostgresBackend
    assert get_config_loader("postgres") is PostgresConfigLoader


def test_loader_keeps_search_sweep_on_one_build(config_dir):
    loader = PostgresConfigLoader(config_dir)
    ds, configs = loader.load(
        "small", str(config_dir), algorithms="pgvector_hnsw", groups="test"
    )
    assert ds.dims == 4
    assert len(configs) == 2
    assert len({cfg.index_name for cfg in configs}) == 2
    assert {cfg.build_params["m"] for cfg in configs} == {8, 16}
    assert all(
        cfg.search_params_list == [{"ef_search": 10}, {"ef_search": 40}]
        for cfg in configs
    )


def test_loader_rejects_unknown_algorithm(config_dir):
    with pytest.raises(ValueError, match="algorithm"):
        PostgresConfigLoader(config_dir).load(
            "small", str(config_dir), algorithms="pgvector_typo"
        )


def test_custom_group_replaces_default_search_grid(config_dir, tmp_path):
    override = tmp_path / "override.yaml"
    override.write_text(
        yaml.safe_dump(
            {
                "name": "pgvector_hnsw",
                "groups": {
                    "test": {
                        "build": {"m": [8], "ef_construction": [64]},
                        "search": {"ef_search": [80]},
                    }
                },
            }
        )
    )
    _, configs = PostgresConfigLoader(config_dir).load(
        "small",
        str(config_dir),
        algorithms="pgvector_hnsw",
        groups="test",
        algorithm_configuration=str(override),
    )
    assert len(configs) == 1
    assert configs[0].build_params == {"m": 8, "ef_construction": 64}
    assert configs[0].search_params_list == [{"ef_search": 80}]


def test_loader_rejects_unknown_group(config_dir):
    with pytest.raises(ValueError, match="configuration"):
        PostgresConfigLoader(config_dir).load(
            "small", str(config_dir), groups="missing"
        )


def test_explicit_algo_group_adds_to_selected_groups(config_dir):
    path = config_dir / "algos" / "pgvector_hnsw.yaml"
    config = yaml.safe_load(path.read_text())
    config["groups"]["base"] = {
        "build": {"m": [32], "ef_construction": [64]},
        "search": {"ef_search": [80]},
    }
    path.write_text(yaml.safe_dump(config))
    _, configs = PostgresConfigLoader(config_dir).load(
        "small",
        str(config_dir),
        algorithms="pgvector_hnsw",
        groups="base",
        algo_groups="pgvector_hnsw.test,pgvector_hnsw.test",
    )
    assert len(configs) == 3
    assert {cfg.build_params["m"] for cfg in configs} == {8, 16, 32}
    assert sum(cfg.backend_config["group"] == "test" for cfg in configs) == 2


def test_explicit_algo_group_selects_only_its_algorithm(config_dir):
    (config_dir / "algos" / "pgvector_ivfflat.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "pgvector_ivfflat",
                "groups": {
                    "base": {
                        "build": {"lists": [16]},
                        "search": {"probes": [4]},
                    },
                    "test": {
                        "build": {"lists": [4]},
                        "search": {"probes": [1]},
                    },
                },
            }
        )
    )
    _, configs = PostgresConfigLoader(config_dir).load(
        "small",
        str(config_dir),
        algorithms="pgvector_hnsw,pgvector_ivfflat",
        groups="base",
        algo_groups="pgvector_hnsw.test",
    )
    assert len(configs) == 3
    assert {
        (cfg.backend_config["algo"], cfg.backend_config["group"])
        for cfg in configs
    } == {("pgvector_hnsw", "test"), ("pgvector_ivfflat", "base")}


@pytest.mark.parametrize(
    "selector",
    ["pgvector_hnsw.missing", "pgvector_typo.test", "pgvector_hnsw", ".test"],
)
def test_loader_rejects_unmatched_explicit_groups(config_dir, selector):
    with pytest.raises(ValueError, match="algo_groups"):
        PostgresConfigLoader(config_dir).load(
            "small",
            str(config_dir),
            algorithms="pgvector_hnsw",
            groups="test",
            algo_groups=selector,
        )


def test_dry_run_does_not_connect(monkeypatch):
    backend = PostgresBackend({"algo": "pgvector_hnsw"})
    monkeypatch.setattr(
        backend, "_connect", lambda: pytest.fail("dry run connected")
    )
    ds, idx = dataset(), index()
    assert backend.build(ds, [idx], dry_run=True).success
    results = backend.search(ds, [idx], 5, dry_run=True)
    assert len(results) == 2
    assert all(result.metadata["dry_run"] for result in results)
    backend.cleanup()


@pytest.mark.parametrize("threads", [None, 1, "1"])
def test_single_client_thread_setting_accepts_cli_string(threads):
    backend = PostgresBackend({"algo": "pgvector_hnsw"})
    results = backend.search(
        dataset(), [index()], 5, search_threads=threads, dry_run=True
    )
    assert all(result.success for result in results)


@pytest.mark.parametrize("threads", [2, "2", "1,2"])
def test_multi_client_thread_setting_is_rejected(threads):
    backend = PostgresBackend({"algo": "pgvector_hnsw"})
    results = backend.search(
        dataset(), [index()], 5, search_threads=threads, dry_run=True
    )
    assert all(not result.success for result in results)


@pytest.mark.parametrize("distance", ["cosine", "inner_product"])
def test_build_rejects_unimplemented_metric(distance):
    ds = dataset()
    ds.distance_metric = distance
    result = PostgresBackend({"algo": "pgvector_hnsw"}).build(ds, [index()])
    assert not result.success
    assert "euclidean" in result.error_message


@pytest.mark.parametrize(
    "params",
    [{"m": "8); DROP SCHEMA public CASCADE; --"}, {"m": 0}, {"typo": 8}],
)
def test_build_rejects_invalid_params_before_connect(params, monkeypatch):
    backend = PostgresBackend({"algo": "pgvector_hnsw"})
    monkeypatch.setattr(
        backend, "_connect", lambda: pytest.fail("invalid params connected")
    )
    result = backend.build(dataset(), [index(build=params)])
    assert not result.success


def test_backend_does_not_export_connection_string():
    backend = PostgresBackend(
        {"algo": "pgvector_hnsw", "conninfo": "password=private"}
    )
    result = backend.build(dataset(), [index()], dry_run=True)
    assert "private" not in str(result.to_json())


@pytest.mark.parametrize("missing_module", ["psycopg", "pgvector.psycopg"])
def test_build_reports_optional_extra_installation(
    missing_module, monkeypatch
):
    monkeypatch.setitem(sys.modules, missing_module, None)
    result = PostgresBackend({"algo": "pgvector_hnsw"}).build(
        dataset(), [index()]
    )
    assert not result.success
    assert "cuvs-bench[postgres]" in result.error_message


def test_cpu_marker_selection_excludes_gpu_with_dsns_set(monkeypatch):
    monkeypatch.setenv("CUVS_BENCH_POSTGRES_DSN", "not-a-live-database")
    monkeypatch.setenv("CUVS_BENCH_POSTGRES_GPU_DSN", "not-a-live-database")
    collected = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(Path(__file__).resolve()),
            "--collect-only",
            "-q",
            "-m",
            "not postgres",
            "-p",
            "no:cacheprovider",
            "-o",
            "markers=postgres: live PostgreSQL tests\n"
            "postgres_gpu: real GPU PostgreSQL tests",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "test_dry_run_does_not_connect" in collected.stdout
    assert "test_real_gpu_build_search_and_hnsw_export" not in collected.stdout


def test_malformed_conninfo_does_not_leak_to_results_or_stdout(
    monkeypatch, capsys
):
    psycopg = pytest.importorskip("psycopg")
    password = "synthetic" + chr(37) + "password"
    conninfo = f"postgresql://benchmark_role:{password}@localhost/cuvs_bench"
    dns = Mock(side_effect=AssertionError("unexpected DNS lookup"))
    connect = Mock(
        side_effect=AssertionError("unexpected database connection")
    )
    monkeypatch.setattr(socket, "getaddrinfo", dns)
    monkeypatch.setattr(psycopg.Connection, "_connect_gen", connect)
    idx = index()
    config = BenchmarkConfig(
        indexes=[idx],
        backend_config={"algo": "pgvector_hnsw", "conninfo": conninfo},
    )
    orchestrator = BenchmarkOrchestrator(backend_type="postgres")
    orchestrator.config_loader = SimpleNamespace(
        load=lambda **kwargs: (None, [config])
    )
    monkeypatch.setattr(
        orchestrator, "_create_dataset", lambda config: dataset()
    )
    results = orchestrator._run_sweep(
        build=True,
        search=False,
        force=False,
        dry_run=False,
        count=5,
        batch_size=10000,
        search_mode="latency",
        search_threads=None,
    )
    assert len(results) == 1 and not results[0].success
    assert password not in results[0].error_message
    assert password not in str(results[0].to_json())
    captured = capsys.readouterr()
    assert password not in captured.out + captured.err
    dns.assert_not_called()
    connect.assert_not_called()


@pytest.mark.parametrize(
    "row",
    [
        None,
        (False, "gpu_cagra", 1),
        (True, "cpu_hnsw", 1),
        (True, "cpu_fallback", 1),
        (True, "brute_force", 1),
    ],
)
def test_cagra_rejects_missing_or_non_gpu_observations(row):
    backend = PostgresBackend({"algo": "pgcuvs_cagra"})
    backend._schema = "isolated_test"
    backend._conn = SimpleNamespace(
        execute=lambda *args: SimpleNamespace(fetchone=lambda: row)
    )
    with pytest.raises(RuntimeError, match="resident GPU CAGRA"):
        backend._gpu_search_count()


def test_cagra_reads_observed_query_count():
    backend = PostgresBackend({"algo": "pgcuvs_cagra"})
    backend._schema = "isolated_test"
    backend._conn = SimpleNamespace(
        execute=lambda *args: SimpleNamespace(
            fetchone=lambda: (True, "gpu_cagra", 42)
        )
    )
    assert backend._gpu_search_count() == 42


def test_cagra_connection_disables_batching_and_records_effective_value(
    monkeypatch,
):
    psycopg = pytest.importorskip("psycopg")
    adapter = pytest.importorskip("pgvector.psycopg")
    conn = Mock()
    conn.execute.return_value.fetchall.return_value = [
        ("vector",),
        ("pg_cuvs",),
    ]
    conn.execute.return_value.fetchone.return_value = ("0",)
    monkeypatch.setattr(psycopg, "connect", Mock(return_value=conn))
    monkeypatch.setattr(adapter, "register_vector", Mock())
    backend = PostgresBackend({"algo": "pgcuvs_cagra"})
    assert backend._connect() is conn
    conn.execute.assert_any_call("SET cuvs.cagra_batch_wait_us = 0")
    conn.execute.assert_any_call("SHOW cuvs.cagra_batch_wait_us")
    assert backend._metadata(index())["cagra_batch_wait_us"] == 0


@pytest.mark.parametrize(
    "algo",
    [
        "pgvector_hnsw",
        "pgvector_ivfflat",
        "pgcuvs_hnsw_import",
        "pgcuvs_cagra",
    ],
)
def test_build_records_resources_and_prewarms_cpu_index(
    algo, monkeypatch, tmp_path
):
    pytest.importorskip("psycopg")
    from cuvs_bench.backends import postgres
    from cuvs_bench.run.data_export import write_results_to_csv

    conn = MagicMock()

    def execute(statement, params=None):
        row = (8192,)
        if isinstance(statement, str):
            if "vram_bytes" in statement:
                row = (65536,)
            elif "pg_prewarm" in statement:
                row = (1,)
        return SimpleNamespace(fetchone=lambda: row)

    conn.execute.side_effect = execute
    backend = PostgresBackend({"algo": algo})
    monkeypatch.setattr(backend, "_connect", lambda: conn)
    monkeypatch.setattr(backend, "_create_index", Mock())
    ticks = iter([10.0, 12.0, 20.0, 23.0])
    monkeypatch.setattr(postgres.time, "perf_counter", lambda: next(ticks))
    ds, idx = dataset(), index(algo, build={"graph_degree": 32})
    if algo == "pgvector_hnsw":
        idx.build_param = {"m": 8}
    elif algo == "pgvector_ivfflat":
        idx.build_param = {"lists": 4}
    built = backend.build(ds, [idx])
    assert built.success, built.error_message
    assert built.build_time_seconds == 2
    assert built.metadata["index_disk_bytes"] == 8192
    if algo == "pgcuvs_cagra":
        assert built.index_size_bytes == 65536
        assert built.metadata["daemon_vram_bytes"] == 65536
        assert built.metadata["index_size_source"] == "daemon_vram_bytes"
        assert not any(
            "pg_prewarm" in str(call.args[0])
            for call in conn.execute.call_args_list
        )
    else:
        assert built.index_size_bytes == 8192
        assert built.metadata["prewarm_method"] == "pg_prewarm_buffer"
        assert built.metadata["prewarm_blocks"] == 1
        assert built.metadata["prewarm_seconds"] == 3
    assert backend._metadata(idx)["index_disk_bytes"] == 8192
    write_results_to_csv([built], ds.name, str(tmp_path), 5, 10000)
    path = tmp_path / ds.name / "result" / "build" / f"{algo},base.csv"
    with path.open() as handle:
        row = next(csv.DictReader(handle))
    assert int(row["index_disk_bytes"]) == 8192
    if algo == "pgcuvs_cagra":
        assert int(row["daemon_vram_bytes"]) == 65536
    else:
        assert row["prewarm_method"] == "pg_prewarm_buffer"


@pytest.mark.postgres
def test_live_prewarm_failure_prevents_search(live_backend, monkeypatch):
    # Exercise failure after CREATE INDEX without changing extension privileges.
    conn = live_backend._connect()
    execute = conn.execute

    def reject_prewarm(statement, params=None):
        if isinstance(statement, str) and "pg_prewarm" in statement:
            raise RuntimeError("prewarm unavailable")
        return execute(statement, params)

    monkeypatch.setattr(conn, "execute", reject_prewarm)
    ds, idx = dataset(), index()
    built = live_backend.build(ds, [idx])
    assert not built.success
    assert "prewarm unavailable" in built.error_message
    result = live_backend.search(ds, [idx], 5)[0]
    assert not result.success
    assert "Build this index" in result.error_message


@pytest.fixture
def live_backend():
    conninfo = os.environ.get("CUVS_BENCH_POSTGRES_DSN")
    if not conninfo:
        pytest.skip("Set CUVS_BENCH_POSTGRES_DSN to a disposable database")
    pytest.importorskip("psycopg")
    backend = PostgresBackend(
        {"algo": "pgvector_hnsw", "conninfo": conninfo, "warmup_queries": 1}
    )
    try:
        yield backend
    finally:
        backend.cleanup()


@pytest.mark.postgres
@pytest.mark.parametrize(
    "algo,build,search",
    [
        (
            "pgvector_hnsw",
            {"m": 8, "ef_construction": 64},
            [{"ef_search": 40}],
        ),
        ("pgvector_ivfflat", {"lists": 4}, [{"probes": 4}]),
    ],
)
def test_live_build_search_and_cleanup(live_backend, algo, build, search):
    ds, idx = dataset(), index(algo, build, search)
    live_backend.config["algo"] = algo
    built = live_backend.build(ds, [idx])
    assert built.success, built.error_message
    assert built.index_size_bytes > 0
    results = live_backend.search(ds, [idx], 5)
    assert len(results) == 1
    result = results[0]
    assert result.success, result.error_message
    assert result.neighbors.shape == (5, 5)
    assert result.distances.shape == (5, 5)
    BenchmarkOrchestrator._finalize_search_result(result, ds, 5)
    assert result.recall == 1.0
    assert result.queries_per_second > 0
    assert result.latency_percentiles["p50_ms"] > 0
    assert result.metadata["measurement_boundary"] == "end_to_end_sql"
    assert built.metadata["prewarm_method"] == "pg_prewarm_buffer"
    block_size = int(
        live_backend._conn.execute("SHOW block_size").fetchone()[0]
    )
    assert (
        built.metadata["prewarm_blocks"] * block_size == built.index_size_bytes
    )
    assert (
        result.metadata["prewarm_blocks"] == built.metadata["prewarm_blocks"]
    )
    schema = live_backend._schema
    conn = live_backend._conn
    live_backend.cleanup()
    assert conn.closed
    import psycopg

    with psycopg.connect(os.environ["CUVS_BENCH_POSTGRES_DSN"]) as check:
        assert (
            check.execute(
                "SELECT 1 FROM pg_namespace WHERE nspname = %s", (schema,)
            ).fetchone()
            is None
        )


@pytest.mark.postgres
def test_live_rejects_seqscan(live_backend):
    from psycopg import sql

    ds, idx = dataset(), index()
    assert live_backend.build(ds, [idx]).success
    live_backend._conn.execute(
        sql.SQL("DROP INDEX {}.ann").format(
            sql.Identifier(live_backend._schema)
        )
    )
    result = live_backend.search(ds, [idx], 5)[0]
    assert not result.success
    assert "Index Scan" in result.error_message


@pytest.mark.postgres
def test_live_rejects_index_loss_after_initial_plan(live_backend, monkeypatch):
    from psycopg import sql

    ds, idx = dataset(), index(search=[{"ef_search": 40}])
    assert live_backend.build(ds, [idx]).success
    original = live_backend._check_plan
    calls = 0

    def drop_after_first_check(statement, params):
        nonlocal calls
        original(statement, params)
        if calls == 0:
            live_backend._conn.execute(
                sql.SQL("DROP INDEX {}.ann").format(
                    sql.Identifier(live_backend._schema)
                )
            )
        calls += 1

    monkeypatch.setattr(live_backend, "_check_plan", drop_after_first_check)
    result = live_backend.search(ds, [idx], 5)[0]
    assert not result.success
    assert "Index Scan" in result.error_message


@pytest.mark.postgres
@pytest.mark.parametrize(
    "after,success", [(0, False), (104, False), (105, True)]
)
def test_cagra_counter_guard_with_cpu_fixture(
    live_backend, monkeypatch, after, success
):
    # Only the telemetry contract is injected; this is not a GPU execution test.
    ds, idx = dataset(), index(search=[{"ef_search": 40}])
    assert live_backend.build(ds, [idx]).success
    live_backend.config["algo"] = "pgcuvs_cagra"
    idx.search_params = [{"candidates": 40}]
    counts = iter([100, after])
    monkeypatch.setattr(
        live_backend, "_gpu_search_count", lambda: next(counts)
    )
    result = live_backend.search(ds, [idx], 5)[0]
    assert result.success is success
    if success:
        assert result.metadata["gpu_queries"] == len(ds.query_vectors)
        assert result.metadata["observed_search_mode"] == "gpu_cagra"
    else:
        assert "GPU query count" in result.error_message


@pytest.mark.postgres
def test_live_missing_neighbors_count_as_misses(live_backend):
    ds = dataset()
    ds.training_vectors = ds.training_vectors[:3]
    idx = index()
    assert live_backend.build(ds, [idx]).success
    result = live_backend.search(ds, [idx], 5)[0]
    assert result.success, result.error_message
    assert np.all(result.neighbors[:, 3:] == -1)
    assert np.all(np.isinf(result.distances[:, 3:]))


@pytest.mark.postgres
def test_live_index_size_survives_csv_export(live_backend, tmp_path):
    from cuvs_bench.run.data_export import write_results_to_csv

    ds, idx = dataset(), index()
    built = live_backend.build(ds, [idx])
    assert built.success, built.error_message
    searched = live_backend.search(ds, [idx], 5)[0]
    assert searched.success, searched.error_message
    write_results_to_csv([built, searched], ds.name, str(tmp_path), 5, 10000)
    path = tmp_path / ds.name / "result" / "build" / "pgvector_hnsw,base.csv"
    with path.open() as handle:
        row = next(csv.DictReader(handle))
    assert int(row["index_size_bytes"]) == built.index_size_bytes > 0
    assert row["index_size_source"] == "pg_relation_size"
    assert int(row["index_disk_bytes"]) == built.index_size_bytes
    assert row["prewarm_method"] == "pg_prewarm_buffer"
    path = (
        tmp_path
        / ds.name
        / "result"
        / "search"
        / "pgvector_hnsw,base,k5,bs10000,raw.csv"
    )
    with path.open() as handle:
        row = next(csv.DictReader(handle))
    assert int(row["index_disk_bytes"]) == built.index_size_bytes
    assert int(row["prewarm_blocks"]) == built.metadata["prewarm_blocks"]


def test_orchestrator_recomputes_recall_from_neighbor_ids():
    ds = dataset()
    result = SearchResult(
        neighbors=ds.groundtruth_neighbors.copy(),
        distances=np.zeros((5, 5)),
        search_time_ms=1,
        queries_per_second=5000,
        recall=0,
        algorithm="pgvector_hnsw",
        search_params=[{}],
    )
    BenchmarkOrchestrator._finalize_search_result(result, ds, 5)
    assert result.recall == 1


@pytest.mark.postgres
@pytest.mark.postgres_gpu
@pytest.mark.parametrize("algo", ["pgcuvs_cagra", "pgcuvs_hnsw_import"])
def test_real_gpu_build_search_and_hnsw_export(algo):
    conninfo = os.environ.get("CUVS_BENCH_POSTGRES_GPU_DSN")
    if not conninfo:
        pytest.skip("Set CUVS_BENCH_POSTGRES_GPU_DSN on a real GPU server")
    pytest.importorskip("psycopg")
    from psycopg.conninfo import make_conninfo

    # Exercise the connection override with a valid conflicting startup value.
    if algo == "pgcuvs_cagra":
        conninfo = make_conninfo(
            conninfo,
            options="-c cuvs.cpu_hnsw_fallback=on -c cuvs.cagra_batch_wait_us=1000",
        )
    rng = np.random.default_rng(20261016)
    base = rng.normal(size=(4096, 32)).astype(np.float32)
    queries = base[:16].copy()
    distances = ((queries[:, None] - base[None]) ** 2).sum(axis=2)
    ds = Dataset(
        "gpu-smoke",
        training_vectors=base,
        query_vectors=queries,
        groundtruth_neighbors=np.argsort(distances, axis=1)[:, :10],
    )
    build = {
        "graph_degree": 32,
        "intermediate_graph_degree": 128,
        "build_algo": "ivf_pq",
    }
    if algo == "pgcuvs_hnsw_import":
        build["mode"] = "nsw"
    search = (
        [{"candidates": 128}]
        if algo == "pgcuvs_cagra"
        else [{"ef_search": 128}]
    )
    idx = index(algo, build, search)
    backend = PostgresBackend(
        {
            "algo": algo,
            "conninfo": conninfo,
            "warmup_queries": 4,
        }
    )
    try:
        built = backend.build(ds, [idx])
        assert built.success, built.error_message
        assert built.index_size_bytes > 0
        assert built.metadata["index_disk_bytes"] > 0
        source = f"{backend._schema}.source_graph"
        if algo == "pgcuvs_cagra":
            assert built.metadata["cagra_batch_wait_us"] == 0
            assert (
                built.metadata["daemon_vram_bytes"] == built.index_size_bytes
            )
            assert (
                backend._conn.execute(
                    "SHOW cuvs.cpu_hnsw_fallback"
                ).fetchone()[0]
                == "off"
            )
        else:
            before = backend._conn.execute(
                "SELECT search_count FROM pg_stat_gpu_search "
                "WHERE index_oid=%s::regclass",
                (source,),
            ).fetchone()[0]
        result = backend.search(ds, [idx], 10)[0]
        assert result.success, result.error_message
        BenchmarkOrchestrator._finalize_search_result(result, ds, 10)
        assert result.neighbors.shape == (16, 10)
        # A broad sanity bound catches broken IDs, not production ANN quality.
        assert result.recall > 0.8
        if algo == "pgcuvs_cagra":
            assert result.metadata["cagra_batch_wait_us"] == 0
            assert (
                result.metadata["index_disk_bytes"]
                == built.metadata["index_disk_bytes"]
            )
            assert result.metadata["observed_search_mode"] == "gpu_cagra"
            assert result.metadata["gpu_queries"] == len(queries)
        else:
            assert (
                result.metadata["prewarm_blocks"]
                == built.metadata["prewarm_blocks"]
            )
            after = backend._conn.execute(
                "SELECT search_count FROM pg_stat_gpu_search "
                "WHERE index_oid=%s::regclass",
                (source,),
            ).fetchone()[0]
            assert after == before
            assert (
                backend._conn.execute("SHOW enable_cuvs").fetchone()[0]
                == "off"
            )
    finally:
        backend.cleanup()
