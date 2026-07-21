#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
"""
Tests for the OpenSearch benchmark backend.
"""

import os
from urllib.parse import urlparse

import numpy as np
import pytest
import requests

pytest.importorskip("opensearchpy")

from cuvs_bench.backends.base import Dataset
from cuvs_bench.backends.opensearch import (
    OpenSearchBackend,
    OpenSearchConfigLoader,
)
from cuvs_bench.orchestrator.config_loaders import IndexConfig


def _make_backend(config_overrides: dict = None) -> OpenSearchBackend:
    """Backend with no network requirement so pre-flight passes without a server."""
    config = {
        "name": "test_index",
        "index_name": "test_index",
        "engine": "faiss",
        "algo": "opensearch_faiss_hnsw",
        **(config_overrides or {}),
    }
    return OpenSearchBackend(config)


def _make_dataset(
    n_base: int = 10, n_queries: int = 5, dims: int = 4, k: int = 10
) -> Dataset:
    rng = np.random.default_rng(0)
    base = rng.random((n_base, dims), dtype=np.float32)
    queries = rng.random((n_queries, dims), dtype=np.float32)
    dists = np.sum((queries[:, None, :] - base[None, :, :]) ** 2, axis=2)
    groundtruth = np.argsort(dists, axis=1)[:, :k].astype(np.int32)
    return Dataset(
        name="test",
        training_vectors=base,
        query_vectors=queries,
        groundtruth_neighbors=groundtruth,
    )


def _make_index_cfg(search_params: list = None) -> IndexConfig:
    return IndexConfig(
        name="test_index",
        algo="opensearch_faiss_hnsw",
        build_param={"m": 16, "ef_construction": 100},
        search_params=search_params or [{"ef_search": 50}, {"ef_search": 100}],
        file="",
    )


def _cleanup_backend(backend: OpenSearchBackend, index_name: str) -> None:
    """Delete the test index and close the client connection."""
    try:
        if backend._client.indices.exists(index=index_name):
            backend._client.indices.delete(index=index_name)
    except Exception:
        pass
    backend.cleanup()


@pytest.fixture(scope="session")
def opensearch_url():
    """Skip integration tests when no live OpenSearch node is reachable."""
    url = os.environ.get("OPENSEARCH_URL", "http://localhost:9200").rstrip("/")
    try:
        requests.get(f"{url}/_cluster/health", timeout=2).raise_for_status()
    except Exception:
        pytest.skip("No OpenSearch node reachable")
    return url


@pytest.fixture
def live_backend(opensearch_url):
    """Backend connected to a live OpenSearch node; cleans up the test index on teardown."""
    parsed = urlparse(opensearch_url)
    index_name = "cuvs_test_index"
    backend = OpenSearchBackend(
        {
            "name": index_name,
            "index_name": index_name,
            "engine": "faiss",
            "algo": "opensearch_faiss_hnsw",
            "host": parsed.hostname,
            "port": parsed.port or 9200,
            "use_ssl": parsed.scheme == "https",
            "verify_certs": False,
            "requires_network": True,
        }
    )
    try:
        yield backend
    finally:
        _cleanup_backend(backend, index_name)


@pytest.fixture
def config_dir(tmp_path):
    """Config directory with a minimal dataset and algo definition."""
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / "datasets.yaml").write_text(
        """\
- name: test-ds
  distance: euclidean
  dims: 4
"""
    )
    (tmp_path / "algos").mkdir()
    (tmp_path / "algos" / "opensearch_faiss_hnsw.yaml").write_text(
        """\
name: opensearch_faiss_hnsw
groups:
  test:
    build:
      m: [16, 32]
      ef_construction: [100, 200]
    search:
      ef_search: [50, 100]
"""
    )
    return tmp_path


class TestOpenSearchConfigLoader:
    def test_load_produces_correct_configs(self, config_dir):
        loader = OpenSearchConfigLoader(config_path=config_dir)
        dataset_config, benchmark_configs = loader.load(
            dataset="test-ds", dataset_path="/data", groups="test"
        )

        assert dataset_config.name == "test-ds"
        assert dataset_config.distance == "euclidean"
        # m=[16,32] x ef_construction=[100,200] = 4 build combos
        assert len(benchmark_configs) == 4
        bc = benchmark_configs[0]
        assert bc.backend_config["engine"] == "faiss"
        assert len(bc.indexes[0].search_params) == 2  # ef_search: [50, 100]

    def test_load_forwards_remote_build_kwargs(self, config_dir):
        loader = OpenSearchConfigLoader(config_path=config_dir)
        _, configs = loader.load(
            dataset="test-ds",
            dataset_path="/data",
            remote_index_build=True,
            remote_build_size_min="2kb",
            remote_build_timeout=123,
            remote_build_s3_endpoint="http://s3:9000",
        )

        bc = configs[0].backend_config
        assert bc["remote_index_build"] is True
        assert bc["remote_build_size_min"] == "2kb"
        assert bc["remote_build_timeout"] == 123
        assert "remote_build_s3_endpoint" not in bc


class TestOpenSearchBackend:
    def test_build_dry_run(self):
        backend = _make_backend()
        result = backend.build(
            _make_dataset(), [_make_index_cfg()], dry_run=True
        )
        assert result.success
        assert result.index_path == backend.config["index_name"]

    def test_search_dry_run(self):
        result = _make_backend().search(
            _make_dataset(), [_make_index_cfg()], k=3, dry_run=True
        )
        assert result.success
        assert len(result.search_params) == 2

    def test_remote_build_requires_faiss_engine(self):
        backend = _make_backend({"engine": "lucene"})
        with pytest.raises(ValueError, match="faiss engine"):
            backend._build_index_mapping(
                dims=4,
                engine="lucene",
                space_type="l2",
                build_param={},
                remote_index_build=True,
            )

    def test_remote_build_uses_opensearch_default_size_when_unspecified(self):
        backend = _make_backend()
        mapping = backend._build_index_mapping(
            dims=4,
            engine="faiss",
            space_type="l2",
            build_param={},
            remote_index_build=True,
        )

        settings = mapping["settings"]["index"]
        assert settings["knn.remote_index_build.enabled"] is True
        assert "knn.remote_index_build.size.min" not in settings

    def test_remote_build_size_min_overrides_default(self):
        backend = _make_backend()
        mapping = backend._build_index_mapping(
            dims=4,
            engine="faiss",
            space_type="l2",
            build_param={},
            remote_index_build=True,
            remote_build_size_min="2kb",
        )

        settings = mapping["settings"]["index"]
        assert settings["knn.remote_index_build.size.min"] == "2kb"

    def test_wait_for_remote_build_raises_on_failure_count(self):
        backend = _make_backend()
        initial_stats = {
            "build_request_success_count": 0,
            "build_request_failure_count": 0,
            "index_build_success_count": 0,
            "index_build_failure_count": 0,
            "remote_index_build_current_merge_operations": 0,
            "remote_index_build_current_flush_operations": 0,
        }
        backend._get_knn_remote_build_stats = lambda: {
            "build_request_success_count": 1,
            "build_request_failure_count": 0,
            "index_build_success_count": 0,
            "index_build_failure_count": 1,
            "remote_index_build_current_merge_operations": 0,
            "remote_index_build_current_flush_operations": 0,
        }

        with pytest.raises(RuntimeError, match="GPU build failed"):
            backend._wait_for_remote_build(
                initial_stats=initial_stats,
                timeout=1,
                poll_interval=0,
            )

    def test_wait_for_remote_build_raises_when_no_build_observed(self):
        backend = _make_backend()
        initial_stats = {
            "build_request_success_count": 0,
            "build_request_failure_count": 0,
            "index_build_success_count": 0,
            "index_build_failure_count": 0,
            "remote_index_build_current_merge_operations": 0,
            "remote_index_build_current_flush_operations": 0,
        }
        backend._get_knn_remote_build_stats = lambda: {
            "build_request_success_count": 0,
            "build_request_failure_count": 0,
            "index_build_success_count": 0,
            "index_build_failure_count": 0,
            "remote_index_build_current_merge_operations": 0,
            "remote_index_build_current_flush_operations": 0,
        }

        with pytest.raises(RuntimeError, match="No remote GPU build"):
            backend._wait_for_remote_build(
                initial_stats=initial_stats,
                timeout=1,
                poll_interval=0,
                no_activity_timeout=0,
            )

    def test_wait_for_remote_build_waits_for_all_submitted_builds(self):
        backend = _make_backend()
        initial_stats = {
            "build_request_success_count": 0,
            "build_request_failure_count": 0,
            "index_build_success_count": 0,
            "index_build_failure_count": 0,
            "remote_index_build_current_merge_operations": 0,
            "remote_index_build_current_flush_operations": 0,
        }
        stats_sequence = iter(
            [
                {
                    "build_request_success_count": 2,
                    "build_request_failure_count": 0,
                    "index_build_success_count": 1,
                    "index_build_failure_count": 0,
                    "remote_index_build_current_merge_operations": 0,
                    "remote_index_build_current_flush_operations": 1,
                },
                {
                    "build_request_success_count": 3,
                    "build_request_failure_count": 0,
                    "index_build_success_count": 2,
                    "index_build_failure_count": 0,
                    "remote_index_build_current_merge_operations": 0,
                    "remote_index_build_current_flush_operations": 1,
                },
                {
                    "build_request_success_count": 3,
                    "build_request_failure_count": 0,
                    "index_build_success_count": 3,
                    "index_build_failure_count": 0,
                    "remote_index_build_current_merge_operations": 0,
                    "remote_index_build_current_flush_operations": 0,
                },
            ]
        )
        backend._get_knn_remote_build_stats = lambda: next(stats_sequence)

        backend._wait_for_remote_build(
            initial_stats=initial_stats,
            timeout=1,
            poll_interval=0,
        )

    def test_search_fails_without_query_vectors(self):
        dataset = Dataset(
            name="empty",
            training_vectors=np.empty((0, 4), dtype=np.float32),
            query_vectors=np.empty((0, 4), dtype=np.float32),
        )
        result = _make_backend().search(dataset, [_make_index_cfg()], k=3)
        assert not result.success
        assert "No query vectors" in result.error_message


@pytest.fixture(scope="session")
def remote_build_env(opensearch_url):
    """
    Skip remote index build tests if required env vars aren't set or services aren't reachable.

    Required environment variables:
      BUILDER_URL      URL of the remote index builder service
      S3_BUCKET        Bucket name used by OpenSearch for vector staging

    Optional environment variables:
      S3_ENDPOINT      Custom S3 endpoint URL (omit to use real AWS S3)

    S3 credentials are not passed through the OpenSearchBackend config or
    snapshot repository registration. The OpenSearch process/container must
    already have access to them, for example through its environment or
    OpenSearch keystore.
    """
    builder_url = os.environ.get("BUILDER_URL")
    # S3_ENDPOINT is optional — omit it (or leave unset) to use real AWS S3.
    s3_endpoint = os.environ.get("S3_ENDPOINT") or None
    s3_bucket = os.environ.get("S3_BUCKET")
    s3_region = (
        os.environ.get("S3_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "us-east-1"
    )
    s3_prefix = "knn-indexes/"

    missing = [
        name
        for name, val in {
            "BUILDER_URL": builder_url,
            "S3_BUCKET": s3_bucket,
        }.items()
        if not val
    ]
    if missing:
        pytest.skip(
            f"Remote index build tests require environment variables: {', '.join(missing)}"
        )

    try:
        requests.get(builder_url, timeout=2)
    except requests.exceptions.RequestException:
        pytest.skip(f"Remote index builder not reachable at {builder_url}")

    if s3_endpoint is not None:
        try:
            requests.get(s3_endpoint, timeout=2)
        except requests.exceptions.RequestException:
            pytest.skip(f"S3 endpoint not reachable at {s3_endpoint}")

    # Register the S3 snapshot repo and enable remote index build
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    repo_name = "vector-repo"
    remote_build_setting_keys = (
        "knn.remote_index_build.enabled",
        "knn.remote_index_build.repository",
        "knn.remote_index_build.service.endpoint",
    )

    cluster_settings_response = session.get(
        f"{opensearch_url}/_cluster/settings"
    )
    cluster_settings_response.raise_for_status()
    cluster_settings = cluster_settings_response.json()
    initial_remote_build_settings = {
        key: cluster_settings.get("persistent", {}).get(key)
        for key in remote_build_setting_keys
    }

    repo_response = session.get(f"{opensearch_url}/_snapshot/{repo_name}")
    if repo_response.status_code == 404:
        initial_snapshot_repo = None
    else:
        repo_response.raise_for_status()
        initial_snapshot_repo = repo_response.json().get(repo_name)

    snapshot_settings = {
        "bucket": s3_bucket,
        "base_path": s3_prefix.rstrip("/"),
        "region": s3_region,
    }
    if s3_endpoint is not None:
        snapshot_settings["endpoint"] = s3_endpoint

    session.put(
        f"{opensearch_url}/_snapshot/{repo_name}",
        json={
            "type": "s3",
            "settings": snapshot_settings,
        },
    ).raise_for_status()

    session.put(
        f"{opensearch_url}/_cluster/settings",
        json={
            "persistent": {
                "knn.remote_index_build.enabled": True,
                "knn.remote_index_build.repository": repo_name,
                "knn.remote_index_build.service.endpoint": builder_url,
            }
        },
    ).raise_for_status()

    try:
        yield {
            "s3_endpoint": s3_endpoint,
            "s3_bucket": s3_bucket,
            "s3_prefix": s3_prefix,
        }
    finally:
        try:
            session.put(
                f"{opensearch_url}/_cluster/settings",
                json={
                    "persistent": initial_remote_build_settings,
                },
            )
            if initial_snapshot_repo is None:
                session.delete(f"{opensearch_url}/_snapshot/{repo_name}")
            else:
                session.put(
                    f"{opensearch_url}/_snapshot/{repo_name}",
                    json=initial_snapshot_repo,
                )
        except requests.exceptions.RequestException:
            pass
        finally:
            session.close()


@pytest.fixture
def live_remote_build_backend(opensearch_url, remote_build_env):
    """Backend with remote_index_build=True pointing at a configured remote build stack."""
    parsed = urlparse(opensearch_url)
    index_name = "cuvs_test_remote_index"
    backend = OpenSearchBackend(
        {
            "name": index_name,
            "index_name": index_name,
            "engine": "faiss",
            "algo": "opensearch_faiss_hnsw",
            "host": parsed.hostname,
            "port": parsed.port or 9200,
            "use_ssl": parsed.scheme == "https",
            "verify_certs": False,
            "requires_network": True,
            "remote_index_build": True,
            # Keep the tiny integration dataset above the remote build threshold.
            "remote_build_size_min": "1kb",
        }
    )
    try:
        yield backend
    finally:
        _cleanup_backend(backend, index_name)


@pytest.mark.opensearch
class TestOpenSearchBackendIntegration:
    def test_build_and_search(self, live_backend):
        k = 3
        dataset = _make_dataset(n_base=100, n_queries=10, dims=4, k=k)
        idx = _make_index_cfg(search_params=[{"ef_search": 50}])

        build_result = live_backend.build(dataset, [idx], force=True)
        assert build_result.success
        assert build_result.build_time_seconds > 0
        assert build_result.index_size_bytes > 0

        search_result = live_backend.search(dataset, [idx], k=k)
        assert search_result.success
        assert search_result.recall == 0.0
        assert search_result.queries_per_second > 0
        assert search_result.neighbors.shape == (10, k)
        assert len(search_result.metadata["per_search_param_results"]) == 1


@pytest.mark.opensearch
class TestOpenSearchRemoteIndexBuildIntegration:
    def test_remote_build_and_search(self, live_remote_build_backend):
        k = 3
        dataset = _make_dataset(n_base=100, n_queries=10, dims=4, k=k)
        idx = _make_index_cfg(search_params=[{"ef_search": 50}])

        build_result = live_remote_build_backend.build(
            dataset, [idx], force=True
        )
        assert build_result.success
        assert build_result.build_time_seconds > 0
        assert build_result.metadata["remote_index_build"] is True

        search_result = live_remote_build_backend.search(dataset, [idx], k=k)
        assert search_result.success
        assert search_result.recall == 0.0
        assert search_result.queries_per_second > 0
        assert search_result.neighbors.shape == (10, k)
