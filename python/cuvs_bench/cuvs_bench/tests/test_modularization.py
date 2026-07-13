#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
"""
Smoke tests for cuvs-bench modularization (optional deps, entry points, lazy loading).

These tests verify the plugin infrastructure without requiring full backend implementations.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cuvs_bench.backends.base import (
    BenchmarkBackend,
    BuildResult,
    Dataset,
    SearchResult,
)
from cuvs_bench.backends.registry import (
    get_backend_class,
    get_config_loader,
    get_registry,
    list_backends,
    list_config_loaders,
    register_backend,
    register_config_loader,
    unregister_config_loader,
)
from cuvs_bench.orchestrator.config_loaders import ConfigLoader, IndexConfig


class TestModularizationSmoke:
    """Smoke tests for optional backend loading and error handling."""

    def test_cpp_gbench_available(self):
        """cpp_gbench (built-in) should always be available."""
        backends = list_backends()
        assert "cpp_gbench" in backends
        cls = get_backend_class("cpp_gbench")
        assert cls is not None

    def test_cpp_gbench_config_loader_available(self):
        """cpp_gbench config loader should always be registered."""
        loaders = list_config_loaders()
        assert "cpp_gbench" in loaders
        loader_cls = get_config_loader("cpp_gbench")
        assert loader_cls is not None

    def test_elastic_without_extra_raises_clear_error(self):
        """Requesting elastic without [elastic] installed raises helpful error."""
        try:
            import elasticsearch  # noqa: F401

            pytest.skip(
                "elasticsearch is installed; cannot test missing-plugin path"
            )
        except ImportError:
            pass
        import importlib.metadata

        all_eps = importlib.metadata.entry_points()
        if hasattr(all_eps, "select"):
            eps = list(
                all_eps.select(group="cuvs_bench.backends", name="elastic")
            )
        else:
            eps = [
                e
                for e in all_eps.get("cuvs_bench.backends", [])
                if e.name == "elastic"
            ]
        if eps:
            pytest.skip(
                "cuvs-bench-elastic is installed; cannot test missing-plugin path"
            )

        with pytest.raises((ImportError, ValueError)) as exc_info:
            get_backend_class("elastic")

        msg = str(exc_info.value)
        assert "elastic" in msg.lower()

    def test_elastic_config_loader_without_extra_raises_clear_error(self):
        """Requesting elastic config loader without [elastic] raises helpful error."""
        try:
            import elasticsearch  # noqa: F401

            pytest.skip(
                "elasticsearch is installed; cannot test missing-plugin path"
            )
        except ImportError:
            pass
        import importlib.metadata

        all_eps = importlib.metadata.entry_points()
        if hasattr(all_eps, "select"):
            eps = list(
                all_eps.select(
                    group="cuvs_bench.config_loaders", name="elastic"
                )
            )
        else:
            eps = [
                e
                for e in all_eps.get("cuvs_bench.config_loaders", [])
                if e.name == "elastic"
            ]
        if eps:
            pytest.skip(
                "cuvs-bench-elastic is installed; cannot test missing-plugin path"
            )

        with pytest.raises((ImportError, ValueError)) as exc_info:
            get_config_loader("elastic")

        msg = str(exc_info.value)
        assert "elastic" in msg.lower()

    def test_unknown_backend_raises_value_error(self):
        """Requesting unknown backend raises ValueError with available backends."""
        with pytest.raises(ValueError) as exc_info:
            get_backend_class("nonexistent_backend_xyz")

        msg = str(exc_info.value)
        assert "nonexistent_backend_xyz" in msg
        assert "cpp_gbench" in msg

    def test_unknown_config_loader_raises_value_error(self):
        """Requesting unknown config loader raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_config_loader("nonexistent_loader_xyz")

        msg = str(exc_info.value)
        assert "nonexistent_loader_xyz" in msg

    def test_orchestrator_cpp_gbench_no_regression(self):
        """Initializing BenchmarkOrchestrator with cpp_gbench should work."""
        from cuvs_bench.orchestrator import BenchmarkOrchestrator

        assert "cpp_gbench" in BenchmarkOrchestrator.available_backends()
        orch = BenchmarkOrchestrator(backend_type="cpp_gbench")
        assert orch.backend_type == "cpp_gbench"
        assert orch.backend_class is not None
        assert orch.config_loader is not None

    def test_orchestrator_respects_authoritative_backend_recall(self):
        """Backends can mark their own recall as authoritative."""
        from cuvs_bench.orchestrator.orchestrator import _should_compute_recall

        result = SearchResult(
            neighbors=np.array([[1, 2, 3]], dtype=np.int64),
            distances=np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
            search_time_ms=1.0,
            queries_per_second=1.0,
            recall=0.5,
            algorithm="elastic_hnsw",
            search_params=[{"num_candidates": 100}],
            metadata={"recall_is_authoritative": True},
            success=True,
        )

        assert not _should_compute_recall(result)


class TestPluginLoaderMocked:
    """
    Tests using mocked entry points (NAT-style).

    These tests do not require elasticsearch or real plugins.
    """

    _MOCK_PLUGIN_NAME = "mock_plugin_test"

    @staticmethod
    def _make_mock_register():
        """Create a register function that registers a minimal stub backend and loader."""

        class StubBackend(BenchmarkBackend):
            def build(self, dataset, indexes, force=False, dry_run=False):
                return BuildResult(
                    index_path="",
                    build_time_seconds=0,
                    index_size_bytes=0,
                    algorithm="stub",
                    build_params={},
                    success=True,
                )

            def search(
                self,
                dataset,
                indexes,
                k=10,
                batch_size=10000,
                mode="latency",
                force=False,
                search_threads=None,
                dry_run=False,
            ):
                import numpy as np

                return SearchResult(
                    neighbors=np.empty((0, k)),
                    distances=np.empty((0, k)),
                    search_time_ms=0,
                    queries_per_second=0,
                    recall=0,
                    algorithm="stub",
                    search_params=[],
                    success=True,
                )

        class StubConfigLoader(ConfigLoader):
            @property
            def backend_type(self):
                return TestPluginLoaderMocked._MOCK_PLUGIN_NAME

            def load(self, **kwargs):
                raise NotImplementedError("Stub loader")

        def register():
            register_backend(
                TestPluginLoaderMocked._MOCK_PLUGIN_NAME, StubBackend
            )
            register_config_loader(
                TestPluginLoaderMocked._MOCK_PLUGIN_NAME, StubConfigLoader
            )

        return register

    def test_valid_plugin_loads_via_mock_entry_point(self):
        """Mock entry point: valid plugin registers and is discoverable."""
        mock_register = self._make_mock_register()
        mock_ep = MagicMock()
        mock_ep.name = self._MOCK_PLUGIN_NAME
        mock_ep.load.return_value = mock_register

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch(
            "cuvs_bench.backends.registry.importlib.metadata.entry_points",
            return_value=mock_eps,
        ):
            cls = get_backend_class(self._MOCK_PLUGIN_NAME)
            assert cls is not None

        loader_cls = get_config_loader(self._MOCK_PLUGIN_NAME)
        assert loader_cls is not None

        # Cleanup
        get_registry().unregister(self._MOCK_PLUGIN_NAME)
        unregister_config_loader(self._MOCK_PLUGIN_NAME)

    def test_config_loader_prefers_loader_entry_point_group(self):
        """Config loader lookup handles plugins split across entry-point groups."""

        class StubSplitBackend(BenchmarkBackend):
            @property
            def algo(self):
                return "stub_split"

            def build(self, dataset, indexes, force=False, dry_run=False):
                return BuildResult(
                    index_path="",
                    build_time_seconds=0,
                    index_size_bytes=0,
                    algorithm=self.algo,
                    build_params={},
                    success=True,
                )

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
                return SearchResult(
                    neighbors=np.empty((0, k)),
                    distances=np.empty((0, k)),
                    search_time_ms=0,
                    queries_per_second=0,
                    recall=0,
                    algorithm=self.algo,
                    search_params=[],
                    success=True,
                )

        class StubSplitLoader(ConfigLoader):
            @property
            def backend_type(self):
                return TestPluginLoaderMocked._MOCK_PLUGIN_NAME

            def load(self, **kwargs):
                raise NotImplementedError("Stub split loader")

        def register_backend_only():
            register_backend(
                TestPluginLoaderMocked._MOCK_PLUGIN_NAME,
                StubSplitBackend,
            )

        def register_loader_only():
            register_config_loader(
                TestPluginLoaderMocked._MOCK_PLUGIN_NAME,
                StubSplitLoader,
            )

        mock_backend_ep = MagicMock()
        mock_backend_ep.name = self._MOCK_PLUGIN_NAME
        mock_backend_ep.load.return_value = register_backend_only

        mock_loader_ep = MagicMock()
        mock_loader_ep.name = self._MOCK_PLUGIN_NAME
        mock_loader_ep.load.return_value = register_loader_only

        def mock_entry_points(*args, **kwargs):
            group = kwargs.get("group")
            if group == "cuvs_bench.backends":
                return [mock_backend_ep]
            if group == "cuvs_bench.config_loaders":
                return [mock_loader_ep]
            return []

        with patch(
            "cuvs_bench.backends.registry.importlib.metadata.entry_points",
            side_effect=mock_entry_points,
        ):
            loader_cls = get_config_loader(self._MOCK_PLUGIN_NAME)
            assert loader_cls is StubSplitLoader

        get_registry().unregister(self._MOCK_PLUGIN_NAME)
        unregister_config_loader(self._MOCK_PLUGIN_NAME)

    def test_import_error_with_elasticsearch_message_raises_helpful_error(
        self,
    ):
        """Mock entry point raising ImportError(elasticsearch) -> our install message."""
        # Ensure elastic is not in registry (e.g. from TestElasticWithExtraInstalled)
        registry = get_registry()
        if "elastic" in registry._backends:
            registry.unregister("elastic")
            unregister_config_loader("elastic")

        mock_ep = MagicMock()
        mock_ep.name = "elastic"
        mock_ep.load.side_effect = ImportError(
            "No module named 'elasticsearch'"
        )

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch(
            "cuvs_bench.backends.registry.importlib.metadata.entry_points",
            return_value=mock_eps,
        ):
            with pytest.raises(ImportError) as exc_info:
                get_backend_class("elastic")

        msg = str(exc_info.value)
        assert "pip install cuvs-bench[elastic]" in msg

    def test_import_error_with_opensearch_message_raises_helpful_error(
        self,
    ):
        """Mock entry point raising ImportError(opensearchpy) -> our install message."""
        registry = get_registry()
        if "opensearch" in registry._backends:
            registry.unregister("opensearch")
            unregister_config_loader("opensearch")

        mock_ep = MagicMock()
        mock_ep.name = "opensearch"
        mock_ep.load.side_effect = ImportError(
            "No module named 'opensearchpy'"
        )

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch(
            "cuvs_bench.backends.registry.importlib.metadata.entry_points",
            return_value=mock_eps,
        ):
            with pytest.raises(ImportError) as exc_info:
                get_backend_class("opensearch")

        msg = str(exc_info.value)
        assert "pip install cuvs-bench[opensearch]" in msg

    def test_import_error_unrelated_propagates(self):
        """Mock entry point: unrelated ImportError propagates unchanged."""
        mock_ep = MagicMock()
        mock_ep.name = "other_plugin"
        mock_ep.load.side_effect = ImportError(
            "No module named 'something_else'"
        )

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch(
            "cuvs_bench.backends.registry.importlib.metadata.entry_points",
            return_value=mock_eps,
        ):
            with pytest.raises(ImportError) as exc_info:
                get_backend_class("other_plugin")

        assert "something_else" in str(exc_info.value)

    def test_unexpected_error_propagates(self):
        """Mock entry point: RuntimeError propagates."""
        mock_ep = MagicMock()
        mock_ep.name = "broken_plugin"
        mock_ep.load.side_effect = RuntimeError("Plugin crashed")

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch(
            "cuvs_bench.backends.registry.importlib.metadata.entry_points",
            return_value=mock_eps,
        ):
            with pytest.raises(RuntimeError) as exc_info:
                get_backend_class("broken_plugin")

        assert "Plugin crashed" in str(exc_info.value)

    def test_no_entry_point_for_name_raises_value_error(self):
        """Mock entry point: no plugin for requested name -> ValueError."""
        mock_eps = MagicMock()
        mock_eps.select.return_value = []  # No matching entry points

        with patch(
            "cuvs_bench.backends.registry.importlib.metadata.entry_points",
            return_value=mock_eps,
        ):
            with pytest.raises(ValueError) as exc_info:
                get_backend_class("nonexistent_mock_xyz")

        msg = str(exc_info.value)
        assert "nonexistent_mock_xyz" in msg
        assert "cpp_gbench" in msg

    def test_unknown_optional_backend_includes_install_hint(self):
        """Missing optional backend names include install hints in the error."""
        mock_eps = MagicMock()
        mock_eps.select.return_value = []

        with patch(
            "cuvs_bench.backends.registry.importlib.metadata.entry_points",
            return_value=mock_eps,
        ):
            with pytest.raises(ValueError) as exc_info:
                get_backend_class("opensearch")

        assert "pip install cuvs-bench[opensearch]" in str(exc_info.value)


def _elasticsearch_installed():
    try:
        import elasticsearch  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _elasticsearch_installed(),
    reason="Requires pip install cuvs-bench[elastic]",
)
class TestElasticWithExtraInstalled:
    """Tests that run only when [elastic] extra is installed."""

    @pytest.fixture(autouse=True)
    def _ensure_elastic_registered(self):
        """Re-register elastic (may have been unregistered by other tests)."""
        registry = get_registry()
        if "elastic" not in registry._backends:
            try:
                from cuvs_bench.backends.elasticsearch import register

                register()
            except ImportError:
                pass
        yield

    def test_elastic_plugin_loads(self):
        """Elastic backend and config loader load when elasticsearch is installed."""
        assert get_backend_class("elastic") is not None
        assert get_config_loader("elastic") is not None

    def test_elastic_config_loader_tune_mode_returns_single_config(self):
        """Tune mode produces one BenchmarkConfig with Optuna-suggested params."""
        loader_cls = get_config_loader("elastic")
        loader = loader_cls()
        dataset_config, benchmark_configs = loader.load(
            dataset="glove-50-angular",
            dataset_path="",
            algorithms="elastic_hnsw",
            _tune_mode=True,
            _tune_build_params={"m": 24, "ef_construction": 150},
            _tune_search_params={"num_candidates": 120},
        )
        assert len(benchmark_configs) == 1
        config = benchmark_configs[0]
        assert config.indexes[0].algo == "elastic_hnsw"
        assert config.indexes[0].build_param["m"] == 24
        assert config.indexes[0].build_param["ef_construction"] == 150
        assert config.indexes[0].build_param["type"] == "hnsw"
        assert config.indexes[0].search_params[0]["num_candidates"] == 120
        assert config.backend_config["name"].startswith("elastic_hnsw_tune")

    def test_elastic_config_loader_sweep_mode_returns_multiple_configs(self):
        """Sweep mode produces multiple BenchmarkConfigs from param cartesian product."""
        loader_cls = get_config_loader("elastic")
        loader = loader_cls()
        dataset_config, benchmark_configs = loader.load(
            dataset="glove-50-angular",
            dataset_path="",
            algorithms="elastic_hnsw",
            groups="test",
        )
        assert len(benchmark_configs) >= 1
        config = benchmark_configs[0]
        assert config.indexes[0].algo == "elastic_hnsw"
        assert "m" in config.indexes[0].build_param
        assert "num_candidates" in config.indexes[0].search_params[0]

    def test_elastic_config_loader_dataset_not_found_raises(self):
        """Config loader raises ValueError for unknown dataset."""
        loader_cls = get_config_loader("elastic")
        loader = loader_cls()
        with pytest.raises(ValueError, match="Could not find"):
            loader.load(dataset="nonexistent_dataset_xyz", dataset_path="")

    def test_elastic_config_loader_group_not_found_raises(self):
        """Config loader raises ValueError for unknown algorithm group."""
        loader_cls = get_config_loader("elastic")
        loader = loader_cls()
        with pytest.raises(ValueError, match="find elastic groups"):
            loader.load(
                dataset="glove-50-angular",
                dataset_path="",
                algorithms="elastic_hnsw",
                groups="nonexistent_group_xyz",
            )

    def test_elastic_config_loader_normalizes_variant_algorithms(self):
        """Variant elastic_* names resolve through the shared elastic config."""
        loader_cls = get_config_loader("elastic")
        loader = loader_cls()
        _, benchmark_configs = loader.load(
            dataset="glove-50-angular",
            dataset_path="",
            algorithms="elastic_int8_hnsw",
            groups="test",
        )

        assert len(benchmark_configs) == 1
        config = benchmark_configs[0]
        assert config.indexes[0].algo == "elastic_int8_hnsw"
        assert config.indexes[0].build_param["type"] == "int8_hnsw"
        assert config.backend_config["type"] == "int8_hnsw"

    def test_elastic_dry_run_build(self):
        """ElasticBackend.build(dry_run=True) returns synthetic result without ES."""
        cls = get_backend_class("elastic")
        backend = cls(
            config={"name": "test", "host": "localhost", "port": 9200}
        )

        base = np.random.rand(100, 32).astype(np.float32)
        queries = np.random.rand(10, 32).astype(np.float32)
        dataset = Dataset(
            name="test",
            training_vectors=base,
            query_vectors=queries,
            distance_metric="euclidean",
        )
        indexes = [
            IndexConfig(
                name="elastic_hnsw_test",
                algo="elastic_hnsw",
                build_param={"m": 16, "ef_construction": 100},
                search_params=[{"num_candidates": 100}],
                file="",
            )
        ]

        result = backend.build(dataset=dataset, indexes=indexes, dry_run=True)

        assert result.success
        assert result.algorithm == "elastic_hnsw"
        assert result.build_time_seconds == 0

    def test_elastic_dry_run_search(self):
        """ElasticBackend.search(dry_run=True) returns synthetic result without ES."""
        cls = get_backend_class("elastic")
        backend = cls(
            config={"name": "test", "host": "localhost", "port": 9200}
        )

        base = np.random.rand(100, 32).astype(np.float32)
        queries = np.random.rand(10, 32).astype(np.float32)
        dataset = Dataset(
            name="test",
            training_vectors=base,
            query_vectors=queries,
            distance_metric="euclidean",
        )
        indexes = [
            IndexConfig(
                name="elastic_hnsw_test",
                algo="elastic_hnsw",
                build_param={},
                search_params=[{"num_candidates": 100}],
                file="",
            )
        ]

        result = backend.search(
            dataset=dataset, indexes=indexes, k=10, dry_run=True
        )

        assert result.success
        assert result.algorithm == "elastic_hnsw"
        assert result.search_time_ms == 0

    def test_elastic_build_requires_training_vectors(self):
        """ElasticBackend.build returns error when no training vectors are available."""
        cls = get_backend_class("elastic")
        backend = cls(
            config={"name": "test", "host": "localhost", "port": 9200}
        )

        queries = np.random.rand(10, 32).astype(np.float32)
        dataset = Dataset(
            name="test",
            query_vectors=queries,
            base_file=None,
            query_file=None,
        )
        indexes = [
            IndexConfig(
                name="elastic_hnsw_test",
                algo="elastic_hnsw",
                build_param={},
                search_params=[{}],
                file="",
            )
        ]

        with patch.object(
            backend, "_check_network_available", return_value=True
        ):
            result = backend.build(
                dataset=dataset, indexes=indexes, dry_run=False
            )

        assert not result.success
        assert "training_vectors" in (result.error_message or "")

    def test_elastic_preflight_fails_when_no_network(self):
        """ElasticBackend.build returns success=False when network is unavailable."""
        cls = get_backend_class("elastic")
        backend = cls(
            config={"name": "test", "host": "localhost", "port": 9200}
        )

        base = np.random.rand(100, 32).astype(np.float32)
        queries = np.random.rand(10, 32).astype(np.float32)
        dataset = Dataset(
            name="test",
            training_vectors=base,
            query_vectors=queries,
            base_file="dummy/base.fbin",
            query_file="dummy/query.fbin",
        )
        indexes = [
            IndexConfig(
                name="elastic_hnsw_test",
                algo="elastic_hnsw",
                build_param={},
                search_params=[{}],
                file="",
            )
        ]

        with patch.object(
            backend, "_check_network_available", return_value=False
        ):
            result = backend.build(dataset=dataset, indexes=indexes)

        assert not result.success
        assert "pre-flight" in (result.error_message or "").lower()

    def test_elastic_search_preflight_fails_when_no_network(self):
        """ElasticBackend.search returns success=False when network is unavailable."""
        cls = get_backend_class("elastic")
        backend = cls(
            config={"name": "test", "host": "localhost", "port": 9200}
        )

        dataset = Dataset(
            name="test",
            training_vectors=np.random.rand(100, 32).astype(np.float32),
            query_vectors=np.random.rand(10, 32).astype(np.float32),
            query_file="dummy/query.fbin",
        )
        indexes = [
            IndexConfig(
                name="elastic_hnsw_test",
                algo="elastic_hnsw",
                build_param={},
                search_params=[{"num_candidates": 100}],
                file="",
            )
        ]

        with patch.object(
            backend, "_check_network_available", return_value=False
        ):
            result = backend.search(dataset=dataset, indexes=indexes, k=10)

        assert not result.success
        assert "pre-flight" in (result.error_message or "").lower()

    def test_elastic_build_skips_existing_index_when_force_false(self):
        """build(force=False) returns success=True immediately when index already exists."""
        cls = get_backend_class("elastic")
        backend = cls(
            config={
                "name": "test",
                "host": "localhost",
                "port": 9200,
                "index_name": "test_index",
            }
        )

        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True
        mock_client.indices.stats.return_value = {
            "_all": {"primaries": {"store": {"size_in_bytes": 1024}}}
        }

        dataset = Dataset(
            name="test",
            training_vectors=np.random.rand(100, 32).astype(np.float32),
            query_vectors=np.random.rand(10, 32).astype(np.float32),
            base_file="dummy/base.fbin",
        )
        indexes = [
            IndexConfig(
                name="test_index",
                algo="elastic_hnsw",
                build_param={
                    "type": "hnsw",
                    "m": 16,
                    "ef_construction": 100,
                    "similarity": "l2_norm",
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "vector_field": "embedding",
                },
                search_params=[{"num_candidates": 100}],
                file="",
            )
        ]

        with patch.object(
            backend, "_check_network_available", return_value=True
        ):
            with patch.object(
                backend, "_get_client", return_value=mock_client
            ):
                result = backend.build(
                    dataset=dataset, indexes=indexes, force=False
                )

        assert result.success
        assert result.index_size_bytes == 1024
        mock_client.indices.delete.assert_not_called()

    def test_elastic_build_uses_lazy_loaded_training_vectors(self):
        """ElasticBackend.build works with Dataset lazy-loading via base_file."""
        cls = get_backend_class("elastic")
        backend = cls(
            config={"name": "test", "host": "localhost", "port": 9200}
        )

        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True
        mock_client.indices.stats.return_value = {
            "_all": {"primaries": {"store": {"size_in_bytes": 2048}}}
        }

        dataset = Dataset(
            name="test",
            query_vectors=np.random.rand(10, 32).astype(np.float32),
            base_file="dummy/base.fbin",
        )
        indexes = [
            IndexConfig(
                name="test_index",
                algo="elastic_hnsw",
                build_param={"m": 16, "ef_construction": 100},
                search_params=[{"num_candidates": 100}],
                file="",
            )
        ]

        with patch.object(
            backend, "_check_network_available", return_value=True
        ):
            with patch.object(
                backend, "_get_client", return_value=mock_client
            ):
                with patch(
                    "cuvs_bench.backends._utils.load_vectors",
                    return_value=np.random.rand(100, 32).astype(np.float32),
                ):
                    result = backend.build(
                        dataset=dataset, indexes=indexes, force=False
                    )

        assert result.success
        assert result.index_size_bytes == 2048

    def test_elastic_search_uses_lazy_loaded_query_vectors(self):
        """ElasticBackend.search works with Dataset lazy-loading via query_file."""
        cls = get_backend_class("elastic")
        backend = cls(
            config={"name": "test", "host": "localhost", "port": 9200}
        )

        mock_client = MagicMock()
        mock_client.search.return_value = {
            "hits": {"hits": [{"_id": "0", "_score": 1.0}]}
        }

        dataset = Dataset(
            name="test",
            training_vectors=np.random.rand(100, 32).astype(np.float32),
            query_file="dummy/query.fbin",
            groundtruth_neighbors=np.array([[0]], dtype=np.int32),
        )
        indexes = [
            IndexConfig(
                name="elastic_hnsw_test",
                algo="elastic_hnsw",
                build_param={},
                search_params=[{"num_candidates": 100}],
                file="",
            )
        ]

        with patch.object(
            backend, "_check_network_available", return_value=True
        ):
            with patch.object(
                backend, "_get_client", return_value=mock_client
            ):
                with patch(
                    "cuvs_bench.backends._utils.load_vectors",
                    return_value=np.random.rand(1, 32).astype(np.float32),
                ):
                    result = backend.search(
                        dataset=dataset, indexes=indexes, k=1
                    )

        assert result.success
        assert result.neighbors.shape == (1, 1)
        assert result.recall == 0.0
        assert "recall_is_authoritative" not in result.metadata
        assert "recall" not in result.metadata["per_search_param_results"][0]

    def test_elastic_algo_from_config(self):
        """ElasticBackend.algo derives from config type (elastic_hnsw, elastic_int8_hnsw)."""
        cls = get_backend_class("elastic")
        backend_hnsw = cls(config={"name": "test", "type": "hnsw"})
        assert backend_hnsw.algo == "elastic_hnsw"

        backend_int8 = cls(config={"name": "test", "type": "int8_hnsw"})
        assert backend_int8.algo == "elastic_int8_hnsw"

    def test_elastic_build_rejects_unsupported_index_type(self):
        """Elastic build fails early with a descriptive unsupported-index error."""
        cls = get_backend_class("elastic")
        backend = cls(
            config={"name": "test", "host": "localhost", "port": 9200}
        )

        dataset = Dataset(
            name="test",
            training_vectors=np.random.rand(8, 4).astype(np.float32),
            query_vectors=np.random.rand(2, 4).astype(np.float32),
            distance_metric="euclidean",
        )
        indexes = [
            IndexConfig(
                name="elastic_scann_test",
                algo="elastic_scann",
                build_param={"type": "scann"},
                search_params=[{"num_candidates": 10}],
                file="",
            )
        ]

        with patch.object(
            backend, "_check_network_available", return_value=True
        ):
            result = backend.build(dataset=dataset, indexes=indexes)

        assert not result.success
        assert "unsupported Elasticsearch index type" in (
            result.error_message or ""
        )
        assert "scann" in (result.error_message or "")

    def test_elastic_build_rejects_unsupported_similarity(self):
        """Elastic build fails early with a descriptive unsupported-similarity error."""
        cls = get_backend_class("elastic")
        backend = cls(
            config={"name": "test", "host": "localhost", "port": 9200}
        )

        dataset = Dataset(
            name="test",
            training_vectors=np.random.rand(8, 4).astype(np.float32),
            query_vectors=np.random.rand(2, 4).astype(np.float32),
            distance_metric="euclidean",
        )
        indexes = [
            IndexConfig(
                name="elastic_hnsw_test",
                algo="elastic_hnsw",
                build_param={"type": "hnsw", "similarity": "dot_product"},
                search_params=[{"num_candidates": 10}],
                file="",
            )
        ]

        with patch.object(
            backend, "_check_network_available", return_value=True
        ):
            result = backend.build(dataset=dataset, indexes=indexes)

        assert not result.success
        assert "unsupported Elasticsearch similarity" in (
            result.error_message or ""
        )
        assert "dot_product" in (result.error_message or "")

    def test_elastic_cleanup_closes_client(self):
        """ElasticBackend.cleanup() closes client and sets _client to None."""
        cls = get_backend_class("elastic")
        backend = cls(
            config={"name": "test", "host": "localhost", "port": 9200}
        )
        mock_client = MagicMock()
        backend._client = mock_client

        backend.cleanup()

        mock_client.close.assert_called_once()
        assert backend._client is None

    def test_orchestrator_elastic_dry_run(self):
        """Initializing the elastic orchestrator should support dry_run."""
        from cuvs_bench.orchestrator import BenchmarkOrchestrator

        orch = BenchmarkOrchestrator(backend_type="elastic")
        results = orch.run_benchmark(
            dataset="glove-50-angular",
            dataset_path="/nonexistent",
            host="localhost",
            port=9200,
            algorithms="elastic_hnsw",
            groups="test",
            build=True,
            search=True,
            dry_run=True,
            count=10,
            batch_size=100,
        )
        assert results is not None
        assert len(results) >= 1


@pytest.mark.skipif(
    not _elasticsearch_installed(),
    reason="Requires pip install cuvs-bench[elastic]",
)
class TestElasticHelpers:
    """Tests for elastic backend helper functions."""

    @pytest.fixture(autouse=True)
    def _ensure_elastic_registered(self):
        """Re-register elastic (may have been unregistered by other tests)."""
        registry = get_registry()
        if "elastic" not in registry._backends:
            try:
                from cuvs_bench.backends.elasticsearch import register

                register()
            except ImportError:
                pass
        yield

    def test_distance_to_similarity(self):
        """_distance_to_similarity maps cuvs distance to ES similarity."""
        from cuvs_bench.backends.elasticsearch import _distance_to_similarity

        assert _distance_to_similarity("euclidean") == "l2_norm"
        assert _distance_to_similarity("inner_product") == "max_inner_product"
        assert _distance_to_similarity("cosine") == "cosine"
        assert _distance_to_similarity("unknown") == "l2_norm"

    def test_validate_elastic_algorithm_rejects_unknown(self):
        """Unknown elastic algorithm names fail with a descriptive error."""
        from cuvs_bench.backends.elasticsearch import (
            _validate_elastic_algorithm,
        )

        with pytest.raises(
            ValueError, match="unsupported algorithm"
        ) as exc_info:
            _validate_elastic_algorithm("elastic_scann")

        assert "elastic_hnsw" in str(exc_info.value)

    def test_validate_elastic_index_type_rejects_unknown(self):
        """Unknown elastic index types fail with a descriptive error."""
        from cuvs_bench.backends.elasticsearch import (
            _validate_elastic_index_type,
        )

        with pytest.raises(
            ValueError,
            match="unsupported Elasticsearch index type",
        ) as exc_info:
            _validate_elastic_index_type("scann")

        assert "hnsw" in str(exc_info.value)

    def test_validate_elastic_similarity_rejects_unknown(self):
        """Unknown elastic similarities fail with a descriptive error."""
        from cuvs_bench.backends.elasticsearch import (
            _validate_elastic_similarity,
        )

        with pytest.raises(
            ValueError,
            match="unsupported Elasticsearch similarity",
        ) as exc_info:
            _validate_elastic_similarity("dot_product")

        assert "max_inner_product" in str(exc_info.value)
