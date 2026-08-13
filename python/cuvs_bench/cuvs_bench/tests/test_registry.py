#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the backend registry system.
"""

import pytest
import numpy as np

from cuvs_bench.backends import (
    Dataset,
    BuildResult,
    SearchResult,
    BenchmarkBackend,
    BackendRegistry,
    get_registry,
    register_backend,
    get_backend,
)


class DummyBackend(BenchmarkBackend):
    """Dummy backend for testing."""

    @property
    def algo(self) -> str:
        return "dummy_algo"

    def build(self, dataset, indexes, force=False, dry_run=False):
        if not indexes:
            raise ValueError("indexes must not be empty")
        first = indexes[0]
        return BuildResult(
            index_path=first.file,
            build_time_seconds=1.0,
            index_size_bytes=1000,
            algorithm=self.algo,
            build_params=first.build_param,
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
        if not indexes:
            raise ValueError("indexes must not be empty")
        n_queries = dataset.n_queries
        neighbors = np.random.randint(0, dataset.n_base, size=(n_queries, k))
        distances = np.random.rand(n_queries, k)
        first = indexes[0]

        return SearchResult(
            neighbors=neighbors,
            distances=distances,
            search_time_ms=0.1,
            queries_per_second=n_queries / 0.1,
            recall=0.95,
            algorithm=self.algo,
            search_params=first.search_params,
            success=True,
        )


class AnotherDummyBackend(BenchmarkBackend):
    """Another dummy backend for testing."""

    @property
    def algo(self) -> str:
        return "another_dummy_algo"

    def build(self, dataset, indexes, force=False, dry_run=False):
        if not indexes:
            raise ValueError("indexes must not be empty")
        first = indexes[0]
        return BuildResult(
            index_path=first.file,
            build_time_seconds=2.0,
            index_size_bytes=2000,
            algorithm=self.algo,
            build_params=first.build_param,
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
        if not indexes:
            raise ValueError("indexes must not be empty")
        n_queries = dataset.n_queries
        neighbors = np.random.randint(0, dataset.n_base, size=(n_queries, k))
        distances = np.random.rand(n_queries, k)
        first = indexes[0]

        return SearchResult(
            neighbors=neighbors,
            distances=distances,
            search_time_ms=0.2,
            queries_per_second=n_queries / 0.2,
            recall=0.90,
            algorithm=self.algo,
            search_params=first.search_params if first else [],
            success=True,
        )


class TestDataset:
    """Tests for Dataset dataclass."""

    def test_dataset_creation(self):
        """Test basic dataset creation."""
        base = np.random.rand(1000, 128).astype(np.float32)
        queries = np.random.rand(100, 128).astype(np.float32)
        gt_neighbors = np.random.randint(0, 1000, size=(100, 10))

        dataset = Dataset(
            name="test_dataset",
            training_vectors=base,
            query_vectors=queries,
            groundtruth_neighbors=gt_neighbors,
            distance_metric="euclidean",
        )

        assert dataset.name == "test_dataset"
        assert dataset.dims == 128
        assert dataset.n_base == 1000
        assert dataset.n_queries == 100
        assert dataset.distance_metric == "euclidean"

    def test_dataset_without_groundtruth(self):
        """Test dataset without ground truth."""
        base = np.random.rand(500, 64).astype(np.float32)
        queries = np.random.rand(50, 64).astype(np.float32)

        dataset = Dataset(
            name="test_dataset_no_gt",
            training_vectors=base,
            query_vectors=queries,
        )

        assert dataset.groundtruth_neighbors is None
        assert dataset.groundtruth_distances is None


class TestBuildResult:
    """Tests for BuildResult dataclass."""

    def test_build_result_creation(self):
        """Test basic build result creation."""
        result = BuildResult(
            index_path="/path/to/index",
            build_time_seconds=5.5,
            index_size_bytes=1024000,
            algorithm="test_algo",
            build_params={"nlist": 1024},
            metadata={"gpu_time": 4.2},
            success=True,
        )

        assert result.index_path == "/path/to/index"
        assert result.build_time_seconds == 5.5
        assert result.algorithm == "test_algo"
        assert result.success is True

    def test_build_result_to_json(self):
        """Test conversion to JSON format."""
        result = BuildResult(
            index_path="/path/to/index",
            build_time_seconds=5.5,
            index_size_bytes=1024000,
            algorithm="test_algo",
            build_params={"nlist": 1024},
            metadata={"gpu_time": 4.2},
        )

        json_result = result.to_json()

        assert json_result["name"] == "test_algo/build"
        assert json_result["real_time"] == 5.5
        assert json_result["nlist"] == 1024
        assert json_result["gpu_time"] == 4.2


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_search_result_creation(self):
        """Test basic search result creation."""
        neighbors = np.array([[1, 2, 3], [4, 5, 6]])
        distances = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        result = SearchResult(
            neighbors=neighbors,
            distances=distances,
            search_time_ms=0.1,
            queries_per_second=20.0,
            recall=0.95,
            algorithm="test_algo",
            search_params=[{"nprobe": 10}],
        )

        assert result.recall == 0.95
        assert result.queries_per_second == 20.0
        assert result.success is True

    def test_search_result_to_json(self):
        """Test conversion to JSON format."""
        neighbors = np.array([[1, 2, 3]])
        distances = np.array([[0.1, 0.2, 0.3]])

        result = SearchResult(
            neighbors=neighbors,
            distances=distances,
            search_time_ms=0.05,
            queries_per_second=20.0,
            recall=0.95,
            algorithm="test_algo",
            search_params=[{"nprobe": 10}],
            gpu_time_seconds=0.04,
            latency_percentiles={"p50": 50.0, "p95": 95.0, "p99": 99.0},
        )

        json_result = result.to_json()

        assert json_result["name"] == "test_algo/search"
        assert json_result["Recall"] == 0.95
        assert json_result["search_params"][0]["nprobe"] == 10
        assert json_result["GPU"] == 0.04
        assert json_result["p50"] == 50.0
        assert json_result["p95"] == 95.0


class TestBackendRegistry:
    """Tests for BackendRegistry."""

    def test_registry_creation(self):
        """Test registry creation."""
        registry = BackendRegistry()
        assert len(registry.list_backends()) == 0

    def test_register_backend(self):
        """Test backend registration."""
        registry = BackendRegistry()
        registry.register("dummy", DummyBackend)

        assert registry.is_registered("dummy")
        assert len(registry.list_backends()) == 1

    def test_register_duplicate_backend(self):
        """Test that registering duplicate backends raises error."""
        registry = BackendRegistry()
        registry.register("dummy", DummyBackend)

        with pytest.raises(ValueError, match="already registered"):
            registry.register("dummy", AnotherDummyBackend)

    def test_register_non_backend_class(self):
        """Test that registering non-backend class raises error."""
        registry = BackendRegistry()

        class NotABackend:
            pass

        with pytest.raises(
            TypeError, match="must inherit from BenchmarkBackend"
        ):
            registry.register("invalid", NotABackend)

    def test_unregister_backend(self):
        """Test backend unregistration."""
        registry = BackendRegistry()
        registry.register("dummy", DummyBackend)

        assert registry.is_registered("dummy")

        registry.unregister("dummy")

        assert not registry.is_registered("dummy")

    def test_unregister_nonexistent_backend(self):
        """Test that unregistering non-existent backend raises error."""
        registry = BackendRegistry()

        with pytest.raises(KeyError):
            registry.unregister("nonexistent")

    def test_get_backend(self):
        """Test getting a backend instance."""
        registry = BackendRegistry()
        registry.register("dummy", DummyBackend)

        backend = registry.get_backend("dummy", config={"name": "dummy"})

        assert isinstance(backend, DummyBackend)
        assert backend.name == "dummy"

    def test_get_nonexistent_backend(self):
        """Test that getting non-existent backend raises error."""
        registry = BackendRegistry()

        with pytest.raises(ValueError, match="not found"):
            registry.get_backend("nonexistent", config={})

    def test_list_backends(self):
        """Test listing all backends."""
        registry = BackendRegistry()
        registry.register("dummy", DummyBackend)
        registry.register("another_dummy", AnotherDummyBackend)

        backends = registry.list_backends()

        assert len(backends) == 2
        assert "dummy" in backends
        assert "another_dummy" in backends
        assert backends["dummy"] == DummyBackend
        assert backends["another_dummy"] == AnotherDummyBackend


class TestBackendIntegration:
    """Integration tests for backends."""

    def test_dummy_backend_build(self, tmp_path):
        """Test dummy backend build."""
        from cuvs_bench.orchestrator.config_loaders import IndexConfig

        backend = DummyBackend(config={"name": "dummy"})

        base = np.random.rand(1000, 128).astype(np.float32)
        queries = np.random.rand(100, 128).astype(np.float32)
        dataset = Dataset(
            name="test", training_vectors=base, query_vectors=queries
        )

        indexes = [
            IndexConfig(
                name="test_index",
                algo="dummy_algo",
                build_param={"nlist": 1024},
                search_params=[{"nprobe": 10}],
                file=str(tmp_path / "test_index"),
            )
        ]

        result = backend.build(dataset=dataset, indexes=indexes)

        assert result.success
        assert result.algorithm == "dummy_algo"
        assert result.build_params["nlist"] == 1024

    def test_dummy_backend_search(self, tmp_path):
        """Test dummy backend search."""
        from cuvs_bench.orchestrator.config_loaders import IndexConfig

        backend = DummyBackend(config={"name": "dummy"})

        base = np.random.rand(1000, 128).astype(np.float32)
        queries = np.random.rand(100, 128).astype(np.float32)
        dataset = Dataset(
            name="test", training_vectors=base, query_vectors=queries
        )

        indexes = [
            IndexConfig(
                name="test_index",
                algo="dummy_algo",
                build_param={"nlist": 1024},
                search_params=[{"nprobe": 10}],
                file=str(tmp_path / "test_index"),
            )
        ]

        result = backend.search(dataset=dataset, indexes=indexes, k=10)

        assert result.success
        assert result.recall == 0.95
        assert result.neighbors.shape == (100, 10)
        assert result.search_params[0]["nprobe"] == 10


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_global_registry(self):
        """Test getting global registry."""
        registry1 = get_registry()
        registry2 = get_registry()

        # Should return the same instance (singleton)
        assert registry1 is registry2

    def test_register_backend_global(self):
        """Test registering backend via global function."""
        register_backend("dummy_global", DummyBackend)

        registry = get_registry()
        assert registry.is_registered("dummy_global")

    def test_get_backend_global(self):
        """Test getting backend via global function."""
        register_backend("dummy_global2", DummyBackend)

        backend = get_backend("dummy_global2", config={"name": "dummy"})

        assert isinstance(backend, DummyBackend)
        assert backend.name == "dummy"
