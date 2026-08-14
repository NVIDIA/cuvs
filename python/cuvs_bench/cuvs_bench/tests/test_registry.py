#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the backend registry system.
"""

import json

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
from cuvs_bench.orchestrator import (
    BenchmarkConfig,
    BenchmarkOrchestrator,
    DatasetConfig,
)
from cuvs_bench.orchestrator.config_loaders import IndexConfig


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

        return [
            SearchResult(
                neighbors=neighbors,
                distances=distances,
                search_time_ms=0.1,
                queries_per_second=n_queries / 0.1,
                recall=0.95,
                algorithm=self.algo,
                search_params=first.search_params,
                success=True,
            )
        ]


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

        return [
            SearchResult(
                neighbors=neighbors,
                distances=distances,
                search_time_ms=0.2,
                queries_per_second=n_queries / 0.2,
                recall=0.90,
                algorithm=self.algo,
                search_params=first.search_params if first else [],
                success=True,
            )
        ]


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

    def test_build_result_core_fields_cannot_be_overridden(self):
        result = BuildResult(
            index_path="/path/to/index",
            build_time_seconds=5.5,
            index_size_bytes=1024000,
            algorithm="test_algo",
            build_params={"name": "wrong", "real_time": -1},
            metadata={
                "time_unit": "ms",
                "success": False,
                "error_message": "wrong",
            },
        )

        json_result = result.to_json()

        assert json_result["name"] == "test_algo/build"
        assert json_result["real_time"] == 5.5
        assert json_result["time_unit"] == "s"
        assert json_result["success"] is True
        assert "error_message" not in json_result


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

    def test_search_result_preserves_legacy_positional_arguments(self):
        neighbors = np.array([[1]])
        distances = np.array([[0.1]])
        latency_percentiles = {"p50": 1.0}
        metadata = {"source": "legacy"}

        result = SearchResult(
            neighbors,
            distances,
            2.0,
            500.0,
            0.9,
            "test_algo",
            [{}],
            latency_percentiles,
            0.5,
            0.75,
            metadata,
            False,
            "failed",
        )

        assert result.latency_percentiles is latency_percentiles
        assert result.gpu_time_seconds == 0.5
        assert result.cpu_time_seconds == 0.75
        assert result.metadata is metadata
        assert result.success is False
        assert result.error_message == "failed"
        assert result.latency_seconds is None

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

    def test_search_result_core_fields_cannot_be_overridden(self):
        result = SearchResult(
            neighbors=np.array([[1]]),
            distances=np.array([[0.1]]),
            search_time_ms=6.0,
            queries_per_second=500.0,
            recall=0.95,
            algorithm="test_algo",
            search_params=[{}],
            latency_seconds=0.003,
            gpu_time_seconds=0.004,
            cpu_time_seconds=0.005,
            metadata={
                "name": "wrong",
                "real_time": -1,
                "Latency": -1,
                "GPU": -1,
                "cpu_time": -1,
                "Recall": -1,
                "success": False,
                "error_message": "wrong",
            },
        )

        json_result = result.to_json()

        assert json_result["name"] == "test_algo/search"
        assert json_result["real_time"] == 3.0
        assert json_result["Latency"] == 0.003
        assert json_result["GPU"] == 0.004
        assert json_result["cpu_time"] == 0.005
        assert json_result["Recall"] == 0.95
        assert json_result["success"] is True
        assert "error_message" not in json_result

    def test_search_result_optional_metadata_fields_are_fallbacks(self):
        result = SearchResult(
            neighbors=np.array([[1]]),
            distances=np.array([[0.1]]),
            search_time_ms=6.0,
            queries_per_second=500.0,
            recall=0.95,
            algorithm="test_algo",
            search_params=[{}],
            metadata={
                "Latency": 0.006,
                "GPU": 0.004,
                "cpu_time": 0.005,
            },
        )

        json_result = result.to_json()

        assert json_result["Latency"] == 0.006
        assert json_result["GPU"] == 0.004
        assert json_result["cpu_time"] == 0.005


def test_backend_without_result_stems_runs_without_generic_persistence(
    tmp_path, monkeypatch
):
    class DummyConfigLoader:
        def load(self, **_kwargs):
            return DatasetConfig(name="dummy"), [
                BenchmarkConfig(
                    indexes=[
                        IndexConfig(
                            name="dummy-index",
                            algo="dummy",
                            build_param={},
                            search_params=[{}],
                            file=str(tmp_path / "dummy-index"),
                        )
                    ],
                    backend_config={"name": "dummy-index"},
                )
            ]

    monkeypatch.setattr(
        "cuvs_bench.orchestrator.orchestrator.get_backend_class",
        lambda _backend_type: DummyBackend,
    )
    monkeypatch.setattr(
        "cuvs_bench.orchestrator.orchestrator.get_config_loader",
        lambda _backend_type: DummyConfigLoader,
    )
    orchestrator = BenchmarkOrchestrator(backend_type="dummy")

    results = orchestrator.run_benchmark(
        build=True,
        search=False,
        dataset="dummy",
        dataset_path=str(tmp_path),
    )

    assert len(results) == 1
    assert results[0].success is True
    assert not (tmp_path / "dummy" / "result").exists()


def test_failed_opt_in_build_clears_only_matching_stale_search_artifacts(
    tmp_path, monkeypatch
):
    build_stem = "dummy,base"
    search_stem = f"{build_stem},k1,bs1"

    class DummyConfigLoader:
        def load(self, **_kwargs):
            return DatasetConfig(name="dummy"), [
                BenchmarkConfig(
                    indexes=[
                        IndexConfig(
                            name="dummy-index",
                            algo="dummy",
                            build_param={},
                            search_params=[{}],
                            file=str(tmp_path / "dummy-index"),
                        )
                    ],
                    backend_config={
                        "name": "dummy-index",
                        "output_filename": (build_stem, search_stem),
                    },
                )
            ]

    class FailedBuildBackend(DummyBackend):
        orchestrator_persists_results = True

        def build(self, dataset, indexes, force=False, dry_run=False):
            return BuildResult(
                index_path=indexes[0].file,
                build_time_seconds=0.0,
                index_size_bytes=0,
                algorithm=self.algo,
                build_params={},
                success=False,
                error_message="expected build failure",
            )

        def search(self, *args, **kwargs):
            raise AssertionError("search must not run after a failed build")

    result_root = tmp_path / "dummy" / "result"
    search_dir = result_root / "search"
    search_dir.mkdir(parents=True)
    stale_paths = [
        search_dir / f"{search_stem}.json",
        search_dir / f"{search_stem},raw.csv",
        search_dir / f"{search_stem},throughput.csv",
        search_dir / f"{search_stem},latency.csv",
    ]
    unrelated_paths = [
        search_dir / "other,base,k1,bs1.json",
        search_dir / "other,base,k1,bs1,raw.csv",
    ]
    for path in [*stale_paths, *unrelated_paths]:
        path.write_text("stale\n", encoding="utf-8")

    monkeypatch.setattr(
        "cuvs_bench.orchestrator.orchestrator.get_backend_class",
        lambda _backend_type: FailedBuildBackend,
    )
    monkeypatch.setattr(
        "cuvs_bench.orchestrator.orchestrator.get_config_loader",
        lambda _backend_type: DummyConfigLoader,
    )

    results = BenchmarkOrchestrator(backend_type="dummy").run_benchmark(
        build=True,
        search=True,
        count=1,
        batch_size=1,
        dataset="dummy",
        dataset_path=str(tmp_path),
    )

    assert len(results) == 1
    assert results[0].success is False
    assert not any(path.exists() for path in stale_paths)
    assert all(
        path.read_text(encoding="utf-8") == "stale\n"
        for path in unrelated_paths
    )

    build_path = result_root / "build" / f"{build_stem}.json"
    build_rows = json.loads(build_path.read_text(encoding="utf-8"))[
        "benchmarks"
    ]
    assert len(build_rows) == 1
    assert build_rows[0]["success"] is False
    assert build_rows[0]["error_message"] == "expected build failure"


@pytest.mark.parametrize(
    ("index_count", "output_filenames", "expected_message"),
    [
        (
            2,
            ("dummy,base", "dummy,base,k1,bs1"),
            "exactly one index",
        ),
        (
            1,
            ("dummy", "dummy,k1,bs1"),
            "Build result filename stem",
        ),
        (
            1,
            ("dummy,base", "other,base,k1,bs1"),
            "Search result filename stem",
        ),
        (
            1,
            ("dummy,base",),
            "exactly two result filename stems",
        ),
        (
            1,
            ("../dummy,base", "../dummy,base,k1,bs1"),
            "Invalid benchmark result filename",
        ),
    ],
)
def test_opt_in_sweep_persistence_validates_artifact_identity(
    tmp_path,
    monkeypatch,
    index_count,
    output_filenames,
    expected_message,
):
    indexes = [
        IndexConfig(
            name=f"dummy-index-{position}",
            algo="dummy",
            build_param={},
            search_params=[{}],
            file=str(tmp_path / f"dummy-index-{position}"),
        )
        for position in range(index_count)
    ]

    class DummyConfigLoader:
        def load(self, **_kwargs):
            return DatasetConfig(name="dummy"), [
                BenchmarkConfig(
                    indexes=indexes,
                    backend_config={
                        "name": "dummy-index",
                        "output_filename": output_filenames,
                    },
                )
            ]

    class PersistedDummyBackend(DummyBackend):
        orchestrator_persists_results = True

    monkeypatch.setattr(
        "cuvs_bench.orchestrator.orchestrator.get_backend_class",
        lambda _backend_type: PersistedDummyBackend,
    )
    monkeypatch.setattr(
        "cuvs_bench.orchestrator.orchestrator.get_config_loader",
        lambda _backend_type: DummyConfigLoader,
    )

    with pytest.raises(ValueError, match=expected_message):
        BenchmarkOrchestrator(backend_type="dummy").run_benchmark(
            build=True,
            search=False,
            count=1,
            batch_size=1,
            dataset="dummy",
            dataset_path=str(tmp_path),
        )

    assert not (tmp_path / "dummy" / "result").exists()


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

        result = backend.search(dataset=dataset, indexes=indexes, k=10)[0]

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
