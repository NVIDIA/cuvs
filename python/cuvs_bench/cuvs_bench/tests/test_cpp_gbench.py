#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the C++ Google Benchmark backend.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from cuvs_bench.backends import (
    Dataset,
    CppGoogleBenchmarkBackend,
    get_registry,
)
from cuvs_bench.orchestrator.config_loaders import IndexConfig


class TestCppGoogleBenchmarkBackend:
    """Tests for CppGoogleBenchmarkBackend."""

    def test_backend_auto_registered(self):
        """Test that cpp_gbench backend is auto-registered on import."""
        registry = get_registry()
        assert registry.is_registered("cpp_gbench")

        backends = registry.list_backends()
        assert "cpp_gbench" in backends
        assert backends["cpp_gbench"] == CppGoogleBenchmarkBackend

    def test_backend_instantiation_missing_executable(self):
        """Test that backend raises error if executable doesn't exist."""
        config = {
            "name": "test_backend",
            "executable_path": "/nonexistent/path/to/executable",
        }

        with pytest.raises(FileNotFoundError, match="executable not found"):
            CppGoogleBenchmarkBackend(config)

    def test_backend_name_from_config(self):
        """Test that backend name comes from config."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_exec = Path(f.name)

        try:
            temp_exec.chmod(0o755)

            config = {
                "name": "my_custom_backend",
                "executable_path": str(temp_exec),
            }
            backend = CppGoogleBenchmarkBackend(config)

            # Name should come from config
            assert backend.name == "my_custom_backend"

        finally:
            temp_exec.unlink(missing_ok=True)

    def test_backend_requires_gpu_from_config(self):
        """
        Test requires_gpu property reads from config.

        GPU requirement is determined by config["requires_gpu"], matching
        the algorithms.yaml structure. If requires_gpu=True and no GPU is
        available, the backend will be skipped during pre-flight checks.
        """
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_exec = Path(f.name)

        try:
            temp_exec.chmod(0o755)

            # Test GPU backend (requires_gpu=True in config)
            gpu_config = {
                "name": "cuvs_ivf_flat",
                "executable_path": str(temp_exec),
                "requires_gpu": True,
            }
            gpu_backend = CppGoogleBenchmarkBackend(gpu_config)
            assert gpu_backend.requires_gpu is True

            # Test CPU backend (requires_gpu=False or not set)
            cpu_config = {
                "name": "faiss_cpu_flat",
                "executable_path": str(temp_exec),
                "requires_gpu": False,
            }
            cpu_backend = CppGoogleBenchmarkBackend(cpu_config)
            assert cpu_backend.requires_gpu is False

            # Test default (requires_gpu not set, defaults to False)
            default_config = {
                "name": "some_backend",
                "executable_path": str(temp_exec),
            }
            default_backend = CppGoogleBenchmarkBackend(default_config)
            assert default_backend.requires_gpu is False

        finally:
            temp_exec.unlink(missing_ok=True)

    def test_backend_config_defaults(self):
        """Test that backend uses default config values."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_exec = Path(f.name)

        try:
            temp_exec.chmod(0o755)

            # Minimal config
            config = {
                "name": "test_backend",
                "executable_path": str(temp_exec),
            }
            backend = CppGoogleBenchmarkBackend(config)

            # Check defaults
            assert backend.data_prefix == ""
            assert backend.warmup_time == 1.0
            assert backend.dataset_name == ""
            assert backend.output_filename == ("", "")
        finally:
            temp_exec.unlink(missing_ok=True)

    def test_backend_config_custom_values(self):
        """Test that backend respects custom config values."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_exec = Path(f.name)

        try:
            temp_exec.chmod(0o755)

            config = {
                "name": "custom_backend",
                "executable_path": str(temp_exec),
                "data_prefix": "custom_data/",
                "warmup_time": 2.5,
                "dataset": "sift-128-euclidean",
                "output_filename": (
                    "cuvs_ivf_flat,test",
                    "cuvs_ivf_flat,test,k10,bs1000",
                ),
                "algo": "cuvs_ivf_flat",
            }
            backend = CppGoogleBenchmarkBackend(config)

            assert backend.data_prefix == "custom_data/"
            assert backend.warmup_time == 2.5
            assert backend.dataset_name == "sift-128-euclidean"
            assert backend.output_filename == (
                "cuvs_ivf_flat,test",
                "cuvs_ivf_flat,test,k10,bs1000",
            )
            assert backend.algo == "cuvs_ivf_flat"
        finally:
            temp_exec.unlink(missing_ok=True)

    def test_get_backend_from_registry(self):
        """Test getting cpp_gbench backend from registry."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_exec = Path(f.name)

        try:
            temp_exec.chmod(0o755)

            registry = get_registry()
            backend = registry.get_backend(
                "cpp_gbench",
                {
                    "name": "registry_test_backend",
                    "executable_path": str(temp_exec),
                },
            )

            assert isinstance(backend, CppGoogleBenchmarkBackend)
            assert backend.name == "registry_test_backend"
        finally:
            temp_exec.unlink(missing_ok=True)


class TestCppBackendBuildSearch:
    """Test build and search operations."""

    def test_build_with_no_indexes(self):
        """Test that build returns skipped result when no indexes provided."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_exec = Path(f.name)

        try:
            temp_exec.chmod(0o755)

            backend = CppGoogleBenchmarkBackend(
                {"name": "test_backend", "executable_path": str(temp_exec)}
            )

            # Create dataset with file paths (C++ backend reads from files)
            dataset = Dataset(
                name="test",
                training_vectors=np.empty((0, 0)),
                query_vectors=np.empty((0, 0)),
                base_file="/path/to/base.fbin",
                query_file="/path/to/query.fbin",
            )

            # Build with empty indexes list
            result = backend.build(
                dataset=dataset, indexes=[], force=False, dry_run=False
            )

            assert result.success is True
            assert result.metadata.get("skipped") is True
            assert result.metadata.get("reason") == "no_indexes"
            assert result.build_time_seconds == 0.0

        finally:
            temp_exec.unlink(missing_ok=True)

    def test_search_with_no_indexes(self):
        """Test that search returns skipped result when no indexes provided."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_exec = Path(f.name)

        try:
            temp_exec.chmod(0o755)

            backend = CppGoogleBenchmarkBackend(
                {"name": "test_backend", "executable_path": str(temp_exec)}
            )

            dataset = Dataset(
                name="test",
                training_vectors=np.empty((0, 0)),
                query_vectors=np.empty((0, 0)),
                base_file="/path/to/base.fbin",
                query_file="/path/to/query.fbin",
            )

            # Search with empty indexes list
            result = backend.search(
                dataset=dataset, indexes=[], k=10, batch_size=1000
            )

            assert result.success is True
            assert result.metadata.get("skipped") is True
            assert result.metadata.get("reason") == "no_indexes"
            assert result.search_time_ms == 0.0
            assert result.search_params == []  # Should be empty list, not dict

        finally:
            temp_exec.unlink(missing_ok=True)

    def test_build_dry_run(self):
        """Test build dry run mode."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_exec = Path(f.name)

        try:
            temp_exec.chmod(0o755)

            backend = CppGoogleBenchmarkBackend(
                {
                    "name": "test_backend",
                    "executable_path": str(temp_exec),
                    "data_prefix": "/data/",
                    "dataset": "sift-128-euclidean",
                    "output_filename": (
                        "cuvs_ivf_flat,test",
                        "cuvs_ivf_flat,test,k10,bs1000",
                    ),
                }
            )

            dataset = Dataset(
                name="sift-128-euclidean",
                training_vectors=np.empty((0, 0)),
                query_vectors=np.empty((0, 0)),
                base_file="sift-128-euclidean/base.fbin",
                query_file="sift-128-euclidean/query.fbin",
                groundtruth_neighbors_file="sift-128-euclidean/groundtruth.neighbors.ibin",
            )

            indexes = [
                IndexConfig(
                    name="cuvs_ivf_flat.nlist1024",
                    algo="cuvs_ivf_flat",
                    build_param={"nlist": 1024},
                    search_params=[{"nprobe": 10}, {"nprobe": 50}],
                    file="/data/sift-128-euclidean/index/cuvs_ivf_flat.nlist1024",
                )
            ]

            result = backend.build(
                dataset=dataset, indexes=indexes, force=False, dry_run=True
            )

            assert result.success is True
            assert result.metadata.get("dry_run") is True
            assert result.metadata.get("num_indexes") == 1

        finally:
            temp_exec.unlink(missing_ok=True)

    def test_search_dry_run(self):
        """Test search dry run mode."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_exec = Path(f.name)

        try:
            temp_exec.chmod(0o755)

            backend = CppGoogleBenchmarkBackend(
                {
                    "name": "test_backend",
                    "executable_path": str(temp_exec),
                    "data_prefix": "/data/",
                    "dataset": "sift-128-euclidean",
                    "output_filename": (
                        "cuvs_ivf_flat,test",
                        "cuvs_ivf_flat,test,k10,bs1000",
                    ),
                }
            )

            dataset = Dataset(
                name="sift-128-euclidean",
                training_vectors=np.empty((0, 0)),
                query_vectors=np.empty((0, 0)),
                base_file="sift-128-euclidean/base.fbin",
                query_file="sift-128-euclidean/query.fbin",
                groundtruth_neighbors_file="sift-128-euclidean/groundtruth.neighbors.ibin",
            )

            indexes = [
                IndexConfig(
                    name="cuvs_ivf_flat.nlist1024",
                    algo="cuvs_ivf_flat",
                    build_param={"nlist": 1024},
                    search_params=[{"nprobe": 10}, {"nprobe": 50}],
                    file="/data/sift-128-euclidean/index/cuvs_ivf_flat.nlist1024",
                )
            ]

            result = backend.search(
                dataset=dataset,
                indexes=indexes,
                k=10,
                batch_size=1000,
                dry_run=True,
            )

            assert result.success is True
            assert result.metadata.get("dry_run") is True
            assert result.metadata.get("num_indexes") == 1
            assert (
                result.metadata.get("total_search_configs") == 2
            )  # 2 search params
            assert isinstance(result.search_params, list)

        finally:
            temp_exec.unlink(missing_ok=True)


class TestDatasetValidation:
    """Test dataset validation for C++ backend."""

    def test_build_requires_base_file(self):
        """Test that build raises error if base_file is missing."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_exec = Path(f.name)

        try:
            temp_exec.chmod(0o755)

            backend = CppGoogleBenchmarkBackend(
                {"name": "test_backend", "executable_path": str(temp_exec)}
            )

            # Dataset without base_file
            dataset = Dataset(
                name="test",
                training_vectors=np.empty((0, 0)),
                query_vectors=np.empty((0, 0)),
                query_file="/path/to/query.fbin",
                # base_file is None
            )

            indexes = [
                IndexConfig(
                    name="test_index",
                    algo="cuvs_ivf_flat",
                    build_param={"nlist": 1024},
                    search_params=[{"nprobe": 10}],
                    file="/path/to/index",
                )
            ]

            with pytest.raises(ValueError, match="base_file is required"):
                backend.build(dataset=dataset, indexes=indexes)

        finally:
            temp_exec.unlink(missing_ok=True)

    def test_build_requires_query_file(self):
        """Test that build raises error if query_file is missing."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_exec = Path(f.name)

        try:
            temp_exec.chmod(0o755)

            backend = CppGoogleBenchmarkBackend(
                {"name": "test_backend", "executable_path": str(temp_exec)}
            )

            # Dataset without query_file
            dataset = Dataset(
                name="test",
                training_vectors=np.empty((0, 0)),
                query_vectors=np.empty((0, 0)),
                base_file="/path/to/base.fbin",
                # query_file is None
            )

            indexes = [
                IndexConfig(
                    name="test_index",
                    algo="cuvs_ivf_flat",
                    build_param={"nlist": 1024},
                    search_params=[{"nprobe": 10}],
                    file="/path/to/index",
                )
            ]

            with pytest.raises(ValueError, match="query_file is required"):
                backend.build(dataset=dataset, indexes=indexes)

        finally:
            temp_exec.unlink(missing_ok=True)
