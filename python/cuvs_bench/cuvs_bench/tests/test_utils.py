#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for shared backend utilities and Dataset transparent loading.
"""

import struct

import numpy as np
import pytest
import yaml

from cuvs_bench._bin_format import (
    EXTENDED_HEADER_BYTES,
    LEGACY_HEADER_BYTES,
    UINT32_MAX,
    read_bin_header,
    write_bin_header,
)
from cuvs_bench.backends import Dataset
from cuvs_bench.backends._utils import (
    compute_recall,
    dtype_from_filename,
    expand_param_grid,
    load_vectors,
)
from cuvs_bench.generate_groundtruth.utils import (
    groundtruth_neighbors_filename,
    memmap_bin_file,
    neighbor_index_accumulator_dtype,
    neighbor_index_dtype,
    offset_neighbor_indices,
    write_groundtruth_neighbors,
)
from cuvs_bench.orchestrator.config_loaders import CppGBenchConfigLoader


def _write_test_bin(path, data, *, size_dtype=np.uint32):
    """Write a numpy array in cuvs-bench binary format."""
    with open(path, "wb") as f:
        write_bin_header(
            f, data.shape[0], data.shape[1], size_dtype=size_dtype
        )
        data.tofile(f)


class TestDtypeFromFilename:
    """Tests for dtype_from_filename."""

    def test_fbin(self):
        """Test .fbin maps to float32."""
        assert dtype_from_filename("vectors.fbin") == np.float32

    def test_f16bin(self):
        """Test .f16bin maps to float16."""
        assert dtype_from_filename("vectors.f16bin") == np.float16

    def test_ibin(self):
        """Test .ibin maps to int32."""
        assert dtype_from_filename("groundtruth.ibin") == np.int32

    def test_u64bin(self):
        """Test .u64bin maps to uint64."""
        assert dtype_from_filename("groundtruth.neighbors.u64bin") == np.uint64

    def test_u8bin(self):
        """Test .u8bin maps to uint8."""
        assert dtype_from_filename("vectors.u8bin") == np.ubyte

    def test_i8bin(self):
        """Test .i8bin maps to int8."""
        assert dtype_from_filename("vectors.i8bin") == np.byte

    def test_unsupported_extension(self):
        """Test that unsupported extensions raise RuntimeError."""
        with pytest.raises(RuntimeError, match="Unsupported file extension"):
            dtype_from_filename("vectors.txt")

    def test_full_path(self):
        """Test that full paths are handled correctly."""
        assert (
            dtype_from_filename("/data/datasets/sift/base.fbin") == np.float32
        )


class TestLoadVectors:
    """Tests for load_vectors."""

    def test_load_float32(self, tmp_path):
        """Test loading float32 vectors."""
        data = np.random.rand(100, 128).astype(np.float32)
        path = str(tmp_path / "test.fbin")
        _write_test_bin(path, data)

        loaded = load_vectors(path)
        np.testing.assert_array_equal(loaded, data)

    def test_load_int32(self, tmp_path):
        """Test loading int32 vectors."""
        data = np.random.randint(0, 1000, size=(50, 10)).astype(np.int32)
        path = str(tmp_path / "test.ibin")
        _write_test_bin(path, data)

        loaded = load_vectors(path)
        np.testing.assert_array_equal(loaded, data)

    def test_load_uint8(self, tmp_path):
        """Test loading uint8 vectors."""
        data = np.random.randint(0, 255, size=(30, 64)).astype(np.uint8)
        path = str(tmp_path / "test.u8bin")
        _write_test_bin(path, data)

        loaded = load_vectors(path)
        np.testing.assert_array_equal(loaded, data)

    def test_load_int8(self, tmp_path):
        """Test loading int8 vectors."""
        data = np.random.randint(-128, 127, size=(30, 64)).astype(np.int8)
        path = str(tmp_path / "test.i8bin")
        _write_test_bin(path, data)

        loaded = load_vectors(path)
        np.testing.assert_array_equal(loaded, data)

    def test_subset_size(self, tmp_path):
        """Test that subset_size limits the number of rows loaded."""
        data = np.random.rand(100, 128).astype(np.float32)
        path = str(tmp_path / "test.fbin")
        _write_test_bin(path, data)

        loaded = load_vectors(path, subset_size=10)
        assert loaded.shape == (10, 128)
        np.testing.assert_array_equal(loaded, data[:10])

    def test_subset_size_larger_than_file(self, tmp_path):
        """Test that subset_size larger than file returns all rows."""
        data = np.random.rand(5, 16).astype(np.float32)
        path = str(tmp_path / "test.fbin")
        _write_test_bin(path, data)

        loaded = load_vectors(path, subset_size=100)
        assert loaded.shape == (5, 16)
        np.testing.assert_array_equal(loaded, data)

    def test_negative_subset_size(self, tmp_path):
        """Test that negative subset_size raises ValueError."""
        data = np.random.rand(10, 4).astype(np.float32)
        path = str(tmp_path / "test.fbin")
        _write_test_bin(path, data)

        with pytest.raises(
            ValueError, match="subset_size must be a positive integer"
        ):
            load_vectors(path, subset_size=-1)

    def test_zero_subset_size(self, tmp_path):
        """Test that zero subset_size raises ValueError."""
        data = np.random.rand(10, 4).astype(np.float32)
        path = str(tmp_path / "test.fbin")
        _write_test_bin(path, data)

        with pytest.raises(
            ValueError, match="subset_size must be a positive integer"
        ):
            load_vectors(path, subset_size=0)

    def test_unsupported_extension(self, tmp_path):
        """Test that unsupported file extensions raise RuntimeError."""
        path = str(tmp_path / "test.txt")
        with open(path, "wb") as f:
            f.write(b"dummy")

        with pytest.raises(RuntimeError, match="Unsupported file extension"):
            load_vectors(path)

    def test_truncated_header(self, tmp_path):
        """Test that truncated headers raise ValueError."""
        path = str(tmp_path / "test.fbin")
        with open(path, "wb") as f:
            f.write(b"\x00\x00")

        with pytest.raises(
            ValueError, match="File too small to contain a valid header"
        ):
            load_vectors(path)

    def test_truncated_data(self, tmp_path):
        """Test that truncated data raises ValueError."""
        path = str(tmp_path / "test.fbin")
        with open(path, "wb") as f:
            np.array([10, 4], dtype=np.uint32).tofile(f)
            np.random.rand(5, 4).astype(np.float32).tofile(f)

        with pytest.raises(ValueError, match="does not match either"):
            load_vectors(path)

    def test_file_not_found(self):
        """Test that missing files raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_vectors("/nonexistent/path/vectors.fbin")

    def test_load_uint64_header(self, tmp_path):
        """``load_vectors`` reads files written with the extended uint64 header."""
        data = np.random.rand(40, 16).astype(np.float32)
        path = str(tmp_path / "test.fbin")
        _write_test_bin(path, data, size_dtype=np.uint64)

        # Sanity check: file really uses the extended layout.
        assert (
            tmp_path.joinpath("test.fbin").stat().st_size
            == EXTENDED_HEADER_BYTES + data.nbytes
        )
        loaded = load_vectors(path)
        np.testing.assert_array_equal(loaded, data)

    def test_load_uint64_header_with_subset(self, tmp_path):
        """``subset_size`` works regardless of which header layout was used."""
        data = np.random.rand(50, 8).astype(np.float32)
        path = str(tmp_path / "test.fbin")
        _write_test_bin(path, data, size_dtype=np.uint64)

        loaded = load_vectors(path, subset_size=12)
        assert loaded.shape == (12, 8)
        np.testing.assert_array_equal(loaded, data[:12])

    @pytest.mark.parametrize(
        "ext, dtype, size_dtype",
        [
            (".fbin", np.float32, np.uint32),
            (".fbin", np.float32, np.uint64),
            (".f16bin", np.float16, np.uint32),
            (".f16bin", np.float16, np.uint64),
            (".ibin", np.int32, np.uint32),
            (".ibin", np.int32, np.uint64),
            (".u64bin", np.uint64, np.uint32),
            (".u64bin", np.uint64, np.uint64),
            (".u8bin", np.uint8, np.uint32),
            (".u8bin", np.uint8, np.uint64),
            (".i8bin", np.int8, np.uint32),
            (".i8bin", np.int8, np.uint64),
        ],
    )
    def test_load_roundtrip_all_dtypes(self, tmp_path, ext, dtype, size_dtype):
        """Round-trip every supported dtype through both header layouts."""
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            data = np.random.randint(
                info.min, info.max, size=(25, 7), dtype=dtype
            )
            if dtype == np.uint64:
                data[0, 0] = np.iinfo(np.int32).max + 42
        else:
            data = np.random.rand(25, 7).astype(dtype)
        path = str(tmp_path / f"test{ext}")
        _write_test_bin(path, data, size_dtype=size_dtype)

        loaded = load_vectors(path)
        np.testing.assert_array_equal(loaded, data)


class TestGroundtruthNeighborFormat:
    """Tests for large-base ground-truth neighbor index format selection."""

    def test_neighbor_index_dtype_small_base(self):
        assert neighbor_index_dtype(1_000_000) == np.int32

    def test_neighbor_index_dtype_large_base(self):
        assert neighbor_index_dtype(np.iinfo(np.int32).max + 1) == np.uint64

    def test_neighbor_index_accumulator_dtype_large_base(self):
        assert (
            neighbor_index_accumulator_dtype(np.iinfo(np.int32).max + 1)
            == np.int64
        )

    def test_groundtruth_neighbors_filename_small_base(self):
        assert (
            groundtruth_neighbors_filename(1_000_000)
            == "groundtruth.neighbors.ibin"
        )

    def test_groundtruth_neighbors_filename_large_base(self):
        assert (
            groundtruth_neighbors_filename(np.iinfo(np.int32).max + 1)
            == "groundtruth.neighbors.u64bin"
        )

    def test_load_u64bin_preserves_large_indices(self, tmp_path):
        """uint64 GT files preserve neighbor IDs above INT32_MAX."""
        large_id = np.iinfo(np.int32).max + 12345
        indices = np.array([[large_id, 0, 1]], dtype=np.uint64)
        path = str(tmp_path / "gt.u64bin")
        _write_test_bin(path, indices)

        loaded = load_vectors(path)
        np.testing.assert_array_equal(loaded, indices)

    def test_offset_neighbor_indices_small_base(self):
        local = np.array([[0, 1, 2]], dtype=np.uint32)
        offset = offset_neighbor_indices(local, 1000, 1_000_000)
        assert offset.dtype == np.int32
        np.testing.assert_array_equal(offset, [[1000, 1001, 1002]])

    def test_offset_neighbor_indices_large_batch_offset(self):
        """Search-local IDs must not wrap when batch offset exceeds INT32_MAX."""
        batch_offset = np.iinfo(np.int32).max + 1
        n_base = batch_offset + 10
        local = np.array([[0, 1, 2]], dtype=np.int64)
        offset = offset_neighbor_indices(local, batch_offset, n_base)
        assert offset.dtype == np.int64
        np.testing.assert_array_equal(
            offset,
            [[batch_offset, batch_offset + 1, batch_offset + 2]],
        )

    def test_write_groundtruth_neighbors_round_trip(self, tmp_path):
        """GT write/load preserves neighbor IDs above INT32_MAX."""
        n_base = np.iinfo(np.int32).max + 1
        large_id = n_base + 999
        indices = np.array([[large_id, large_id - 1, 0]], dtype=np.int64)
        path = str(tmp_path / groundtruth_neighbors_filename(n_base))
        write_groundtruth_neighbors(path, indices, n_base)

        loaded = load_vectors(path)
        assert loaded.dtype == np.uint64
        np.testing.assert_array_equal(loaded, indices.astype(np.uint64))

    def test_dataset_lazy_load_u64bin_groundtruth(self, tmp_path):
        """Dataset loads .u64bin ground truth with IDs above INT32_MAX."""
        large_id = np.iinfo(np.int32).max + 12345
        gt = np.array([[large_id, 0, 1]], dtype=np.uint64)
        path = str(tmp_path / "groundtruth.neighbors.u64bin")
        _write_test_bin(path, gt)

        dataset = Dataset(name="test", groundtruth_neighbors_file=path)
        np.testing.assert_array_equal(dataset.groundtruth_neighbors, gt)


class TestBinHeaderHelpers:
    """Tests for ``cuvs_bench._bin_format.read_bin_header`` / ``write_bin_header``."""

    def test_write_legacy_returns_8_bytes(self, tmp_path):
        """Small shapes should write the 8-byte uint32 header by default."""
        path = tmp_path / "h.bin"
        with open(path, "wb") as f:
            n = write_bin_header(f, 7, 3)
        assert n == LEGACY_HEADER_BYTES
        assert path.stat().st_size == LEGACY_HEADER_BYTES

    def test_write_size_dtype_uint64_returns_16_bytes(self, tmp_path):
        """``size_dtype=np.uint64`` should write the 16-byte uint64 header."""
        path = tmp_path / "h.bin"
        with open(path, "wb") as f:
            n = write_bin_header(f, 7, 3, size_dtype=np.uint64)
        assert n == EXTENDED_HEADER_BYTES
        assert path.stat().st_size == EXTENDED_HEADER_BYTES

    def test_write_auto_promotes_to_uint64_when_overflowing(self, tmp_path):
        """Shapes that don't fit in uint32 should auto-promote to uint64."""
        path = tmp_path / "h.bin"
        with open(path, "wb") as f:
            n = write_bin_header(f, UINT32_MAX + 1, 4)
        assert n == EXTENDED_HEADER_BYTES

    def test_write_negative_raises(self, tmp_path):
        """Negative dimensions are rejected."""
        path = tmp_path / "h.bin"
        with open(path, "wb") as f:
            with pytest.raises(ValueError, match="non-negative"):
                write_bin_header(f, -1, 4)

    def test_read_legacy_round_trip(self, tmp_path):
        """Legacy round-trip: write 8-byte header, read it back."""
        path = tmp_path / "x.fbin"
        data = np.random.rand(11, 5).astype(np.float32)
        with open(path, "wb") as f:
            write_bin_header(f, data.shape[0], data.shape[1])
            data.tofile(f)
        n_rows, n_cols, hbytes = read_bin_header(str(path), itemsize=4)
        assert (n_rows, n_cols, hbytes) == (11, 5, LEGACY_HEADER_BYTES)

    def test_read_extended_round_trip(self, tmp_path):
        """Extended round-trip: write 16-byte header, read it back."""
        path = tmp_path / "x.fbin"
        data = np.random.rand(11, 5).astype(np.float32)
        with open(path, "wb") as f:
            write_bin_header(
                f, data.shape[0], data.shape[1], size_dtype=np.uint64
            )
            data.tofile(f)
        n_rows, n_cols, hbytes = read_bin_header(str(path), itemsize=4)
        assert (n_rows, n_cols, hbytes) == (11, 5, EXTENDED_HEADER_BYTES)

    def test_read_synthesized_huge_extended_header(self, tmp_path):
        """Extended-header file with >UINT32_MAX rows and positive n_cols.

        We can't materialize the full data section, so write the header and
        truncate to the exact file size ``read_bin_header`` expects.
        """
        path = tmp_path / "huge.fbin"
        n_rows = UINT32_MAX + 17
        n_cols = 4
        itemsize = 4
        expected_size = EXTENDED_HEADER_BYTES + n_rows * n_cols * itemsize
        with open(path, "wb") as f:
            write_bin_header(f, n_rows, n_cols)
            f.truncate(expected_size)

        assert path.stat().st_size == expected_size
        got_rows, got_cols, hbytes = read_bin_header(
            str(path), itemsize=itemsize
        )
        assert got_rows == n_rows
        assert got_cols == n_cols
        assert hbytes == EXTENDED_HEADER_BYTES

    def test_read_file_too_small_raises(self, tmp_path):
        """A file shorter than the legacy header raises a clear error."""
        path = tmp_path / "x.fbin"
        path.write_bytes(b"\x00\x00\x00")
        with pytest.raises(ValueError, match="File too small"):
            read_bin_header(str(path), itemsize=4)

    def test_read_size_mismatch_raises(self, tmp_path):
        """Header values that don't balance the file size are rejected."""
        path = tmp_path / "x.fbin"
        with open(path, "wb") as f:
            f.write(struct.pack("<II", 10, 4))
            f.write(b"\x00" * (5 * 4 * 4))
        with pytest.raises(ValueError, match="does not match either"):
            read_bin_header(str(path), itemsize=4)

    def test_read_dispatch_prefers_legacy(self, tmp_path):
        """When the legacy interpretation balances, it wins.

        Guards against accidentally treating a small legacy file as
        extended (which would silently mis-interpret the first 16 bytes
        as two uint64s).
        """
        path = tmp_path / "x.fbin"
        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        with open(path, "wb") as f:
            write_bin_header(f, 3, 4)
            data.tofile(f)
        _, _, hbytes = read_bin_header(str(path), itemsize=4)
        assert hbytes == LEGACY_HEADER_BYTES


class TestMemmapBinFile:
    """Tests for ``generate_groundtruth.utils.memmap_bin_file``."""

    def test_read_legacy_header(self, tmp_path):
        """Read mode auto-detects the legacy 8-byte header offset."""
        data = np.random.rand(30, 8).astype(np.float32)
        path = str(tmp_path / "test.fbin")
        _write_test_bin(path, data)

        mm = memmap_bin_file(path, np.float32, mode="r")
        assert mm.shape == (30, 8)
        np.testing.assert_array_equal(mm[:], data)

    def test_read_extended_header(self, tmp_path):
        """Read mode auto-detects the extended 16-byte header offset."""
        data = np.random.rand(30, 8).astype(np.float32)
        path = str(tmp_path / "test.fbin")
        _write_test_bin(path, data, size_dtype=np.uint64)

        mm = memmap_bin_file(path, np.float32, mode="r")
        assert mm.shape == (30, 8)
        np.testing.assert_array_equal(mm[:], data)

    def test_read_partial_shape_override(self, tmp_path):
        """Read mode fills ``None`` shape entries from the header."""
        data = np.random.rand(50, 8).astype(np.float32)
        path = str(tmp_path / "test.fbin")
        _write_test_bin(path, data)

        mm = memmap_bin_file(path, np.float32, shape=(10, None), mode="r")
        assert mm.shape == (10, 8)
        np.testing.assert_array_equal(mm[:], data[:10])

    def test_write_read_roundtrip_legacy(self, tmp_path):
        """Write mode with uint32 header, then read back via memmap."""
        path = str(tmp_path / "test.fbin")
        shape = (20, 8)
        data = np.random.rand(*shape).astype(np.float32)

        mm = memmap_bin_file(path, np.float32, shape=shape, mode="w+")
        mm[:] = data
        mm.flush()
        del mm

        loaded = memmap_bin_file(path, np.float32, mode="r")
        assert loaded.shape == shape
        np.testing.assert_array_equal(loaded[:], data)

    def test_write_read_roundtrip_extended(self, tmp_path):
        """Write mode with uint64 header, then read back via memmap."""
        path = str(tmp_path / "test.fbin")
        shape = (20, 8)
        data = np.random.rand(*shape).astype(np.float32)

        mm = memmap_bin_file(
            path, np.float32, shape=shape, mode="w+", size_dtype=np.uint64
        )
        mm[:] = data
        mm.flush()
        del mm

        loaded = memmap_bin_file(path, np.float32, mode="r")
        assert loaded.shape == shape
        np.testing.assert_array_equal(loaded[:], data)
        assert (
            tmp_path.joinpath("test.fbin").stat().st_size
            == EXTENDED_HEADER_BYTES + data.nbytes
        )


class TestDatasetLazyLoading:
    """Tests for Dataset transparent vector loading."""

    def test_lazy_load_training_vectors(self, tmp_path):
        """Test that base vectors are loaded from file on first access."""
        data = np.random.rand(50, 32).astype(np.float32)
        path = str(tmp_path / "base.fbin")
        _write_test_bin(path, data)

        dataset = Dataset(name="test", base_file=path)
        np.testing.assert_array_equal(dataset.training_vectors, data)

    def test_lazy_load_query_vectors(self, tmp_path):
        """Test that query vectors are loaded from file on first access."""
        data = np.random.rand(10, 32).astype(np.float32)
        path = str(tmp_path / "query.fbin")
        _write_test_bin(path, data)

        dataset = Dataset(name="test", query_file=path)
        np.testing.assert_array_equal(dataset.query_vectors, data)

    def test_lazy_load_groundtruth(self, tmp_path):
        """Test that ground truth is loaded from file on first access."""
        data = np.random.randint(0, 100, size=(10, 5)).astype(np.int32)
        path = str(tmp_path / "gt.ibin")
        _write_test_bin(path, data)

        dataset = Dataset(name="test", groundtruth_neighbors_file=path)
        np.testing.assert_array_equal(dataset.groundtruth_neighbors, data)

    def test_lazy_load_groundtruth_distances(self, tmp_path):
        """Test that ground truth distances are loaded from file on first access."""
        data = np.random.rand(10, 5).astype(np.float32)
        path = str(tmp_path / "gt_dist.fbin")
        _write_test_bin(path, data)

        dataset = Dataset(name="test", groundtruth_distances_file=path)
        np.testing.assert_array_equal(dataset.groundtruth_distances, data)

    def test_preloaded_vectors_returned_directly(self):
        """Test that pre-loaded vectors are returned without file access."""
        base = np.random.rand(20, 16).astype(np.float32)
        query = np.random.rand(5, 16).astype(np.float32)

        dataset = Dataset(
            name="test",
            training_vectors=base,
            query_vectors=query,
        )
        np.testing.assert_array_equal(dataset.training_vectors, base)
        np.testing.assert_array_equal(dataset.query_vectors, query)

    def test_no_file_returns_empty(self):
        """Test that Dataset without files returns empty arrays."""
        dataset = Dataset(name="test")
        assert dataset.training_vectors.size == 0
        assert dataset.query_vectors.size == 0
        assert dataset.groundtruth_neighbors is None

    def test_lazy_load_with_subset_size(self, tmp_path):
        """Test that subset_size is respected during lazy loading."""
        data = np.random.rand(100, 32).astype(np.float32)
        path = str(tmp_path / "base.fbin")
        _write_test_bin(path, data)

        dataset = Dataset(
            name="test",
            base_file=path,
            metadata={"subset_size": 10},
        )
        assert dataset.training_vectors.shape == (10, 32)
        np.testing.assert_array_equal(dataset.training_vectors, data[:10])

    def test_file_path_still_accessible(self):
        """Test that file paths are accessible without triggering loading."""
        dataset = Dataset(
            name="test",
            base_file="/path/to/base.fbin",
            query_file="/path/to/query.fbin",
        )
        assert dataset.base_file == "/path/to/base.fbin"
        assert dataset.query_file == "/path/to/query.fbin"
        assert dataset.name == "test"

    def test_file_path_access_does_not_trigger_loading(self, tmp_path):
        """Test that accessing file paths does not load vectors into memory."""
        data = np.random.rand(50, 32).astype(np.float32)
        path = str(tmp_path / "base.fbin")
        _write_test_bin(path, data)

        dataset = Dataset(name="test", base_file=path)

        _ = dataset.base_file
        _ = dataset.name
        _ = dataset.distance_metric

        assert dataset._training_vectors.size == 0

    def test_dims_and_counts(self, tmp_path):
        """Test dims, n_base, and n_queries properties."""
        data = np.random.rand(50, 32).astype(np.float32)
        path = str(tmp_path / "base.fbin")
        _write_test_bin(path, data)

        queries = np.random.rand(10, 32).astype(np.float32)
        qpath = str(tmp_path / "query.fbin")
        _write_test_bin(qpath, queries)

        dataset = Dataset(name="test", base_file=path, query_file=qpath)
        assert dataset.dims == 32
        assert dataset.n_base == 50
        assert dataset.n_queries == 10

    def test_caching(self, tmp_path):
        """Test that vectors are loaded once and cached."""
        data = np.random.rand(10, 4).astype(np.float32)
        path = str(tmp_path / "base.fbin")
        _write_test_bin(path, data)

        dataset = Dataset(name="test", base_file=path)
        first_access = dataset.training_vectors
        second_access = dataset.training_vectors
        assert first_access is second_access


class TestConfigLoaderMethods:
    """Tests for base ConfigLoader inherited methods."""

    def test_load_yaml_file(self, tmp_path):
        """Test loading a YAML file."""
        yaml_path = str(tmp_path / "test.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump({"name": "test", "value": 42}, f)

        loader = CppGBenchConfigLoader(config_path=str(tmp_path))
        result = loader.load_yaml_file(yaml_path)
        assert result["name"] == "test"
        assert result["value"] == 42

    def test_get_dataset_configuration(self):
        """Test finding a dataset by name."""
        loader = CppGBenchConfigLoader()
        datasets = [
            {"name": "sift-128", "dims": 128},
            {"name": "glove-100", "dims": 100},
        ]

        result = loader.get_dataset_configuration("glove-100", datasets)
        assert result["name"] == "glove-100"
        assert result["dims"] == 100

    def test_get_dataset_configuration_not_found(self):
        """Test that missing datasets raise ValueError."""
        loader = CppGBenchConfigLoader()
        datasets = [{"name": "sift-128"}]

        with pytest.raises(
            ValueError, match="Could not find a dataset configuration"
        ):
            loader.get_dataset_configuration("nonexistent", datasets)


class TestExpandParamGrid:
    """Tests for expand_param_grid."""

    def test_single_param(self):
        """Test expansion of a single parameter."""
        result = expand_param_grid({"m": [16, 32]})
        assert result == [{"m": 16}, {"m": 32}]

    def test_two_params(self):
        """Test Cartesian product of two parameters."""
        result = expand_param_grid({"m": [16, 32], "ef": [100, 200]})
        assert len(result) == 4
        assert {"m": 16, "ef": 100} in result
        assert {"m": 16, "ef": 200} in result
        assert {"m": 32, "ef": 100} in result
        assert {"m": 32, "ef": 200} in result

    def test_empty_spec(self):
        """Test that empty spec returns a single empty dict."""
        result = expand_param_grid({})
        assert result == [{}]

    def test_single_values(self):
        """Test that single-element lists produce one combination."""
        result = expand_param_grid({"m": [16], "ef": [100]})
        assert result == [{"m": 16, "ef": 100}]

    def test_three_params(self):
        """Test Cartesian product of three parameters."""
        result = expand_param_grid({"a": [1, 2], "b": [3], "c": [4, 5]})
        assert len(result) == 4
        assert {"a": 1, "b": 3, "c": 4} in result
        assert {"a": 2, "b": 3, "c": 5} in result


class TestComputeRecall:
    """Tests for compute_recall."""

    def test_perfect_recall(self):
        """Test recall is 1.0 when neighbors match ground truth exactly."""
        neighbors = np.array([[0, 1, 2], [3, 4, 5]])
        groundtruth = np.array([[0, 1, 2], [3, 4, 5]])
        assert compute_recall(neighbors, groundtruth, k=3) == 1.0

    def test_zero_recall(self):
        """Test recall is 0.0 when no neighbors match ground truth."""
        neighbors = np.array([[10, 11, 12], [13, 14, 15]])
        groundtruth = np.array([[0, 1, 2], [3, 4, 5]])
        assert compute_recall(neighbors, groundtruth, k=3) == 0.0

    def test_partial_recall(self):
        """Test recall with partial overlap."""
        neighbors = np.array([[0, 1, 99]])
        groundtruth = np.array([[0, 1, 2]])
        recall = compute_recall(neighbors, groundtruth, k=3)
        assert abs(recall - 2.0 / 3.0) < 1e-9

    def test_k_smaller_than_groundtruth(self):
        """Test recall when k is smaller than ground truth columns."""
        neighbors = np.array([[0, 1]])
        groundtruth = np.array([[0, 1, 2, 3, 4]])
        recall = compute_recall(neighbors, groundtruth, k=2)
        assert recall == 1.0

    def test_k_larger_than_groundtruth(self):
        """Test recall when k is larger than ground truth columns."""
        neighbors = np.array([[0, 1, 2, 3, 4]])
        groundtruth = np.array([[0, 1]])
        recall = compute_recall(neighbors, groundtruth, k=5)
        assert recall == 1.0

    def test_empty_groundtruth(self):
        """Test recall is 0.0 when ground truth has zero columns."""
        neighbors = np.array([[0, 1, 2]])
        groundtruth = np.empty((1, 0), dtype=np.int32)
        assert compute_recall(neighbors, groundtruth, k=3) == 0.0

    def test_empty_queries(self):
        """Test recall is 0.0 when there are no queries."""
        neighbors = np.empty((0, 3), dtype=np.int64)
        groundtruth = np.empty((0, 3), dtype=np.int32)
        assert compute_recall(neighbors, groundtruth, k=3) == 0.0

    def test_multiple_queries(self):
        """Test recall averaged across multiple queries."""
        neighbors = np.array([[0, 1, 2], [3, 4, 99]])
        groundtruth = np.array([[0, 1, 2], [3, 4, 5]])
        recall = compute_recall(neighbors, groundtruth, k=3)
        assert abs(recall - 5.0 / 6.0) < 1e-9

    def test_large_uint64_neighbor_ids(self):
        """Recall works when GT neighbor IDs exceed INT32_MAX."""
        large_id = np.iinfo(np.int32).max + 999
        neighbors = np.array([[large_id, 0, 1]], dtype=np.int64)
        groundtruth = np.array([[large_id, 0, 1]], dtype=np.uint64)
        assert compute_recall(neighbors, groundtruth, k=3) == 1.0
