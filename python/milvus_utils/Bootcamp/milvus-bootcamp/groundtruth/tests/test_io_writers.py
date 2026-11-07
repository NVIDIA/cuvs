import numpy as np
import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from groundtruth.main import write_gt_ibin, write_gt_parquet


def test_write_gt_ibin_accepts_int64_ids(tmp_path):
    nq, k = 2, 3
    neighbors = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    distances = np.zeros((nq, k), dtype=np.float32)

    out_prefix = tmp_path / "gt"
    out_path = write_gt_ibin(
        str(out_prefix),
        neighbors,
        distances,
        nq=nq,
        k=k,
    )

    assert out_path == str(out_prefix) + ".ibin"
    assert (tmp_path / "gt.ibin").exists()
    assert (tmp_path / "gt.ibin").stat().st_size > 0


def test_write_gt_ibin_rejects_large_ids(tmp_path):
    nq, k = 1, 1
    neighbors = np.array([[2**31]], dtype=np.int64)
    distances = np.zeros((nq, k), dtype=np.float32)

    out_prefix = tmp_path / "gt"

    with pytest.raises(ValueError, match="IBIN format does not support"):
        write_gt_ibin(
            str(out_prefix),
            neighbors,
            distances,
            nq=nq,
            k=k,
        )


def test_write_gt_parquet_schema_and_rows(tmp_path):
    nq, k, dim = 3, 2, 128
    neighbors = np.arange(nq * k, dtype=np.int64).reshape(nq, k)
    distances = np.arange(nq * k, dtype=np.float32).reshape(nq, k)

    out_prefix = tmp_path / "gt"
    out_path = write_gt_parquet(
        str(out_prefix),
        neighbors,
        distances,
        nq=nq,
        k=k,
        dim=dim,
    )

    assert out_path == str(out_prefix) + ".parquet"
    assert (tmp_path / "gt.parquet").exists()

    table = pq.read_table(out_path)

    # one row per query
    assert table.num_rows == nq

    # list-typed columns
    assert table.schema.field("neighbors_id").type == pa.list_(pa.int64())
    assert table.schema.field("distances").type == pa.list_(pa.float32())

    # each row contains exactly K elements
    for i in range(nq):
        assert len(table["neighbors_id"][i].as_py()) == k
        assert len(table["distances"][i].as_py()) == k


def test_parquet_metadata(tmp_path):
    nq, k, dim = 2, 3, 64
    neighbors = np.zeros((nq, k), dtype=np.int64)
    distances = np.zeros((nq, k), dtype=np.float32)

    out_prefix = tmp_path / "gt"
    out_path = write_gt_parquet(
        str(out_prefix),
        neighbors,
        distances,
        nq=nq,
        k=k,
        dim=dim,
    )

    meta = pq.ParquetFile(out_path).schema_arrow.metadata

    assert meta[b"K"] == b"3"
    assert meta[b"Q"] == b"2"
    assert meta[b"D"] == b"64"


def test_parquet_row_ordering(tmp_path):
    nq, k = 2, 3
    neighbors = np.array(
        [
            [10, 11, 12],
            [20, 21, 22],
        ],
        dtype=np.int64,
    )
    distances = np.zeros((nq, k), dtype=np.float32)

    out_prefix = tmp_path / "gt"
    out_path = write_gt_parquet(
        str(out_prefix),
        neighbors,
        distances,
        nq=nq,
        k=k,
        dim=1,
    )

    table = pq.read_table(out_path)

    # row 0 → query 0
    assert table["neighbors_id"][0].as_py() == [10, 11, 12]

    # row 1 → query 1
    assert table["neighbors_id"][1].as_py() == [20, 21, 22]
