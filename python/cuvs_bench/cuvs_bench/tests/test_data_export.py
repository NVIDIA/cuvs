#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Regression tests for in-process benchmark result persistence and export."""

import json
import stat

import numpy as np
import pandas as pd
import pytest

from cuvs_bench.backends.base import BuildResult, SearchResult
from cuvs_bench.orchestrator.result_files import (
    ResultRecord,
    write_result_files,
)
from cuvs_bench.run.data_export import (
    convert_json_to_csv_build,
    convert_json_to_csv_search,
    read_json_files,
)


_DATASET = "test-data"
_ALGORITHM = "pylucene_cuvs_hnsw"
_BUILD_STEM = f"{_ALGORITHM},base"
_SEARCH_STEM = f"{_BUILD_STEM},k5,bs2"
_CODECS = (
    "Lucene101AcceleratedHNSWCodec",
    "Lucene101AcceleratedHNSWBaseLayerCodec",
    "Lucene101AcceleratedHNSWMultiLayerCodec",
)


def _index_name(codec):
    return f"{_ALGORITHM}.codec{codec}"


def _build_result(
    codec,
    *,
    build_time=1.25,
    metadata=None,
    success=True,
    error_message=None,
):
    return BuildResult(
        index_path=f"/indexes/{_index_name(codec)}",
        build_time_seconds=build_time,
        index_size_bytes=4096,
        algorithm=_ALGORITHM,
        build_params={"codec": codec},
        metadata=metadata or {},
        success=success,
        error_message=error_message,
    )


def _search_result(
    codec,
    *,
    search_time_ms=6.0,
    latency_seconds=0.003,
    recall=0.9,
    queries_per_second=500.0,
    metadata=None,
    success=True,
    error_message=None,
):
    result_metadata = {"codec": codec, "num_batches": 2}
    if metadata:
        result_metadata.update(metadata)
    return SearchResult(
        neighbors=np.empty((0, 5), dtype=np.int64),
        distances=np.empty((0, 5), dtype=np.float32),
        search_time_ms=search_time_ms,
        queries_per_second=queries_per_second,
        recall=recall,
        algorithm=_ALGORITHM,
        search_params=[{}],
        latency_seconds=latency_seconds,
        latency_percentiles={"p50": 2.5, "p95": 3.5, "p99": 3.9},
        metadata=result_metadata,
        success=success,
        error_message=error_message,
    )


def _record(result, codec, output_filename):
    return ResultRecord(
        result=result,
        index_name=_index_name(codec),
        output_filename=output_filename,
    )


def _load_payload(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def test_write_result_files_groups_three_codecs_with_expected_schema(tmp_path):
    records = []
    for position, codec in enumerate(_CODECS, start=1):
        records.extend(
            [
                _record(
                    _build_result(codec, build_time=float(position)),
                    codec,
                    _BUILD_STEM,
                ),
                _record(
                    _search_result(
                        codec,
                        search_time_ms=float(position * 4),
                        latency_seconds=float(position) / 500.0,
                        recall=0.8 + position / 100.0,
                        queries_per_second=1000.0 / position,
                    ),
                    codec,
                    _SEARCH_STEM,
                ),
            ]
        )

    write_result_files(records, dataset=_DATASET, dataset_path=str(tmp_path))

    result_root = tmp_path / _DATASET / "result"
    build_path = result_root / "build" / f"{_BUILD_STEM}.json"
    search_path = result_root / "search" / f"{_SEARCH_STEM}.json"
    assert sorted(
        str(path.relative_to(result_root))
        for path in result_root.rglob("*.json")
    ) == [
        f"build/{_BUILD_STEM}.json",
        f"search/{_SEARCH_STEM}.json",
    ]

    build_rows = _load_payload(build_path)["benchmarks"]
    search_rows = _load_payload(search_path)["benchmarks"]
    assert len(build_rows) == len(_CODECS)
    assert len(search_rows) == len(_CODECS)
    assert {row["name"] for row in build_rows} == {
        f"{_index_name(codec)}/build" for codec in _CODECS
    }
    assert {row["name"] for row in search_rows} == {
        f"{_index_name(codec)}/search" for codec in _CODECS
    }

    first_build = build_rows[0]
    assert first_build == {
        "codec": _CODECS[0],
        "name": f"{_index_name(_CODECS[0])}/build",
        "real_time": 1.0,
        "time_unit": "s",
        "index_size": 4096,
        "success": True,
    }

    first_search = search_rows[0]
    assert first_search["real_time"] == pytest.approx(2.0)
    assert first_search["time_unit"] == "ms"
    assert first_search["search_time_ms"] == pytest.approx(4.0)
    assert first_search["Latency"] == pytest.approx(0.002)
    assert first_search["items_per_second"] == pytest.approx(1000.0)
    assert first_search["Recall"] == pytest.approx(0.81)
    assert first_search["p50"] == pytest.approx(2.5)
    assert first_search["p95"] == pytest.approx(3.5)
    assert first_search["p99"] == pytest.approx(3.9)
    assert first_search["codec"] == _CODECS[0]


def test_skipped_build_preserves_existing_positive_result(tmp_path):
    codec = _CODECS[0]
    output_path = (
        tmp_path / _DATASET / "result" / "build" / f"{_BUILD_STEM}.json"
    )
    write_result_files(
        [_record(_build_result(codec, build_time=4.5), codec, _BUILD_STEM)],
        dataset=_DATASET,
        dataset_path=str(tmp_path),
    )

    write_result_files(
        [
            _record(
                _build_result(
                    codec,
                    build_time=0.0,
                    metadata={"skipped": True},
                ),
                codec,
                _BUILD_STEM,
            )
        ],
        dataset=_DATASET,
        dataset_path=str(tmp_path),
    )

    rows = _load_payload(output_path)["benchmarks"]
    assert len(rows) == 1
    assert rows[0]["real_time"] == pytest.approx(4.5)
    assert "skipped" not in rows[0]


def test_skipped_build_replaces_existing_failed_result(tmp_path):
    codec = _CODECS[0]
    output_path = (
        tmp_path / _DATASET / "result" / "build" / f"{_BUILD_STEM}.json"
    )
    write_result_files(
        [
            _record(
                _build_result(
                    codec,
                    build_time=0.0,
                    success=False,
                    error_message="old failure",
                ),
                codec,
                _BUILD_STEM,
            )
        ],
        dataset=_DATASET,
        dataset_path=str(tmp_path),
    )

    write_result_files(
        [
            _record(
                _build_result(
                    codec,
                    build_time=0.0,
                    metadata={"skipped": True},
                ),
                codec,
                _BUILD_STEM,
            )
        ],
        dataset=_DATASET,
        dataset_path=str(tmp_path),
    )

    rows = _load_payload(output_path)["benchmarks"]
    assert len(rows) == 1
    assert rows[0]["success"] is True
    assert rows[0]["skipped"] is True
    assert "error_message" not in rows[0]

    csv_path = output_path.with_suffix(".csv")
    csv_path.write_text("stale\n", encoding="utf-8")
    convert_json_to_csv_build(_DATASET, str(tmp_path))
    assert not csv_path.exists()


def test_build_upsert_replaces_by_name_without_duplicates(tmp_path):
    initial_records = [
        _record(_build_result(codec, build_time=1.0), codec, _BUILD_STEM)
        for codec in _CODECS
    ]
    write_result_files(
        initial_records, dataset=_DATASET, dataset_path=str(tmp_path)
    )

    updated_codec = _CODECS[1]
    write_result_files(
        [
            _record(
                _build_result(
                    updated_codec,
                    build_time=7.5,
                    metadata={"writer_path": "gpu-hnsw"},
                ),
                updated_codec,
                _BUILD_STEM,
            )
        ],
        dataset=_DATASET,
        dataset_path=str(tmp_path),
    )

    path = tmp_path / _DATASET / "result" / "build" / f"{_BUILD_STEM}.json"
    rows = _load_payload(path)["benchmarks"]
    assert len(rows) == len(_CODECS)
    assert len({row["name"] for row in rows}) == len(_CODECS)
    updated = next(
        row
        for row in rows
        if row["name"] == f"{_index_name(updated_codec)}/build"
    )
    assert updated["real_time"] == pytest.approx(7.5)
    assert updated["writer_path"] == "gpu-hnsw"


def test_search_write_replaces_previous_rows(tmp_path):
    write_result_files(
        [
            _record(_search_result(codec), codec, _SEARCH_STEM)
            for codec in _CODECS
        ],
        dataset=_DATASET,
        dataset_path=str(tmp_path),
    )

    replacement_codec = _CODECS[-1]
    write_result_files(
        [
            _record(
                _search_result(
                    replacement_codec,
                    recall=0.99,
                    queries_per_second=750.0,
                ),
                replacement_codec,
                _SEARCH_STEM,
            )
        ],
        dataset=_DATASET,
        dataset_path=str(tmp_path),
    )

    path = tmp_path / _DATASET / "result" / "search" / f"{_SEARCH_STEM}.json"
    rows = _load_payload(path)["benchmarks"]
    assert len(rows) == 1
    assert rows[0]["name"] == f"{_index_name(replacement_codec)}/search"
    assert rows[0]["Recall"] == pytest.approx(0.99)
    assert rows[0]["items_per_second"] == pytest.approx(750.0)


@pytest.mark.parametrize(
    "output_filename",
    ["", ".", "..", "../escape", "nested/result", "/absolute/result", None],
)
def test_write_result_files_rejects_invalid_output_filename(
    tmp_path, output_filename
):
    codec = _CODECS[0]
    record = _record(_build_result(codec), codec, output_filename)

    with pytest.raises(ValueError, match="Invalid benchmark result filename"):
        write_result_files(
            [record], dataset=_DATASET, dataset_path=str(tmp_path)
        )

    assert not (tmp_path / _DATASET / "result").exists()


@pytest.mark.parametrize(
    "dataset",
    ["", ".", "..", "../escape", "nested/dataset", "/absolute/dataset", None],
)
def test_write_result_files_rejects_invalid_dataset_name(tmp_path, dataset):
    codec = _CODECS[0]
    record = _record(_build_result(codec), codec, _BUILD_STEM)

    with pytest.raises(ValueError, match="Invalid benchmark dataset name"):
        write_result_files(
            [record], dataset=dataset, dataset_path=str(tmp_path)
        )


@pytest.mark.parametrize(
    "dataset",
    ["", ".", "..", "../escape", "nested/dataset", "/absolute/dataset", None],
)
@pytest.mark.parametrize(
    "converter", [convert_json_to_csv_build, convert_json_to_csv_search]
)
def test_data_export_rejects_invalid_dataset_name(
    tmp_path, dataset, converter
):
    with pytest.raises(ValueError, match="Invalid benchmark dataset name"):
        converter(dataset, str(tmp_path))


def test_build_merge_rejects_invalid_existing_payload(tmp_path):
    build_dir = tmp_path / _DATASET / "result" / "build"
    build_dir.mkdir(parents=True)
    output_path = build_dir / f"{_BUILD_STEM}.json"
    output_path.write_text('{"benchmarks": {}}\n', encoding="utf-8")
    codec = _CODECS[0]

    with pytest.raises(ValueError, match="'benchmarks' list"):
        write_result_files(
            [_record(_build_result(codec), codec, _BUILD_STEM)],
            dataset=_DATASET,
            dataset_path=str(tmp_path),
        )

    assert _load_payload(output_path) == {"benchmarks": {}}


@pytest.mark.parametrize(
    "benchmarks",
    [
        [None],
        [{"real_time": 1.0}],
        [{"name": 42}],
        [{"name": "duplicate"}, {"name": "duplicate"}],
    ],
)
def test_build_merge_rejects_invalid_existing_rows(tmp_path, benchmarks):
    build_dir = tmp_path / _DATASET / "result" / "build"
    build_dir.mkdir(parents=True)
    output_path = build_dir / f"{_BUILD_STEM}.json"
    original_payload = {"benchmarks": benchmarks}
    output_path.write_text(json.dumps(original_payload), encoding="utf-8")
    codec = _CODECS[0]

    with pytest.raises(ValueError, match="Benchmark row|duplicate name"):
        write_result_files(
            [_record(_build_result(codec), codec, _BUILD_STEM)],
            dataset=_DATASET,
            dataset_path=str(tmp_path),
        )

    assert _load_payload(output_path) == original_payload


def test_atomic_result_write_uses_readable_and_preserved_file_modes(tmp_path):
    codec = _CODECS[0]
    output_path = (
        tmp_path / _DATASET / "result" / "build" / f"{_BUILD_STEM}.json"
    )
    record = _record(_build_result(codec), codec, _BUILD_STEM)

    write_result_files([record], dataset=_DATASET, dataset_path=str(tmp_path))
    assert stat.S_IMODE(output_path.stat().st_mode) == 0o644

    output_path.chmod(0o640)
    write_result_files([record], dataset=_DATASET, dataset_path=str(tmp_path))
    assert stat.S_IMODE(output_path.stat().st_mode) == 0o640


def test_result_write_invalidates_only_derived_csvs(tmp_path):
    result_root = tmp_path / _DATASET / "result"
    build_dir = result_root / "build"
    search_dir = result_root / "search"
    build_dir.mkdir(parents=True)
    search_dir.mkdir(parents=True)
    derived_paths = [
        build_dir / f"{_BUILD_STEM}.csv",
        search_dir / f"{_SEARCH_STEM},raw.csv",
        search_dir / f"{_SEARCH_STEM},throughput.csv",
        search_dir / f"{_SEARCH_STEM},latency.csv",
    ]
    unrelated_path = search_dir / "unrelated.csv"
    for path in [*derived_paths, unrelated_path]:
        path.write_text("stale\n", encoding="utf-8")

    codec = _CODECS[0]
    write_result_files(
        [
            _record(_build_result(codec), codec, _BUILD_STEM),
            _record(_search_result(codec), codec, _SEARCH_STEM),
        ],
        dataset=_DATASET,
        dataset_path=str(tmp_path),
    )

    assert not any(path.exists() for path in derived_paths)
    assert unrelated_path.read_text(encoding="utf-8") == "stale\n"


def test_missing_phase_directories_are_noop(tmp_path):
    assert list(read_json_files(_DATASET, str(tmp_path), "build")) == []
    assert list(read_json_files(_DATASET, str(tmp_path), "search")) == []

    convert_json_to_csv_build(_DATASET, str(tmp_path))
    convert_json_to_csv_search(_DATASET, str(tmp_path))

    assert not (tmp_path / _DATASET / "result").exists()


def test_failed_rows_remain_diagnostic_json_but_are_filtered_from_csv(
    tmp_path,
):
    good_codec, failed_codec = _CODECS[:2]
    write_result_files(
        [
            _record(_build_result(good_codec), good_codec, _BUILD_STEM),
            _record(
                _build_result(
                    failed_codec,
                    build_time=0.0,
                    success=False,
                    error_message="build failed",
                ),
                failed_codec,
                _BUILD_STEM,
            ),
            _record(_search_result(good_codec), good_codec, _SEARCH_STEM),
            _record(
                _search_result(
                    failed_codec,
                    search_time_ms=0.0,
                    latency_seconds=None,
                    recall=0.0,
                    queries_per_second=0.0,
                    success=False,
                    error_message="search failed",
                ),
                failed_codec,
                _SEARCH_STEM,
            ),
        ],
        dataset=_DATASET,
        dataset_path=str(tmp_path),
    )

    result_root = tmp_path / _DATASET / "result"
    build_json = result_root / "build" / f"{_BUILD_STEM}.json"
    search_json = result_root / "search" / f"{_SEARCH_STEM}.json"
    build_rows = _load_payload(build_json)["benchmarks"]
    search_rows = _load_payload(search_json)["benchmarks"]
    assert (
        next(row for row in build_rows if not row["success"])["error_message"]
        == "build failed"
    )
    assert (
        next(row for row in search_rows if not row["success"])["error_message"]
        == "search failed"
    )

    convert_json_to_csv_build(_DATASET, str(tmp_path))
    convert_json_to_csv_search(_DATASET, str(tmp_path))

    build_csv = pd.read_csv(build_json.with_suffix(".csv"))
    search_csv = pd.read_csv(search_json.with_name(f"{_SEARCH_STEM},raw.csv"))
    assert build_csv["index_name"].tolist() == [_index_name(good_codec)]
    assert search_csv["index_name"].tolist() == [_index_name(good_codec)]
    assert "success" not in build_csv.columns
    assert "error_message" not in build_csv.columns
    assert "success" not in search_csv.columns
    assert "error_message" not in search_csv.columns


def test_all_failed_json_removes_only_matching_derived_csvs(tmp_path):
    codec = _CODECS[0]
    write_result_files(
        [
            _record(
                _build_result(
                    codec,
                    build_time=0.0,
                    success=False,
                    error_message="build failed",
                ),
                codec,
                _BUILD_STEM,
            ),
            _record(
                _search_result(
                    codec,
                    search_time_ms=0.0,
                    latency_seconds=None,
                    recall=0.0,
                    queries_per_second=0.0,
                    success=False,
                    error_message="search failed",
                ),
                codec,
                _SEARCH_STEM,
            ),
        ],
        dataset=_DATASET,
        dataset_path=str(tmp_path),
    )

    result_root = tmp_path / _DATASET / "result"
    derived_paths = [
        result_root / "build" / f"{_BUILD_STEM}.csv",
        result_root / "search" / f"{_SEARCH_STEM},raw.csv",
        result_root / "search" / f"{_SEARCH_STEM},throughput.csv",
        result_root / "search" / f"{_SEARCH_STEM},latency.csv",
    ]
    unrelated_path = result_root / "search" / "unrelated.csv"
    for path in [*derived_paths, unrelated_path]:
        path.write_text("stale\n", encoding="utf-8")

    convert_json_to_csv_build(_DATASET, str(tmp_path))
    convert_json_to_csv_search(_DATASET, str(tmp_path))

    assert not any(path.exists() for path in derived_paths)
    assert unrelated_path.read_text(encoding="utf-8") == "stale\n"


def test_named_build_join_preserves_search_codec_and_build_metadata(tmp_path):
    codec = _CODECS[0]
    index_name = _index_name(codec)
    build_codec = "build-codec-sentinel"
    search_codec = "search-codec-sentinel"
    build_result = BuildResult(
        index_path=f"/indexes/{index_name}",
        build_time_seconds=2.75,
        index_size_bytes=8192,
        algorithm=_ALGORITHM,
        build_params={"codec": build_codec},
        metadata={"writer_path": "gpu-hnsw"},
    )
    search_result = _search_result(
        search_codec,
        metadata={"mode": "latency"},
    )
    write_result_files(
        [
            ResultRecord(build_result, index_name, _BUILD_STEM),
            ResultRecord(search_result, index_name, _SEARCH_STEM),
        ],
        dataset=_DATASET,
        dataset_path=str(tmp_path),
    )

    convert_json_to_csv_build(_DATASET, str(tmp_path))
    convert_json_to_csv_search(_DATASET, str(tmp_path))

    raw_path = (
        tmp_path / _DATASET / "result" / "search" / f"{_SEARCH_STEM},raw.csv"
    )
    row = pd.read_csv(raw_path).iloc[0]
    assert row["index_name"] == index_name
    assert row["codec"] == search_codec
    assert row["build time"] == pytest.approx(2.75)
    assert row["writer_path"] == "gpu-hnsw"
    assert row["mode"] == "latency"
    assert pd.isna(row["build threads"])
    assert pd.isna(row["build cpu_time"])


def test_scoped_search_joins_matching_build_and_preserves_legacy_stem(
    tmp_path,
):
    codec = _CODECS[0]
    full_index_name = _index_name(codec)
    subset_scope = "subset128"
    subset_index_name = f"{_ALGORITHM}.{subset_scope}.codec{codec}"
    subset_build_stem = f"{_BUILD_STEM},{subset_scope}"
    subset_search_stem = f"{subset_build_stem},k5,bs2"

    write_result_files(
        [
            ResultRecord(
                _build_result(codec, build_time=1.25),
                full_index_name,
                _BUILD_STEM,
            ),
            ResultRecord(
                _search_result(codec),
                full_index_name,
                _SEARCH_STEM,
            ),
            ResultRecord(
                _build_result(codec, build_time=4.5),
                subset_index_name,
                subset_build_stem,
            ),
            ResultRecord(
                _search_result(codec),
                subset_index_name,
                subset_search_stem,
            ),
        ],
        dataset=_DATASET,
        dataset_path=str(tmp_path),
    )

    convert_json_to_csv_build(_DATASET, str(tmp_path))
    convert_json_to_csv_search(_DATASET, str(tmp_path))

    result_root = tmp_path / _DATASET / "result"
    full_row = pd.read_csv(
        result_root / "search" / f"{_SEARCH_STEM},raw.csv"
    ).iloc[0]
    subset_row = pd.read_csv(
        result_root / "search" / f"{subset_search_stem},raw.csv"
    ).iloc[0]
    assert full_row["index_name"] == full_index_name
    assert full_row["build time"] == pytest.approx(1.25)
    assert subset_row["index_name"] == subset_index_name
    assert subset_row["build time"] == pytest.approx(4.5)
