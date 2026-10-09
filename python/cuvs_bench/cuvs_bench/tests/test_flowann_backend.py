# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from cuvs_bench.backends.flowann import parse_measurements


def measurement(
    kind="MEASURE", recall="0.95", round_id=0, query_count="10000"
):
    return (
        f"{kind} round={round_id} batch=1 query_count={query_count} "
        "search_ms=1 rerank_prep_ms=0.1 "
        f"rerank_ms=0.2 total_ms=1.3 recall={recall} qps=769.23 "
        "p50_ms=1.2 p95_ms=1.5 p99_ms=2"
    )


def test_excludes_warmup_and_summary():
    text = "\n".join(
        [
            measurement("WARMUP", "0.1"),
            measurement(),
            "SUMMARY recall_median=0.1",
        ]
    )
    assert parse_measurements(text, 1)[0]["recall"] == 0.95


@pytest.mark.parametrize(
    "text,count",
    [
        (measurement(recall="nan"), 1),
        (measurement(recall="1.1"), 1),
        (measurement(), 3),
        (measurement() + "\n" + measurement(), 2),
        (measurement(round_id=-1), 1),
        (measurement(round_id=0) + "\n" + measurement(round_id=2), 2),
    ],
)
def test_rejects_invalid_or_incomplete_results(text, count):
    with pytest.raises(ValueError):
        parse_measurements(text, count)


@pytest.mark.parametrize("first", [0, 1, 2])
def test_accepts_native_numbering_after_warmup(first):
    text = "\n".join(measurement(round_id=i) for i in range(first, first + 3))
    assert len(parse_measurements(text, 3)) == 3


@pytest.mark.parametrize("query_count", ["0", "1.5", "nan", "invalid"])
def test_rejects_invalid_query_count(query_count):
    with pytest.raises(ValueError, match="query_count"):
        parse_measurements(measurement(query_count=query_count), 1)


def test_rejects_missing_query_count():
    text = measurement().replace(" query_count=10000", "")
    with pytest.raises(ValueError, match="Missing.*query_count"):
        parse_measurements(text, 1)


def test_rejects_inconsistent_query_count():
    text = "\n".join(
        [
            measurement(round_id=0, query_count="10000"),
            measurement(round_id=1, query_count="5000"),
        ]
    )
    with pytest.raises(ValueError, match="consistent"):
        parse_measurements(text, 2)


@pytest.mark.parametrize(
    "requested_query_count,actual_query_count",
    [(0, 8000), (20000, 8000)],
)
def test_backend_uses_actual_query_count(
    tmp_path, monkeypatch, requested_query_count, actual_query_count
):
    from cuvs_bench.backends import Dataset, get_backend
    from cuvs_bench.orchestrator.config_loaders import IndexConfig
    import cuvs_bench.backends.flowann as module

    def fake_run(command, stdout, **kwargs):
        assert "--include-query-transfer" in command
        assert "True" not in command
        assert command[command.index("--cpu-threads") + 1] == "128"
        stdout.write(measurement("WARMUP", "0.1") + "\n")
        for i in range(3):
            stdout.write(
                measurement(round_id=i, query_count=str(actual_query_count))
                + "\n"
            )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    backend = get_backend(
        "flowann",
        config={
            "name": "test",
            "executable_path": "/unused/test_binary",
            "output_dir": str(tmp_path),
        },
    )
    dataset = Dataset(
        "test",
        base_file="base",
        query_file="queries",
        groundtruth_neighbors_file="truth",
    )
    index = IndexConfig(
        "test",
        "flowann",
        {},
        [
            {
                "include_query_transfer": True,
                "query_count": requested_query_count,
            }
        ],
        "index",
    )
    (result,) = backend.search(
        dataset, [index], k=10, batch_size=1, search_threads=128
    )
    assert result.recall == 0.95
    assert result.search_time_ms == 1.3 * actual_query_count
    assert len(result.metadata["rounds"]) == 3
    assert result.metadata["batch_latency_ms"] == 1.3
    assert result.queries_per_second == 769.23
    assert result.metadata["query_count"] == actual_query_count
