# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Search existing FlowANN indexes through FLOWANN_BENCHMARK."""

import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import uuid

import numpy as np

from .base import BenchmarkBackend, SearchResult


def parse_measurements(text, expected_rounds):
    """Read measured rounds only, rejecting incomplete or invalid runs."""
    rounds = []
    for line in text.splitlines():
        if not line.startswith("MEASURE "):
            continue
        fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
        row = {}
        for key in (
            "round",
            "batch",
            "query_count",
            "search_ms",
            "rerank_prep_ms",
            "rerank_ms",
            "total_ms",
            "recall",
            "qps",
            "p50_ms",
            "p95_ms",
            "p99_ms",
        ):
            if key not in fields:
                raise ValueError(f"Missing benchmark measurement field: {key}")
            try:
                row[key] = float(fields[key])
            except ValueError as error:
                raise ValueError(
                    f"Benchmark measurement field {key} must be numeric"
                ) from error
        if (
            not math.isfinite(row["query_count"])
            or row["query_count"] <= 0
            or not row["query_count"].is_integer()
        ):
            raise ValueError(
                "Measurement query_count must be a positive integer"
            )
        if not all(math.isfinite(value) for value in row.values()):
            raise ValueError("Non-finite benchmark measurement")
        if (
            not 0 <= row["recall"] <= 1
            or row["qps"] <= 0
            or row["total_ms"] <= 0
        ):
            raise ValueError("Invalid recall or timing")
        rounds.append(row)
    if len(rounds) != expected_rounds:
        raise ValueError(
            f"Expected {expected_rounds} measured rounds, got {len(rounds)}"
        )
    # Native programs use different starting round numbers.
    round_ids = sorted(row["round"] for row in rounds)
    first = round_ids[0]
    if (
        first < 0
        or not first.is_integer()
        or round_ids != list(range(int(first), int(first) + expected_rounds))
    ):
        raise ValueError(
            "Measured round IDs must be consecutive nonnegative integers"
        )
    if len({row["query_count"] for row in rounds}) != 1:
        raise ValueError(
            "Measurement query_count must be consistent across rounds"
        )
    return rounds


class FlowannBackend(BenchmarkBackend):
    """Keep native warmup/session semantics and return cuvs-bench results.

    config requires executable_path and output_dir. Optional command_prefix
    (an argv list) supports taskset; env supplies process-local environment.
    IndexConfig.file is an existing index, never an index to rebuild.
    """

    @property
    def algo(self):
        return self.config.get("algo", "flowann")

    def build(self, dataset, indexes, force=False, dry_run=False):
        raise NotImplementedError(
            "This backend searches existing indexes only"
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
        if mode != "latency":
            raise ValueError(
                "Use latency mode; batch size controls query batching"
            )
        executable = str(Path(self.config["executable_path"]).resolve())
        output = Path(self.config["output_dir"])
        results = []
        for index in indexes:
            for params in index.search_params:
                options = dict(params)
                forbidden = {
                    "index",
                    "base",
                    "queries",
                    "ground_truth",
                    "batch_size",
                    "top_k",
                    "no_rerank",
                    "no_session",
                    "queue_statistics",
                }
                if forbidden.intersection(options):
                    raise ValueError(
                        "Search parameters override required benchmark semantics"
                    )
                options.setdefault("warmup_rounds", 1)
                options.setdefault("measured_rounds", 3)
                options.setdefault("query_count", 10000)
                if (
                    options["warmup_rounds"] < 1
                    or options["measured_rounds"] < 1
                ):
                    raise ValueError(
                        "Warmup and measurement rounds must be positive"
                    )
                if search_threads is not None:
                    if (
                        "cpu_threads" in options
                        and options["cpu_threads"] != search_threads
                    ):
                        raise ValueError("Conflicting CPU thread counts")
                    options["cpu_threads"] = search_threads
                required = {
                    "index": index.file,
                    "base": dataset.base_file,
                    "queries": dataset.query_file,
                    "ground_truth": dataset.groundtruth_neighbors_file,
                    "top_k": k,
                    "batch_size": batch_size,
                }
                if any(value is None for value in required.values()):
                    raise ValueError(
                        "Index, base, queries and ground truth paths are required"
                    )
                command = list(self.config.get("command_prefix", [])) + [
                    executable
                ]
                for key, value in (required | options).items():
                    if not re.fullmatch(r"[a-z][a-z0-9_]*", key):
                        raise ValueError(f"Invalid option name: {key}")
                    flag = "--" + key.replace("_", "-")
                    if isinstance(value, bool):
                        if value:
                            command.append(flag)
                    else:
                        command += [flag, str(value)]
                if dry_run:
                    print(shlex.join(command))
                    continue
                output.mkdir(parents=True, exist_ok=True)
                stem = f"{index.algo}-bs{batch_size}-{uuid.uuid4().hex[:12]}"
                log = output / (stem + ".log")
                metadata = {
                    "command": command,
                    "params": options,
                    "index": index.file,
                    "batch_size": batch_size,
                }
                (output / (stem + ".command.json")).write_text(
                    json.dumps(metadata, indent=2) + "\n"
                )
                env = os.environ | self.config.get("env", {})
                with log.open("w") as stream:
                    subprocess.run(
                        command,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=True,
                        env=env,
                    )
                rounds = parse_measurements(
                    log.read_text(), options["measured_rounds"]
                )
                if any(row["batch"] != batch_size for row in rounds):
                    raise ValueError(
                        "Output batch does not match requested batch"
                    )
                medians = {
                    key: statistics.median(row[key] for row in rounds)
                    for key in rounds[0]
                }
                query_count = int(rounds[0]["query_count"])
                result = SearchResult(
                    neighbors=np.empty((0, k), dtype=np.int64),
                    distances=np.empty((0, k), dtype=np.float32),
                    search_time_ms=medians["total_ms"]
                    * query_count
                    / batch_size,
                    queries_per_second=medians["qps"],
                    recall=medians["recall"],
                    algorithm=index.algo,
                    search_params=[options],
                    latency_percentiles={
                        key: medians[key]
                        for key in ("p50_ms", "p95_ms", "p99_ms")
                    },
                    metadata={
                        "native_log": str(log),
                        "rounds": rounds,
                        "batch_latency_ms": medians["total_ms"],
                        "recall_min": min(row["recall"] for row in rounds),
                        "query_count": query_count,
                        "batch_size": batch_size,
                        "index": index.file,
                        "command": command,
                    },
                )
                (output / (stem + ".json")).write_text(
                    json.dumps(result.to_json(), indent=2) + "\n"
                )
                results.append(result)
        return results
