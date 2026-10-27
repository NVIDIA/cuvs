#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import csv
import json
import os
import stat
import sys

import numpy as np
import pytest
import yaml
from click.testing import CliRunner

from cuvs_bench.orchestrator.config_loaders import CppGBenchConfigLoader
from cuvs_bench.run.data_export import convert_json_to_csv_build
from cuvs_bench.run_cluster.__main__ import main
from cuvs_bench.run_cluster.run_cluster import cluster_name, prepare_conf

EXECUTABLE = "CUVS_KMEANS_CLUSTER_BENCH"


@pytest.fixture
def gpu_present(monkeypatch):
    monkeypatch.setattr(
        CppGBenchConfigLoader, "gpu_present", property(lambda self: True)
    )


@pytest.fixture
def dataset_dir(tmp_path):
    """A tiny blobs dataset plus a datasets.yaml pointing at it."""
    rng = np.random.default_rng(0)
    centers = rng.uniform(-10, 10, (8, 16)).astype(np.float32)
    data = centers[rng.integers(0, 8, 2000)] + rng.standard_normal(
        (2000, 16), dtype=np.float32
    )
    os.makedirs(tmp_path / "blobs")
    with open(tmp_path / "blobs" / "base.fbin", "wb") as f:
        np.array(data.shape, dtype=np.uint32).tofile(f)
        data.astype(np.float32).tofile(f)
    with open(tmp_path / "datasets.yaml", "w") as f:
        yaml.safe_dump([{"name": "blobs", "base_file": "blobs/base.fbin"}], f)
    return tmp_path


def _fake_executable(directory, exit_code=0):
    """A stand-in for the C++ binary that writes canned gbench output."""
    path = os.path.join(directory, EXECUTABLE)
    script = f"""#!{sys.executable}
import json, sys
args = dict(a.split("=", 1) for a in sys.argv[1:-1] if "=" in a)
conf = json.load(open(sys.argv[-1]))
benchmarks = [
    {{"name": c["name"] + "/process_time/real_time", "run_type": "iteration",
      "iterations": 3, "real_time": 1.5, "cpu_time": 1.6, "time_unit": "ms",
      "inertia": 42.0, **c["param"]}}
    for c in conf["clustering"]
]
json.dump({{"context": {{"dataset": conf["dataset"]["name"]}},
           "benchmarks": benchmarks}}, open(args["--benchmark_out"], "w"))
sys.exit({exit_code})
"""
    with open(path, "w") as f:
        f.write(script)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
    return directory


def test_cluster_name():
    params = {"n_clusters": 16, "max_iter": 20}
    assert (
        cluster_name("cuvs_kmeans", "base", params)
        == "cuvs_kmeans.n_clusters16.max_iter20"
    )
    assert (
        cluster_name("cuvs_kmeans", "test", params)
        == "cuvs_kmeans_test.n_clusters16.max_iter20"
    )


def test_prepare_conf():
    dataset_conf = {
        "name": "d",
        "base_file": "d/base.u8bin",
        "subset_size": 1000,
    }
    group_conf = {"fit": {"n_clusters": [16, 32], "n_iters": [10]}}
    conf = prepare_conf(
        dataset_conf, "cuvs_kmeans_balanced", "base", group_conf
    )
    assert conf["dataset"] == {
        "name": "d",
        "base_file": "d/base.u8bin",
        "dtype": "uint8",
        "subset_size": 1000,
    }
    assert [c["param"] for c in conf["clustering"]] == [
        {"n_clusters": 16, "n_iters": 10},
        {"n_clusters": 32, "n_iters": 10},
    ]
    assert {c["algo"] for c in conf["clustering"]} == {"cuvs_kmeans_balanced"}

    # An explicit subset size overrides the dataset's.
    conf = prepare_conf(dataset_conf, "a", "base", group_conf, 10)
    assert conf["dataset"]["subset_size"] == 10


def test_convert_json_to_csv_cluster(tmp_path):
    os.makedirs(tmp_path / "d" / "result" / "cluster")
    out = tmp_path / "d" / "result" / "cluster" / "cuvs_kmeans,test.json"
    out.write_text(
        json.dumps(
            {
                "benchmarks": [
                    {
                        "name": "cuvs_kmeans_test.n_clusters16/process_time/real_time",
                        "iterations": 10,
                        "real_time": 5.3,
                        "cpu_time": 5.5,
                        "time_unit": "ms",
                        "inertia": 320556.0,
                        "n_iter": 2.0,
                        "n_clusters": 16.0,
                    },
                    {
                        "name": "cuvs_kmeans_test.n_clusters32/process_time/real_time",
                        "error_occurred": True,
                        "error_message": "boom",
                    },
                ]
            }
        )
    )
    convert_json_to_csv_build("d", str(tmp_path), method="cluster")
    with open(str(out).removesuffix(".json") + ".csv") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["algo_name"] == "cuvs_kmeans_test"
    assert rows[0]["index_name"] == "cuvs_kmeans_test.n_clusters16"
    assert rows[0]["time"] == "5.3"
    assert rows[0]["inertia"] == "320556.0"
    assert "iterations" not in rows[0]
    assert rows[1]["error_message"] == "boom"


def test_cli_dry_run(dataset_dir, gpu_present):
    result = CliRunner().invoke(
        main,
        [
            "--dataset=blobs",
            f"--dataset-path={dataset_dir}",
            f"--dataset-configuration={dataset_dir / 'datasets.yaml'}",
            "--groups=test",
            f"--executable-dir={_fake_executable(dataset_dir)}",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "cuvs_kmeans,test" in result.output
    assert "cuvs_kmeans_balanced,test" in result.output
    assert not (dataset_dir / "blobs" / "result").exists()


def test_cli_fake_executable(dataset_dir, gpu_present):
    args = [
        "--dataset=blobs",
        f"--dataset-path={dataset_dir}",
        f"--dataset-configuration={dataset_dir / 'datasets.yaml'}",
        "--algorithms=cuvs_kmeans",
        "--groups=test",
        f"--executable-dir={_fake_executable(dataset_dir)}",
    ]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    csv_file = (
        dataset_dir / "blobs" / "result" / "cluster" / "cuvs_kmeans,test.csv"
    )
    with open(csv_file) as f:
        rows = list(csv.DictReader(f))
    assert [r["index_name"] for r in rows] == [
        "cuvs_kmeans_test.n_clusters16.max_iter20.tol0.0001.initkmeans++"
    ]
    assert rows[0]["inertia"] == "42.0"

    # Existing results are kept unless --force is given.
    os.remove(csv_file)
    assert CliRunner().invoke(main, args).exit_code == 0
    assert not csv_file.exists()
    assert CliRunner().invoke(main, args + ["--force"]).exit_code == 0
    assert csv_file.exists()


def test_cli_executable_failure(dataset_dir, gpu_present):
    result = CliRunner().invoke(
        main,
        [
            "--dataset=blobs",
            f"--dataset-path={dataset_dir}",
            f"--dataset-configuration={dataset_dir / 'datasets.yaml'}",
            "--algorithms=cuvs_kmeans",
            "--groups=test",
            f"--executable-dir={_fake_executable(dataset_dir, exit_code=1)}",
        ],
    )
    assert result.exit_code != 0
    assert "cuvs_kmeans,test" in str(result.exception)


def test_cli_end_to_end(dataset_dir):
    """Runs the real CUVS_KMEANS_CLUSTER_BENCH; skipped if it is not found."""
    executable_dir = os.environ.get("CUVS_CLUSTER_BENCH_DIR")
    if not CppGBenchConfigLoader(executable_subdir="cluster").get_build_path(
        EXECUTABLE, executable_dir
    ):
        pytest.skip(f"{EXECUTABLE} not found")
    args = [
        "--dataset=blobs",
        f"--dataset-path={dataset_dir}",
        f"--dataset-configuration={dataset_dir / 'datasets.yaml'}",
        "--groups=test",
    ]
    if executable_dir:
        args.append(f"--executable-dir={executable_dir}")
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    for algo in ["cuvs_kmeans", "cuvs_kmeans_balanced"]:
        with open(
            dataset_dir / "blobs" / "result" / "cluster" / f"{algo},test.csv"
        ) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert not rows[0].get("error_message")
        assert float(rows[0]["inertia"]) > 0
        assert float(rows[0]["time"]) > 0
