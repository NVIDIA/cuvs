#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from click.testing import CliRunner
from cuvs_bench.get_dataset.__main__ import main


@pytest.fixture(scope="session")
def temp_datasets_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("datasets")


def test_get_dataset_creates_expected_files(temp_datasets_dir: Path):
    runner = CliRunner()
    dataset_path_arg = str(temp_datasets_dir)

    # Invoke the CLI command as if calling:
    # python -m cuvs_bench.get_dataset --dataset test-data \
    # --dataset-path <temp_datasets_dir>
    result = runner.invoke(
        main, ["--dataset", "test-data", "--dataset-path", dataset_path_arg]
    )

    assert result.exit_code == 0, f"CLI call failed: {result.output}"

    expected_files = [
        "test-data/ann_benchmarks_like.groundtruth.distances.fbin",
        "test-data/ann_benchmarks_like.base.fbin",
        "test-data/ann_benchmarks_like.groundtruth.neighbors.ibin",
        "test-data/ann_benchmarks_like.query.fbin",
        "test-data/ann_benchmarks_like.hdf5",
    ]

    # Verify that each expected file exists in the datasets directory.
    for filename in expected_files:
        file_path = temp_datasets_dir / filename
        assert file_path.exists(), (
            f"Expected file {filename} was not generated."
        )


def test_run_command_creates_results(temp_datasets_dir: Path):
    """
    This test simulates running the command:

        python -m cuvs_bench.run --dataset test-data --dataset-path datasets/ \
            --algorithms faiss_gpu_ivf_flat,faiss_gpu_ivf_sq,cuvs_ivf_flat,\
            cuvs_cagra,ggnn,cuvs_cagra_hnswlib,cuvs_ivf_pq \
            --batch-size 100 -k 10 --groups test -m latency --force

    It then verifies that the set of expected result files
         (both under result/build and result/search)
         are created under datasets/test-data/ and are not empty.
    """

    dataset_path_arg = str(temp_datasets_dir)

    from cuvs_bench.run.__main__ import main as run_main

    runner = CliRunner()
    run_args = [
        "--dataset",
        "test-data",
        "--dataset-path",
        dataset_path_arg,
        "--algorithms",
        "faiss_gpu_ivf_flat,faiss_gpu_ivf_sq,cuvs_ivf_flat,cuvs_cagra,ggnn,cuvs_cagra_hnswlib,cuvs_ivf_pq",  # noqa: E501
        "--batch-size",
        "100",
        "-k",
        "10",
        "--groups",
        "test",
        "-m",
        "latency",
        "--force",
    ]
    result = runner.invoke(run_main, run_args)
    assert result.exit_code == 0, (
        f"Run command failed with output:\n{result.output}"
    )

    common_build_header = [
        "algo_name",
        "index_name",
        "time",
        "threads",
        "cpu_time",
    ]

    common_search_header = [
        "algo_name",
        "index_name",
        "recall",
        "throughput",
        "latency",
        "threads",
        "cpu_time",
    ]

    # --- Verify that the expected result files exist and are not empty ---
    expected_files = {
        # Build files:
        "test-data/result/build/cuvs_ivf_flat,test.csv": {
            "header": common_build_header
            + [
                "GPU",
                "niter",
                "nlist",
                "ratio",
            ],
            "rows": 1,
        },
        "test-data/result/build/cuvs_cagra_hnswlib,test.csv": {
            "header": common_build_header
            + [
                "ef_construction",
                "graph_degree",
                "intermediate_graph_degree",
                "label",
            ],
            "rows": 2,
        },
        "test-data/result/build/faiss_gpu_ivf_flat,test.csv": {
            "header": common_build_header
            + [
                "GPU",
                "nlist",
                "ratio",
                "use_cuvs",
            ],
            "rows": 1,
        },
        "test-data/result/build/cuvs_cagra,test.csv": {
            "header": common_build_header
            + [
                "GPU",
                "graph_degree",
                "intermediate_graph_degree",
                "label",
            ],
            "rows": 1,
        },
        "test-data/result/build/cuvs_ivf_pq,test.csv": {
            "header": common_build_header
            + [
                "GPU",
                "niter",
                "nlist",
                "pq_bits",
                "pq_dim",
                "ratio",
            ],
            "rows": 1,
        },
        # Search files:
        "test-data/result/search/cuvs_cagra_hnswlib,test,k10,bs100,raw.csv": {
            "header": common_search_header
            + [
                "ef",
                "end_to_end",
                "k",
                "n_queries",
                "total_queries",
                "build time",
                "build threads",
                "build cpu_time",
                "ef_construction",
                "graph_degree",
                "intermediate_graph_degree",
                "label",
            ],
            "rows": 4,
        },
        "test-data/result/search/cuvs_cagra,test,k10,bs100,latency.csv": {
            "header": common_search_header
            + [
                "GPU",
                "end_to_end",
                "itopk",
                "k",
                "n_queries",
                "search_width",
                "total_queries",
                "build time",
                "build threads",
                "build cpu_time",
                "build GPU",
                "graph_degree",
                "intermediate_graph_degree",
                "label",
            ],
            "rows": 2,
        },
        "test-data/result/search/cuvs_cagra,test,k10,bs100,throughput.csv": {
            "header": common_search_header
            + [
                "GPU",
                "end_to_end",
                "itopk",
                "k",
                "n_queries",
                "search_width",
                "total_queries",
                "build time",
                "build threads",
                "build cpu_time",
                "build GPU",
                "graph_degree",
                "intermediate_graph_degree",
                "label",
            ],
            "rows": 2,
        },
        "test-data/result/search/cuvs_cagra,test,k10,bs100,raw.csv": {
            "header": common_search_header
            + [
                "GPU",
                "end_to_end",
                "itopk",
                "k",
                "n_queries",
                "search_width",
                "total_queries",
                "build time",
                "build threads",
                "build cpu_time",
                "build GPU",
                "graph_degree",
                "intermediate_graph_degree",
                "label",
            ],
            "rows": 2,
        },
        "test-data/result/search/cuvs_ivf_flat,test,k10,bs100,latency.csv": {
            "header": common_search_header
            + [
                "GPU",
                "end_to_end",
                "k",
                "n_queries",
                "nprobe",
                "total_queries",
                "build time",
                "build threads",
                "build cpu_time",
                "build GPU",
                "niter",
                "nlist",
                "ratio",
            ],
            "rows": 2,
        },
        "test-data/result/search/cuvs_ivf_flat,test,k10,bs100,raw.csv": {
            "header": common_search_header
            + [
                "GPU",
                "end_to_end",
                "k",
                "n_queries",
                "nprobe",
                "total_queries",
                "build time",
                "build threads",
                "build cpu_time",
                "build GPU",
                "niter",
                "nlist",
                "ratio",
            ],
            "rows": 2,
        },
        "test-data/result/search/cuvs_ivf_flat,test,k10,bs100,throughput.csv": {  # noqa: E501
            "header": common_search_header
            + [
                "GPU",
                "end_to_end",
                "k",
                "n_queries",
                "nprobe",
                "total_queries",
                "build time",
                "build threads",
                "build cpu_time",
                "build GPU",
                "niter",
                "nlist",
                "ratio",
            ],
            "rows": 2,
        },
        "test-data/result/search/faiss_gpu_ivf_flat,test,k10,bs100,latency.csv": {  # noqa: E501
            "header": common_search_header
            + [
                "GPU",
                "end_to_end",
                "k",
                "n_queries",
                "nprobe",
                "total_queries",
                "build time",
                "build threads",
                "build cpu_time",
                "build GPU",
                "nlist",
                "ratio",
                "use_cuvs",
            ],
            "rows": 2,
        },
        "test-data/result/search/faiss_gpu_ivf_flat,test,k10,bs100,raw.csv": {
            "header": common_search_header
            + [
                "GPU",
                "end_to_end",
                "k",
                "n_queries",
                "nprobe",
                "total_queries",
                "build time",
                "build threads",
                "build cpu_time",
                "build GPU",
                "nlist",
                "ratio",
                "use_cuvs",
            ],
            "rows": 2,
        },
        "test-data/result/search/faiss_gpu_ivf_flat,test,k10,bs100,throughput.csv": {  # noqa: E501
            "header": common_search_header
            + [
                "GPU",
                "end_to_end",
                "k",
                "n_queries",
                "nprobe",
                "total_queries",
                "build time",
                "build threads",
                "build cpu_time",
                "build GPU",
                "nlist",
                "ratio",
                "use_cuvs",
            ],
            "rows": 2,
        },
        "test-data/result/search/cuvs_ivf_pq,test,k10,bs100,raw.csv": {
            "header": common_search_header
            + [
                "GPU",
                "end_to_end",
                "k",
                "n_queries",
                "nprobe",
                "refine_ratio",
                "total_queries",
                "search_label",
                "build time",
                "build threads",
                "build cpu_time",
                "build GPU",
                "niter",
                "nlist",
                "pq_bits",
                "pq_dim",
                "ratio",
            ],
            "rows": 2,
        },
        "test-data/result/search/cuvs_ivf_pq,test,k10,bs100,latency.csv": {
            "header": common_search_header
            + [
                "GPU",
                "end_to_end",
                "k",
                "n_queries",
                "nprobe",
                "refine_ratio",
                "total_queries",
                "search_label",
                "build time",
                "build threads",
                "build cpu_time",
                "build GPU",
                "niter",
                "nlist",
                "pq_bits",
                "pq_dim",
                "ratio",
            ],
            "rows": 2,
        },
        "test-data/result/search/cuvs_ivf_pq,test,k10,bs100,throughput.csv": {
            "header": common_search_header
            + [
                "GPU",
                "end_to_end",
                "k",
                "n_queries",
                "nprobe",
                "refine_ratio",
                "total_queries",
                "search_label",
                "build time",
                "build threads",
                "build cpu_time",
                "build GPU",
                "niter",
                "nlist",
                "pq_bits",
                "pq_dim",
                "ratio",
            ],
            "rows": 2,
        },
    }

    for rel_path, expectations in expected_files.items():
        file_path = temp_datasets_dir / rel_path
        assert file_path.exists(), f"Expected file {file_path} does not exist."
        assert file_path.stat().st_size > 0, (
            f"Expected file {file_path} is empty."
        )

        df = pd.read_csv(file_path)

        actual_header = list(df.columns)
        actual_rows = len(df)

        # breakpoint()
        assert actual_header == expectations["header"], (
            f"Wrong header produced in file f{rel_path}"
        )
        is_frontier = rel_path.endswith(("latency.csv", "throughput.csv"))
        if is_frontier:
            # Frontier files may have fewer rows than the raw results
            # because the Pareto frontier drops dominated points.
            assert 1 <= actual_rows <= expectations["rows"], (
                f"Frontier file {rel_path} has {actual_rows} row(s), "
                f"expected between 1 and {expectations['rows']}"
            )
            if actual_rows < expectations["rows"]:
                print(
                    f"Note: {rel_path} has {actual_rows} row(s), "
                    f"expected {expectations['rows']} "
                    f"(Pareto frontier dropped dominated points)"
                )
        else:
            assert actual_rows == expectations["rows"], (
                f"Expected {expectations['rows']} rows in {rel_path}, "
                f"got {actual_rows}"
            )


def test_plot_command_creates_png_files(temp_datasets_dir: Path):
    """
    This test simulates running the command:

      python -m cuvs_bench.plot --dataset test-data --dataset-path datasets/ \
          --algorithms faiss_gpu_ivf_flat,faiss_gpu_ivf_sq, \
          cuvs_ivf_flat,cuvs_cagra,ggnn,cuvs_cagra_hnswlib,cuvs_ivf_pq \
          --batch-size 100 -k 10 --groups test -m latency

    and then verifies that the following files are produced in the
    working directory:
      - search-test-data-k10-batch_size100.png
      - build-test-data-k10-batch_size100.png

    It also checks that these files are not empty.
    """

    dataset_path_arg = str(temp_datasets_dir)

    from cuvs_bench.plot.__main__ import main as plot_main

    runner = CliRunner()
    args = [
        "--dataset",
        "test-data",
        "--dataset-path",
        dataset_path_arg,
        "--output-filepath",
        dataset_path_arg,
        "--algorithms",
        "faiss_gpu_ivf_flat,faiss_gpu_ivf_sq,cuvs_ivf_flat,cuvs_cagra,ggnn,cuvs_cagra_hnswlib,cuvs_ivf_pq",  # noqa: E501
        "--batch-size",
        "100",
        "-k",
        "10",
        "--groups",
        "test",
        "-m",
        "latency",
    ]
    result = runner.invoke(plot_main, args)
    assert result.exit_code == 0, (
        f"Plot command failed with output:\n{result.output}"
    )

    # Expected output file names.
    expected_files = [
        "search-test-data-k10-batch_size100.png",
        "build-test-data-k10-batch_size100.png",
    ]

    for filename in expected_files:
        file_path = temp_datasets_dir / filename
        assert file_path.exists(), f"Expected file {filename} does not exist."
        assert file_path.stat().st_size > 0, (
            f"Expected file {filename} is empty."
        )


# The mocked-result tests below isolate CLI outcome and export semantics.
# Later tests use --dry-run to verify flag parsing and orchestrator routing
# without requiring native benchmark execution.


def _invoke_run_with_results(monkeypatch, tmp_path, mode, results):
    from cuvs_bench.run import __main__ as run_module

    orchestrator = SimpleNamespace(
        run_benchmark=lambda **_kwargs: results,
    )
    monkeypatch.setattr(
        run_module,
        "BenchmarkOrchestrator",
        lambda backend_type: orchestrator,
    )
    exported = []
    monkeypatch.setattr(
        run_module,
        "convert_json_to_csv_build",
        lambda dataset, dataset_path: exported.append(
            ("build", dataset, dataset_path)
        ),
    )
    monkeypatch.setattr(
        run_module,
        "convert_json_to_csv_search",
        lambda dataset, dataset_path: exported.append(
            ("search", dataset, dataset_path)
        ),
    )

    result = CliRunner().invoke(
        run_module.main,
        [
            "--dataset",
            "test-data",
            "--dataset-path",
            str(tmp_path),
            "--algorithms",
            "fake",
            "--groups",
            "base",
            "--batch-size",
            "1",
            "-k",
            "1",
            "-m",
            "latency",
            "--mode",
            mode,
        ],
    )
    return result, exported


def _benchmark_result(success, error_message=None):
    return SimpleNamespace(
        success=success,
        algorithm="fake",
        error_message=error_message,
    )


def test_tune_mixed_trial_results_exit_zero_and_export(monkeypatch, tmp_path):
    result, exported = _invoke_run_with_results(
        monkeypatch,
        tmp_path,
        "tune",
        [
            _benchmark_result(False, "trial failed"),
            _benchmark_result(True),
        ],
    )

    assert result.exit_code == 0, result.output
    assert exported == [
        ("build", "test-data", str(tmp_path)),
        ("search", "test-data", str(tmp_path)),
    ]


@pytest.mark.parametrize(
    ("results", "expected_message"),
    [
        ([_benchmark_result(False, "trial failed")], "trial failed"),
        ([], "tune mode produced no benchmark results"),
    ],
)
def test_tune_without_successful_results_exits_nonzero(
    monkeypatch, tmp_path, results, expected_message
):
    result, exported = _invoke_run_with_results(
        monkeypatch, tmp_path, "tune", results
    )

    assert result.exit_code != 0
    assert expected_message in result.output
    assert exported == []


def test_tune_all_constraint_pruned_measurements_exit_zero_and_export(
    monkeypatch, tmp_path
):
    # _run_tune retains successful measurements even when Optuna prunes every
    # trial for violating a hard constraint.
    result, exported = _invoke_run_with_results(
        monkeypatch,
        tmp_path,
        "tune",
        [_benchmark_result(True), _benchmark_result(True)],
    )

    assert result.exit_code == 0, result.output
    assert exported == [
        ("build", "test-data", str(tmp_path)),
        ("search", "test-data", str(tmp_path)),
    ]


def test_sweep_mixed_results_exit_nonzero(monkeypatch, tmp_path):
    result, exported = _invoke_run_with_results(
        monkeypatch,
        tmp_path,
        "sweep",
        [
            _benchmark_result(False, "benchmark failed"),
            _benchmark_result(True),
        ],
    )

    assert result.exit_code != 0
    assert "benchmark failed" in result.output
    assert exported == []


def test_sweep_without_results_exits_nonzero(monkeypatch, tmp_path):
    result, exported = _invoke_run_with_results(
        monkeypatch, tmp_path, "sweep", []
    )

    assert result.exit_code != 0
    assert "sweep mode produced no benchmark results" in result.output
    assert exported == []


@pytest.mark.parametrize(
    "dataset", ["..", "../escape", "nested/dataset", "/absolute/dataset"]
)
def test_data_export_rejects_invalid_dataset_name(tmp_path, dataset):
    from cuvs_bench.run.__main__ import main as run_main

    runner = CliRunner()
    result = runner.invoke(
        run_main,
        [
            "--data-export",
            "--dataset",
            dataset,
            "--dataset-path",
            str(tmp_path),
            "--count",
            "10",
            "--batch-size",
            "100",
            "--algorithms",
            "cuvs_cagra",
            "--groups",
            "base",
            "--search-mode",
            "latency",
        ],
    )

    assert result.exit_code == 2
    assert "Invalid benchmark dataset name" in result.output


def test_run_with_mode_sweep(temp_datasets_dir):
    """Verify --mode sweep is accepted (default behavior)."""
    from cuvs_bench.run.__main__ import main as run_main

    runner = CliRunner()
    result = runner.invoke(
        run_main,
        [
            "--dataset",
            "test-data",
            "--dataset-path",
            str(temp_datasets_dir),
            "--algorithms",
            "cuvs_cagra",
            "--batch-size",
            "10",
            "-k",
            "10",
            "--groups",
            "test",
            "-m",
            "latency",
            "--mode",
            "sweep",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, (
        f"--mode sweep failed with output:\n{result.output}"
    )


def test_run_with_backend_config(temp_datasets_dir, tmp_path):
    """Verify --backend-config YAML is parsed correctly."""
    from cuvs_bench.run.__main__ import main as run_main

    config_file = tmp_path / "backend.yaml"
    config_file.write_text("backend: cpp_gbench\n")
    runner = CliRunner()
    result = runner.invoke(
        run_main,
        [
            "--dataset",
            "test-data",
            "--dataset-path",
            str(temp_datasets_dir),
            "--algorithms",
            "cuvs_cagra",
            "--batch-size",
            "10",
            "-k",
            "10",
            "--groups",
            "test",
            "-m",
            "latency",
            "--backend-config",
            str(config_file),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, (
        f"--backend-config failed with output:\n{result.output}"
    )


def test_run_with_invalid_backend_config(tmp_path):
    """Verify missing backend config file raises error."""
    from cuvs_bench.run.__main__ import main as run_main

    runner = CliRunner()
    result = runner.invoke(
        run_main,
        [
            "--dataset",
            "test-data",
            "--dataset-path",
            str(tmp_path),
            "--algorithms",
            "cuvs_cagra",
            "--batch-size",
            "10",
            "-k",
            "10",
            "--groups",
            "test",
            "-m",
            "latency",
            "--backend-config",
            "/nonexistent/config.yaml",
        ],
    )
    assert result.exit_code != 0


def test_run_with_n_trials_flag(temp_datasets_dir):
    """Verify --n-trials flag is accepted by the CLI parser."""
    from cuvs_bench.run.__main__ import main as run_main

    runner = CliRunner()
    result = runner.invoke(
        run_main,
        [
            "--dataset",
            "test-data",
            "--dataset-path",
            str(temp_datasets_dir),
            "--algorithms",
            "cuvs_cagra",
            "--batch-size",
            "10",
            "-k",
            "10",
            "--groups",
            "test",
            "-m",
            "latency",
            "--n-trials",
            "50",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, (
        f"--n-trials failed with output:\n{result.output}"
    )


def test_run_with_constraints_flag(temp_datasets_dir):
    """Verify --constraints flag accepts valid JSON."""
    from cuvs_bench.run.__main__ import main as run_main

    runner = CliRunner()
    result = runner.invoke(
        run_main,
        [
            "--dataset",
            "test-data",
            "--dataset-path",
            str(temp_datasets_dir),
            "--algorithms",
            "cuvs_cagra",
            "--batch-size",
            "10",
            "-k",
            "10",
            "--groups",
            "test",
            "-m",
            "latency",
            "--constraints",
            '{"recall": "maximize", "latency": {"max": 10}}',
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, (
        f"--constraints failed with output:\n{result.output}"
    )
