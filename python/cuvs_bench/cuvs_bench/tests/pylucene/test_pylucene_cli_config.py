#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""CLI and configuration tests for the PyLucene backend."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner

import cuvs_bench.backends.pylucene as pylucene_backend
from cuvs_bench.backends import get_registry
from cuvs_bench.backends.base import SearchResult
from cuvs_bench.backends.pylucene import (
    PyLuceneConfigLoader,
    _SearchHit,
)
from cuvs_bench.backends.registry import list_config_loaders
from cuvs_bench.backends.search_spaces import get_search_space
from cuvs_bench.orchestrator.orchestrator import BenchmarkOrchestrator
from cuvs_bench.tests.pylucene._pylucene_test_utils import (
    _CAGRA_CODEC,
    _HNSW_CODEC,
    _FakeRuntime,
    _write_test_bin,
)


_DEFAULT_HNSW_BUILD_PARAMETERS = {
    "codec": _HNSW_CODEC,
    "m": 32,
    "ef_construction": 32,
    "direct_single_segment": False,
}


def _install_single_trial_optuna(monkeypatch):
    suggested_ranges = {}
    selected_parameters = {}

    def suggest_int(name, minimum, maximum, *, log=False):
        suggested_ranges[name] = (minimum, maximum, log)
        selected_parameters[name] = minimum
        return minimum

    trial = SimpleNamespace(
        number=0,
        suggest_int=suggest_int,
        suggest_float=lambda name, minimum, maximum, **kwargs: minimum,
        suggest_categorical=lambda name, choices: choices[0],
    )
    study = SimpleNamespace(
        best_trial=trial,
        best_params=selected_parameters,
        best_value=0.0,
    )

    def optimize(objective, **kwargs):
        study.best_value = objective(trial)

    study.optimize = optimize
    fake_optuna = SimpleNamespace(
        TrialPruned=RuntimeError,
        create_study=lambda **kwargs: study,
        logging=SimpleNamespace(WARNING=0, set_verbosity=lambda level: None),
    )
    monkeypatch.setitem(sys.modules, "optuna", fake_optuna)
    return suggested_ranges


def _hnsw_index_name(
    algorithm: str,
    group: str,
    *,
    subset_scope: str | None = None,
    m: int = 32,
    ef_construction: int = 32,
    direct_single_segment: bool = False,
) -> str:
    identity = f"{algorithm}[group={group}]"
    if subset_scope is not None:
        identity = f"{identity}[scope={subset_scope}]"
    return (
        f"{identity}[codec={_HNSW_CODEC}]"
        f"[m={m}]"
        f"[ef_construction={ef_construction}]"
        f"[direct_single_segment={str(direct_single_segment).lower()}]"
    )


def _prepare_cli_dataset(dataset_path: Path) -> None:
    training_vectors = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    query_vectors = training_vectors[:2].copy()
    groundtruth_neighbors = np.asarray([[0], [1]], dtype=np.int32)
    dataset_dir = dataset_path / "test-dataset"
    _write_test_bin(dataset_dir / "base.fbin", training_vectors)
    _write_test_bin(dataset_dir / "query.fbin", query_vectors)
    _write_test_bin(
        dataset_dir / "groundtruth.neighbors.ibin", groundtruth_neighbors
    )


def _pylucene_cli_args(
    backend_config: Path,
    config_dir: Path,
    dataset_path: Path,
    *,
    dry_run: bool = False,
) -> list[str]:
    args = [
        "--backend-config",
        str(backend_config),
        "--dataset-configuration",
        str(config_dir / "datasets" / "datasets.yaml"),
        "--configuration",
        str(config_dir / "algos" / "pylucene_test.yaml"),
        "--dataset",
        "test-dataset",
        "--dataset-path",
        str(dataset_path),
        "--algorithms",
        "pylucene_test",
        "--groups",
        "test",
        "--batch-size",
        "2",
        "-k",
        "1",
        "-m",
        "latency",
        "--build",
        "--search",
        "--force",
    ]
    if dry_run:
        args.append("--dry-run")
    return args


@pytest.fixture
def config_dir(tmp_path):
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / "datasets.yaml").write_text(
        """\
- name: test-dataset
  base_file: test-dataset/base.fbin
  query_file: test-dataset/query.fbin
  groundtruth_neighbors_file: test-dataset/groundtruth.neighbors.ibin
  distance: euclidean
  dims: 4
"""
    )
    (tmp_path / "algos").mkdir()
    (tmp_path / "algos" / "pylucene_test.yaml").write_text(
        f"""\
name: pylucene_test
groups:
  base:
    build:
      codec: [{_HNSW_CODEC}, {_CAGRA_CODEC}]
    search: {{}}
  test:
    build:
      codec: [{_HNSW_CODEC}]
    search: {{}}
"""
    )
    (tmp_path / "algos" / "unrelated.yaml").write_text(
        """\
name: unrelated
groups:
  base:
    build: {}
    search: {}
"""
    )
    return tmp_path


def test_backend_and_loader_are_registered():
    assert get_registry().is_registered("pylucene")
    assert list_config_loaders()["pylucene"] is PyLuceneConfigLoader


def test_import_does_not_load_pylucene():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import cuvs_bench.backends; "
                "assert 'lucene' not in sys.modules"
            ),
        ],
        check=True,
    )


def test_cli_backend_config_supports_pylucene_dry_run(config_dir, tmp_path):
    from cuvs_bench.run.__main__ import main as run_main

    backend_config = tmp_path / "backend.yaml"
    backend_config.write_text("backend: pylucene\n")
    dataset_path = tmp_path / "runtime-datasets"
    result_path = dataset_path / "test-dataset" / "result"
    assert not result_path.exists()

    result = CliRunner().invoke(
        run_main,
        _pylucene_cli_args(
            backend_config,
            config_dir,
            dataset_path,
            dry_run=True,
        ),
    )

    assert result.exit_code == 0, result.output
    assert "Would build PyLucene index" in result.output
    assert "Would search PyLucene index" in result.output
    assert not result_path.exists()


def test_cli_build_search_persists_metrics_and_build_join(
    config_dir, tmp_path, monkeypatch
):
    from cuvs_bench.run.__main__ import main as run_main

    backend_config = tmp_path / "backend.yaml"
    backend_config.write_text("backend: pylucene\n")
    dataset_path = tmp_path / "runtime-datasets"
    _prepare_cli_dataset(dataset_path)
    result_path = dataset_path / "test-dataset" / "result"
    assert not result_path.exists()

    runtime = _FakeRuntime(
        hits=[
            [_SearchHit(document_id=0, score=1.0)],
            [_SearchHit(document_id=1, score=1.0)],
        ]
    )
    monkeypatch.setattr(
        pylucene_backend._PyLuceneRuntime,
        "create",
        staticmethod(lambda _config: runtime),
    )

    result = CliRunner().invoke(
        run_main,
        _pylucene_cli_args(backend_config, config_dir, dataset_path),
    )

    assert result.exit_code == 0, result.output
    index_name = _hnsw_index_name("pylucene_test", "test")
    index_path = dataset_path / "test-dataset" / "index" / index_name
    assert (index_path / "segments_1").is_file()
    assert (index_path / pylucene_backend._HNSW_PROVENANCE_FILE).is_file()
    build_csv = result_path / "build" / "pylucene_test,test.csv"
    with build_csv.open(newline="") as file:
        build_rows = list(csv.DictReader(file))
    assert len(build_rows) == 1
    build_row = build_rows[0]
    assert build_row["index_name"] == index_name
    assert float(build_row["time"]) > 0.0
    assert build_row["codec"] == _HNSW_CODEC
    assert build_row["writer_policy"] == "gpu-with-cpu-fallback"

    raw_csv = result_path / "search" / "pylucene_test,test,k1,bs2,raw.csv"
    with raw_csv.open(newline="") as file:
        csv_rows = list(csv.DictReader(file))
    assert len(csv_rows) == 1
    csv_row = csv_rows[0]
    assert csv_row["index_name"] == index_name
    assert float(csv_row["recall"]) == pytest.approx(1.0)
    assert float(csv_row["throughput"]) == pytest.approx(2000.0)
    assert float(csv_row["latency"]) == pytest.approx(0.001)
    assert float(csv_row["build time"]) > 0.0
    assert float(csv_row["p50"]) == pytest.approx(1.0)
    assert (raw_csv.parent / "pylucene_test,test,k1,bs2,latency.csv").is_file()
    assert (
        raw_csv.parent / "pylucene_test,test,k1,bs2,throughput.csv"
    ).is_file()


def test_cli_returns_nonzero_without_persisting_failed_measurement(
    config_dir, tmp_path, monkeypatch
):
    from cuvs_bench.run.__main__ import main as run_main

    backend_config = tmp_path / "backend.yaml"
    backend_config.write_text("backend: pylucene\n")
    dataset_path = tmp_path / "runtime-datasets"
    _prepare_cli_dataset(dataset_path)
    runtime = _FakeRuntime()
    runtime.build_error = RuntimeError("intentional build failure")
    monkeypatch.setattr(
        pylucene_backend._PyLuceneRuntime,
        "create",
        staticmethod(lambda _config: runtime),
    )

    result = CliRunner().invoke(
        run_main,
        _pylucene_cli_args(backend_config, config_dir, dataset_path),
    )

    assert result.exit_code != 0
    assert "intentional build failure" in result.output
    build_csv = (
        dataset_path
        / "test-dataset"
        / "result"
        / "build"
        / "pylucene_test,test.csv"
    )
    assert not build_csv.exists()
    assert not (dataset_path / "test-dataset" / "result" / "search").exists()


def test_config_loader_expands_codecs_and_forwards_runtime_config(config_dir):
    loader = PyLuceneConfigLoader(config_path=config_dir)
    dataset_config, configs = loader.load(
        dataset="test-dataset",
        dataset_path="/datasets",
        algorithms="pylucene_test",
        groups="base",
        cuvs_java_jar="/artifacts/cuvs-java.jar",
        cuvs_lucene_jar="/artifacts/cuvs-lucene.jar",
        java_library_path="/native",
        jvm_args=["-Xms1g"],
    )

    assert dataset_config.distance == "euclidean"
    assert len(configs) == 2
    assert {config.indexes[0].build_param["codec"] for config in configs} == {
        _HNSW_CODEC,
        _CAGRA_CODEC,
    }
    for config in configs:
        assert config.indexes[0].search_params == [{}]
        assert config.backend_config["cuvs_java_jar"].endswith("cuvs-java.jar")
        assert config.backend_config["cuvs_lucene_jar"].endswith(
            "cuvs-lucene.jar"
        )
        assert config.backend_config["java_library_path"] == "/native"
        assert config.backend_config["jvm_args"] == ["-Xms1g"]
        codec = config.indexes[0].build_param["codec"]
        assert config.backend_config["requires_gpu"] is (codec == _CAGRA_CODEC)
        assert config.backend_config["group"] == "base"
        assert config.backend_config["index_name"] == config.index_name
        assert config.backend_config["index_root"] == str(
            Path("/datasets/test-dataset/index")
        )
        assert config.backend_config["result_scope"] is None


def test_hnsw_build_defaults_are_canonical_and_share_one_identity(config_dir):
    _, configs = PyLuceneConfigLoader(config_path=config_dir).load(
        dataset="test-dataset",
        dataset_path="/datasets",
        algorithms="pylucene_test",
        groups="test",
    )

    assert len(configs) == 1
    config = configs[0]
    assert config.indexes[0].build_param == _DEFAULT_HNSW_BUILD_PARAMETERS
    assert config.index_name == _hnsw_index_name("pylucene_test", "test")
    assert PyLuceneConfigLoader._index_label(
        "pylucene_test",
        "test",
        None,
        {"codec": _HNSW_CODEC},
    ) == PyLuceneConfigLoader._index_label(
        "pylucene_test",
        "test",
        None,
        _DEFAULT_HNSW_BUILD_PARAMETERS,
    )


def test_config_loader_expands_hnsw_build_and_search_sweeps(
    config_dir, tmp_path
):
    algorithm_config = tmp_path / "pylucene_hnsw_sweep.yaml"
    algorithm_config.write_text(
        f"""\
backend: pylucene
name: pylucene_hnsw_sweep
groups:
  requested:
    build:
      codec: [{_HNSW_CODEC}]
      m: [16, 24, 32]
    search:
      num_candidates: [150, 200, 300, 600]
  build_grid:
    build:
      codec: [{_HNSW_CODEC}]
      m: [16, 32]
      ef_construction: [48, 64]
    search: {{}}
"""
    )
    loader = PyLuceneConfigLoader(config_path=config_dir)

    _, requested_configs = loader.load(
        dataset="test-dataset",
        dataset_path="/datasets",
        algorithms="pylucene_hnsw_sweep",
        groups="requested",
        algorithm_configuration=str(algorithm_config),
    )

    assert len(requested_configs) == 3
    assert len({config.index_path for config in requested_configs}) == 3
    assert (
        sum(
            len(config.indexes[0].search_params)
            for config in requested_configs
        )
        == 12
    )
    for config, m in zip(requested_configs, (16, 24, 32), strict=True):
        assert config.indexes[0].build_param == {
            **_DEFAULT_HNSW_BUILD_PARAMETERS,
            "m": m,
        }
        assert config.index_name == _hnsw_index_name(
            "pylucene_hnsw_sweep", "requested", m=m
        )
        assert config.indexes[0].search_params == [
            {"num_candidates": 150},
            {"num_candidates": 200},
            {"num_candidates": 300},
            {"num_candidates": 600},
        ]

    _, build_grid_configs = loader.load(
        dataset="test-dataset",
        dataset_path="/datasets",
        algorithms="pylucene_hnsw_sweep",
        groups="build_grid",
        algorithm_configuration=str(algorithm_config),
    )

    assert len(build_grid_configs) == 4
    assert {
        (
            config.indexes[0].build_param["m"],
            config.indexes[0].build_param["ef_construction"],
        )
        for config in build_grid_configs
    } == {(16, 48), (16, 64), (32, 48), (32, 64)}


def test_pylucene_tune_space_uses_runtime_top_k_for_candidates():
    assert get_search_space("pylucene_cuvs_hnsw")["search"] == {
        "num_candidates": {
            "type": "int",
            "min": "top_k",
            "max": 500,
        }
    }


def test_pylucene_tune_resolves_candidate_minimum_from_top_k(monkeypatch):
    suggested_ranges = _install_single_trial_optuna(monkeypatch)
    orchestrator = BenchmarkOrchestrator.__new__(BenchmarkOrchestrator)
    trial_arguments = {}

    def run_trial(**kwargs):
        trial_arguments.update(kwargs)
        return [
            SearchResult(
                neighbors=np.empty((0, 0), dtype=np.int64),
                distances=np.empty((0, 0), dtype=np.float32),
                search_time_ms=1.0,
                queries_per_second=1.0,
                recall=1.0,
                algorithm="pylucene_cuvs_hnsw",
                search_params=[kwargs["search_params"]],
            )
        ]

    orchestrator._run_trial = run_trial
    results = orchestrator._run_tune(
        constraints={"recall": "maximize"},
        n_trials=1,
        build=True,
        search=True,
        force=False,
        dry_run=False,
        count=150,
        batch_size=1,
        search_mode="latency",
        search_threads=1,
        algorithms="pylucene_cuvs_hnsw",
    )

    assert suggested_ranges["num_candidates"] == (150, 500, False)
    assert trial_arguments["search_params"] == {"num_candidates": 150}
    assert results[0].search_params == [{"num_candidates": 150}]


def test_pylucene_tune_rejects_top_k_above_candidate_ceiling(monkeypatch):
    _install_single_trial_optuna(monkeypatch)
    orchestrator = BenchmarkOrchestrator.__new__(BenchmarkOrchestrator)

    with pytest.raises(ValueError, match="minimum 501 exceeds maximum 500"):
        orchestrator._run_tune(
            constraints={"recall": "maximize"},
            n_trials=1,
            build=True,
            search=True,
            force=False,
            dry_run=False,
            count=501,
            batch_size=1,
            search_mode="latency",
            search_threads=1,
            algorithms="pylucene_cuvs_hnsw",
        )


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("m", True),
        ("m", 0),
        ("m", 513),
        ("ef_construction", False),
        ("ef_construction", 0),
        ("ef_construction", 513),
    ],
)
def test_hnsw_build_parameters_reject_booleans_and_values_outside_range(
    field_name, field_value
):
    with pytest.raises(ValueError, match=field_name):
        pylucene_backend._normalize_build_params(
            {"codec": _HNSW_CODEC, field_name: field_value}
        )


def test_hnsw_build_parameter_boundaries_are_supported():
    assert pylucene_backend._normalize_build_params(
        {
            "codec": _HNSW_CODEC,
            "m": 1,
            "ef_construction": 512,
        }
    ) == {
        **_DEFAULT_HNSW_BUILD_PARAMETERS,
        "m": 1,
        "ef_construction": 512,
    }


@pytest.mark.parametrize("field_name", ["m", "ef_construction"])
def test_hnsw_build_parameters_are_rejected_for_cagra(field_name):
    with pytest.raises(ValueError, match=field_name):
        pylucene_backend._normalize_build_params(
            {"codec": _CAGRA_CODEC, field_name: 32}
        )


@pytest.mark.parametrize(
    ("build_yaml", "search_yaml", "error"),
    [
        (
            f"codec: [{_HNSW_CODEC}]\n      ignored: [1]",
            "{}",
            "Unsupported PyLucene build parameter.*ignored",
        ),
        (
            f"codec: [{_HNSW_CODEC}]",
            "ignored: [1]",
            "Unsupported PyLucene search parameter.*ignored",
        ),
    ],
)
def test_config_loader_rejects_unsupported_parameters(
    config_dir, tmp_path, build_yaml, search_yaml, error
):
    custom_config = tmp_path / "unsupported-parameters.yaml"
    custom_config.write_text(
        f"""\
backend: pylucene
name: pylucene_unsupported
groups:
  test:
    build:
      {build_yaml}
    search:
      {search_yaml}
"""
    )

    with pytest.raises(ValueError, match=error):
        PyLuceneConfigLoader(config_path=config_dir).load(
            dataset="test-dataset",
            dataset_path="/datasets",
            algorithms="pylucene_unsupported",
            groups="test",
            algorithm_configuration=str(custom_config),
        )


@pytest.mark.parametrize(
    ("algorithm_name", "group_name", "match"),
    [
        ("../../../escaped", "test", "PyLucene algorithm name"),
        ("pylucene_safe", "../escaped", "PyLucene group name"),
        ("pylucene,ambiguous", "test", "PyLucene algorithm name"),
    ],
)
def test_config_loader_rejects_unsafe_artifact_identity(
    config_dir, tmp_path, algorithm_name, group_name, match
):
    custom_config = tmp_path / "unsafe-identity.yaml"
    custom_config.write_text(
        f"""\
backend: pylucene
name: {algorithm_name}
groups:
  {group_name}:
    build:
      codec: [{_HNSW_CODEC}]
    search: {{}}
"""
    )

    with pytest.raises(ValueError, match=match):
        PyLuceneConfigLoader(config_path=config_dir).load(
            dataset="test-dataset",
            dataset_path="/datasets",
            algorithms=(None if "," in algorithm_name else algorithm_name),
            groups=group_name,
            algorithm_configuration=str(custom_config),
        )


def test_config_loader_scopes_artifact_identity_by_dataset_subset(config_dir):
    loader = PyLuceneConfigLoader(config_path=config_dir)
    identities = []

    for subset_size, subset_scope in (
        (None, None),
        (2, "subset2"),
        (3, "subset3"),
    ):
        dataset_config, configs = loader.load(
            dataset="test-dataset",
            dataset_path="/datasets",
            algorithms="pylucene_test",
            groups="test",
            subset_size=subset_size,
            count=1,
            batch_size=2,
        )

        assert dataset_config.subset_size == subset_size
        assert len(configs) == 1
        config = configs[0]
        index_name = _hnsw_index_name(
            "pylucene_test", "test", subset_scope=subset_scope
        )
        assert config.index_name == index_name
        assert config.index_path == (
            Path("/datasets/test-dataset/index") / index_name
        )
        assert config.backend_config["group"] == "test"
        assert config.backend_config["index_name"] == index_name
        assert config.backend_config["result_scope"] == subset_scope
        identities.append(
            (
                config.index_name,
                config.index_path,
                config.backend_config["result_scope"],
            )
        )

    assert len(set(identities)) == 3


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (("foo.subset2", "base", None), ("foo", "base", "subset2")),
        (("a_b", "c", None), ("a", "b_c", None)),
    ],
)
def test_index_labels_are_injective(first, second):
    build_params = _DEFAULT_HNSW_BUILD_PARAMETERS

    first_label = PyLuceneConfigLoader._index_label(
        first[0], first[1], first[2], build_params
    )
    second_label = PyLuceneConfigLoader._index_label(
        second[0], second[1], second[2], build_params
    )

    assert first_label != second_label


@pytest.mark.parametrize("subset_size", [0, -1, True, "../other"])
def test_config_loader_rejects_unsafe_subset_identity(config_dir, subset_size):
    loader = PyLuceneConfigLoader(config_path=config_dir)

    with pytest.raises(ValueError, match="positive integer"):
        loader.load(
            dataset="test-dataset",
            dataset_path="/datasets",
            algorithms="pylucene_test",
            groups="test",
            subset_size=subset_size,
        )


@pytest.mark.parametrize(
    ("backend_declaration", "algorithm_name"),
    [
        ("", "pylucene_custom"),
        ("backend: pylucene\n", "custom_lucene"),
    ],
    ids=["parsed-name", "declared-backend"],
)
def test_config_loader_discovers_custom_filename_by_parsed_identity(
    config_dir,
    tmp_path,
    backend_declaration,
    algorithm_name,
):
    custom_config = tmp_path / "arbitrary-custom-name.yaml"
    custom_config.write_text(
        f"""\
{backend_declaration}name: {algorithm_name}
groups:
  test:
    build:
      codec: [{_HNSW_CODEC}]
    search: {{}}
"""
    )
    loader = PyLuceneConfigLoader(config_path=config_dir)

    _, configs = loader.load(
        dataset="test-dataset",
        dataset_path="/datasets",
        algorithms=algorithm_name,
        groups="test",
        algorithm_configuration=str(custom_config),
    )

    assert len(configs) == 1
    assert configs[0].indexes[0].algo == algorithm_name
    assert configs[0].index_name == _hnsw_index_name(algorithm_name, "test")


def test_config_loader_excludes_explicit_other_backend(config_dir, tmp_path):
    custom_config = tmp_path / "misleading-pylucene-name.yaml"
    custom_config.write_text(
        f"""\
backend: cpp_gbench
name: pylucene_other_backend
groups:
  test:
    build:
      codec: [{_HNSW_CODEC}]
    search: {{}}
"""
    )
    loader = PyLuceneConfigLoader(config_path=config_dir)

    with pytest.raises(
        ValueError,
        match="Unknown PyLucene algorithm selector.*pylucene_other_backend",
    ):
        loader.load(
            dataset="test-dataset",
            dataset_path="/datasets",
            algorithms="pylucene_other_backend",
            groups="test",
            algorithm_configuration=str(custom_config),
        )


def test_config_loader_honors_algorithm_and_group_filters(config_dir):
    loader = PyLuceneConfigLoader(config_path=config_dir)

    _, configs = loader.load(
        dataset="test-dataset",
        dataset_path="/datasets",
        algorithms="pylucene_test",
        groups="test",
    )
    assert len(configs) == 1
    assert configs[0].index_name == _hnsw_index_name("pylucene_test", "test")


def test_algorithm_config_discovery_is_deterministic(tmp_path):
    config_path = tmp_path / "config"
    bundled_algorithms = config_path / "algos"
    bundled_algorithms.mkdir(parents=True)
    bundled_zeta = bundled_algorithms / "zeta.yaml"
    bundled_alpha = bundled_algorithms / "alpha.yml"
    bundled_zeta.touch()
    bundled_alpha.touch()
    (bundled_algorithms / "ignored.txt").touch()

    custom_algorithms = tmp_path / "custom"
    custom_algorithms.mkdir()
    custom_zeta = custom_algorithms / "zeta.yml"
    custom_alpha = custom_algorithms / "alpha.yaml"
    custom_zeta.touch()
    custom_alpha.touch()

    files = PyLuceneConfigLoader(
        config_path=config_path
    ).gather_algorithm_configs(config_path, str(custom_algorithms))

    assert files == [
        str(bundled_alpha),
        str(bundled_zeta),
        str(custom_alpha),
        str(custom_zeta),
    ]


def test_duplicate_algorithm_config_uses_last_definition_and_position(
    config_dir, monkeypatch
):
    loader = PyLuceneConfigLoader(config_path=config_dir)
    config_files = ["alpha-original", "beta", "alpha-override"]
    configs_by_file = {
        "alpha-original": {
            "name": "pylucene_alpha",
            "groups": {"original": {}},
        },
        "beta": {
            "name": "pylucene_beta",
            "groups": {"base": {}},
        },
        "alpha-override": {
            "name": "pylucene_alpha",
            "groups": {"override": {}},
        },
    }
    monkeypatch.setattr(loader, "load_yaml_file", configs_by_file.__getitem__)

    algorithm_configs = loader._load_algorithm_configs(config_files)

    assert list(algorithm_configs) == ["pylucene_beta", "pylucene_alpha"]
    assert algorithm_configs["pylucene_alpha"] == {"override": {}}


def test_config_loader_unions_global_and_algorithm_specific_groups(config_dir):
    loader = PyLuceneConfigLoader(config_path=config_dir)

    _, configs = loader.load(
        dataset="test-dataset",
        dataset_path="/datasets",
        algorithms="pylucene_test",
        groups="base",
        algo_groups="pylucene_test.test",
    )

    assert len(configs) == 3
    assert (
        sum(
            config.index_name.startswith("pylucene_test[group=test]")
            for config in configs
        )
        == 1
    )


def test_config_loader_selects_only_explicit_algorithm_group(config_dir):
    loader = PyLuceneConfigLoader(config_path=config_dir)

    _, configs = loader.load(
        dataset="test-dataset",
        dataset_path="/datasets",
        algo_groups="pylucene_test.test",
    )

    assert len(configs) == 1
    assert configs[0].index_name.startswith("pylucene_test[group=test]")


@pytest.mark.parametrize(
    ("selectors", "expected_groups"),
    [
        (
            {},
            ["pylucene_alpha,base", "pylucene_beta,base"],
        ),
        (
            {"algorithms": "pylucene_beta,pylucene_alpha"},
            ["pylucene_alpha,base", "pylucene_beta,base"],
        ),
        (
            {"groups": "shared"},
            ["pylucene_alpha,shared", "pylucene_beta,shared"],
        ),
        (
            {
                "algorithms": "pylucene_alpha,pylucene_beta",
                "groups": "alpha_only,shared",
            },
            [
                "pylucene_alpha,shared",
                "pylucene_alpha,alpha_only",
                "pylucene_beta,shared",
            ],
        ),
        (
            {"algo_groups": "pylucene_beta.beta_only"},
            ["pylucene_beta,beta_only"],
        ),
        (
            {
                "algorithms": "pylucene_alpha",
                "algo_groups": "pylucene_beta.beta_only",
            },
            ["pylucene_alpha,base", "pylucene_beta,beta_only"],
        ),
        (
            {
                "groups": "alpha_only",
                "algo_groups": "pylucene_beta.beta_only",
            },
            ["pylucene_alpha,alpha_only", "pylucene_beta,beta_only"],
        ),
        (
            {
                "algorithms": "pylucene_beta",
                "groups": "shared",
                "algo_groups": "pylucene_alpha.alpha_only",
            },
            ["pylucene_alpha,alpha_only", "pylucene_beta,shared"],
        ),
        (
            {
                "algorithms": "pylucene_alpha",
                "groups": "shared",
                "algo_groups": "pylucene_alpha.shared",
            },
            ["pylucene_alpha,shared"],
        ),
        (
            {
                "algo_groups": (
                    "pylucene_beta.beta_only,pylucene_alpha.alpha_only"
                )
            },
            ["pylucene_alpha,alpha_only", "pylucene_beta,beta_only"],
        ),
        (
            {
                "algorithms": " pylucene_beta, pylucene_beta ",
                "groups": " base,base ",
            },
            ["pylucene_beta,base"],
        ),
    ],
    ids=[
        "defaults",
        "algorithm",
        "group",
        "algorithm-and-group",
        "explicit-pair",
        "algorithm-plus-explicit-pair",
        "group-plus-explicit-pair",
        "all-selectors",
        "overlapping-selectors",
        "explicit-pair-order-follows-config",
        "duplicate-selectors",
    ],
)
def test_config_loader_combines_global_and_explicit_selectors(
    config_dir, monkeypatch, selectors, expected_groups
):
    alpha_config = config_dir / "algos" / "pylucene_alpha.yaml"
    alpha_config.write_text(
        f"""\
name: pylucene_alpha
groups:
  base:
    build:
      codec: [{_HNSW_CODEC}]
    search: {{}}
  shared:
    build:
      codec: [{_HNSW_CODEC}]
    search: {{}}
  alpha_only:
    build:
      codec: [{_HNSW_CODEC}]
    search: {{}}
"""
    )
    beta_config = config_dir / "algos" / "pylucene_beta.yaml"
    beta_config.write_text(
        f"""\
name: pylucene_beta
groups:
  base:
    build:
      codec: [{_HNSW_CODEC}]
    search: {{}}
  shared:
    build:
      codec: [{_HNSW_CODEC}]
    search: {{}}
  beta_only:
    build:
      codec: [{_HNSW_CODEC}]
    search: {{}}
"""
    )
    loader = PyLuceneConfigLoader(config_path=config_dir)
    monkeypatch.setattr(
        loader,
        "gather_algorithm_configs",
        lambda *_args: [str(alpha_config), str(beta_config)],
    )

    _, configs = loader.load(
        dataset="test-dataset",
        dataset_path="/datasets",
        **selectors,
    )

    selected_groups = [
        f"{config.algo},{config.backend_config['group']}" for config in configs
    ]
    assert selected_groups == expected_groups


def test_algorithm_selection_resolution_preserves_inputs_and_source_order():
    alpha_base = {"build": {"codec": [_HNSW_CODEC]}}
    beta_base = {"build": {"codec": [_HNSW_CODEC]}}
    algorithm_configs = {
        "pylucene_beta": {"base": beta_base},
        "pylucene_alpha": {"base": alpha_base},
    }
    original_items = [
        (algorithm, list(groups.items()))
        for algorithm, groups in algorithm_configs.items()
    ]
    selection = pylucene_backend._AlgorithmSelection(
        algorithms=frozenset({"pylucene_alpha", "pylucene_beta"}),
        groups=None,
        explicit_groups=frozenset(),
    )

    selected = selection.resolve(algorithm_configs)

    assert [(group.algorithm, group.group) for group in selected] == [
        ("pylucene_beta", "base"),
        ("pylucene_alpha", "base"),
    ]
    assert selected[0].configuration is beta_base
    assert selected[1].configuration is alpha_base
    assert [
        (algorithm, list(groups.items()))
        for algorithm, groups in algorithm_configs.items()
    ] == original_items


def test_algorithm_selection_rejects_group_outside_selected_algorithms():
    selection = pylucene_backend._AlgorithmSelection(
        algorithms=frozenset({"pylucene_alpha"}),
        groups=frozenset({"beta_only"}),
        explicit_groups=frozenset(),
    )
    algorithm_configs = {
        "pylucene_alpha": {"base": {}},
        "pylucene_beta": {"beta_only": {}},
    }

    with pytest.raises(
        ValueError,
        match="Unknown PyLucene group selector\\(s\\): beta_only",
    ):
        selection.resolve(algorithm_configs)


def test_algorithm_selection_reports_explicit_errors_deterministically():
    selection = pylucene_backend._AlgorithmSelection(
        algorithms=None,
        groups=None,
        explicit_groups=frozenset(
            {
                ("pylucene_zeta", "base"),
                ("pylucene_beta", "base"),
            }
        ),
    )

    with pytest.raises(ValueError) as error:
        selection.resolve({"pylucene_alpha": {"base": {}}})

    assert str(error.value) == (
        "Unknown PyLucene algorithm in --algo-groups: pylucene_beta"
    )


def test_config_loader_rejects_malformed_algorithm_group(config_dir):
    loader = PyLuceneConfigLoader(config_path=config_dir)

    with pytest.raises(ValueError, match="<algorithm>.<group>"):
        loader.load(
            dataset="test-dataset",
            dataset_path="/datasets",
            algo_groups="pylucene_test",
        )


@pytest.mark.parametrize(
    ("selectors", "error"),
    [
        (
            {"algorithms": "not-pylucene", "groups": "base"},
            "Unknown PyLucene algorithm selector.*not-pylucene",
        ),
        (
            {"algorithms": "pylucene_test", "groups": "missing"},
            "Unknown PyLucene group selector.*missing",
        ),
        (
            {
                "algorithms": "pylucene_test",
                "groups": "base",
                "algo_groups": "not-pylucene.test",
            },
            "Unknown PyLucene algorithm in --algo-groups.*not-pylucene",
        ),
        (
            {
                "algorithms": "pylucene_test",
                "groups": "base",
                "algo_groups": "pylucene_test.missing",
            },
            "Unknown PyLucene group for pylucene_test.*missing",
        ),
    ],
    ids=[
        "algorithm",
        "global-group",
        "algo-group-algorithm",
        "algo-group-group",
    ],
)
def test_config_loader_rejects_unknown_selectors(config_dir, selectors, error):
    loader = PyLuceneConfigLoader(config_path=config_dir)

    with pytest.raises(ValueError, match=error):
        loader.load(
            dataset="test-dataset",
            dataset_path="/datasets",
            **selectors,
        )


@pytest.mark.parametrize(
    ("algorithm", "expected_codecs"),
    [
        (
            "pylucene_cuvs_hnsw",
            {_HNSW_CODEC},
        ),
        ("pylucene_cuvs_cagra", {_CAGRA_CODEC}),
    ],
)
def test_shipped_algorithm_configs_load(algorithm, expected_codecs):
    _, configs = PyLuceneConfigLoader().load(
        dataset="test-data",
        dataset_path="/datasets",
        algorithms=algorithm,
        groups="base",
    )

    assert {
        config.indexes[0].build_param["codec"] for config in configs
    } == expected_codecs
