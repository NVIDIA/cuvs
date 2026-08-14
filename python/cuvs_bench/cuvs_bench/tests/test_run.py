#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#


import itertools
import os
from unittest.mock import MagicMock, mock_open, patch

import pytest
from cuvs_bench.run.run import (
    find_executable,
    gather_algorithm_configs,
    get_dataset_configuration,
    load_algorithms_conf,
    load_yaml_file,
    prepare_conf_file,
    prepare_executables,
    prepare_indexes,
    rmm_present,
    validate_algorithm,
    validate_constraints,
    validate_search_params,
)


def test_load_yaml_file():
    yaml_content = """
    key: value
    """
    with patch("builtins.open", mock_open(read_data=yaml_content)):
        result = load_yaml_file("dummy_path.yaml")
        assert result == {"key": "value"}


def test_get_dataset_configuration():
    dataset_conf_all = [{"name": "dataset1"}, {"name": "dataset2"}]
    result = get_dataset_configuration("dataset1", dataset_conf_all)
    assert result == {"name": "dataset1"}
    with pytest.raises(ValueError):
        get_dataset_configuration("non_existent_dataset", dataset_conf_all)


def test_prepare_conf_file():
    dataset_conf = {"name": "dataset1"}
    result = prepare_conf_file(dataset_conf, 1000, 10, 128)
    expected_result = {
        "dataset": {"name": "dataset1", "subset_size": 1000},
        "search_basic_param": {"k": 10, "batch_size": 128},
    }
    assert result == expected_result


def test_gather_algorithm_configs(tmpdir):
    scripts_base_path = tmpdir.mkdir("scripts_base")

    scripts_path = os.path.join(scripts_base_path, "scripts")
    os.mkdir(scripts_path)

    algos_path = os.path.join(scripts_base_path, "config")
    os.mkdir(algos_path)
    algos_path = os.path.join(algos_path, "algos")
    os.mkdir(algos_path)

    with open(os.path.join(algos_path, "algo1.yaml"), "w") as f:
        f.write("key: value")

    with open(os.path.join(algos_path, "algo2.yaml"), "w") as f:
        f.write("key: value")

    result = gather_algorithm_configs(str(scripts_path), None)
    assert len(result) == 2

    custom_conf_dir = tmpdir.mkdir("custom_conf")
    custom_conf_dir.join("custom_algo.yaml").write("key: value")
    result = gather_algorithm_configs(str(scripts_path), str(custom_conf_dir))
    assert len(result) == 3

    custom_conf_file = custom_conf_dir.join("custom_algo_file.yaml")
    custom_conf_file.write("key: value")
    result = gather_algorithm_configs(str(scripts_path), str(custom_conf_file))
    assert len(result) == 3


def test_load_algorithms_conf():
    algos_conf_fs = ["path/to/algo1.yaml", "path/to/algo2.yaml"]
    yaml_content = """
    name: algo1
    groups:
      group1: {}
    """
    with patch("builtins.open", mock_open(read_data=yaml_content)):
        result = load_algorithms_conf(algos_conf_fs, None, None, None)
        assert "algo1" in result

    with patch("builtins.open", mock_open(read_data=yaml_content)):
        result = load_algorithms_conf(algos_conf_fs, ["algo1"], None, None)
        assert "algo1" in result
        result = load_algorithms_conf(algos_conf_fs, ["algo2"], None, None)
        assert "algo1" not in result


@patch(
    "cuvs_bench.run.run.find_executable",
    return_value=("executable", "path", "filename"),
)
@patch("cuvs_bench.run.run.validate_algorithm", return_value=True)
@patch(
    "cuvs_bench.run.run.prepare_indexes",
    return_value=[{"index_key": "index_value"}],
)
def test_prepare_executables(
    mock_prepare_indexes, mock_validate_algorithm, mock_find_executable
):
    algos_conf = {"algo1": {"groups": {"group1": {"build": {}, "search": {}}}}}
    algos_yaml = {"algo1": {}}
    gpu_present = True
    conf_file = {}
    executable_dir = "executable_dir"
    dataset_path = "dataset_path"
    dataset = "dataset"
    count = 10
    batch_size = 128
    result = prepare_executables(
        algos_conf,
        algos_yaml,
        gpu_present,
        conf_file,
        executable_dir,
        dataset_path,
        dataset,
        count,
        batch_size,
    )
    assert ("executable", "path", "filename") in result.keys()
    assert len(result[("executable", "path", "filename")]["index"]) == 1


@patch("cuvs_bench.run.run.validate_constraints", return_value=True)
@patch(
    "cuvs_bench.run.run.validate_search_params",
    return_value=[{"sparam": "dummy_value"}],
)
def test_prepare_indexes_valid(
    mock_validate_search_params, mock_validate_constraints
):
    group_conf = {
        "build": {
            "param1": [1, 2],
            "param2": [3, 4],
        },
        "search": {"sparam1": [True, False]},
    }
    algo = "algo1"
    group = "base"
    conf_file = {"dataset": {"dims": 128}}
    algos_conf = {}
    dataset_path = "/tmp/dataset"
    dataset = "data1"
    count = 10
    batch_size = 32

    indexes = prepare_indexes(
        group_conf,
        algo,
        group,
        conf_file,
        algos_conf,
        dataset_path,
        dataset,
        count,
        batch_size,
    )

    # There are 2 build parameters with 2 values each, so we expect 4 indexes.
    assert len(indexes) == 4

    for index in indexes:
        assert index["algo"] == algo

        expected_filename = (
            index["name"]
            if len(index["name"]) < 128
            else str(hash(index["name"]))
        )
        expected_file = os.path.join(
            dataset_path, dataset, "index", expected_filename
        )
        assert index["file"] == expected_file
        # Verify that our dummy search parameters were set.
        assert index["search_params"] == [{"sparam": "dummy_value"}]


@patch("cuvs_bench.run.run.validate_constraints", return_value=False)
@patch(
    "cuvs_bench.run.run.validate_search_params",
    return_value=[{"sparam": "dummy_value"}],
)
def test_prepare_indexes_invalid(
    mock_validate_search_params, mock_validate_constraints
):
    group_conf = {
        "build": {
            "param1": [1, 2],
        },
        "search": {},
    }
    algo = "algo1"
    group = "base"
    conf_file = {"dataset": {"dims": 128}}
    algos_conf = {}
    dataset_path = "/tmp/dataset"
    dataset = "data1"
    count = 10
    batch_size = 32

    indexes = prepare_indexes(
        group_conf,
        algo,
        group,
        conf_file,
        algos_conf,
        dataset_path,
        dataset,
        count,
        batch_size,
    )

    # Since constraints fail, no indexes should be created.
    assert indexes == []


def test_validate_search_params():
    all_search_params = itertools.product([1, 2], [3, 4])
    search_param_names = ["param1", "param2"]
    group_conf = {}
    conf_file = {"dataset": {"dims": 128}}
    result = validate_search_params(
        all_search_params,
        search_param_names,
        {},
        "algo",
        group_conf,
        {"algo": {"constraints": []}},
        conf_file,
        10,
        128,
    )
    assert len(result) == 4


def test_rmm_present():
    with patch.dict("sys.modules", {"rmm": MagicMock()}):
        assert rmm_present() is True
    with patch.dict("sys.modules", {"rmm": None}):
        assert rmm_present() is False


@patch("cuvs_bench.run.run.get_build_path", return_value="build_path")
def test_find_executable(mock_get_build_path):
    algos_conf = {"algo1": {"executable": "executable1"}}
    result = find_executable(algos_conf, "algo1", "group1", 10, 128, None)
    assert result == (
        "executable1",
        "build_path",
        ("algo1,group1", "algo1,group1,k10,bs128"),
    )
    mock_get_build_path.return_value = None
    with pytest.raises(FileNotFoundError):
        find_executable(algos_conf, "algo1", "group1", 10, 128, None)


def test_validate_algorithm():
    algos_conf = {"algo1": {"requires_gpu": False}}
    result = validate_algorithm(algos_conf, "algo1", gpu_present=True)
    assert result is True
    result = validate_algorithm(algos_conf, "algo1", gpu_present=False)
    assert result is True
    algos_conf["algo1"]["requires_gpu"] = True
    result = validate_algorithm(algos_conf, "algo1", gpu_present=False)
    assert result is False


@patch("cuvs_bench.run.run.import_module")
def test_validate_constraints(mock_import_module):
    # Create a mock validator and have import_module return it.
    mock_validator = MagicMock()
    mock_import_module.return_value = mock_validator

    # Test case 1: The constraint function returns True.
    mock_validator.constraint_func.return_value = True
    algos_conf = {
        "algo1": {"constraints": {"build": "module.constraint_func"}}
    }
    result = validate_constraints(
        algos_conf, "algo1", "build", {"param1": "value1"}, {}, 128, None, None
    )
    assert result is True

    # Test case 2: No constraints are specified; should return True.
    algos_conf = {"algo1": {"constraints": {}}}
    result = validate_constraints(
        algos_conf, "algo1", "build", {"param1": "value1"}, {}, 128, None, None
    )
    assert result is True

    # Test case 3: The constraint function returns False.
    mock_validator.constraint_func.return_value = False
    algos_conf["algo1"]["constraints"]["build"] = "module.constraint_func"
    result = validate_constraints(
        algos_conf, "algo1", "build", {"param1": "value1"}, {}, 128, None, None
    )
    assert result is False
