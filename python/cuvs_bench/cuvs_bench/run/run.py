#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import itertools
import os
import warnings
from importlib import import_module
from typing import Any, Dict, Optional, Tuple

import yaml

from .runners import cuvs_bench_cpp


def rmm_present() -> bool:
    """
    Check if RMM (RAPIDS Memory Manager) is present.

    Returns
    -------
    bool
        True if RMM is present, False otherwise.
    """
    try:
        import rmm  # noqa: F401

        return True
    except ImportError:
        return False


def load_yaml_file(file_path: str) -> dict:
    """
    Load a YAML file and return its contents as a dictionary.

    Parameters
    ----------
    file_path : str
        The path to the YAML file.

    Returns
    -------
    dict
        The contents of the YAML file.
    """
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def get_dataset_configuration(dataset: str, dataset_conf_all: list) -> dict:
    """
    Retrieve the configuration for a specific dataset.

    Parameters
    ----------
    dataset : str
        The name of the dataset to retrieve the configuration for.
    dataset_conf_all : list
        A list of dataset configurations.

    Returns
    -------
    dict
        The configuration for the specified dataset.

    Raises
    ------
    ValueError
        If the dataset configuration is not found.
    """
    for dset in dataset_conf_all:
        if dataset == dset["name"]:
            return dset
    raise ValueError("Could not find a dataset configuration")


def prepare_conf_file(
    dataset_conf: dict, subset_size: Optional[int], count: int, batch_size: int
) -> dict:
    """
    Prepare the main configuration file for the benchmark.

    Parameters
    ----------
    dataset_conf : dict
        The configuration for the dataset.
    subset_size : Optional[int]
        The subset size of the dataset.
    count : int
        The number of nearest neighbors to search for.
    batch_size : int
        The size of each batch for processing.

    Returns
    -------
    dict
        The prepared configuration file.
    """
    conf_file = {"dataset": dataset_conf}
    if subset_size:
        conf_file["dataset"]["subset_size"] = subset_size
    conf_file["search_basic_param"] = {"k": count, "batch_size": batch_size}
    return conf_file


def gather_algorithm_configs(
    scripts_path: str, configuration: Optional[str]
) -> list:
    """
    Gather the list of algorithm configuration files.

    Parameters
    ----------
    scripts_path : str
        The path to the script directory.
    configuration : Optional[str]
        The path to the algorithm configuration directory or file.

    Returns
    -------
    list
        A list of paths to the algorithm configuration files.
    """
    algos_conf_fs = os.listdir(
        os.path.join(scripts_path, "../config", "algos")
    )
    algos_conf_fs = [
        os.path.join(scripts_path, "../config", "algos", f)
        for f in algos_conf_fs
        if ".json" not in f
        and "constraint" not in f
        and ".py" not in f
        and "__pycache__" not in f
    ]

    if configuration:
        if os.path.isdir(configuration):
            algos_conf_fs += [
                os.path.join(configuration, f)
                for f in os.listdir(configuration)
                if ".json" not in f
            ]
        elif os.path.isfile(configuration):
            algos_conf_fs.append(configuration)
    return algos_conf_fs


def load_algorithms_conf(
    algos_conf_fs: list,
    allowed_algos: Optional[list],
    allowed_groups: Optional[list],
    allowed_algo_groups: Optional[tuple],
) -> dict:
    """
    Load and filter the algorithm configurations.

    Parameters
    ----------
    algos_conf_fs : list
        A list of paths to algorithm configuration files.
    allowed_algos : Optional[list]
        A list of allowed algorithm names to filter by.
    allowed_algo_groups : Optional[tuple]
        A tuple of allowed algorithm groups to filter by.

    Returns
    -------
    dict
        A dictionary containing the loaded and filtered algorithm
        configurations.
    """
    algos_conf = {}
    for algo_f in algos_conf_fs:
        try:
            algo = load_yaml_file(algo_f)
        except Exception as e:
            warnings.warn(f"Could not load YAML config {algo_f} due to {e}")
            continue
        if allowed_algos and algo["name"] not in allowed_algos:
            continue
        groups = algo.get("groups", {})
        if allowed_groups:
            groups = {k: v for k, v in groups.items() if k in allowed_groups}
        algos_conf[algo["name"]] = {
            "groups": groups,
            "constraints": algo.get("constraints", {}),
        }
        if allowed_algo_groups and algo["name"] in allowed_algo_groups[0]:
            algos_conf[algo["name"]]["groups"].update(
                {
                    group: algo["groups"][group]
                    for group in allowed_algo_groups[1]
                    if group in algo["groups"]
                }
            )
    return algos_conf


def prepare_executables(
    algos_conf: dict,
    algos_yaml: dict,
    gpu_present: bool,
    conf_file: dict,
    executable_dir: str,
    dataset_path: str,
    dataset: str,
    count: int,
    batch_size: int,
) -> dict:
    """
    Prepare the list of executables to run based on the configurations.

    Parameters
    ----------
    algos_conf : dict
        The loaded algorithm configurations.
    algos_yaml : dict
        The global algorithms configuration.
    gpu_present : bool
        Whether a GPU is present.
    conf_file : dict
        The main configuration file.
    dataset_path : str
        The path to the dataset directory.
    dataset : str
        The name of the dataset.
    count : int
        The number of nearest neighbors to search for.
    batch_size : int
        The size of each batch for processing.

    Returns
    -------
    dict
        A dictionary of executables to run with their associated
        configurations.
    """
    executables_to_run = {}

    for algo, algo_conf in algos_conf.items():
        validate_algorithm(algos_yaml, algo, gpu_present)
        for group, group_conf in algo_conf["groups"].items():
            executable = find_executable(
                algos_yaml, algo, group, count, batch_size, executable_dir
            )
            if executable not in executables_to_run:
                executables_to_run[executable] = {"index": []}
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
            executables_to_run[executable]["index"].extend(indexes)
    return executables_to_run


def validate_algorithm(algos_conf: dict, algo: str, gpu_present: bool) -> bool:
    """
    Validate the algorithm based on the available hardware (GPU presence).

    Parameters
    ----------
    algos_conf : dict
        The configuration dictionary for the algorithms.
    algo : str
        The name of the algorithm.
    gpu_present : bool
        Whether a GPU is present.

    Returns
    -------
    bool
        True if the algorithm is valid for the current hardware
        configuration, False otherwise.
    """
    algos_conf_keys = set(algos_conf.keys())
    if gpu_present:
        return algo in algos_conf_keys
    return (
        algo in algos_conf_keys and algos_conf[algo]["requires_gpu"] is False
    )


def find_executable(
    algos_conf: dict,
    algo: str,
    group: str,
    k: int,
    batch_size: int,
    executable_dir: str,
) -> Tuple[str, str, Tuple[str, str]]:
    """
    Find the executable for the given algorithm and group.

    Parameters
    ----------
    algos_conf : dict
        The configuration dictionary for the algorithms.
    algo : str
        The name of the algorithm.
    group : str
        The name of the group.
    k : int
        The number of nearest neighbors to search for.
    batch_size : int
        The size of each batch for processing.

    Returns
    -------
    Tuple[str, str, Tuple[str, str]]
        A tuple containing the executable name, the path to the executable,
        and the file name.
    """
    executable = algos_conf[algo]["executable"]
    file_name = (f"{algo},{group}", f"{algo},{group},k{k},bs{batch_size}")
    build_path = get_build_path(executable, executable_dir)
    if build_path:
        return executable, build_path, file_name
    raise FileNotFoundError(executable)


def get_build_path(executable: str, executable_dir: str) -> Optional[str]:
    """
    Get the build path for the given executable.

    Parameters
    ----------
    executable : str
        The name of the executable.

    Returns
    -------
    Optional[str]
        The build path for the executable, if found.
    """
    if executable_dir is not None:
        build_path = os.path.join(executable_dir, executable)
        if os.path.exists(build_path):
            print(f"-- Using cuVS bench from {build_path}.")
            return build_path

    devcontainer_path = "/home/coder/cuvs/cpp/build/latest/bench/ann"
    if os.path.exists(devcontainer_path):
        devc_executable = os.path.join(devcontainer_path, executable)
        print(f"-- Detected devcontainer artifact {devc_executable}.")
        return devc_executable

    build_path = os.getenv("CUVS_HOME")
    if build_path:
        build_path = os.path.join(
            build_path, "cpp", "build", "release", executable
        )
        if os.path.exists(build_path):
            print(f"-- Using cuVS bench from repository in {build_path}.")
            return build_path

    conda_path = os.getenv("CONDA_PREFIX")
    if conda_path:
        conda_executable = os.path.join(conda_path, "bin", "ann", executable)
        if os.path.exists(conda_executable):
            print("-- Using cuVS bench found in conda environment.")
            return conda_executable

    return None


def prepare_indexes(
    group_conf: dict,
    algo: str,
    group: str,
    conf_file: dict,
    algos_conf: dict,
    dataset_path: str,
    dataset: str,
    count: int,
    batch_size: int,
) -> list:
    """
    Prepare the index configurations for the given algorithm and group.

    Parameters
    ----------
    group_conf : dict
        The configuration for the algorithm group.
    algo : str
        The name of the algorithm.
    group : str
        The name of the group.
    conf_file : dict
        The main configuration file.
    dataset_path : str
        The path to the dataset directory.
    dataset : str
        The name of the dataset.
    count : int
        The number of nearest neighbors to search for.
    batch_size : int
        The size of each batch for processing.

    Returns
    -------
    list
        A list of index configurations.
    """
    indexes = []
    build_params = group_conf.get("build", {})
    search_params = group_conf.get("search", {})
    all_build_params = itertools.product(*build_params.values())
    search_param_names, search_param_lists = (
        zip(*search_params.items()) if search_params else ([], [])
    )
    param_names = list(build_params.keys())
    for params in all_build_params:
        index = {
            "algo": algo,
            "build_param": dict(zip(build_params.keys(), params)),
        }
        index_name = f"{algo}_{group}" if group != "base" else f"{algo}"
        for i in range(len(params)):
            index["build_param"][param_names[i]] = params[i]
            index_name += "." + f"{param_names[i]}{params[i]}"

        if not validate_constraints(
            algos_conf,
            algo,
            "build",
            index["build_param"],
            None,
            conf_file["dataset"].get("dims"),
            count,
            batch_size,
        ):
            continue

        index_filename = (
            index_name if len(index_name) < 128 else str(hash(index_name))
        )
        index["name"] = index_name
        index["file"] = os.path.join(
            dataset_path, dataset, "index", index_filename
        )
        index["search_params"] = validate_search_params(
            itertools.product(*search_param_lists),
            search_param_names,
            index["build_param"],
            algo,
            group_conf,
            algos_conf,
            conf_file,
            count,
            batch_size,
        )
        if index["search_params"]:
            indexes.append(index)
    return indexes


def validate_search_params(
    all_search_params,
    search_param_names,
    build_params,
    algo,
    group_conf,
    algos_conf,
    conf_file,
    count,
    batch_size,
) -> list:
    """
    Validate and prepare the search parameters for the given algorithm
    and group.

    Parameters
    ----------
    all_search_params : itertools.product
        The Cartesian product of search parameter values.
    search_param_names : list
        The names of the search parameters.
    algo : str
        The name of the algorithm.
    group_conf : dict
        The configuration for the algorithm group.
    conf_file : dict
        The main configuration file.
    count : int
        The number of nearest neighbors to search for.
    batch_size : int
        The size of each batch for processing.

    Returns
    -------
    list
        A list of validated search parameters.
    """
    search_params_list = []
    for search_params in all_search_params:
        search_dict = dict(zip(search_param_names, search_params))
        if validate_constraints(
            algos_conf,
            algo,
            "search",
            search_dict,
            build_params,
            conf_file["dataset"].get("dims"),
            count,
            batch_size,
        ):
            search_params_list.append(search_dict)
    return search_params_list


def validate_constraints(
    algos_conf: Dict[str, Any],
    algo: str,
    constraint_type: str,
    param: Dict[str, Any],
    build_param: dict,
    dims: Any,
    k: Optional[int],
    batch_size: Optional[int],
) -> bool:
    """
    Validate the constraints for the given algorithm and constraint type.

    Parameters
    ----------
    algos_conf : Dict[str, Any]
        The configuration dictionary for the algorithms.
    algo : str
        The name of the algorithm.
    constraint_type : str
        The type of constraint to validate ('build' or 'search').
    param : Dict[str, Any]
        The parameters to validate against the constraints.
    dims : Any
        The dimensions required for the constraints.
    k : Optional[int]
        The number of nearest neighbors to search for.
    batch_size : Optional[int]
        The size of each batch for processing.

    Returns
    -------
    bool
        True if the constraints are valid, False otherwise.

    Raises
    ------
    ValueError
        If `dims` are needed for build constraints but not specified in the
        dataset configuration.
    """
    if constraint_type in algos_conf[algo]["constraints"]:
        importable = algos_conf[algo]["constraints"][constraint_type]
        module, func = (
            ".".join(importable.split(".")[:-1]),
            importable.split(".")[-1],
        )
        validator = import_module(module)
        constraints_func = getattr(validator, func)
        if constraint_type == "build":
            return constraints_func(param, dims)
        else:
            return constraints_func(param, build_param, k, batch_size)
    return True


def run_benchmark(
    subset_size: int,
    count: int,
    batch_size: int,
    dataset_configuration: Optional[str],
    configuration: Optional[str],
    dataset: str,
    dataset_path: str,
    executable_dir: str,
    build: Optional[bool],
    search: Optional[bool],
    algorithms: Optional[str],
    groups: str,
    algo_groups: Optional[str],
    force: bool,
    search_mode: str,
    search_threads: int,
    dry_run: bool,
    data_export: bool,
) -> None:
    """
    Runs a benchmarking process based on the provided configurations.

    Parameters
    ----------
    subset_size : int
        The subset size of the dataset.
    count : int
        The number of nearest neighbors to search for.
    batch_size : int
        The size of each batch for processing.
    dataset_configuration : Optional[str]
        Path to the dataset configuration file.
    configuration : Optional[str]
        Path to the algorithm configuration directory or file.
    dataset : str
        The name of the dataset to use.
    dataset_path : str
        The path to the dataset directory.
    build : Optional[bool]
        Whether to build the indices.
    search : Optional[bool]
        Whether to perform the search.
    algorithms : Optional[str]
        Comma-separated list of algorithm names to use.
    groups : str
        Comma-separated list of groups to consider.
    algo_groups : Optional[str]
        Comma-separated list of algorithm groups to consider.
    force : bool
        Whether to force the execution regardless of warnings.
    search_mode : str
        The mode of search to perform.
    search_threads : int
        The number of threads to use for searching.
    dry_run : bool
        Whether to perform a dry run without actual execution.

    Returns
    -------
    None
    """
    warnings.warn(
        "run_benchmark() is deprecated and will be removed in a future release. "
        "Use BenchmarkOrchestrator.run_benchmark() from cuvs_bench.orchestrator instead.",
        FutureWarning,
        stacklevel=2,
    )

    scripts_path = os.path.dirname(os.path.realpath(__file__))
    gpu_present = rmm_present()

    if not build and not search:
        build, search = True, True

    dataset_conf_all = load_yaml_file(
        dataset_configuration
        or os.path.join(scripts_path, "../config/datasets", "datasets.yaml")
    )
    dataset_conf = get_dataset_configuration(dataset, dataset_conf_all)
    conf_file = prepare_conf_file(dataset_conf, subset_size, count, batch_size)
    algos_conf_fs = gather_algorithm_configs(scripts_path, configuration)

    allowed_algos = algorithms.split(",") if algorithms else None
    allowed_groups = groups.split(",") if groups else None
    allowed_algo_groups = (
        [algo_group.split(".") for algo_group in algo_groups.split(",")]
        if algo_groups
        else None
    )
    algos_conf = load_algorithms_conf(
        algos_conf_fs,
        allowed_algos,
        allowed_groups,
        tuple(zip(*allowed_algo_groups)) if allowed_algo_groups else None,
    )

    executables_to_run = prepare_executables(
        algos_conf,
        load_yaml_file(
            os.path.join(scripts_path, "../config", "algorithms.yaml")
        ),
        gpu_present,
        conf_file,
        executable_dir,
        dataset_path,
        dataset,
        count,
        batch_size,
    )

    cuvs_bench_cpp(
        conf_file,
        dataset,
        os.path.dirname(configuration)
        if configuration and os.path.isfile(configuration)
        else os.path.join(scripts_path, "conf", "algos"),
        executables_to_run,
        dataset_path,
        force,
        build,
        search,
        dry_run,
        count,
        batch_size,
        search_threads,
        search_mode,
    )
