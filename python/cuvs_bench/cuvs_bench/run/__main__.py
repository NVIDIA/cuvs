#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import json
import os
from pathlib import Path
from typing import Optional

import click
import yaml

from .data_export import convert_json_to_csv_build, convert_json_to_csv_search
from ..orchestrator import BenchmarkOrchestrator


@click.command()
@click.option(
    "--subset-size",
    type=click.IntRange(min=1),
    help="The number of subset rows of the dataset to build the index",
)
@click.option(
    "-k",
    "--count",
    default=10,
    show_default=True,
    type=click.IntRange(min=1),
    prompt="Enter the number of  neighbors to search for",
    help="The number of nearest neighbors to search for",
)
@click.option(
    "-bs",
    "--batch-size",
    default=10000,
    show_default=True,
    type=click.IntRange(min=1),
    prompt="Enter the batch size",
    help="Number of query vectors to use in each query trial",
)
@click.option(
    "--dataset-configuration",
    default=None,
    show_default=True,
    help="Path to YAML configuration file for datasets",
)
@click.option(
    "--configuration",
    help="Path to YAML configuration file or directory for algorithms. "
    "Any run groups found in the specified file/directory will "
    "automatically override groups of the same name present in the "
    "default configurations, including `base`.",
)
@click.option(
    "--dataset",
    default="glove-100-inner",
    show_default=True,
    prompt="Enter the name of dataset",
    help="Name of dataset",
)
@click.option(
    "--dataset-path",
    default=lambda: os.environ.get(
        "RAPIDS_DATASET_ROOT_DIR",
        os.path.join(Path(__file__).parent, "datasets/"),
    ),
    show_default=True,
    prompt="Enter the path to dataset folder",
    help="Path to dataset folder, by default will look in "
    "RAPIDS_DATASET_ROOT_DIR if defined, otherwise a datasets "
    "subdirectory from the calling directory.",
)
@click.option(
    "--executable-dir",
    default=None,
    show_default=True,
    help="Path to executable folder, by default we will look in the "
    "devcontainer folder (/home/coder/cuvs/cpp/build/latest/bench/ann), in"
    "$CUVS_HOME/cpp/build/release and in $CONDA_PREFIX/bin/ann (in this "
    "order).",
)
@click.option("--build", is_flag=True, help="Build the index")
@click.option("--search", is_flag=True, help="Perform the search")
@click.option(
    "--algorithms",
    default="cuvs_cagra",
    show_default=True,
    prompt="Enter the comma separated list of named algorithms to run",
    help="Run only comma separated list of named algorithms. If parameters "
    "`groups` and `algo-groups` are both undefined, then group `base` "
    "is run by default.",
)
@click.option(
    "--groups",
    default="base",
    show_default=True,
    prompt="Enter the comma separated groups of parameters",
    help="Run only comma separated groups of parameters",
)
@click.option(
    "--algo-groups",
    help="Add comma separated <algorithm>.<group> to run. Example usage: "
    ' "--algo-groups=cuvs_cagra.large,hnswlib.large".',
)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    help="Re-run algorithms even if their results already exist",
)
@click.option(
    "-m",
    "--search-mode",
    type=click.Choice(["latency", "throughput"], case_sensitive=False),
    default="latency",
    show_default=True,
    prompt='Enter the search mode ("latency" or "throughput")',
    help="Run search in 'latency' (measure individual batches) or "
    "'throughput' (pipeline batches and measure end-to-end) mode.",
)
@click.option(
    "-t",
    "--search-threads",
    default=None,
    show_default=True,
    help="Specify the number threads to use for throughput benchmark. "
    "Single value or a pair of min and max separated by ':'. "
    "Example: --search-threads=1:4. Power of 2 values between 'min' "
    "and 'max' will be used. If only 'min' is specified, then a single "
    "test is run with 'min' threads. By default min=1, "
    "max=<num hyper threads>.",
)
@click.option(
    "-r",
    "--dry-run",
    is_flag=True,
    help="Dry-run mode will convert the yaml config for the specified "
    "algorithms and datasets to the json format that's consumed "
    "by the lower-level c++ binaries and then print the command to "
    "run execute the benchmarks but will not actually execute "
    "the command.",
)
@click.option(
    "--data-export",
    is_flag=True,
    help="By default, the intermediate JSON outputs produced by "
    "cuvs_bench.run to more easily readable CSV files is done "
    "automatically, which are needed to build charts made by "
    "cuvs_bench.plot. But if some of the benchmark runs failed or "
    "were interrupted, use this option to convert those intermediate "
    "files manually.",
)
@click.option(
    "--mode",
    type=click.Choice(["sweep", "tune"], case_sensitive=False),
    default="sweep",
    show_default=True,
    help="Benchmark mode: 'sweep' runs all parameter combinations from "
    "YAML configs. 'tune' uses Optuna to intelligently search the "
    "parameter space (requires --constraints).",
)
@click.option(
    "--constraints",
    default=None,
    help="Tune mode constraints as JSON string. One metric should have "
    "'maximize' or 'minimize', others have min/max bounds. "
    'Example: \'{"recall": "maximize", "latency": {"max": 10}}\'',
)
@click.option(
    "--n-trials",
    type=int,
    default=None,
    help="Number of Optuna trials for tune mode (default: 100).",
)
@click.option(
    "--backend-config",
    default=None,
    help="Path to YAML configuration file for non-C++ backends. "
    "The file must contain a 'backend' field specifying the backend "
    "type (e.g., 'opensearch', 'elastic'). All other fields are "
    "passed as backend-specific parameters.",
)
def main(
    subset_size: Optional[int],
    count: int,
    batch_size: int,
    dataset_configuration: Optional[str],
    configuration: Optional[str],
    dataset: str,
    dataset_path: str,
    executable_dir: str,
    build: bool,
    search: bool,
    algorithms: Optional[str],
    groups: str,
    algo_groups: Optional[str],
    force: bool,
    search_mode: str,
    search_threads: Optional[str],
    dry_run: bool,
    data_export: bool,
    mode: str,
    constraints: Optional[str],
    n_trials: Optional[int],
    backend_config: Optional[str],
) -> None:
    """
    Main function to run the benchmark with the provided options.

    Parameters
    ----------
    subset_size : Optional[int]
        The number of subset rows of the dataset to build the index.
    count : int
        The number of nearest neighbors to search for.
    batch_size : int
        Number of query vectors to use in each query trial.
    dataset_configuration : Optional[str]
        Path to YAML configuration file for datasets.
    configuration : Optional[str]
        Path to YAML configuration file or directory for algorithms.
    dataset : str
        Name of the dataset to use.
    dataset_path : str
        Path to the dataset folder.
    build : bool
        Whether to build the indices.
    search : bool
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
        The mode of search to perform ('latency' or 'throughput').
    search_threads : Optional[str]
        The number of threads to use for throughput benchmark.
    dry_run : bool
        Whether to perform a dry run without actual execution.
    data_export : bool
        Whether to export intermediate JSON results to CSV.
    mode : str
        Benchmark mode: 'sweep' (exhaustive) or 'tune' (Optuna-based).
    constraints : Optional[str]
        Tune mode constraints as JSON string.
    n_trials : Optional[int]
        Number of Optuna trials for tune mode.
    backend_config : Optional[str]
        Path to YAML config for non-C++ backends. If not provided,
        defaults to the C++ Google Benchmark backend. The YAML file
        must contain a 'backend' field (e.g., 'opensearch', 'elastic')
        and any backend-specific connection parameters (host, port, etc.).

    """
    if not data_export:
        # Determine backend type and extra kwargs from --backend-config
        backend_type = "cpp_gbench"
        backend_kwargs = {}
        if backend_config:
            with open(backend_config, "r") as f:
                cfg = yaml.safe_load(f)
            if not isinstance(cfg, dict):
                raise ValueError(
                    f"--backend-config must parse to a mapping, "
                    f"got {type(cfg).__name__}"
                )
            if "backend" not in cfg:
                raise ValueError(
                    "--backend-config must include a 'backend' field"
                )
            backend_type = cfg.pop("backend")
            backend_kwargs = cfg

        orchestrator = BenchmarkOrchestrator(backend_type=backend_type)
        orchestrator.run_benchmark(
            mode=mode,
            constraints=json.loads(constraints) if constraints else None,
            n_trials=n_trials,
            dataset=dataset,
            dataset_path=dataset_path,
            build=build,
            search=search,
            force=force,
            dry_run=dry_run,
            count=count,
            batch_size=batch_size,
            search_mode=search_mode,
            search_threads=search_threads,
            dataset_configuration=dataset_configuration,
            algorithm_configuration=configuration,
            algorithms=algorithms,
            groups=groups,
            algo_groups=algo_groups,
            subset_size=subset_size,
            executable_dir=executable_dir,
            **backend_kwargs,
        )

    convert_json_to_csv_build(dataset, dataset_path)
    convert_json_to_csv_search(dataset, dataset_path)


if __name__ == "__main__":
    main()
