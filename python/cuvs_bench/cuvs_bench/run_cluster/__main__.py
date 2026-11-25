#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import os

import click

from .run_cluster import run_cluster_benchmark


@click.command()
@click.option("--dataset", required=True, help="Name of dataset")
@click.option(
    "--dataset-path",
    default=lambda: os.environ.get(
        "RAPIDS_DATASET_ROOT_DIR", os.path.join(os.getcwd(), "datasets")
    ),
    show_default="RAPIDS_DATASET_ROOT_DIR or ./datasets",
    help="Path to dataset folder",
)
@click.option(
    "--dataset-configuration",
    default=None,
    help="Path to YAML configuration file for datasets",
)
@click.option(
    "--configuration",
    default=None,
    help="Path to YAML configuration file or directory for clustering "
    "algorithms. Run groups found there override groups of the same name "
    "in the default configurations, including `base`.",
)
@click.option(
    "--algorithms",
    default=None,
    help="Comma separated list of algorithms to run (default: all)",
)
@click.option(
    "--groups",
    default="base",
    show_default=True,
    help="Comma separated groups of parameters to run",
)
@click.option(
    "--algo-groups",
    default=None,
    help="Add comma separated <algorithm>.<group> to run, e.g. "
    '"cuvs_kmeans.large,cuvs_kmeans_balanced.large".',
)
@click.option(
    "--subset-size",
    type=click.IntRange(min=1),
    default=None,
    help="Number of rows of the dataset to cluster",
)
@click.option(
    "--executable-dir",
    default=None,
    help="Path to the executable folder; by default the devcontainer build "
    "(/home/coder/cuvs/cpp/build/latest/bench/cluster), "
    "$CUVS_HOME/cpp/build/release and $CONDA_PREFIX/bin/cluster are searched.",
)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    help="Re-run and overwrite existing results",
)
@click.option(
    "-r",
    "--dry-run",
    is_flag=True,
    help="Print the commands and configurations without running them",
)
def main(
    dataset,
    dataset_path,
    dataset_configuration,
    configuration,
    algorithms,
    groups,
    algo_groups,
    subset_size,
    executable_dir,
    force,
    dry_run,
):
    run_cluster_benchmark(
        dataset=dataset,
        dataset_path=dataset_path,
        dataset_configuration=dataset_configuration,
        algorithm_configuration=configuration,
        algorithms=algorithms,
        groups=groups,
        algo_groups=algo_groups,
        subset_size=subset_size,
        executable_dir=executable_dir,
        force=force,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    main()
