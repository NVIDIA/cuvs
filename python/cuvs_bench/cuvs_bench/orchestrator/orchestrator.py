#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

"""
Benchmark Orchestrator for cuvs-bench.

This module provides the BenchmarkOrchestrator class that coordinates
benchmark runs across different backends using the registry pattern.
"""

from typing import List, Optional, Union

import numpy as np

from ..backends.base import Dataset, BuildResult, SearchResult
from ..backends.registry import (
    get_backend_class,
    get_config_loader,
    list_backends,
)
from ..backends._utils import compute_recall
from .config_loaders import DatasetConfig


class BenchmarkOrchestrator:
    """
    Orchestrator for running benchmarks using the pluggable backend system.

    This class coordinates benchmark runs by:
    1. Using a ConfigLoader to load backend-specific configurations
    2. Creating Backend instances from the registry
    3. Running build/search operations
    4. Collecting and returning results

    Parameters
    ----------
    backend_type : str
        Type of backend to use (e.g., 'cpp_gbench', 'milvus')
    **loader_kwargs
        Additional arguments passed to the ConfigLoader constructor

    Examples
    --------
    >>> # For C++ benchmarks
    >>> from cuvs_bench.orchestrator import BenchmarkOrchestrator
    >>> orchestrator = BenchmarkOrchestrator(backend_type="cpp_gbench")
    >>> results = orchestrator.run_benchmark(
    ...     dataset="glove-100-inner",
    ...     dataset_path="/data/",
    ...     algorithms="cuvs_ivf_flat",
    ...     build=True,
    ...     search=True
    ... )

    >>> # For Milvus benchmarks (when implemented)
    >>> orchestrator = BenchmarkOrchestrator(backend_type="milvus")
    >>> results = orchestrator.run_benchmark(
    ...     host="localhost",
    ...     port=19530,
    ...     ...
    ... )
    """

    def __init__(self, backend_type: str = "cpp_gbench"):
        """
        Initialize the orchestrator.

        Parameters
        ----------
        backend_type : str
            Type of backend to use
        """
        self.backend_type = backend_type

        # Get classes from registry
        self.backend_class = get_backend_class(backend_type)
        # Get the config loader for this specific backend
        loader_class = get_config_loader(backend_type)

        # Instantiate config loader
        self.config_loader = loader_class()

    def run_benchmark(
        self,
        mode: str = "sweep",
        constraints: dict = None,
        n_trials: int = None,
        build: bool = True,
        search: bool = True,
        force: bool = False,
        dry_run: bool = False,
        count: int = 10,
        batch_size: int = 10000,
        search_mode: str = "latency",
        search_threads: Optional[int] = None,
        **loader_kwargs,
    ) -> List[Union[BuildResult, SearchResult]]:
        """
        Run benchmarks using the configured backend.

        Parameters
        ----------
        mode : str
            Benchmark mode: "sweep" (default, exhaustive Cartesian product) or
            "tune" (intelligent search using Optuna)
        constraints : dict
            For tune mode: optimization target and hard limits.
            - One metric should have "maximize" or "minimize" as value (the target)
            - Other metrics have {"min": X} or {"max": X} as bounds (hard limits)
            E.g., {"recall": "maximize", "latency": {"max": 10}} or
            {"latency": "minimize", "recall": {"min": 0.95}}
        n_trials : int
            For tune mode: maximum number of Optuna trials per parameter configuration.
            default is 100.
            Ignored in sweep mode.
        build : bool
            Whether to build indices
        search : bool
            Whether to perform searches
        force : bool
            Whether to force rebuild even if index exists
        dry_run : bool
            Whether to perform a dry run without actual execution
        count : int
            Number of neighbors (k) to search for
        batch_size : int
            Batch size for search
        search_mode : str
            Search mode ('throughput' or 'latency')
        search_threads : Optional[int]
            Number of threads for search
        **loader_kwargs
            Backend-specific arguments passed to ConfigLoader.load().

            These vary by backend type to keep the orchestrator backend-agnostic:
            - cpp_gbench: dataset, dataset_path, algorithms, groups, algo_groups,
              dataset_configuration, algorithm_configuration, subset_size
            - milvus (future): host, port, collection, api_key
            - qdrant (future): host, api_key, collection

        Returns
        -------
        List[Union[BuildResult, SearchResult]]
            List of result objects, one per benchmark run:
            - sweep mode: One result per IndexConfig (Cartesian product of params)
            - tune mode: One result per Optuna trial (n_trials total)

            Each SearchResult contains: recall, search_time_ms,
            queries_per_second, success, metadata, etc.
        """
        # Validate mode parameter
        valid_modes = ("sweep", "tune")
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of: {valid_modes}"
            )

        # If build and search are not provided, set them to True by default
        if not build and not search:
            build, search = True, True

        # Branch based on mode
        if mode == "sweep":
            return self._run_sweep(
                build=build,
                search=search,
                force=force,
                dry_run=dry_run,
                count=count,
                batch_size=batch_size,
                search_mode=search_mode,
                search_threads=search_threads,
                **loader_kwargs,
            )
        else:  # mode == "tune"
            return self._run_tune(
                constraints=constraints,
                n_trials=n_trials,
                build=build,
                search=search,
                force=force,
                dry_run=dry_run,
                count=count,
                batch_size=batch_size,
                search_mode=search_mode,
                search_threads=search_threads,
                **loader_kwargs,
            )

    def _run_sweep(
        self,
        build: bool,
        search: bool,
        force: bool,
        dry_run: bool,
        count: int,
        batch_size: int,
        search_mode: str,
        search_threads: Optional[int],
        **loader_kwargs,
    ) -> List[Union[BuildResult, SearchResult]]:
        """
        Run exhaustive sweep over all parameter combinations.

        This is the default mode that runs all configurations from the
        Cartesian product of build/search parameters defined in YAML.
        """
        # Load configurations using the backend-specific loader
        dataset_config, benchmark_configs = self.config_loader.load(
            count=count, batch_size=batch_size, **loader_kwargs
        )

        # Create Dataset object from config
        bench_dataset = self._create_dataset(dataset_config)

        # Collect results
        results: List[Union[BuildResult, SearchResult]] = []

        # Run benchmarks
        # Each config contains ALL indexes for one executable (matches runners.py)
        for config in benchmark_configs:
            # Create backend instance
            backend = self.backend_class(config.backend_config)
            try:
                backend.initialize()

                if build:
                    # Pass ALL indexes at once - ONE C++ command builds all
                    build_result = backend.build(
                        dataset=bench_dataset,
                        indexes=config.indexes,
                        force=force,
                        dry_run=dry_run,
                    )
                    results.append(build_result)

                    if not build_result.success:
                        print(
                            f"Build failed for {config.index_name}: {build_result.error_message}"
                        )
                        continue

                if search:
                    # Pass ALL indexes at once - ONE C++ command searches all
                    # Each index has its own search_params list
                    # Total benchmarks = sum(len(idx.search_params) for idx in indexes)
                    search_result = backend.search(
                        dataset=bench_dataset,
                        indexes=config.indexes,
                        k=count,
                        batch_size=batch_size,
                        mode=search_mode,
                        force=force,
                        search_threads=search_threads,
                        dry_run=dry_run,
                    )

                    # Compute recall for backends that return actual neighbors.
                    # The C++ backend computes recall in the subprocess and returns
                    # empty neighbors, so this is skipped for it.
                    # Empty neighbors or nonzero recall indicate that the backend
                    # already handled recall itself.
                    if (
                        search_result.success
                        and search_result.neighbors.size > 0
                        and search_result.recall == 0.0
                    ):
                        gt = bench_dataset.groundtruth_neighbors
                        if gt is not None:
                            search_result.recall = compute_recall(
                                search_result.neighbors, gt, count
                            )

                    results.append(search_result)

                    if not search_result.success:
                        print(
                            f"Search failed for {config.index_name}: {search_result.error_message}"
                        )
            finally:
                backend.cleanup()

        return results

    def _run_tune(
        self,
        constraints: dict,
        n_trials: int,
        build: bool,
        search: bool,
        force: bool,
        dry_run: bool,
        count: int,
        batch_size: int,
        search_mode: str,
        search_threads: Optional[int],
        **loader_kwargs,
    ) -> List[Union[BuildResult, SearchResult]]:
        """
        Run intelligent parameter tuning using Optuna.

        Instead of exhaustive sweep, uses Optuna to intelligently explore
        the parameter space and find optimal configurations based on
        the provided constraints.

        Parameters
        ----------
        constraints : dict
            Optimization target and hard limits.
            - One metric should have "maximize" or "minimize" as value
            - Other metrics have {"min": X} or {"max": X} as bounds
        n_trials : int
            Maximum number of Optuna trials (default: 100 if not provided).
        **loader_kwargs
            Backend-specific arguments passed to ConfigLoader.load().
            Required for tune mode:
            - algorithms : str - Algorithm name (e.g., "cuvs_ivf_flat")
            - dataset : str - Dataset name
            - dataset_path : str - Path to dataset directory

            Optional:
            - groups, algo_groups, dataset_configuration, algorithm_configuration

            Note: Internally, tune mode adds _tune_mode, _tune_build_params,
            and _tune_search_params to these kwargs for each Optuna trial.
        """
        # Tune mode requires search to be enabled (metrics come from SearchResult)
        if not search:
            raise ValueError(
                "tune mode requires search=True to optimize metrics (recall, latency, throughput)"
            )

        # Import Optuna (optional dependency)
        try:
            import optuna

            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise ImportError(
                "Optuna is required for tune mode. Install with: pip install optuna"
            )

        from ..backends.search_spaces import get_search_space

        # Get algorithm from loader_kwargs
        algorithm = loader_kwargs.get("algorithms")
        if not algorithm:
            raise ValueError("'algorithms' must be specified for tune mode")

        # Get search space for this algorithm
        search_space = get_search_space(algorithm)

        # Set default n_trials
        if n_trials is None:
            n_trials = 100

        # Parse constraints to find optimization target and direction
        if not constraints:
            raise ValueError(
                "constraints must be specified for tune mode. "
                "Example: {'recall': 'maximize', 'latency': {'max': 10}}"
            )

        optimize_metric = None
        direction = None
        hard_constraints = {}

        for metric, value in constraints.items():
            if value == "maximize":
                optimize_metric = metric
                direction = "maximize"
            elif value == "minimize":
                optimize_metric = metric
                direction = "minimize"
            elif isinstance(value, dict):
                hard_constraints[metric] = value
            else:
                raise ValueError(
                    f"Invalid constraint for '{metric}': {value}. "
                    "Use 'maximize', 'minimize', or {{'min': X}} / {{'max': X}}"
                )

        if not optimize_metric:
            raise ValueError(
                "One metric must have 'maximize' or 'minimize' as its value. "
                "Example: {'recall': 'maximize', 'latency': {'max': 10}}"
            )

        # Collect all results for pareto plot
        all_results: List[Union[BuildResult, SearchResult]] = []

        def suggest_params(
            trial, param_space: dict, build_params: dict = None
        ) -> dict:
            """Suggest parameters from search space using Optuna trial."""
            params = {}
            for param, spec in param_space.items():
                if spec["type"] == "int":
                    max_val = spec["max"]
                    # Handle dynamic constraints (e.g., nprobe <= nlist)
                    if isinstance(max_val, str) and build_params:
                        max_val = build_params.get(max_val, 1000)
                    params[param] = trial.suggest_int(
                        param, spec["min"], max_val, log=spec.get("log", False)
                    )
                elif spec["type"] == "float":
                    params[param] = trial.suggest_float(
                        param,
                        spec["min"],
                        spec["max"],
                        log=spec.get("log", False),
                    )
                elif spec["type"] == "categorical":
                    params[param] = trial.suggest_categorical(
                        param, spec["choices"]
                    )
            return params

        def objective(trial) -> float:
            """Optuna objective function for a single trial."""
            # Suggest build parameters
            build_params = suggest_params(trial, search_space.get("build", {}))

            # Suggest search parameters (may depend on build params)
            search_params_dict = suggest_params(
                trial, search_space.get("search", {}), build_params
            )

            # Run single trial with these specific parameters
            # First trial (trial.number=0) overwrites, subsequent trials append
            result = self._run_trial(
                algorithm=algorithm,
                build_params=build_params,
                search_params=search_params_dict,
                build=build,
                search=search,
                force=force,
                dry_run=dry_run,
                count=count,
                batch_size=batch_size,
                search_mode=search_mode,
                search_threads=search_threads,
                append_results=(trial.number > 0),
                **loader_kwargs,
            )

            # Store result for pareto plot
            all_results.append(result)

            # Check if trial failed
            if not result.success:
                raise optuna.TrialPruned()

            # Build metrics dict from SearchResult attributes
            # No fallbacks - if metrics are missing, let it fail loudly so we can fix the root cause
            metrics = {
                "recall": result.recall,
                "latency": result.search_time_ms,
                "throughput": result.queries_per_second,
            }
            # Also include latency from metadata if available (in microseconds)
            if hasattr(result, "metadata") and result.metadata:
                if (
                    "latency_us" in result.metadata
                    and result.metadata["latency_us"]
                ):
                    metrics["latency_us"] = result.metadata["latency_us"]
                metrics.update(
                    {
                        k: v
                        for k, v in result.metadata.items()
                        if k not in metrics and v is not None
                    }
                )

            # Check hard constraints - prune if violated
            for metric, bounds in hard_constraints.items():
                metric_value = metrics.get(metric)
                if metric_value is None:
                    continue
                if "min" in bounds and metric_value < bounds["min"]:
                    raise optuna.TrialPruned()
                if "max" in bounds and metric_value > bounds["max"]:
                    raise optuna.TrialPruned()

            # Return optimization target
            return metrics[optimize_metric]

        # Create and run Optuna study
        study = optuna.create_study(direction=direction)

        print(f"Starting tune mode: {n_trials} trials for '{algorithm}'")
        print(f"Optimizing: {optimize_metric} ({direction})")
        print(f"Constraints: {hard_constraints}")

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Report best results (handle case where all trials were pruned)
        try:
            print(f"\nBest trial: #{study.best_trial.number}")
            print(f"Best params: {study.best_params}")
            print(f"Best {optimize_metric}: {study.best_value:.4f}")
        except ValueError:
            print(
                f"\n⚠️ All {n_trials} trials were pruned (constraints not met)"
            )
            print(f"   Consider relaxing constraints: {hard_constraints}")

        return all_results

    def _run_trial(
        self,
        algorithm: str,
        build_params: dict,
        search_params: dict,
        build: bool,
        search: bool,
        force: bool,
        dry_run: bool,
        count: int,
        batch_size: int,
        search_mode: str,
        search_threads: Optional[int],
        append_results: bool = False,
        **loader_kwargs,
    ) -> Union[BuildResult, SearchResult]:
        """
        Run a single benchmark trial with specific parameters.

        Used by tune mode to evaluate one parameter configuration.

        Parameters
        ----------
        algorithm : str
            Algorithm name
        build_params : dict
            Build parameters suggested by Optuna
        search_params : dict
            Search parameters suggested by Optuna
        append_results : bool
            If True, append results to existing JSON file.
        """
        # Override loader_kwargs with tune-specific params
        tune_kwargs = loader_kwargs.copy()
        tune_kwargs["_tune_mode"] = True
        tune_kwargs["_tune_build_params"] = build_params
        tune_kwargs["_tune_search_params"] = search_params

        # Load config (config_loader needs to handle _tune_* kwargs)
        dataset_config, benchmark_configs = self.config_loader.load(
            count=count, batch_size=batch_size, **tune_kwargs
        )

        # Create dataset
        bench_dataset = self._create_dataset(dataset_config)

        # Should have exactly one config for single trial
        if not benchmark_configs:
            return SearchResult(
                success=False,
                error_message="No config generated for trial",
                metrics={},
                search_params=[],
            )

        config = benchmark_configs[0]
        # Pass append_results via config (backend-specific, not in base class)
        backend_config = {
            **config.backend_config,
            "append_results": append_results,
        }
        backend = self.backend_class(backend_config)
        try:
            backend.initialize()

            result = None

            if build:
                result = backend.build(
                    dataset=bench_dataset,
                    indexes=config.indexes,
                    force=force,
                    dry_run=dry_run,
                )
                if not result.success:
                    return result

            if search:
                result = backend.search(
                    dataset=bench_dataset,
                    indexes=config.indexes,
                    k=count,
                    batch_size=batch_size,
                    mode=search_mode,
                    force=force,
                    search_threads=search_threads,
                    dry_run=dry_run,
                )

                # Compute recall for backends that return actual neighbors.
                # Empty neighbors or nonzero recall indicate that the backend
                # already handled recall itself.
                if (
                    result.success
                    and result.neighbors.size > 0
                    and result.recall == 0.0
                ):
                    gt = bench_dataset.groundtruth_neighbors
                    if gt is not None:
                        result.recall = compute_recall(
                            result.neighbors, gt, count
                        )

            return result
        finally:
            backend.cleanup()

    def _create_dataset(self, dataset_config: DatasetConfig) -> Dataset:
        """
        Create a Dataset object from DatasetConfig.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Dataset configuration

        Returns
        -------
        Dataset
            Dataset object for use with backends
        """
        return Dataset(
            name=dataset_config.name,
            # C++ backend loads vectors from files; other backends may need actual arrays
            training_vectors=np.empty((0, 0)),
            query_vectors=np.empty((0, 0)),
            # Ground truth arrays - used by Python-native backends for recall calculation
            groundtruth_neighbors=None,
            groundtruth_distances=None,
            # File paths - used by C++ backend
            base_file=dataset_config.base_file,
            query_file=dataset_config.query_file,
            groundtruth_neighbors_file=dataset_config.groundtruth_neighbors_file,
            groundtruth_distances_file=dataset_config.groundtruth_distances_file,
            distance_metric=dataset_config.distance,
            metadata={"subset_size": dataset_config.subset_size},
        )

    @staticmethod
    def available_backends() -> List[str]:
        """
        List all available backend types.

        Returns
        -------
        List[str]
            List of registered backend names
        """
        return list(list_backends().keys())
