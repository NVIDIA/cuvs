#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""
C++ Google Benchmark backend for cuvs-bench.

This backend wraps existing C++ Google Benchmark executables to maintain
backward compatibility with the current cuvs-bench infrastructure.
"""

import subprocess
import json
import tempfile
import time
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

from .base import BenchmarkBackend, Dataset, BuildResult, SearchResult
from ..orchestrator.config_loaders import IndexConfig


class CppGoogleBenchmarkBackend(BenchmarkBackend):
    """
    Backend for existing C++ Google Benchmark executables.

    This wraps the current cuvs_bench_cpp() logic into the plugin interface,
    ensuring 100% backward compatibility with existing C++ benchmarks.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary with:
        - name: str - User-defined name for this index configuration (required)
        - executable_path: str - Path to C++ benchmark executable
        - data_prefix: str - Prefix for dataset paths (default: "")
        - warmup_time: float - Warmup time in seconds (default: 1.0)

    Examples
    --------
    >>> config = {
    ...     "name": "ivf_flat_experiment",
    ...     "executable_path": "/path/to/CUVS_IVF_FLAT_ANN_BENCH",
    ...     "data_prefix": "data/",
    ...     "algo": "cuvs_ivf_flat"
    ... }
    >>> backend = CppGoogleBenchmarkBackend(config)
    >>> print(backend.name)  # "ivf_flat_experiment" (user-defined)
    >>> print(backend.algo)  # "cuvs_ivf_flat" (from config)
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize C++ benchmark backend."""
        super().__init__(config)

        self.executable_path = Path(config["executable_path"])
        self.data_prefix = config.get("data_prefix", "")
        self.warmup_time = config.get("warmup_time", 1.0)
        self.dataset_name = config.get("dataset", "")
        # Tuple: (build_name, search_name) e.g., ("cuvs_ivf_flat,test", "cuvs_ivf_flat,test,k10,bs1000")
        self.output_filename = config.get("output_filename", ("", ""))

        if not self.executable_path.exists():
            raise FileNotFoundError(
                f"C++ benchmark executable not found: {self.executable_path}"
            )

    @staticmethod
    def _gbench_error_message(
        benchmarks: List[Dict[str, Any]],
        phase: str,
        output_path: Optional[Path] = None,
    ) -> Optional[str]:
        """Return a compact message if Google Benchmark reported row errors."""
        errors = [b for b in benchmarks if b.get("error_occurred")]
        if not errors:
            return None

        location = f" in {output_path}" if output_path else ""
        details = []
        for bench in errors:
            name = bench.get("name", "<unknown benchmark>")
            message = bench.get("error_message", "<missing error_message>")
            details.append(f"{name}: {message}")
        return f"{phase} benchmark error(s){location}: " + "; ".join(details)

    @staticmethod
    def _index_debug_summary(
        indexes: List[IndexConfig],
    ) -> List[Dict[str, Any]]:
        """Return the C++ benchmark index payload fields useful in CI logs."""
        return [
            {
                "name": idx.name,
                "algo": idx.algo,
                "file": idx.file,
                "build_param": idx.build_param,
                "search_params": idx.search_params,
            }
            for idx in indexes
        ]

    def build(
        self,
        dataset: Dataset,
        indexes: List[IndexConfig],
        force: bool = False,
        dry_run: bool = False,
    ) -> BuildResult:
        """
        Build indexes using C++ Google Benchmark executable.

        ALL indexes are batched into ONE temp config and ONE C++ command,
        matching the old workflow (runners.py).

        Parameters
        ----------
        dataset : Dataset
            Dataset with base vectors
        indexes : List[IndexConfig]
            List of index configurations to build together
        force : bool
            Whether to force the execution regardless of existing results.
        dry_run : bool
            Whether to perform a dry run without actual execution.

        Returns
        -------
        BuildResult
            Build timing and metadata (aggregated across all indexes)

        Example
        -------
        >>> indexes = [
        ...     IndexConfig(name="cagra_32", build_param={"graph_degree": 32}, ...),
        ...     IndexConfig(name="cagra_64", build_param={"graph_degree": 64}, ...),
        ... ]
        >>> result = backend.build(dataset, indexes)
        # Both indexes built in ONE C++ command (dataset loaded once)
        """
        if not indexes:
            return BuildResult(
                index_path="",
                build_time_seconds=0.0,
                index_size_bytes=0,
                algorithm="",
                build_params={},
                metadata={"skipped": True, "reason": "no_indexes"},
                success=True,
            )

        first_index = indexes[0]

        # Pre-flight check (GPU, network, etc.)
        skip_reason = self._pre_flight_check()
        if skip_reason:
            return BuildResult(
                index_path=first_index.file,
                build_time_seconds=0.0,
                index_size_bytes=0,
                algorithm=first_index.algo,
                build_params=first_index.build_param,
                metadata={"skipped": True, "reason": skip_reason},
                success=True,
            )

        # Note: runners.py doesn't validate and lets C++ fail. We validate here for
        # better Python-side error messages.
        # C++ requires: name, base_file, query_file, distance (see conf.hpp parse_dataset)
        # C++ optional: groundtruth_neighbors_file
        if not dataset.base_file:
            raise ValueError(
                "dataset.base_file is required (C++ parser requires it)"
            )
        if not dataset.query_file:
            raise ValueError(
                "dataset.query_file is required (C++ parser requires it)"
            )

        # Create temporary JSON config (Google Benchmark format)
        # Structure matches runners.py temp_conf exactly
        # Contains ALL indexes in one config (matches old workflow)
        # delete=False because C++ subprocess needs to read file after Python closes it
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix=f"{self.dataset_name}_build_",
            delete=False,
        ) as f:
            temp_config_path = f.name
            dataset_config = {
                "name": dataset.name,
                "base_file": dataset.base_file,
                "query_file": dataset.query_file,
                "distance": dataset.distance_metric,
            }
            # groundtruth_neighbors_file is optional in C++
            if dataset.groundtruth_neighbors_file:
                dataset_config["groundtruth_neighbors_file"] = (
                    dataset.groundtruth_neighbors_file
                )

            # Build index list from ALL IndexConfig objects (matches runners.py)
            index_list = [
                {
                    "name": idx.name,
                    "algo": idx.algo,
                    "build_param": idx.build_param,
                    "file": idx.file,
                    "search_params": idx.search_params,  # Required by C++ even for build
                }
                for idx in indexes
            ]

            config = {
                "dataset": dataset_config,
                "search_basic_param": {"k": 10, "batch_size": 10000},
                "index": index_list,
            }
            json.dump(config, f, indent=2)

        # Prepare output directory and file (matches runners.py path structure)
        # OLD: {dataset_path}/{dataset}/result/build/{algo},{group}.json.lock
        result_folder = Path(self.data_prefix) / self.dataset_name / "result"
        build_folder = result_folder / "build"
        build_folder.mkdir(parents=True, exist_ok=True)
        build_file = f"{self.output_filename[0]}.json"
        temp_build_file = f"{build_file}.lock"
        benchmark_out = build_folder / temp_build_file

        # Construct C++ command
        cmd = [
            str(self.executable_path),
            "--build",
            f"--data_prefix={self.data_prefix}",
            "--benchmark_out_format=json",
            "--benchmark_counters_tabular=true",
            f"--benchmark_out={benchmark_out}",
        ]

        if force:
            cmd.append("--force")

        cmd.append(temp_config_path)

        # Dry run: print command and return without executing
        if dry_run:
            print(
                f"Benchmark command for {self.output_filename[0]}:\n{' '.join(cmd)}\n"
            )
            Path(temp_config_path).unlink(missing_ok=True)
            return BuildResult(
                index_path=first_index.file,
                build_time_seconds=0.0,
                index_size_bytes=0,
                algorithm=first_index.algo,
                build_params=first_index.build_param,
                metadata={"dry_run": True, "num_indexes": len(indexes)},
                success=True,
            )

        # Execute subprocess
        start_time = time.perf_counter()

        try:
            print("[cuvs-bench] C++ build launch", flush=True)
            print(f"  executable: {self.executable_path}", flush=True)
            print(f"  temp_config: {temp_config_path}", flush=True)
            print(f"  benchmark_out: {benchmark_out}", flush=True)
            print(f"  data_prefix: {self.data_prefix}", flush=True)
            print(
                "  indexes: "
                + json.dumps(self._index_debug_summary(indexes), default=str),
                flush=True,
            )
            print(f"  command: {' '.join(cmd)}", flush=True)
            subprocess.run(
                cmd,
                check=True,
                # Scale timeout with number of indexes
                timeout=3600 * len(indexes),
            )
            elapsed_time = time.perf_counter() - start_time

            # Parse Google Benchmark JSON output
            with open(benchmark_out) as f:
                gbench_results = json.load(f)

            # Merge with existing results (if any)
            self.merge_build_files(
                str(build_folder), build_file, temp_build_file
            )

            # Extract build metrics
            benchmarks = gbench_results.get("benchmarks", [])
            if not benchmarks:
                raise ValueError(
                    "No benchmarks found in Google Benchmark output"
                )
            print(
                f"[cuvs-bench] C++ build parsed {len(benchmarks)} "
                f"benchmark row(s) from {benchmark_out}",
                flush=True,
            )

            error_message = self._gbench_error_message(
                benchmarks, "Build", benchmark_out
            )
            if error_message:
                print(f"[cuvs-bench] {error_message}", flush=True)
                raise RuntimeError(error_message)

            # Return aggregated result
            total_build_time = sum(b.get("real_time", 0) for b in benchmarks)

            return BuildResult(
                index_path=first_index.file,
                build_time_seconds=total_build_time,
                index_size_bytes=0,  # Multiple indexes, can't report single size
                algorithm=first_index.algo,
                build_params=first_index.build_param,
                metadata={
                    "num_indexes": len(indexes),
                    "num_benchmarks": len(benchmarks),
                    "elapsed_time": elapsed_time,
                    "context": gbench_results.get("context", {}),
                },
                success=True,
            )

        except subprocess.CalledProcessError as e:
            print(
                f"[cuvs-bench] C++ build subprocess failed: "
                f"returncode={e.returncode} stderr={e.stderr}",
                flush=True,
            )
            return BuildResult(
                index_path=first_index.file,
                build_time_seconds=time.perf_counter() - start_time,
                index_size_bytes=0,
                algorithm=first_index.algo,
                build_params=first_index.build_param,
                success=False,
                error_message=f"Build failed: {e.stderr}",
            )

        except Exception as e:
            print(f"[cuvs-bench] C++ build exception: {e}", flush=True)
            return BuildResult(
                index_path=first_index.file,
                build_time_seconds=time.perf_counter() - start_time,
                index_size_bytes=0,
                algorithm=first_index.algo,
                build_params=first_index.build_param,
                success=False,
                error_message=f"Build error: {str(e)}",
            )

        finally:
            # Cleanup temporary files
            Path(temp_config_path).unlink(missing_ok=True)
            Path(benchmark_out).unlink(missing_ok=True)

    def search(
        self,
        dataset: Dataset,
        indexes: List[IndexConfig],
        k: int,
        batch_size: int = 10000,
        mode: str = "latency",
        force: bool = False,
        search_threads: Optional[int] = None,
        dry_run: bool = False,
    ) -> SearchResult:
        """
        Search using C++ Google Benchmark executable.

        ALL indexes are batched into ONE temp config and ONE C++ command,
        matching the old workflow (runners.py). Each index has its own
        search_params list, so total benchmarks = sum(len(idx.search_params) for idx).

        Parameters
        ----------
        dataset : Dataset
            Dataset with query vectors and ground truth
        indexes : List[IndexConfig]
            List of index configurations, each with its own search_params
        k : int
            The number of nearest neighbors to search for.
        batch_size : int
            The size of each batch for processing.
        mode : str
            The mode of search to perform ('latency' or 'throughput'),
            by default 'latency'.
        force : bool
            Whether to force the execution regardless of existing results.
        search_threads : Optional[int]
            The number of threads to use for searching.
        dry_run : bool
            Whether to perform a dry run without actual execution.

        Returns
        -------
        SearchResult
            Search timing, recall, and QPS (aggregated across all indexes)
        """
        if not indexes:
            return SearchResult(
                neighbors=np.array([]),
                distances=np.array([]),
                search_time_ms=0.0,
                queries_per_second=0.0,
                recall=0.0,
                algorithm="",
                search_params=[],
                metadata={"skipped": True, "reason": "no_indexes"},
                success=True,
            )

        first_index = indexes[0]

        # Pre-flight check (GPU, network, etc.)
        skip_reason = self._pre_flight_check()
        if skip_reason:
            return SearchResult(
                neighbors=np.array([]),
                distances=np.array([]),
                search_time_ms=0.0,
                queries_per_second=0.0,
                recall=0.0,
                algorithm=first_index.algo,
                search_params=first_index.search_params
                if first_index.search_params
                else [],
                metadata={"skipped": True, "reason": skip_reason},
                success=True,
            )

        # Note: runners.py doesn't validate and lets C++ fail. We validate here for
        # better Python-side error messages.
        # C++ requires: name, base_file, query_file, distance (see conf.hpp parse_dataset)
        # C++ optional: groundtruth_neighbors_file (but needed for recall calculation)
        if not dataset.base_file:
            raise ValueError(
                "dataset.base_file is required (C++ parser requires it)"
            )
        if not dataset.query_file:
            raise ValueError(
                "dataset.query_file is required (C++ parser requires it)"
            )

        # Create temporary JSON config
        # Structure matches runners.py temp_conf exactly
        # Contains ALL indexes with their search_params (matches old workflow)
        # delete=False because C++ subprocess needs to read file after Python closes it
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix=f"{self.dataset_name}_search_",
            delete=False,
        ) as f:
            temp_config_path = f.name
            dataset_config = {
                "name": dataset.name,
                "base_file": dataset.base_file,
                "query_file": dataset.query_file,
                "distance": dataset.distance_metric,
            }
            # groundtruth_neighbors_file is optional in C++, but needed for recall calculation
            if dataset.groundtruth_neighbors_file:
                dataset_config["groundtruth_neighbors_file"] = (
                    dataset.groundtruth_neighbors_file
                )

            # Build index list from ALL IndexConfig objects (matches runners.py)
            index_list = [
                {
                    "name": idx.name,
                    "algo": idx.algo,
                    "build_param": idx.build_param,
                    "file": idx.file,
                    "search_params": idx.search_params,
                }
                for idx in indexes
            ]

            config = {
                "dataset": dataset_config,
                "search_basic_param": {"k": k, "batch_size": batch_size},
                "index": index_list,
            }
            json.dump(config, f, indent=2)

        # Prepare output file (matches runners.py path structure)
        # OLD: {dataset_path}/{dataset}/result/search/{algo},{group},k{k},bs{batch_size}.json
        result_folder = Path(self.data_prefix) / self.dataset_name / "result"
        output_dir = result_folder / "search"
        output_dir.mkdir(parents=True, exist_ok=True)
        search_file = f"{self.output_filename[1]}.json"
        output_json = output_dir / search_file

        # Use temp file for C++ output, then merge into accumulated results
        temp_output_json = output_dir / f"{self.output_filename[1]}_temp.json"

        # Construct C++ command
        cmd = [
            str(self.executable_path),
            "--search",
            f"--data_prefix={self.data_prefix}",
            f"--mode={mode}",
            f"--benchmark_min_warmup_time={self.warmup_time}",
            f"--override_kv=k:{k}",
            f"--override_kv=n_queries:{batch_size}",
            "--benchmark_out_format=json",
            "--benchmark_counters_tabular=true",
            f"--benchmark_out={temp_output_json}",
        ]

        if force:
            cmd.append("--force")
        if search_threads:
            cmd.append(f"--threads={search_threads}")

        cmd.append(temp_config_path)

        # Calculate expected number of benchmarks
        total_search_configs = sum(len(idx.search_params) for idx in indexes)

        # Dry run: print command and return without executing
        if dry_run:
            print(
                f"Benchmark command for {self.output_filename[1]}:\n{' '.join(cmd)}\n"
            )
            Path(temp_config_path).unlink(missing_ok=True)
            return SearchResult(
                neighbors=np.array([]),
                distances=np.array([]),
                search_time_ms=0.0,
                queries_per_second=0.0,
                recall=0.0,
                algorithm=first_index.algo,
                search_params=first_index.search_params,
                metadata={
                    "dry_run": True,
                    "num_indexes": len(indexes),
                    "total_search_configs": total_search_configs,
                },
                success=True,
            )

        # Execute subprocess
        start_time = time.perf_counter()

        try:
            print("[cuvs-bench] C++ search launch", flush=True)
            print(f"  executable: {self.executable_path}", flush=True)
            print(f"  temp_config: {temp_config_path}", flush=True)
            print(f"  temp_output_json: {temp_output_json}", flush=True)
            print(f"  output_json: {output_json}", flush=True)
            print(f"  data_prefix: {self.data_prefix}", flush=True)
            print(
                f"  total_search_configs: {total_search_configs}", flush=True
            )
            print(
                "  indexes: "
                + json.dumps(self._index_debug_summary(indexes), default=str),
                flush=True,
            )
            print(f"  command: {' '.join(cmd)}", flush=True)
            subprocess.run(
                cmd,
                check=True,
                # Scale timeout with number of indexes
                timeout=3600 * len(indexes),
            )
            elapsed_time = time.perf_counter() - start_time

            # Parse results from temp file
            with open(temp_output_json) as f:
                gbench_results = json.load(f)

            new_benchmarks = gbench_results.get("benchmarks", [])

            # Handle result accumulation based on append_results config
            # (set by orchestrator for tune mode: trial 0 overwrites, trial 1+ appends)
            append_results = self.config.get("append_results", False)
            if append_results and output_json.exists():
                # Append new benchmarks to accumulated file (tune mode)
                with open(output_json) as f:
                    accumulated = json.load(f)
                accumulated["benchmarks"].extend(new_benchmarks)
                with open(output_json, "w") as f:
                    json.dump(accumulated, f, indent=2)
            else:
                # Overwrite with new results (sweep mode, or first tune trial)
                with open(output_json, "w") as f:
                    json.dump(gbench_results, f, indent=2)

            # Clean up temp file
            temp_output_json.unlink(missing_ok=True)

            benchmarks = new_benchmarks
            if not benchmarks:
                raise ValueError(
                    "No benchmarks found in Google Benchmark output"
                )
            print(
                f"[cuvs-bench] C++ search parsed {len(benchmarks)} "
                f"benchmark row(s) from {output_json}",
                flush=True,
            )

            error_message = self._gbench_error_message(
                benchmarks, "Search", output_json
            )
            if error_message:
                print(f"[cuvs-bench] {error_message}", flush=True)
                raise RuntimeError(error_message)

            # Aggregate metrics across all benchmarks
            total_search_time = sum(b.get("real_time", 0) for b in benchmarks)
            avg_recall = (
                sum(b.get("Recall", 0) for b in benchmarks) / len(benchmarks)
                if benchmarks
                else 0
            )
            avg_qps = (
                sum(b.get("items_per_second", 0) for b in benchmarks)
                / len(benchmarks)
                if benchmarks
                else 0
            )

            # Note: C++ Google Benchmark doesn't return actual neighbors/distances
            # This is a limitation of the current system
            return SearchResult(
                neighbors=np.array([]),  # Not available from C++ benchmark
                distances=np.array([]),  # Not available from C++ benchmark
                search_time_ms=total_search_time,
                queries_per_second=avg_qps,
                recall=avg_recall,
                algorithm=first_index.algo,
                search_params=first_index.search_params,
                metadata={
                    "num_indexes": len(indexes),
                    "num_benchmarks": len(benchmarks),
                    "elapsed_time": elapsed_time,
                    "latency_us": benchmarks[0].get("Latency")
                    if benchmarks
                    else None,
                    "end_to_end": benchmarks[0].get("end_to_end")
                    if benchmarks
                    else None,
                    "context": gbench_results.get("context", {}),
                },
                success=True,
            )

        except subprocess.CalledProcessError as e:
            print(
                f"[cuvs-bench] C++ search subprocess failed: "
                f"returncode={e.returncode} stderr={e.stderr}",
                flush=True,
            )
            return SearchResult(
                neighbors=np.array([]),
                distances=np.array([]),
                search_time_ms=time.perf_counter() - start_time,
                queries_per_second=0.0,
                recall=0.0,
                algorithm=first_index.algo,
                search_params=first_index.search_params,
                success=False,
                error_message=f"Search failed: {e.stderr}",
            )

        except Exception as e:
            print(f"[cuvs-bench] C++ search exception: {e}", flush=True)
            return SearchResult(
                neighbors=np.array([]),
                distances=np.array([]),
                search_time_ms=time.perf_counter() - start_time,
                queries_per_second=0.0,
                recall=0.0,
                algorithm=first_index.algo,
                search_params=first_index.search_params,
                success=False,
                error_message=f"Search error: {str(e)}",
            )

        finally:
            # Cleanup temporary config
            Path(temp_config_path).unlink(missing_ok=True)

    def merge_build_files(
        self, build_dir: str, build_file: str, temp_build_file: str
    ) -> None:
        """
        Merge temporary build files into the main build file.

        Parameters
        ----------
        build_dir : str
            The directory of the build files.
        build_file : str
            The main build file.
        temp_build_file : str
            The temporary build file to merge.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the temporary build file is not found.
        """
        build_dict = {}

        # If build file exists, read it
        build_json_path = os.path.join(build_dir, build_file)
        tmp_build_json_path = os.path.join(build_dir, temp_build_file)
        if os.path.isfile(build_json_path):
            try:
                with open(build_json_path, "r") as f:
                    build_dict = json.load(f)
            except Exception as e:
                print(
                    f"Error loading existing build file: {build_json_path} ({e})"
                )

        temp_build_dict = {}
        if os.path.isfile(tmp_build_json_path):
            with open(tmp_build_json_path, "r") as f:
                temp_build_dict = json.load(f)
        else:
            raise ValueError(
                f"Temp build file not found: {tmp_build_json_path}"
            )

        tmp_benchmarks = temp_build_dict.get("benchmarks", {})
        benchmarks = build_dict.get("benchmarks", {})

        # If the build time is absolute 0 then an error occurred
        final_bench_dict = {
            b["name"]: b for b in benchmarks if b["real_time"] > 0
        }

        for tmp_bench in tmp_benchmarks:
            if tmp_bench["real_time"] > 0:
                final_bench_dict[tmp_bench["name"]] = tmp_bench

        temp_build_dict["benchmarks"] = list(final_bench_dict.values())
        with open(build_json_path, "w") as f:
            json_str = json.dumps(temp_build_dict, indent=2)
            f.write(json_str)

    @property
    def algo(self) -> str:
        """Algorithm name from config."""
        return self.config.get("algo", "")
