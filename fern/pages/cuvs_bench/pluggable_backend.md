# cuVS Bench Backends

This page explains how cuVS Bench separates benchmark orchestration from the system that actually builds and searches an index. Use it when you want to understand the built-in C++ benchmark backend, add a new backend for another product or service, or add a new indexing algorithm to the existing C++ backend.

cuVS Bench uses two pieces for each backend:

| Piece | Purpose |
| --- | --- |
| Config loader | Reads user inputs and configuration files, expands parameter sweeps, and returns the datasets and index configurations to run. |
| Backend | Uses those configurations to build indexes, run searches, and return benchmark results. |

Both pieces are registered under the same backend type name. The default backend type is `cpp_gbench`, which runs the C++ Google Benchmark executables.

```python
from cuvs_bench.orchestrator import BenchmarkOrchestrator

orchestrator = BenchmarkOrchestrator(backend_type="cpp_gbench")
results = orchestrator.run_benchmark(
    dataset="deep-image-96-inner",
    algorithms="cuvs_cagra",
    count=10,
    batch_size=10,
    build=True,
    search=True,
)
```

## How a benchmark run works

1. The user calls `BenchmarkOrchestrator(...).run_benchmark(...)`.
2. The orchestrator finds the config loader registered for the requested backend type.
3. The config loader returns a `DatasetConfig` and one or more `BenchmarkConfig` objects.
4. The orchestrator creates the backend registered for the same backend type.
5. The backend runs `build(...)` and `search(...)`, then returns a `BuildResult`
   and a list of `SearchResult` objects.

The config loader decides what to run. The backend decides how to run it.

## Configuration contract

A config loader receives the arguments passed to `run_benchmark()`, such as `dataset`, `dataset_path`, `algorithms`, `count`, `batch_size`, `groups`, and backend-specific options. It returns:

| Return value | Purpose |
| --- | --- |
| `DatasetConfig` | Dataset metadata, including vector files, ground-truth files, distance metric, dimensions, and optional subset size. |
| `List[BenchmarkConfig]` | One or more benchmark configurations. Each contains index configurations and backend-specific options. |

Each `IndexConfig` describes one index to benchmark:

| Field | Purpose |
| --- | --- |
| `name` | Human-readable index name, usually including parameter values. |
| `algo` | Algorithm name. |
| `build_param` | Build parameters for the index. |
| `search_params` | Search parameter combinations to benchmark. |
| `file` | Path or identifier where the backend stores the index. |

`ConfigLoader.load()` owns dataset-configuration loading and parameter expansion. A backend
loader implements the two template hooks below. This minimal network loader
creates one index configuration for each requested algorithm and group:

```python
from pathlib import Path

from cuvs_bench.orchestrator.config_loaders import (
    BenchmarkConfig,
    ConfigLoader,
    IndexConfig,
)

class MyConfigLoader(ConfigLoader):
    def __init__(self, config_path=None):
        self.config_path = str(
            Path(config_path) if config_path else Path(__file__).parent / "config"
        )

    @property
    def backend_type(self) -> str:
        return "my_backend"

    def _discover_algo_groups(
        self, dataset_conf, dataset, dataset_path, **kwargs
    ):
        algorithms = (kwargs.get("algorithms") or "my_algorithm").split(",")
        groups = (kwargs.get("groups") or "default").split(",")
        group_config = {
            "build": {},
            "search": {"ef_search": [100]},
        }
        return [
            (algorithm.strip(), group.strip(), group_config, {})
            for algorithm in algorithms
            for group in groups
        ]

    def _build_benchmark_configs(
        self,
        dataset_config,
        dataset_conf,
        dataset,
        dataset_path,
        expanded_groups,
        **kwargs,
    ):
        configurations = []
        for (
            algorithm,
            group,
            _group_config,
            build_combinations,
            search_combinations,
            _group_metadata,
        ) in expanded_groups:
            indexes = [
                IndexConfig(
                    name=f"{algorithm}.{group}",
                    algo=algorithm,
                    build_param=build_parameters,
                    search_params=search_combinations,
                    file=str(
                        Path(dataset_path)
                        / dataset
                        / "index"
                        / f"{algorithm}.{group}"
                    ),
                )
                for build_parameters in build_combinations
            ]
            configurations.append(
                BenchmarkConfig(
                    indexes=indexes,
                    backend_config={
                        "host": kwargs.get("host", "localhost"),
                        "port": kwargs.get("port", 9200),
                        "algo": algorithm,
                    },
                )
            )
        return configurations
```

## Adding a backend

Add a new backend when cuVS Bench needs to drive a different execution path, such as a vector database, remote service, or custom benchmark runner.

1. Implement a config loader by subclassing `ConfigLoader` and defining
   `_discover_algo_groups()`, `_build_benchmark_configs()`, and `backend_type`.
   The inherited `load()` method returns
   `(DatasetConfig, List[BenchmarkConfig])`.
2. Implement a backend by subclassing `BenchmarkBackend`. Its `build()` method
   returns `BuildResult`; its `search()` method returns `List[SearchResult]`.
3. Register both pieces with the same backend type name.
4. Select an installed backend with a YAML `backend:` field passed through
   `--backend-config`, or construct `BenchmarkOrchestrator` directly with its
   `backend_type`.

The registration example below uses an idempotent function to register both pieces.

## Example: network backend

This is interface scaffolding, not a working benchmark. The zero-valued results
below are fabricated placeholders: replace them with real service calls and
measurements before use. The loader also needs a packaged
`config/datasets/datasets.yaml` describing the datasets it accepts.

```python
import numpy as np
from cuvs_bench.backends.base import (
    BenchmarkBackend,
    BuildResult,
    SearchResult,
)

class MyBackend(BenchmarkBackend):
    default_algorithm = "my_algorithm"

    @property
    def algo(self) -> str:
        return self.config.get("algo", "my_algorithm")

    def build(self, dataset, indexes, force=False, dry_run=False):
        return BuildResult(
            index_path=indexes[0].file if indexes else "",
            build_time_seconds=0.0,
            index_size_bytes=0,
            algorithm=self.algo,
            build_params=indexes[0].build_param if indexes else {},
            metadata={},
            success=True,
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
        n_queries = dataset.n_queries
        return [
            SearchResult(
                neighbors=np.zeros((n_queries, k), dtype=np.int64),
                distances=np.zeros((n_queries, k), dtype=np.float32),
                search_time_ms=0.0,
                queries_per_second=0.0,
                recall=0.0,
                algorithm=self.algo,
                search_params=indexes[0].search_params if indexes else [],
                success=True,
            )
        ]
```

```python
from cuvs_bench.backends.registry import (
    get_registry,
    list_config_loaders,
    register_backend,
    register_config_loader,
)

def register():
    registry = get_registry()
    if not registry.is_registered("my_backend"):
        register_backend("my_backend", MyBackend)
    if "my_backend" not in list_config_loaders():
        register_config_loader("my_backend", MyConfigLoader)
```

Installed plugins publish the same idempotent registration function in both
entry-point groups:

```toml
[project.entry-points."cuvs_bench.backends"]
my_backend = "my_package.backend:register"

[project.entry-points."cuvs_bench.config_loaders"]
my_backend = "my_package.backend:register"
```

Select the installed backend with a small YAML file:

```yaml
backend: my_backend
host: localhost
port: 9200
```

```bash
python -m cuvs_bench.run --backend-config my-backend.yaml
```

`BenchmarkBackend.default_algorithm` supplies the CLI's algorithm default for
the selected backend. The default value is `None`, which retains the cuVS Bench
CLI default. A backend may also override `result_failure_message(results)` to
return a message when failed result objects should make the CLI exit nonzero
after export processing. Unsuccessful, skipped, and dry-run results are
excluded from the CSV output and do not replace prior measurements. Returning
`None` keeps the default nonfatal policy.

## Components at a glance

| Component | Description |
| --- | --- |
| `ConfigLoader` | Template base class whose `load(**kwargs)` method returns `(DatasetConfig, List[BenchmarkConfig])`; plugins implement its discovery/build hooks. Register with `register_config_loader(backend_type, loader_class)`. |
| `BenchmarkBackend` | Abstract class whose `build(...)` method returns `BuildResult` and whose `search(...)` method returns `List[SearchResult]`. Its optional class hooks control the CLI algorithm default and fatal-result policy. |
| `BackendRegistry` | Singleton registry returned by `get_registry()` that stores registered backend classes. The `get_backend_class()` and `get_config_loader()` helpers discover matching installed entry points on demand. |

## C++ Backend

The built-in `CppGoogleBenchmarkBackend` uses `backend_type="cpp_gbench"`. Its config loader reads YAML under `config/datasets` and `config/algos`, expands parameter combinations, and validates constraints. Its backend runs the C++ benchmark executables and merges their results.

Adding a new C++ algorithm usually means adding another executable and YAML config for this backend. It does not require a new backend type.

### Implementation and configuration

New algorithms should be C++ classes that inherit `class ANN` from `cpp/bench/ann/src/ann.h` and implement all pure virtual functions.

Define separate build and search parameter structs. The search parameter struct should inherit `struct ANN<T>::AnnSearchParam`.

```c++
template<typename T>
class HnswLib : public ANN<T> {
public:
  struct BuildParam {
    int M;
    int ef_construction;
    int num_threads;
  };

  using typename ANN<T>::AnnSearchParam;
  struct SearchParam : public AnnSearchParam {
    int ef;
    int num_threads;
  };

  // ...
};
```

The benchmark program consumes generated JSON files for indexes, build parameters, and search parameters. The JSON objects map to YAML `build_param` objects and `search_param` arrays.

```json
{
  "name": "hnswlib.M12.ef500.th32",
  "algo": "hnswlib",
  "build_param": {"M": 12, "efConstruction": 500, "numThreads": 32},
  "file": "/path/to/file",
  "search_params": [
    {"ef": 10, "numThreads": 1},
    {"ef": 20, "numThreads": 1},
    {"ef": 40, "numThreads": 1}
  ],
  "search_result_file": "/path/to/file"
}
```

Parse build and search parameters from JSON:

```c++
template<typename T>
void parse_build_param(const nlohmann::json& conf,
                       typename cuann::HnswLib<T>::BuildParam& param) {
  param.ef_construction = conf.at("efConstruction");
  param.M = conf.at("M");
  if (conf.contains("numThreads")) {
    param.num_threads = conf.at("numThreads");
  }
}

template<typename T>
void parse_search_param(const nlohmann::json& conf,
                        typename cuann::HnswLib<T>::SearchParam& param) {
  param.ef = conf.at("ef");
  if (conf.contains("numThreads")) {
    param.num_threads = conf.at("numThreads");
  }
}
```

Add matching `if` cases to `create_algo()` and `create_search_param()` in `cpp/bench/ann/`. The string literal must match the `algo` value in the configuration file.

```c++
if (algo == "hnswlib") {
   // ...
}
```

### Adding a CMake target

`cuvs/cpp/bench/ann/CMakeLists.txt` provides a CMake helper for new benchmark targets:

```cmake
ConfigureAnnBench(
  NAME <algo_name>
  PATH </path/to/algo/benchmark/source/file>
  INCLUDES <additional_include_directories>
  CXXFLAGS <additional_cxx_flags>
  LINKS <additional_link_library_targets>
)
```

Example target for `HNSWLIB`:

```cmake
ConfigureAnnBench(
  NAME HNSWLIB PATH bench/ann/src/hnswlib/hnswlib_benchmark.cpp INCLUDES
  ${CMAKE_CURRENT_BINARY_DIR}/_deps/hnswlib-src/hnswlib CXXFLAGS "${HNSW_CXX_FLAGS}"
)
```

This creates `HNSWLIB_ANN_BENCH`, which runs `HNSWLIB` benchmarks.

Add an `algos.yaml` entry that maps the algorithm name to its executable and declares whether the algorithm requires a GPU:

```yaml
cuvs_ivf_pq:
  executable: CUVS_IVF_PQ_ANN_BENCH
  requires_gpu: true
```

`executable` specifies the binary used to build and search the index. cuVS Bench expects it to be available in `cuvs/cpp/build/`. `requires_gpu` tells cuVS Bench whether the algorithm must run on a GPU node.

## Summary

cuVS Bench backends let the same benchmark workflow run against different execution targets. A config loader describes the dataset and parameter combinations, while a backend performs the build and search work. Use a new backend type for a new execution environment, and use the existing C++ backend when you are only adding another C++ ANN benchmark executable.
