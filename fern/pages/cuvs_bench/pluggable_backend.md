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
5. The backend runs `build(...)` and `search(...)`, then returns `BuildResult` and `SearchResult` objects.

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

The following minimal loader creates one dataset, one index, and one search configuration:

```python
from cuvs_bench.orchestrator.config_loaders import (
    ConfigLoader,
    DatasetConfig,
    BenchmarkConfig,
    IndexConfig,
)

class MyConfigLoader(ConfigLoader):
    @property
    def backend_type(self) -> str:
        return "my_backend"

    def load(self, dataset, dataset_path, algorithms, count=10, batch_size=10000, **kwargs):
        dataset_config = DatasetConfig(
            name=dataset,
            base_file=...,
            query_file=...,
            groundtruth_neighbors_file=...,
            distance="euclidean",
            dims=128,
        )
        index = IndexConfig(
            name=f"{algorithms}.default",
            algo=algorithms,
            build_param={"nlist": 1024},
            search_params=[{"nprobe": 10}],
            file=...,
        )
        benchmark_config = BenchmarkConfig(
            indexes=[index],
            backend_config={
                "host": ...,
                "port": ...,
                "index_name": ...,
            },
        )
        return dataset_config, [benchmark_config]
```

## Adding a backend

Add a new backend when cuVS Bench needs to drive a different execution path, such as a vector database, remote service, or custom benchmark runner.

1. Implement a config loader by subclassing `ConfigLoader` from `cuvs_bench.orchestrator.config_loaders`. Its `load()` method should return `(DatasetConfig, List[BenchmarkConfig])`.
2. Implement a backend by subclassing `BenchmarkBackend` from `cuvs_bench.backends.base`. Its `build()` method should return `BuildResult`; its `search()` method should return `SearchResult`.
3. Register both pieces with the same backend type name.
4. Run benchmarks with `BenchmarkOrchestrator(backend_type="my_backend")`.

```python
from cuvs_bench.orchestrator import register_config_loader
from cuvs_bench.backends import get_registry

register_config_loader("my_backend", MyConfigLoader)
get_registry().register("my_backend", MyBackend)
```

## Example: Elasticsearch backend

This example shows the shape of a network backend. The loader creates the dataset and benchmark configs. The backend uses `backend_config` to connect to the service, build the index, run search, and return cuVS Bench result objects.

```python
from cuvs_bench.orchestrator.config_loaders import (
    ConfigLoader,
    DatasetConfig,
    BenchmarkConfig,
    IndexConfig,
)

class ElasticsearchConfigLoader(ConfigLoader):
    @property
    def backend_type(self) -> str:
        return "elasticsearch"

    def load(self, dataset, dataset_path, algorithms, count=10, batch_size=10000, **kwargs):
        dataset_config = DatasetConfig(
            name=dataset,
            base_file=...,
            query_file=...,
            groundtruth_neighbors_file=...,
            distance="euclidean",
            dims=kwargs.get("dims", 128),
        )
        index = IndexConfig(
            name=f"{algorithms}.es",
            algo=algorithms,
            build_param={},
            search_params=[{"ef_search": 100}],
            file=...,
        )
        benchmark_config = BenchmarkConfig(
            indexes=[index],
            backend_config={
                "host": ...,
                "port": ...,
                "index_name": ...,
                "algo": algorithms,
            },
        )
        return dataset_config, [benchmark_config]
```

```python
import numpy as np
from cuvs_bench.backends.base import (
    BenchmarkBackend,
    BuildResult,
    SearchResult,
)

class ElasticsearchBackend(BenchmarkBackend):
    @property
    def algo(self) -> str:
        return self.config.get("algo", "elasticsearch")

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
        return SearchResult(
            neighbors=np.zeros((n_queries, k), dtype=np.int64),
            distances=np.zeros((n_queries, k), dtype=np.float32),
            search_time_ms=0.0,
            queries_per_second=0.0,
            recall=0.0,
            algorithm=self.algo,
            search_params=indexes[0].search_params if indexes else [],
            success=True,
        )
```

```python
from cuvs_bench.orchestrator import register_config_loader
from cuvs_bench.backends import get_registry

register_config_loader("elasticsearch", ElasticsearchConfigLoader)
get_registry().register("elasticsearch", ElasticsearchBackend)
```

## Components at a glance

| Component | Description |
| --- | --- |
| `ConfigLoader` | Abstract class whose `load(**kwargs)` method returns `(DatasetConfig, List[BenchmarkConfig])`. Register with `register_config_loader(backend_type, loader_class)`. |
| `BenchmarkBackend` | Abstract class whose `build(...)` method returns `BuildResult` and whose `search(...)` method returns `SearchResult`. Register with `BackendRegistry.register(name, backend_class)`. |
| `BackendRegistry` | Singleton registry returned by `get_registry()`. It maps backend type names to backend classes. |

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
