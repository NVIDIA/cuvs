# Usage

This page shows how to configure algorithms, run cuVS Bench from the command line, use Docker containers, and read the build and search results. The command-line runner uses the cuVS Bench orchestrator and the default `cpp_gbench` backend, so the end-to-end workflow can stay fully CLI-based. For dataset formats, built-in datasets, custom dataset descriptors, and ground-truth generation, see [Benchmark Datasets](/user-guide/benchmarking-guide/cu-vs-bench-tool/datasets).

## Creating and customizing algorithm configurations

Algorithm YAML files define the build and search parameter sweeps for each algorithm. Dataset YAML files are covered in [Benchmark Datasets](/user-guide/benchmarking-guide/cu-vs-bench-tool/datasets#dataset-configurations).

Algorithm configs live in `${CUVS_HOME}/python/cuvs_bench/cuvs_bench/config/algos`. A `cuvs_cagra` config looks like this:

```yaml
name: cuvs_cagra
constraints:
  build: cuvs_bench.config.algos.constraints.cuvs_cagra_build
  search: cuvs_bench.config.algos.constraints.cuvs_cagra_search
groups:
  base:
    build:
      graph_degree: [32, 64, 96, 128]
      intermediate_graph_degree: [32, 64, 96, 128]
      graph_build_algo: ["NN_DESCENT"]
    search:
      itopk: [32, 64, 128, 256, 512]
      search_width: [1, 2, 4, 8, 16, 32, 64]

  test:
    build:
      graph_degree: [32]
      intermediate_graph_degree: [32]
      graph_build_algo: ["NN_DESCENT"]
    search:
      itopk: [32]
      search_width: [1, 2]

  persistent:
    build:
      graph_degree: [32, 64, 96]
      intermediate_graph_degree: [128]
      graph_build_algo: ["IVF_PQ"]
    search:
      persistent: [true]
      persistent_device_usage: [0.95]
      algo: ["single_cta"]
      itopk: [32, 64, 128, 256, 512]
      max_iterations: [0, 16]
      search_width: [1, 2, 4, 8]
```

Each config has three main fields:

| Field | Purpose |
| --- | --- |
| `name` | Algorithm name. |
| `constraints` | Optional Python functions that reject invalid build or search parameter combinations. |
| `groups` | Named parameter sweeps. Each group expands the cross product of its `build` and `search` values. |

Create a custom YAML file with a `base` group to override the default benchmark parameters. For parameter guidance, see the [ANN Algorithm Parameter Tuning Guide](/user-guide/benchmarking-guide/cu-vs-bench-tool/cu-vs-bench-parameter-tuning-guide).

| Library | Algorithms |
| --- | --- |
| FAISS_GPU | `faiss_gpu_flat`, `faiss_gpu_ivf_flat`, `faiss_gpu_ivf_pq`, `faiss_gpu_cagra` |
| FAISS_CPU | `faiss_cpu_flat`, `faiss_cpu_ivf_flat`, `faiss_cpu_ivf_pq`, `faiss_cpu_hnsw_flat` |
| GGNN | `ggnn` |
| HNSWLIB | `hnswlib` |
| DiskANN | `diskann_memory`, `diskann_ssd` |
| NVIDIA cuVS | `cuvs_brute_force`, `cuvs_cagra`, `cuvs_ivf_flat`, `cuvs_ivf_pq`, `cuvs_cagra_hnswlib`, `cuvs_vamana` |

### Multi-GPU algorithms

cuVS Bench includes single-node multi-GPU versions of IVF-Flat, IVF-PQ, and CAGRA.

| Index type | Multi-GPU algo name |
| --- | --- |
| IVF-Flat | `cuvs_mg_ivf_flat` |
| IVF-PQ | `cuvs_mg_ivf_pq` |
| CAGRA | `cuvs_mg_cagra` |

## Smaller-scale benchmarks (&lt;1M to 10M vectors)

Use `cuvs_bench.get_dataset` to prepare a built-in dataset. By default, datasets are stored under `RAPIDS_DATASET_ROOT_DIR` when that environment variable is set, or under a local `datasets` directory otherwise.

```bash
# (1) Prepare dataset.
python -m cuvs_bench.get_dataset --dataset deep-image-96-angular --normalize

# (2) Build and search index.
python -m cuvs_bench.run --dataset deep-image-96-inner --algorithms cuvs_cagra --batch-size 10 -k 10 --build --search

# (3) Export data.
python -m cuvs_bench.run --data-export --dataset deep-image-96-inner

# (4) Plot results.
python -m cuvs_bench.plot --dataset deep-image-96-inner
```

For available built-in datasets and ground-truth details, see [Benchmark Datasets](/user-guide/benchmarking-guide/cu-vs-bench-tool/datasets#built-in-datasets).

## Large-scale benchmarks (>10M vectors)

`cuvs_bench.get_dataset` does not download billion-scale datasets. Prepare large datasets first using [Benchmark Datasets](/user-guide/benchmarking-guide/cu-vs-bench-tool/datasets#dataset-sources), then run the same build, search, export, and plot workflow. Datasets at this scale are best suited to large-memory GPUs such as A100 or H100.

The following example prepares ground truth for the Yandex Deep-1B dataset and then runs benchmarks on a 100M-vector subset:

```bash
mkdir -p datasets/deep-1B

# (1) Prepare dataset.
# Download the Yandex DEEP "Ground Truth" file manually.
# Suppose the file name is deep_new_groundtruth.public.10K.bin.
python -m cuvs_bench.split_groundtruth --groundtruth datasets/deep-1B/deep_new_groundtruth.public.10K.bin

# The split step creates groundtruth.neighbors.ibin and groundtruth.distances.fbin.
```

```bash
# (2) Build and search index.
python -m cuvs_bench.run --dataset deep-1B --algorithms cuvs_cagra --batch-size 10 -k 10 --build --search

# (3) Export data.
python -m cuvs_bench.run --data-export --dataset deep-1B

# (4) Plot results.
python -m cuvs_bench.plot --dataset deep-1B
```

Use `python -m cuvs_bench.split_groundtruth --help` to see the full CLI help for the ground-truth split command.

## Running with Docker containers

Docker images can run the full workflow or open a shell for manual commands. See [Installation](/user-guide/benchmarking-guide/cu-vs-bench-tool/installation#docker) for image and tag guidance.

### End-to-end run on GPU

Set `DATA_FOLDER` to a local directory. Datasets are stored in `$DATA_FOLDER/datasets` and results in `$DATA_FOLDER/result`.

```bash
export DATA_FOLDER=path/to/store/datasets/and/results
docker run --gpus all --rm -it -u $(id -u)                      \
    -v $DATA_FOLDER:/data/benchmarks                            \
    rapidsai/cuvs-bench:26.06a-cuda12-py3.13              \
    "--dataset deep-image-96-angular"                           \
    "--normalize"                                               \
    "--algorithms cuvs_cagra,cuvs_ivf_pq --batch-size 10 -k 10" \
    ""
```

| Argument | Description |
| --- | --- |
| `rapidsai/cuvs-bench:26.06a-cuda12-py3.13` | Image to use. See [Installation](/user-guide/benchmarking-guide/cu-vs-bench-tool/installation#docker) for available tags. |
| `"--dataset deep-image-96-angular"` | Dataset name. |
| `"--normalize"` | Normalizes the dataset before benchmarking. |
| `"--algorithms cuvs_cagra,hnswlib --batch-size 10 -k 10"` | Arguments passed to the benchmark run script. |
| `""` | Optional arguments passed to the plot script. |

The `-u $(id -u)` flag lets the container user match the host user, so files written to the mounted volume keep usable permissions.

### End-to-end run on CPU

Use the CPU image and omit `--gpus all` on systems without a GPU.

```bash
export DATA_FOLDER=path/to/store/datasets/and/results
docker run  --rm -it -u $(id -u)                  \
    -v $DATA_FOLDER:/data/benchmarks              \
    rapidsai/cuvs-bench-cpu:26.06a-py3.13     \
     "--dataset deep-image-96-angular"            \
     "--normalize"                                \
     "--algorithms hnswlib --batch-size 10 -k 10" \
     ""
```

### Manual container workflow

All `cuvs-bench` images include the Conda packages. Start a shell when you want to run individual commands yourself:

```bash
export DATA_FOLDER=path/to/store/datasets/and/results
docker run --gpus all --rm -it -u $(id -u)          \
    --entrypoint /bin/bash                          \
    --workdir /data/benchmarks                      \
    -v $DATA_FOLDER:/data/benchmarks                \
    rapidsai/cuvs-bench:26.06a-cuda12-py3.13
```

Inside the container, run the same Python modules directly:

```bash
(base) root@00b068fbb862:/data/benchmarks# python -m cuvs_bench.get_dataset --dataset deep-image-96-angular --normalize
```

Containers can also run in detached mode.

## Evaluating results

Build benchmarks report:

| Name | Description |
| --- | --- |
| Benchmark | Name that identifies the benchmark instance. |
| Time | Wall time spent training the index. |
| CPU | CPU time spent training the index. |
| Iterations | Number of iterations, usually 1. |
| GPU | GPU time spent building. |
| index_size | Number of vectors used to train the index. |

Search benchmarks report:

| Name | Description |
| --- | --- |
| Benchmark | Name that identifies the benchmark instance. |
| Time | Wall time for a single batch divided by the number of threads. |
| CPU | Average CPU time, not including idle time while waiting for GPU synchronization. |
| Iterations | Total number of batches, equal to `total_queries / n_queries`. |
| GPU | GPU latency for a single batch. In throughput mode, this is averaged across threads. |
| Latency | Batch latency from wall-clock time. In throughput mode, this is averaged across threads. |
| Recall | Fraction of returned neighbors that match ground truth. Present only when ground truth is configured. |
| items_per_second | Total throughput, or queries per second, approximately `total_queries / end_to_end`. |
| k | Number of neighbors requested per query. |
| end_to_end | Total time to run all batches across all iterations. |
| n_queries | Number of query vectors in each batch. |
| total_queries | Total query count, equal to `iterations * n_queries`. |

`Time` and `end_to_end` are measured differently, so `end_to_end = Time * Iterations` is only approximate. Output tables may also include the hyper-parameters for each benchmarked configuration. Recall can fluctuate when fewer queries are processed than the benchmark contains, because processed query count depends on iteration count.

## Summary

cuVS Bench usage has three main steps: configure datasets and algorithm sweeps, run build and search through Python or Docker, and compare the reported build and search measurements. Start with built-in datasets for smaller tests, prepare large datasets separately for scale testing, and use the result tables to compare quality, latency, throughput, build time, and resource behavior across parameter settings.
