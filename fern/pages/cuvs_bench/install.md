# Install cuVS Bench

Use these instructions when you want to run cuVS Bench with pre-built packages or containers. Conda is usually the simplest option for local Python workflows, while Docker is useful when you want a reproducible container image with the benchmark environment already included.

There are two main ways pre-compiled benchmarks are distributed:

- [Conda](#conda): best when you want a Python package without containers. Pip wheels are planned for users who cannot use conda.
- [Docker](#docker): best when you want a containerized workflow. It needs Docker and, for GPU runs, [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker).

## Conda

```bash
conda create --name cuvs_benchmarks
conda activate cuvs_benchmarks

# to install GPU package:
conda install -c rapidsai -c conda-forge cuvs-bench=<rapids_version> 'cuda-version=13.3.*'

# to install CPU package for usage in CPU-only systems:
conda install -c rapidsai -c conda-forge  cuvs-bench-cpu
```

Use `rapidsai-nightly` instead of `rapidsai` for nightly benchmarks. The CPU package currently supports HNSW benchmarks.

## Docker

Images are available for GPU and CPU-only systems:

- `cuvs-bench`: includes GPU and CPU benchmarks, supports all algorithms, downloads million-scale datasets as needed, and requires the NVIDIA Container Toolkit for GPU algorithms.
- `cuvs-bench-cpu`: includes only CPU benchmarks and is the smallest image for systems without GPUs.

Nightly images are located on [Docker Hub](https://hub.docker.com/r/rapidsai/cuvs-bench/tags).

The following command pulls the nightly container for Python version 3.13, CUDA version 12.9, and NVIDIA cuVS version 26.06:

```bash
docker pull rapidsai/cuvs-bench:26.06a-cuda12-py3.13 # substitute cuvs-bench for the exact desired container.
```

CUDA and Python versions can be changed to supported values:

- Supported CUDA versions: 12, 13
- Supported Python versions: 3.11, 3.12, 3.13, and 3.14

Exact tags are listed on Docker Hub:

- [NVIDIA cuVS bench images](https://hub.docker.com/r/rapidsai/cuvs-bench/tags)
- [NVIDIA cuVS bench CPU only images](https://hub.docker.com/r/rapidsai/cuvs-bench-cpu/tags)

**Note:** GPU containers use the CUDA toolkit inside the container. The host only needs a compatible driver, so CUDA 12 containers can run on systems with CUDA 13.x-capable drivers. GPU access also requires the NVIDIA Docker runtime from the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker).

## Build from Source

Build cuVS Bench from source when you need local benchmark executables that match a development checkout, include custom algorithm targets, or use dependencies that are not available in the pre-built packages.

### Dependencies

CUDA 12+ and a GPU with Volta architecture or later are required to run the benchmarks.

Please refer to the  [installation docs](/installation) for the base requirements to build NVIDIA cuVS.

In addition to the base requirements for building NVIDIA cuVS, additional dependencies needed to build the ANN benchmarks include:

1. FAISS GPU >= 1.7.1
2. Google Logging (GLog)
3. H5Py
4. HNSWLib
5. nlohmann_json
6. GGNN

[rapids-cmake](https://github.com/rapidsai/rapids-cmake) is used to build the ANN benchmarks so the code for dependencies not already supplied in the CUDA toolkit will be downloaded and built automatically.

The easiest and most reproducible way to install the dependencies needed to build the ANN benchmarks is to use the conda environment file located in the `conda/environments` directory of the NVIDIA cuVS repository. The following command will use `mamba` to build and activate a new environment for compiling the benchmarks:

```bash
conda env create --name cuvs_benchmarks -f conda/environments/bench_ann_cuda-133_arch-$(uname -m).yaml
conda activate cuvs_benchmarks
```

The above conda environment will also reduce the compile times as dependencies like FAISS will already be installed and not need to be compiled with `rapids-cmake`.

### Compiling the Benchmarks

After the needed dependencies are satisfied, the easiest way to compile ANN benchmarks is through the `build.sh` script in the root of the NVIDIA cuVS source code repository. The following will build the executables for all the supported algorithms:

```bash
./build.sh bench-ann
```

You can limit the algorithms that are built by providing a semicolon-delimited list of executable names. Each algorithm is suffixed with `_ANN_BENCH`:

```bash
./build.sh bench-ann -n --limit-bench-ann=HNSWLIB_ANN_BENCH;CUVS_IVF_PQ_ANN_BENCH
```

Available targets to use with `--limit-bench-ann` are:

- FAISS_GPU_IVF_FLAT_ANN_BENCH
- FAISS_GPU_IVF_PQ_ANN_BENCH
- FAISS_CPU_IVF_FLAT_ANN_BENCH
- FAISS_CPU_IVF_PQ_ANN_BENCH
- FAISS_GPU_FLAT_ANN_BENCH
- FAISS_CPU_FLAT_ANN_BENCH
- GGNN_ANN_BENCH
- HNSWLIB_ANN_BENCH
- CUVS_CAGRA_ANN_BENCH
- CUVS_IVF_PQ_ANN_BENCH
- CUVS_IVF_FLAT_ANN_BENCH

By default, the `*_ANN_BENCH` executables infer the dataset datatype from the filename extension. For example, an extension of `fbin` uses a `float` datatype, `f16bin` uses a `float16` datatype, `i8bin` uses an `int8_t` datatype, and `u8bin` uses a `uint8_t` type. Currently, only `float`, `float16`, `int8_t`, and `uint8_t` are supported.
