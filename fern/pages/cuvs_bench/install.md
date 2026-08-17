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

## PyLucene backend prerequisites

The optional `pylucene` backend requires components that cuVS Bench does not install automatically:

- For GPU indexing or search, an NVIDIA GPU supported by cuVS plus matching CUDA and cuVS native libraries. The accelerated HNSW codec intentionally falls back to Lucene's CPU writer when cuVS is unavailable; the CAGRA codec requires cuVS GPU support.
- JDK 22, including `javac`, and a custom PyLucene wrapper generated against Lucene 10.2.0. cuVS Bench compiles its small PyLucene codec adapter before starting the JVM. Apache does not publish PyLucene 10.2.0, and the official PyLucene 10.0.0 distribution is incompatible with the Lucene 10.2 APIs used here. The [PyLucene source-build instructions](https://lucene.apache.org/pylucene/install.html) describe the general build mechanics, but neither Apache nor cuVS currently provides a ready-to-use 10.2.0 wrapper.
- [Maven 3.9.6 or newer](https://maven.apache.org/download.cgi) to build cuVS-Lucene.
- The base `cuvs-java` JAR, standard `cuvs-lucene` JAR, and, for GPU execution, native libraries from the same cuVS version. The cuVS-Lucene PR declares cuVS Java 26.10.0.

cuVS-Lucene now lives in this repository under `java/cuvs-lucene`. [NVIDIA/cuvs-lucene#174](https://github.com/NVIDIA/cuvs-lucene/pull/174), which predates that move, contains the remaining PyLucene 10.2 compatibility changes and its Python end-to-end coverage. Its standalone branch does not include the newer HNSW heuristic delegation from [NVIDIA/cuvs-lucene#177](https://github.com/NVIDIA/cuvs-lucene/pull/177), which is already present in cuVS and is required for the `m` and `ef_construction` benchmark parameters. Until the PyLucene changes are ported and merged, use the validated source combination below; neither source by itself is sufficient.

Build the dependencies in separate checkouts so this does not change your cuVS Bench working tree. While the cuVS-Lucene PR is under review, record the exact cuVS and PR revisions used for a reproducible environment and keep their cuVS Java versions aligned.

```bash
git clone https://github.com/NVIDIA/cuvs.git cuvs-pylucene-deps
cd cuvs-pylucene-deps
git switch --detach be8ab314d044aee0f80fbe2c2277a893561288de
./build.sh libcuvs java
cd ..
```

If matching native cuVS libraries are already built and installed, `./build.sh java` is sufficient. The Java build installs the base and native-classifier JARs into the local Maven repository; see the [cuVS Java build guide](https://github.com/NVIDIA/cuvs/blob/main/java/README.md).

```bash
git clone https://github.com/NVIDIA/cuvs-lucene.git cuvs-lucene-pylucene
cd cuvs-lucene-pylucene
git fetch origin pull/174/head
git switch --detach 6fe2c2824408a4ff2ac8f201df05308fd2404b76
git -C ../cuvs-pylucene-deps show \
    --relative=java/cuvs-lucene --format=email --binary \
    65b4ae5f1ac5b5916d49709f6c16231df8d26188 \
    -- java/cuvs-lucene | git apply
mvn clean package -DskipTests
```

After the build, the conventional JAR paths are:

```text
~/.m2/repository/com/nvidia/cuvs/cuvs-java/<version>/cuvs-java-<version>.jar
<cuvs-lucene-pylucene-checkout>/target/cuvs-lucene-<version>.jar
```

Use the base `cuvs-java` JAR, not a native-classifier JAR. Use the standard cuVS-Lucene JAR, not its `-jar-with-dependencies`, sources, or Javadoc variants. Native-library paths must resolve `libcuvs.so`, `libcuvs_c.so`, their dependencies, and the CUDA runtime libraries from the matching cuVS build.

Use a clean environment without another cuVS native installation on its library path; otherwise, the JVM can load the other `libcuvs_c.so` first and reject the Java/native version mismatch.

The backend checks that `lucene.VERSION` is exactly `10.2.0` before starting the process-wide JVM. It also compiles its configured-codec adapter against the selected JAR, which fails early when the HNSW heuristic API is missing. Then validate the combined artifacts from the temporary cuVS-Lucene checkout. The upstream PyLucene suite is a pytest module; its Java test adapter is compiled into `target/test-classes` by the Maven build and is not part of the production JAR.

```bash
python -m pip install pytest

CUVS_NATIVE_BUILD="$(cd ../cuvs-pylucene-deps/cpp/build && pwd)"
export JAVA_LIBRARY_PATH="$CUVS_NATIVE_BUILD:$CUVS_NATIVE_BUILD/c:/usr/local/cuda/lib64"
export LD_LIBRARY_PATH="$JAVA_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python -m pytest -q -s src/test/python/test_pylucene_end_to_end.py
```

See [Running the PyLucene backend](/user-guide/benchmarking-guide/cu-vs-bench-tool/usage#running-the-pylucene-backend) for a smoke workflow after these prerequisites are prepared.

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
