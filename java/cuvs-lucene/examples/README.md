# Examples

This maven project contains basic examples that showcase how `cuvs-lucene` can be used.

You can build the prerequisites either inside a RAPIDS conda container (the quickest way to try
things out, since it provides a prebuilt `libcuvs` without compiling the C/C++ libraries) or
entirely from source.

## Option A — RAPIDS container (no local libcuvs build)

### Prerequisites

- [Docker](https://www.docker.com/)
- [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- A machine with an Nvidia GPU

### Steps

Launch the container **from the repository root** (the directory containing `build.sh` and
`dependencies.yaml`) — not from this `examples/` directory — so the whole repo is mounted and
becomes the container's working directory:

```sh
docker run --rm --gpus all --pull=always --volume $PWD:$PWD --workdir $PWD -it rapidsai/ci-conda:26.10-cuda13.3.0-ubuntu24.04-py3.13
```

Inside the container you are now at the repository root. Create a conda environment with `libcuvs`
and the Java toolchain (this reads the repo's `dependencies.yaml`, so it must be run from the
repository root), then build `cuvs-java` and `cuvs-lucene` against it:

```sh
rapids-dependency-file-generator --output conda --file-key java \
  --matrix "cuda=13.3;arch=$(arch)" | tee /tmp/java_env.yaml
rapids-mamba-retry env create --yes -f /tmp/java_env.yaml -n java
conda activate java
export CMAKE_PREFIX_PATH=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
./build.sh java lucene
```

## Option B — build everything from source

### Prerequisites

- A local [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit-archive) and a machine with an Nvidia GPU
- [Maven 3.9.6+](https://maven.apache.org/download.cgi) and [JDK 22](https://jdk.java.net/archive/)

### Steps

From the repository root, build `libcuvs`, the `cuvs-java` bindings, and `cuvs-lucene` (see the
[top-level Java README](../../README.md)), then point `LD_LIBRARY_PATH` at the in-tree libraries:

```sh
./build.sh libcuvs java lucene
export LD_LIBRARY_PATH=$PWD/cpp/build:$LD_LIBRARY_PATH
```

## Build and run the examples

With either option done (and its `LD_LIBRARY_PATH`/conda env active), build the examples from this
directory:

```sh
cd java/cuvs-lucene/examples
mvn clean install
```

To run the Accelerated HNSW example do:

```sh
java -Djava.util.logging.config.file=src/main/resources/logging.properties -cp target/examples-26.10.0-jar-with-merged-services.jar com.nvidia.cuvs.lucene.examples.AcceleratedHnswExample
```

To run the Index and Search on GPU example do:

```sh
java -Djava.util.logging.config.file=src/main/resources/logging.properties -cp target/examples-26.10.0-jar-with-merged-services.jar com.nvidia.cuvs.lucene.examples.IndexAndSearchonGPUExample
```
