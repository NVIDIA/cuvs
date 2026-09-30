# cuVS Lucene

This is a project for using [cuVS](https://github.com/NVIDIA/cuvs), NVIDIA's GPU accelerated vector search library, with [Apache Lucene](https://github.com/apache/lucene).

## Contents

1. [What is cuvs-lucene?](#what-is-cuvs-lucene)
2. [Installing cuvs-lucene](#installing-cuvs-lucene)
3. [Getting Started](#getting-started)
4. [Contributing](#contributing)
5. [Supporting several Lucene releases](#supporting-several-lucene-releases)
6. [References](#references)

## What is cuvs-lucene?

`cuvs-lucene` provides a pluggable [KnnVectorsFormat](https://lucene.apache.org/core/10_5_0/core/org/apache/lucene/codecs/KnnVectorsFormat.html) that uses cuVS to offload vector index build — and optionally search — to NVIDIA GPUs. Because it plugs in through a standard Lucene codec, existing Lucene applications can take advantage of GPU acceleration with minimal code changes. It supports Lucene 10.2 through 10.5.

Four codecs are currently provided, created through `CuVSCodecs`:

- `CuVSCodecs.acceleratedHNSW()` — GPU-accelerated HNSW build with CPU HNSW search. The on-disk format is standard Lucene HNSW, so indexes built on the GPU are read back through the standard Lucene reader and searching them needs no GPU. Lucene still resolves the codec by name, so search nodes need the `cuvs-lucene` and `cuvs-java` jars on their classpath. Falls back to building the graph on the CPU when no GPU is present.
  - `CuVSCodecs.acceleratedHNSWScalarQuantized()` — scalar-quantized vectors for a smaller index footprint.
  - `CuVSCodecs.acceleratedHNSWBinaryQuantized()` — graph built over binary-quantized vectors.
- `CuVSCodecs.gpuSearch()` — GPU-accelerated build and GPU search. Requires a GPU.

Each codec wraps the default codec of one Lucene release, so the codec that writes changes with Lucene: for
example `Lucene101AcceleratedHNSWCodec` on Lucene 10.2, `Lucene103AcceleratedHNSWCodec` on 10.3, and
`Lucene104AcceleratedHNSWCodec` on 10.4 and 10.5. Codecs of earlier releases stay available so that existing
indexes can still be read after an upgrade, but they refuse to write. `CuVSCodecs` always returns the codecs
that write on the Lucene release in use.

For guidance on choosing between them, configuring builds, and tuning GPU resources, see the
[Lucene Integration](https://docs.nvidia.com/cuvs/user-guide/lucene) guide.

## Installing cuvs-lucene

### Prerequisites

- An NVIDIA GPU, to use the GPU-accelerated paths (the accelerated HNSW codecs fall back to CPU index construction without one; `CuVS2510GPUSearchCodec` requires a GPU)
- [CUDA Toolkit 12.2+](https://developer.nvidia.com/cuda-toolkit-archive) and an Ampere architecture GPU or newer, matching the [cuVS requirements](https://docs.nvidia.com/cuvs/installation)
- JDK 21 or newer. GPU acceleration needs a [JDK 22](https://jdk.java.net/archive/) or newer runtime; on JDK 21
  the accelerated HNSW codecs build graphs on the CPU, and the GPU search codec is unavailable
- [Maven 3.9.6+](https://maven.apache.org/download.cgi)
- A matching version of the [cuVS libraries](https://docs.nvidia.com/cuvs/installation/java). For Maven usage, install the cuVS libraries and add them to your system library load path.

### Maven

Lucene changes its codec APIs between minor releases, so there is one `cuvs-lucene` artifact per supported
Lucene minor release: `cuvs-lucene-10.2`, `cuvs-lucene-10.3`, `cuvs-lucene-10.4` and `cuvs-lucene-10.5`. Pick
the one matching the Lucene version your application uses. For Lucene 10.5, add the following dependency to
your `pom.xml`:

```xml
<dependency>
  <groupId>com.nvidia.cuvs.lucene</groupId>
  <artifactId>cuvs-lucene-10.5</artifactId>
  <version>26.12.0</version>
</dependency>
```

An artifact fails to load against another Lucene minor release, with an error that names the artifact to use.

### Building from source

`cuvs-lucene` lives in the [cuVS repository](https://github.com/NVIDIA/cuvs) and builds against the cuVS
Java bindings. If the libcuvs libraries and the Java bindings have not been built and installed, use
`./build.sh libcuvs java lucene` in the top level directory.

Alternatively, if libcuvs is already built and the `cuvs-java` artifact is already installed in your local
Maven repository, do `./build.sh lucene` in the top level directory or just do `./build.sh` in this directory.

This builds, and installs into your local Maven repository, every `cuvs-lucene-10.X` artifact. They are
written to `lucene-10.X/target/`.

To run the tests, add `--run-java-tests` to any of the commands above. Be sure to set (manually, if needed)
your `LD_LIBRARY_PATH` to include the directory with the appropriate (matching) version of `libcuvs.so`, as
described in the [cuVS installation instructions](https://docs.nvidia.com/cuvs/installation/java#cuvs-lucene).

## Getting Started

The [Lucene Integration](https://docs.nvidia.com/cuvs/user-guide/lucene) guide walks through plugging a
codec into a standard Lucene `IndexWriter`, searching on the GPU, tuning index builds, and managing GPU
resources in a long-lived application. Class-level documentation is in the
[Lucene API reference](https://docs.nvidia.com/cuvs/api-reference/lucene-api-documentation).

Runnable examples of CAGRA-accelerated HNSW indexing, and of indexing and searching entirely on the GPU, are
in the [`examples/`](../../examples/java/cuvs-lucene) directory.

## Contributing

If you are interested in contributing to cuvs-lucene, please read the cuVS [Contributing guide](https://docs.nvidia.com/cuvs/developer-guide/contributing).

> [!NOTE]
> The code style format is enforced using the [Spotless maven plugin](https://github.com/diffplug/spotless/tree/main/plugin-maven), which runs as a `pre-commit` hook. Run `pre-commit run --all-files`, or `mvn spotless:apply` in this directory, to format the sources.

## Supporting several Lucene releases

All `cuvs-lucene-10.X` artifacts are built from the same sources:

```
java/cuvs-lucene/
  pom.xml                 parent pom, lists one module per Lucene minor release
  src/main/java           shared code: everything that compiles against every supported Lucene release
  src/test/java           shared tests
  src/test/resources/backcompat   indexes written by each release, see below
  compat/<variant>/src    the adapter for one Lucene class as it looks from one release on, shared by
                          the releases that have it, such as knn-vectors-reader-102-bits and
                          knn-vectors-reader-103-acceptdocs; see compat/README.md
  lucene-10.X/
    pom.xml               the Lucene 10.X dependencies, and which compat/ variant of each API it uses
    src/main/java         LuceneCompat: the calls that differ on Lucene 10.X
    src/main/resources    the META-INF/services files: the codecs and formats Lucene 10.X can load
    src/since/java        code first needed on Lucene 10.X, such as the codecs wrapping its default
                          codec; compiled by this module and every later one
```

Each module compiles the shared sources against its own Lucene release, so the compiler checks every release.
The shared code never refers to a Lucene API that differs between releases. Instead it goes through:

- `LuceneCompat`, for calls whose signature or class location differs, such as the codecs of earlier releases,
  which move to Lucene's backward codecs.
- `CompatKnnVectorsReader`, `CompatKnnVectorsWriter` and `CompatKnnFloatVectorQuery`, which override the Lucene
  methods whose signatures changed and forward to a `do*` method with the same signature on every release.
  They live in `compat/`, one directory per variant, and each module picks its variants in its `pom.xml`.

Do not look up Lucene classes or constants by reflection. It hides API changes from the compiler, and constants
such as `Lucene99HnswVectorsFormat.VERSION_CURRENT` change meaning between releases.

`TestBackCompatIndices` reads indexes written by each module, stored as zips in `src/test/resources/backcompat`,
and checks that every later module can still read and search them. `TestBackCompat` checks that only the codecs
from `CuVSCodecs` write.

### Adding a Lucene release

[MAINTAINING.md](MAINTAINING.md) lists exactly what to change when Lucene publishes a new patch or minor
release, and how to drop support for one.

## References

- [Bring Massive-Scale Vector Search to the GPU with Apache Lucene](https://www.nvidia.com/en-us/on-demand/session/gtc25-S71286/) — NVIDIA GTC 2025 session video
- [cuVS and Lucene: GPU-based Vector Search](https://www.youtube.com/watch?v=qiW7iIDFJC0) — Berlin Buzzwords 2024 session video
- [Exploring GPU-accelerated vector search in Elasticsearch with NVIDIA](https://www.elastic.co/search-labs/blog/gpu-accelerated-vector-search-elasticsearch-nvidia) — Elasticsearch Blog
- [Apache Lucene Accelerated with the NVIDIA cuVS 25.06 Release](https://searchscale.com/blog/apache-lucene-accelerated-with-nvidia-cuvs-25.06-release/) — SearchScale Blog
