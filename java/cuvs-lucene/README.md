# cuVS Lucene

This is a project for using [cuVS](https://github.com/NVIDIA/cuvs), NVIDIA's GPU accelerated vector search library, with [Apache Lucene](https://github.com/apache/lucene).

## Contents

1. [What is cuvs-lucene?](#what-is-cuvs-lucene)
2. [Installing cuvs-lucene](#installing-cuvs-lucene)
3. [Getting Started](#getting-started)
4. [Contributing](#contributing)
5. [References](#references)

## What is cuvs-lucene?

`cuvs-lucene` provides a pluggable [KnnVectorsFormat](https://lucene.apache.org/core/10_2_0/core/org/apache/lucene/codecs/KnnVectorsFormat.html) that uses cuVS to offload vector index build — and optionally search — to NVIDIA GPUs. Because it plugs in through a standard Lucene codec, existing Lucene applications can take advantage of GPU acceleration with minimal code changes. `Lucene101AcceleratedHNSWCodec` also falls back to the stock CPU codec when no GPU is present; the other three require a working cuVS installation.

Four codecs are currently provided:

- `Lucene101AcceleratedHNSWCodec` — GPU-accelerated HNSW build with CPU HNSW search. The on-disk format is standard Lucene HNSW, so indexes built on the GPU are read back through the standard Lucene 10.2 reader and searching them needs no GPU. Lucene still resolves the codec by name, so search nodes need the `cuvs-lucene` and `cuvs-java` jars on their classpath.
  - `LuceneAcceleratedHNSWScalarQuantizedCodec` — scalar-quantized vectors for a smaller index footprint.
  - `LuceneAcceleratedHNSWBinaryQuantizedCodec` — binary-quantized vectors for an even smaller index footprint.
- `CuVS2510GPUSearchCodec` — GPU-accelerated HNSW build and GPU search

For guidance on choosing between them, configuring builds, and tuning GPU resources, see the
[Lucene Integration](https://docs.nvidia.com/cuvs/user-guide/lucene) guide.

## Installing cuvs-lucene

### Prerequisites

- An NVIDIA GPU, to use the GPU-accelerated paths (the accelerated HNSW codecs fall back to CPU index construction without one; `CuVS2510GPUSearchCodec` requires a GPU)
- [CUDA Toolkit 12.2+](https://developer.nvidia.com/cuda-toolkit-archive) and an Ampere architecture GPU or newer, matching the [cuVS requirements](https://docs.nvidia.com/cuvs/installation)
- [JDK 22](https://jdk.java.net/archive/)
- [Maven 3.9.6+](https://maven.apache.org/download.cgi)
- A matching version of the [cuVS libraries](https://docs.nvidia.com/cuvs/installation/java). For Maven usage, install the cuVS libraries and add them to your system library load path.

### Maven

To pull `cuvs-lucene` into a Maven project, add the following dependency to your `pom.xml`:

```xml
<dependency>
  <groupId>com.nvidia.cuvs.lucene</groupId>
  <artifactId>cuvs-lucene</artifactId>
  <version>26.12.0</version>
</dependency>
```

### Building from source

`cuvs-lucene` lives in the [cuVS repository](https://github.com/NVIDIA/cuvs) and builds against the cuVS
Java bindings. If the libcuvs libraries and the Java bindings have not been built and installed, use
`./build.sh libcuvs java lucene` in the top level directory.

Alternatively, if libcuvs is already built and the `cuvs-java` artifact is already installed in your local
Maven repository, do `./build.sh lucene` in the top level directory or just do `./build.sh` in this directory.

The resulting artifacts are written to `target/`.

To run the tests, add `--run-java-tests` to any of the commands above. Be sure to set (manually, if needed)
your `LD_LIBRARY_PATH` to include the directory with the appropriate (matching) version of `libcuvs.so`, as
described in the [cuVS installation instructions](https://docs.nvidia.com/cuvs/installation/java#cuvs-lucene).

## Getting Started

The [Lucene Integration](https://docs.nvidia.com/cuvs/user-guide/lucene) guide walks through plugging a
codec into a standard Lucene `IndexWriter`, searching on the GPU, tuning index builds, and managing GPU
resources in a long-lived application. Class-level documentation is in the
[Lucene API reference](https://docs.nvidia.com/cuvs/api-reference/lucene-api-documentation).

Runnable examples of CAGRA-accelerated HNSW indexing, and of indexing and searching entirely on the GPU with
`CuVS2510GPUSearchCodec`, are in the [`examples/`](../../examples/java/cuvs-lucene) directory.

### Parameter bounds

The public Lucene API rejects out-of-range CAGRA parameters in Java before they reach native CAGRA.
Build parameters are checked by `build()`, and search parameters when the query is constructed.
Optional filter over-fetch is capped for `SINGLE_CTA` at search time (see below).
The table below lists what is enforced in Java; it is
**not** a claim that every value in these ranges is supported by native CAGRA — see the caveats
that follow.

| Parameter | Java-level range enforced |
| --- | --- |
| GPU writer threads | 1–512 |
| Intermediate graph degree | 2–512 |
| Graph degree | 1–512 |
| `GPUKnnFloatVectorQuery` `iTopK` | minimum 1; at most 512 with `SINGLE_CTA` |
| `GPUKnnFloatVectorQuery` `searchWidth` | minimum 1; 4,194,303 numeric-safety ceiling |
| `GPUKnnFloatVectorQuery` `maxIterations` | nonnegative; 0 selects automatically |
| `GPUKnnFloatVectorQuery` `threadBlockSize` | 0 (auto), 64, 128, 256, 512, or 1024 |

Under `CUSTOM`, native CAGRA reduces `graphDegree` to `intermediateGraphDegree` when needed,
and may reduce both for a small dataset. Java preserves this behavior rather than rejecting the pair. Under
`HEURISTIC`, the configured `graphDegree`/`intermediateGraphDegree` pair is not what CAGRA
actually builds with, so this relationship is not enforced on the configured pair: for
`AcceleratedHNSWParams`, both degrees are derived from `maxConn`/`beamWidth` and the configured
pair is ignored entirely; for `GPUSearchParams`, the configured `graphDegree` is passed into the
dataset-size heuristic as an input (it is not ignored), while `intermediateGraphDegree` is ignored
and the rest of the build parameters are derived from the heuristic's output.

For `iTopK`, the lower bound of 1 and the `SINGLE_CTA` maximum of 512 are native limits.
`MAX_ITOPK` (`Integer.MAX_VALUE`) is simply the largest value representable by the
public Java API, and `MAX_SEARCH_WIDTH` (4,194,303) only keeps CAGRA's result buffer within its
unsigned 32-bit indexing limit. Neither is a promise that native CAGRA supports every value up
to that ceiling, and in practice values anywhere near `MAX_ITOPK` are not usable. The true upper
limit for a given search depends on the resolved CAGRA algorithm, `max_iterations`, graph degree,
filtering, and available GPU memory. In particular, `MULTI_CTA` (which `AUTO` typically selects
for a single query unless there are enough partitions to use `SINGLE_CTA`) sizes an internal
traversal hash table from `search_width`, `iTopK`,
`max_iterations`, and the graph degree. This API does not replicate that calculation, since
`max_iterations` is itself auto-derived from the graph degree and dataset size, values not known
at query-construction time, so out-of-range combinations are caught by native CAGRA at search
time rather than here.

Note that native CAGRA only reports some of those combinations cleanly. Moderately oversized
values raise a clear exception, but above roughly `iTopK` 1e9 the native hash-table sizing loop
fails to terminate and the search hangs instead of returning an error
([#2523](https://github.com/NVIDIA/cuvs/issues/2523)). Treat these constants as representational
ceilings only, and size `iTopK`/`searchWidth` to what the workload actually needs.

The query uses an effective `iTopK` equal to the greater of the configured value and the requested
Lucene `k`; for `SINGLE_CTA`, that value must be at most 512 at query construction.
The filtered per-segment path can request up to `k + 10` candidates internally, but caps that
optional over-fetch at 512 for `SINGLE_CTA`. A change in filter cardinality therefore does not
turn a valid query into a parameter error.

## Contributing

If you are interested in contributing to cuvs-lucene, please read the cuVS [Contributing guide](https://docs.nvidia.com/cuvs/developer-guide/contributing).

> [!NOTE]
> The code style format is enforced using the [Spotless maven plugin](https://github.com/diffplug/spotless/tree/main/plugin-maven), which runs as a `pre-commit` hook. Run `pre-commit run --all-files`, or `mvn spotless:apply` in this directory, to format the sources.

## References

- [Bring Massive-Scale Vector Search to the GPU with Apache Lucene](https://www.nvidia.com/en-us/on-demand/session/gtc25-S71286/) — NVIDIA GTC 2025 session video
- [cuVS and Lucene: GPU-based Vector Search](https://www.youtube.com/watch?v=qiW7iIDFJC0) — Berlin Buzzwords 2024 session video
- [Exploring GPU-accelerated vector search in Elasticsearch with NVIDIA](https://www.elastic.co/search-labs/blog/gpu-accelerated-vector-search-elasticsearch-nvidia) — Elasticsearch Blog
- [Apache Lucene Accelerated with the NVIDIA cuVS 25.06 Release](https://searchscale.com/blog/apache-lucene-accelerated-with-nvidia-cuvs-25.06-release/) — SearchScale Blog
