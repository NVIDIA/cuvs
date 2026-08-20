# Examples

This maven project contains basic examples that showcase how `cuvs-lucene` can be used.

## Prerequisites

- The [`cuvs-lucene` prerequisites](../README.md#prerequisites)

## Steps

First build `cuvs-lucene` and install it into your local Maven repository, as described in
[Building from source](../README.md#building-from-source). From the cuVS repository root:

```sh
./build.sh libcuvs java lucene
```

Then return to this directory:

```sh
cd java/cuvs-lucene/examples
```

To run Accelerated HNSW example do:

```sh
mvn clean install && java -Djava.util.logging.config.file=src/main/resources/logging.properties -cp target/examples-26.10.0-jar-with-merged-services.jar com.nvidia.cuvs.lucene.examples.AcceleratedHnswExample
```

To run the Index and Search on GPU example do:

```sh
mvn clean install && java -Djava.util.logging.config.file=src/main/resources/logging.properties -cp target/examples-26.10.0-jar-with-merged-services.jar com.nvidia.cuvs.lucene.examples.IndexAndSearchonGPUExample
```

To run the optimized CAGRA-HNSW build example (reference pattern for efficiently building an
accelerated HNSW index from a large `.fbin` with every ingest-side knob on — open the file once and
stream sequential prefetched chunks that overlap the disk read with indexing, hold at most two chunks
in memory, reuse a single vector array, size a native flat buffer per segment, auto-select the CAGRA
graph-build algorithm, and optionally partition into K segments built sequentially or overlapped) do:

```sh
mvn clean install && java -Djava.util.logging.config.file=src/main/resources/logging.properties -cp target/examples-26.10.0-jar-with-merged-services.jar com.nvidia.cuvs.lucene.examples.OptimizedCagraHnswBuildExample
```

With no arguments it generates and indexes a small demo `.fbin` as a single segment; pass a real file,
chunk size, segment count, and overlap flag as
`... OptimizedCagraHnswBuildExample <path-to.fbin> <chunkSizeMB> <numSegments> <overlap:true|false>`.
