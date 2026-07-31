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

To run the chunked `.fbin` ingestion example (reference pattern for efficiently streaming a large
vector file into an accelerated HNSW index — open the file once, read sequential prefetched chunks
that overlap the disk read with indexing, hold at most two chunks in memory, and reuse a single
vector array across all documents) do:

```sh
mvn clean install && java -Djava.util.logging.config.file=src/main/resources/logging.properties -cp target/examples-26.10.0-jar-with-merged-services.jar com.nvidia.cuvs.lucene.examples.OptimizedFbinIngestExample
```

With no arguments it generates and indexes a small demo `.fbin`; pass a real file and chunk size as
`... OptimizedFbinIngestExample <path-to.fbin> <chunkSizeMB>`.
