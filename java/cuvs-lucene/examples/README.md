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

To run the chunked `.fbin` ingestion example (reference pattern for streaming a large vector file
into an accelerated HNSW index without per-vector file reopening or holding the whole file in
memory) do:

```sh
mvn clean install && java -Djava.util.logging.config.file=src/main/resources/logging.properties -cp target/examples-26.08.0-jar-with-merged-services.jar com.nvidia.cuvs.lucene.examples.ChunkedFbinIngestExample
```

With no arguments it generates and indexes a small demo `.fbin`; pass a real file and chunk size as
`... ChunkedFbinIngestExample <path-to.fbin> <chunkSizeMB>`.
