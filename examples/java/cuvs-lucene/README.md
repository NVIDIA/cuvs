# Examples

This maven project contains basic examples that showcase how `cuvs-lucene` can be used.

## Prerequisites

- The [`cuvs-lucene` prerequisites](../../../java/cuvs-lucene/README.md#prerequisites)

## Steps

First build `cuvs-lucene` and install it into your local Maven repository, as described in
[Building from source](../../../java/cuvs-lucene/README.md#building-from-source). From the cuVS repository root:

```sh
./build.sh libcuvs java lucene
```

The examples use Lucene 10.5 and the matching `cuvs-lucene-10.5` artifact. To run them against another
supported Lucene release, change the `lucene.version` and `cuvs.lucene.artifactId` properties in `pom.xml`, for
example to `10.3.2` and `cuvs-lucene-10.3`.

Then return to this directory:

```sh
cd examples/java/cuvs-lucene
```

To run Accelerated HNSW example do:

```sh
mvn clean install && java -Djava.util.logging.config.file=src/main/resources/logging.properties -cp target/examples-26.12.0-jar-with-merged-services.jar com.nvidia.cuvs.lucene.examples.AcceleratedHnswExample
```

To run the Index and Search on GPU example do:

```sh
mvn clean install && java -Djava.util.logging.config.file=src/main/resources/logging.properties -cp target/examples-26.12.0-jar-with-merged-services.jar com.nvidia.cuvs.lucene.examples.IndexAndSearchonGPUExample
```
