# Examples

This maven project contains basic examples that showcase how `cuvs-lucene` can be used.

The [Lucene Integration](https://docs.nvidia.com/cuvs/user-guide/lucene) guide (`fern/pages/user_guide/lucene.md`) embeds `AcceleratedHnswExample` and
`IndexAndSearchonGPUExample` as its quickstart programs, from the `package` line on, through Fern's
`<Code src="..." lines={[7-1000]} />`, so every change to them is published in the docs. Keep them short and
self-contained. The guide finds the `package` line by its line number, 7, so adding or removing lines above it
requires updating `lines` in the guide, as the comment above it says.

## Prerequisites

- The [`cuvs-lucene` prerequisites](../../../java/cuvs-lucene/README.md#prerequisites)

## Steps

First build `cuvs-lucene` and install it into your local Maven repository, as described in
[Building from source](../../../java/cuvs-lucene/README.md#building-from-source). From the cuVS repository root:

```sh
./build.sh libcuvs java lucene
```

The examples use Lucene 10.2 and the matching `cuvs-lucene-10.2` artifact, set by the `lucene.version` and
`cuvs.lucene.artifactId` properties in `pom.xml`.

Then return to this directory:

```sh
cd examples/java/cuvs-lucene
```

To run Accelerated HNSW example do:

```sh
mvn clean install && java --enable-native-access=ALL-UNNAMED -cp target/examples-27.02.0-jar-with-merged-services.jar com.nvidia.cuvs.lucene.examples.AcceleratedHnswExample
```

To run the Index and Search on GPU example do:

```sh
mvn clean install && java --enable-native-access=ALL-UNNAMED -cp target/examples-27.02.0-jar-with-merged-services.jar com.nvidia.cuvs.lucene.examples.IndexAndSearchonGPUExample
```

Each run writes its index into a new directory under the system temporary directory (`cuvs-lucene-quickstart*` or
`cuvs-lucene-gpu-search*`) and leaves it there, so delete those directories between runs.
