# Lucene cuVS

This is a project for using [cuVS](https://github.com/rapidsai/cuvs), NVIDIA's GPU accelerated vector search library, with [Apache Lucene](https://github.com/apache/lucene).

## Overview

This library provides a new [KnnVectorFormat](https://lucene.apache.org/core/10_3_1/core/org/apache/lucene/codecs/KnnVectorsFormat.html) which can be plugged into a Lucene codec.

cuvs-lucene is part of the [cuVS](https://github.com/rapidsai/cuvs) repository and depends on
the `com.nvidia.cuvs:cuvs-java` artifact built by [`../cuvs-java`](../cuvs-java). See
[`../README.md`](../README.md) for an overview of the Java projects.

## Building

### Prerequisites

- [CUDA 12.0+](https://developer.nvidia.com/cuda-toolkit-archive)
- [Maven 3.9.6+](https://maven.apache.org/download.cgi)
- [JDK 22](https://jdk.java.net/archive/)
- The `libcuvs` C/C++ libraries and the `cuvs-java` bindings (both built from this repo, see below)

### From the repository root

`lucene` is a build target of the top-level `build.sh`, alongside `libcuvs` and `java`:

```sh
# Build everything cuvs-lucene needs, from scratch:
./build.sh libcuvs java lucene

# If libcuvs and cuvs-java are already built and installed, just (re)build cuvs-lucene:
./build.sh lucene
```

Like `java` (which builds against an already-built `libcuvs`), `lucene` builds against the
already-installed `cuvs-java` artifact, so it can be rebuilt on its own without waiting for
`cuvs-java`. Add `java` (i.e. `./build.sh java lucene`) when `cuvs-java` has changed and needs
rebuilding first. The `cuvs-lucene` jar is produced under `target/`.

### From this directory

Once `cuvs-java` has been installed into your local Maven repository (via `../build.sh java`),
you can build cuvs-lucene directly:

```sh
./build.sh
```

### Running Tests

Append `--run-java-tests` to run the test suite. `build.sh` sets `LD_LIBRARY_PATH` to the in-tree
`../../cpp/build` so the tests can locate `libcuvs`:

```sh
# from the repository root
./build.sh lucene --run-java-tests

# or from this directory
./build.sh --run-java-tests
```

If your `libcuvs` libraries live elsewhere (e.g. a conda environment), set `LD_LIBRARY_PATH` to
the directory containing the matching `libcuvs_c.so` before running the tests.

## Contributing

> [!NOTE]
> The code style format is automatically enforced (including the missing license header, if any) using the [Spotless maven plugin](https://github.com/diffplug/spotless/tree/main/plugin-maven). This currently happens in the maven's `validate` stage.
