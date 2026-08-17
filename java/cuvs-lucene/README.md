# cuVS Lucene

This is a project for using [cuVS](https://github.com/rapidsai/cuvs), NVIDIA's GPU accelerated vector search library, with [Apache Lucene](https://github.com/apache/lucene).

## Contents

1. [What is cuvs-lucene?](#what-is-cuvs-lucene)
2. [Installing cuvs-lucene](#installing-cuvs-lucene)
3. [Getting Started](#getting-started)
4. [Contributing](#contributing)
5. [References](#references)

## What is cuvs-lucene?

`cuvs-lucene` provides a pluggable [KnnVectorsFormat](https://lucene.apache.org/core/10_2_0/core/org/apache/lucene/codecs/KnnVectorsFormat.html) that uses cuVS to offload vector index build — and optionally search — to NVIDIA GPUs. The accelerated-HNSW codecs can fall back to Lucene's CPU HNSW writer when cuVS is unavailable; the GPU-search codec requires cuVS. This development line is compiled and tested against the Lucene 10.2.0 runtime ABI.

Four codecs are currently provided:

- `Lucene101AcceleratedHNSWCodec` — GPU-accelerated HNSW build with CPU HNSW search. Its vector data uses Lucene's standard HNSW format and stock HNSW reader; applications still need a compatible `cuvs-lucene` codec provider to resolve the segment codec.
  - `LuceneAcceleratedHNSWScalarQuantizedCodec` — scalar-quantized vectors for a smaller index footprint.
  - `LuceneAcceleratedHNSWBinaryQuantizedCodec` — binary-quantized vectors for an even smaller index footprint.
- `CuVS2510GPUSearchCodec` — GPU CAGRA build and GPU CAGRA search

## Installing cuvs-lucene

### Prerequisites

- A machine with an NVIDIA GPU
- [CUDA 12.2+](https://developer.nvidia.com/cuda-toolkit-archive)
- [JDK 22](https://jdk.java.net/archive/)
- [Maven 3.9.6+](https://maven.apache.org/download.cgi)
- A matching version of the [cuVS libraries](https://docs.rapids.ai/api/cuvs/stable/build/#build-from-source). For Maven usage, install the cuVS tarball and add it to your system library load path. See the cuVS [tarball install instructions](https://docs.nvidia.com/cuvs/installation/c#tarball).

### Maven

To pull `cuvs-lucene` into a Maven project, add the following dependency to your `pom.xml`:

```xml
<dependency>
  <groupId>com.nvidia.cuvs.lucene</groupId>
  <artifactId>cuvs-lucene</artifactId>
  <version>26.10.0</version>
</dependency>
```

### Building from source

`cuvs-lucene` lives in the [cuVS repository](https://github.com/rapidsai/cuvs) and builds against the cuVS
Java bindings. If the libcuvs libraries and the Java bindings have not been built and installed, use
`./build.sh libcuvs java lucene` in the top level directory.

Alternatively, if libcuvs is already built and the `cuvs-java` artifact is already installed in your local
Maven repository, do `./build.sh lucene` in the top level directory or just do `./build.sh` in this directory.

The resulting artifacts are written to `target/`.

To run the tests, add `--run-java-tests` to any of the commands above. Be sure to set (manually, if needed)
your `LD_LIBRARY_PATH` to include the directory with the appropriate (matching) version of `libcuvs.so`, as
described in the cuVS [tarball install instructions](https://docs.nvidia.com/cuvs/installation/c#tarball).

## Getting Started

The example below plugs the GPU-accelerated HNSW codec into a standard Lucene `IndexWriter`. Once the codec is set on the `IndexWriterConfig`, indexing proceeds exactly as it would with the default Lucene codec, and search uses the stock `KnnFloatVectorQuery`.

Before running it, make sure cuVS is installed and available on your system library load path. The cuVS [tarball install instructions](https://docs.nvidia.com/cuvs/installation/c#tarball) show how to set this up.

### RMM async allocation for GPU search

Applications using `CuVS2510GPUSearchCodec` can opt into RMM's stream-ordered asynchronous device
allocator during startup:

```java
CuVSProvider.provider().enableRMMAsyncMemory();
```

Call this before creating any cuVS resources, codecs, writers, or readers. The setting affects the
entire process on the current CUDA device, so allocator policy belongs to the application rather
than an individual Lucene codec. Async allocation is optional for correctness and recommended for
GPU workloads with repeated device allocations, especially concurrent or multi-stream searches.
Applications that do not opt in use the default RMM device-memory resource.

In a Maven project that includes the `cuvs-lucene` dependency shown above, create `src/main/java/com/nvidia/cuvs/lucene/examples/HelloCuvsLucene.java`:

```java
package com.nvidia.cuvs.lucene.examples;

import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import com.nvidia.cuvs.lucene.AcceleratedHNSWParams;
import com.nvidia.cuvs.lucene.Lucene101AcceleratedHNSWCodec;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

public class HelloCuvsLucene {
  public static void main(String[] args) throws Exception {
    AcceleratedHNSWParams params = new AcceleratedHNSWParams.Builder().build();
    Codec codec = new Lucene101AcceleratedHNSWCodec(params);
    IndexWriterConfig config = new IndexWriterConfig().setCodec(codec);

    Path indexPath = Paths.get("index");
    float[] embedding = new float[] {0.1f, 0.2f, 0.3f, 0.4f};

    try (Directory dir = FSDirectory.open(indexPath);
        IndexWriter writer = new IndexWriter(dir, config)) {
      Document doc = new Document();
      doc.add(new KnnFloatVectorField("vector_field", embedding, EUCLIDEAN));
      writer.addDocument(doc);
    }

    System.out.println("Hello cuVS Lucene ran successfully.");
  }
}
```

The artifacts are built in the `target/` directory.

Run the example with:

```sh
mvn -q compile org.codehaus.mojo:exec-maven-plugin:3.5.1:java \
  -Dexec.mainClass=com.nvidia.cuvs.lucene.examples.HelloCuvsLucene
```

### Using with PyLucene

The complete codec set requires PyLucene generated against Lucene 10.2.0. The
official PyLucene 10.0.0 distribution is not a compatible full-feature runtime:
in particular, GPU search uses Lucene 10.2 APIs and binary quantization uses
Lucene102 vector formats.

Apache does not publish a PyLucene 10.2.0 release. Linux development and
testing therefore require a custom PyLucene wrapper build generated against
the Lucene 10.2.0 sources. Prepare and activate that matching external
environment before using PyLucene or running pytest; Maven can build the jar
independently and does not produce the PyLucene runtime.

Build the standard thin `cuvs-lucene` jar:

```sh
mvn clean package -DskipTests
```

Add that jar and the matching base `cuvs-java` jar to the classpath passed to
`lucene.initVM(...)`. PyLucene can then load the codec through Lucene's service
provider lookup:

```python
from org.apache.lucene.codecs import Codec

codec = Codec.forName("Lucene101AcceleratedHNSWCodec")
```

Use the returned codec with `IndexWriterConfig.setCodec(codec)`. Initialize the
JVM only after the custom 10.2 environment and every application jar are on its
classpath; PyLucene cannot replace that classpath after `lucene.initVM(...)`.

### Running Tests

Run the Java tests with `mvn clean test`. Once the custom PyLucene 10.2
environment is activated and the cuVS classpath and native-library environment
are available, run the full parametrized CPU/GPU end-to-end suite directly
with pytest:

```sh
python3 -m pytest -q -s src/test/python/test_pylucene_end_to_end.py
```

The cases live in `src/test/python/test_pylucene_end_to_end.py`; reusable
runtime helpers are in `pylucene_test_support.py`, and the Java test bridge is
compiled to `target/test-classes`. Set `CUVS_LUCENE_JAR`,
`CUVS_LUCENE_CUVS_JAVA_JAR`, or `CUVS_LUCENE_PYLUCENE_TEST_CLASSES` only when
their standard Maven locations are not appropriate. The helper verifies that
PyLucene's Lucene version exactly matches the `lucene-core` version in this
checkout's POM before starting the JVM, so a mismatched wrapper fails with an
actionable error instead of a later linkage failure.

The pytest IDs identify CPU HNSW, CAGRA-built HNSW, and CAGRA-search cases;
use pytest's `-k` option for a focused run.

For more examples, including one that indexes and searches entirely on the GPU using `CuVS2510GPUSearchCodec`, please refer to the [`examples/`](examples) directory.

## Contributing

If you are interested in contributing to cuvs-lucene, please read the cuVS [Contributing guide](https://docs.nvidia.com/cuvs/developer-guide/contributing).

> [!NOTE]
> The code style format is enforced using the [Spotless maven plugin](https://github.com/diffplug/spotless/tree/main/plugin-maven), which runs as a `pre-commit` hook. Run `pre-commit run --all-files`, or `mvn spotless:apply` in this directory, to format the sources.

## References

- [Bring Massive-Scale Vector Search to the GPU with Apache Lucene](https://www.nvidia.com/en-us/on-demand/session/gtc25-S71286/) — NVIDIA GTC 2025 session video
- [cuVS and Lucene: GPU-based Vector Search](https://www.youtube.com/watch?v=qiW7iIDFJC0) — Berlin Buzzwords 2024 session video
- [Exploring GPU-accelerated vector search in Elasticsearch with NVIDIA](https://www.elastic.co/search-labs/blog/gpu-accelerated-vector-search-elasticsearch-nvidia) — Elasticsearch Blog
- [Apache Lucene Accelerated with the NVIDIA cuVS 25.06 Release](https://searchscale.com/blog/apache-lucene-accelerated-with-nvidia-cuvs-25.06-release/) — SearchScale Blog
