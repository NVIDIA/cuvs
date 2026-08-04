# cuVS Lucene

This is a project for using [cuVS](https://github.com/rapidsai/cuvs), NVIDIA's GPU accelerated vector search library, with [Apache Lucene](https://github.com/apache/lucene).

## Contents

1. [What is cuvs-lucene?](#what-is-cuvs-lucene)
2. [Installing cuvs-lucene](#installing-cuvs-lucene)
3. [Getting Started](#getting-started)
4. [Contributing](#contributing)
5. [References](#references)

## What is cuvs-lucene?

`cuvs-lucene` provides a pluggable [KnnVectorsFormat](https://lucene.apache.org/core/10_2_0/core/org/apache/lucene/codecs/KnnVectorsFormat.html) that uses cuVS to offload vector index build — and optionally search — to NVIDIA GPUs. Because it plugs in through a standard Lucene codec, existing Lucene applications can take advantage of GPU acceleration with minimal code changes and gracefully fall back to the default CPU codec when no GPU is present.

Four codecs are currently provided:

- `Lucene101AcceleratedHNSWCodec` — GPU-accelerated HNSW build with CPU HNSW search. The on-disk format is standard Lucene HNSW, so indexes built on the GPU can be read by any stock Lucene 10.x reader.
  - `LuceneAcceleratedHNSWScalarQuantizedCodec` — scalar-quantized vectors for a smaller index footprint.
  - `LuceneAcceleratedHNSWBinaryQuantizedCodec` — binary-quantized vectors for an even smaller index footprint.
- `CuVS2510GPUSearchCodec` — GPU-accelerated HNSW build and GPU search

## Installing cuvs-lucene

### Prerequisites

- [CUDA 12.0+](https://developer.nvidia.com/cuda-toolkit-archive)
- [JDK 22](https://jdk.java.net/archive/)
- [Maven 3.9.6+](https://maven.apache.org/download.cgi)
- A compatible cuVS installation (26.04 - 26.06). For Maven usage, install the cuVS tarball and add it to your system library load path. See the cuVS [tarball install instructions](https://docs.rapids.ai/api/cuvs/stable/build/#download-extract).

### Maven

To pull `cuvs-lucene` into a Maven project, add the following dependency to your `pom.xml`:

```xml
<dependency>
  <groupId>com.nvidia.cuvs.lucene</groupId>
  <artifactId>cuvs-lucene</artifactId>
  <version>26.06.0</version>
</dependency>
```

### Building from source

```sh
git clone https://github.com/rapidsai/cuvs-lucene.git
cd cuvs-lucene
mvn clean compile package
```

The resulting artifacts are written to `target/`. To run the tests, first install cuVS and add it to your system library load path, as described in the cuVS [tarball install instructions](https://docs.rapids.ai/api/cuvs/stable/build/#download-extract), then run:

```sh
mvn clean test
```

## Getting Started

The example below plugs the GPU-accelerated HNSW codec into a standard Lucene `IndexWriter`. Once the codec is set on the `IndexWriterConfig`, indexing proceeds exactly as it would with the default Lucene codec, and search uses the stock `KnnFloatVectorQuery`.

Before running it, make sure cuVS is installed and available on your system library load path. The cuVS [tarball install instructions](https://docs.rapids.ai/api/cuvs/stable/build/#download-extract) show how to set this up.

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

The artifacts would be built and available in the target / folder.

### Running Tests

```sh
mvn -q compile org.codehaus.mojo:exec-maven-plugin:3.5.1:java \
  -Dexec.mainClass=com.nvidia.cuvs.lucene.examples.HelloCuvsLucene
```

For more examples, including one that indexes and searches entirely on the GPU using `CuVS2510GPUSearchCodec`, please refer to the [`examples/`](examples) directory.

## Contributing

If you are interested in contributing to cuvs-lucene, please read our [Contributing guide](CONTRIBUTING.md).

> [!NOTE]
> The code style format is automatically enforced (including the missing license header, if any) using the [Spotless maven plugin](https://github.com/diffplug/spotless/tree/main/plugin-maven). This currently happens in the maven's `validate` stage.

## References

- [Bring Massive-Scale Vector Search to the GPU with Apache Lucene](https://www.nvidia.com/en-us/on-demand/session/gtc25-S71286/) — NVIDIA GTC 2025 session video
- [cuVS and Lucene: GPU-based Vector Search](https://www.youtube.com/watch?v=qiW7iIDFJC0) — Berlin Buzzwords 2024 session video
- [Exploring GPU-accelerated vector search in Elasticsearch with NVIDIA](https://www.elastic.co/search-labs/blog/gpu-accelerated-vector-search-elasticsearch-nvidia) — Elasticsearch Blog
- [Apache Lucene Accelerated with the NVIDIA cuVS 25.06 Release](https://searchscale.com/blog/apache-lucene-accelerated-with-nvidia-cuvs-25.06-release/) — SearchScale Blog
