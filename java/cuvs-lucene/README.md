# cuVS Lucene

This is a project for using [cuVS](https://github.com/rapidsai/cuvs), NVIDIA's GPU accelerated vector search library, with [Apache Lucene](https://github.com/apache/lucene).

## Contents

1. [What is cuvs-lucene?](#what-is-cuvs-lucene)
2. [Installing cuvs-lucene](#installing-cuvs-lucene)
3. [Getting Started](#getting-started)
4. [CAGRA/HNSW bulk indexing](#cagrahnsw-bulk-indexing)
5. [Contributing](#contributing)
6. [References](#references)

## What is cuvs-lucene?

`cuvs-lucene` provides a pluggable [KnnVectorsFormat](https://lucene.apache.org/core/10_2_0/core/org/apache/lucene/codecs/KnnVectorsFormat.html) that uses cuVS to offload vector index build — and optionally search — to NVIDIA GPUs. Because it plugs in through a standard Lucene codec, existing Lucene applications can take advantage of GPU acceleration with minimal code changes and gracefully fall back to the default CPU codec when no GPU is present.

Four codecs are currently provided:

- `Lucene101AcceleratedHNSWCodec` — GPU-accelerated HNSW build with CPU HNSW search. The on-disk format is standard Lucene HNSW, so indexes built on the GPU can be read by any stock Lucene 10.x reader.
  - `LuceneAcceleratedHNSWScalarQuantizedCodec` — scalar-quantized vectors for a smaller index footprint.
  - `LuceneAcceleratedHNSWBinaryQuantizedCodec` — binary-quantized vectors for an even smaller index footprint.
- `CuVS2510GPUSearchCodec` — GPU-accelerated HNSW build and GPU search

## Installing cuvs-lucene

### Prerequisites

- A machine with an NVIDIA GPU
- [CUDA 12.0+](https://developer.nvidia.com/cuda-toolkit-archive)
- [JDK 22](https://jdk.java.net/archive/)
- [Maven 3.9.6+](https://maven.apache.org/download.cgi)
- A matching version of the [cuVS libraries](https://docs.rapids.ai/api/cuvs/stable/build/#build-from-source). For Maven usage, install the cuVS tarball and add it to your system library load path. See the cuVS [tarball install instructions](https://docs.rapids.ai/api/cuvs/stable/build/#download-extract).

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

`cuvs-lucene` lives in the [cuVS repository](https://github.com/rapidsai/cuvs) and builds against the cuVS
Java bindings. If the libcuvs libraries and the Java bindings have not been built and installed, use
`./build.sh libcuvs java lucene` in the top level directory.

Alternatively, if libcuvs is already built and the `cuvs-java` artifact is already installed in your local
Maven repository, do `./build.sh lucene` in the top level directory or just do `./build.sh` in this directory.

The resulting artifacts are written to `target/`.

To run the tests, add `--run-java-tests` to any of the commands above. Be sure to set (manually, if needed)
your `LD_LIBRARY_PATH` to include the directory with the appropriate (matching) version of `libcuvs.so`, as
described in the cuVS [tarball install instructions](https://docs.rapids.ai/api/cuvs/stable/build/#download-extract).

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

For more examples, including one that indexes and searches entirely on the GPU using `CuVS2510GPUSearchCodec`, please refer to the [`examples/`](../../examples/java/cuvs-lucene) directory.

## CAGRA/HNSW bulk indexing

`CagraHnswBulkIndexWriter` is an opt-in API for controlled, offline creation of
`Lucene101AcceleratedHNSWCodec` indexes. It owns the `IndexWriter` configuration and flush
lifecycle needed by native buffering. It supports GPU CAGRA graph construction followed by CPU
HNSW search; it does not build `CuVS2510GPUSearchCodec` indexes and, unlike the generic codec,
requires cuVS GPU support instead of falling back to CPU indexing.

### Ingestion modes

| Mode | Public entry point | Vector handling | Finished index |
| --- | --- | --- | --- |
| Generic Lucene codec | Configure `Lucene101AcceleratedHNSWCodec` on an application-owned `IndexWriter` | Normal Lucene document ingestion and lifecycle | Self-contained; normal Lucene flush, sort, and merge behavior |
| Streaming/native-buffered bulk | Construct `CagraHnswBulkIndexWriter` directly, call `build(VectorSource, Config)`, or call `indexFbin(Path, Config)` | Copies each vector into an exactly sized native host matrix before that segment's GPU build | Self-contained; the source can be removed after a successful build |
| Mapped, self-contained bulk | `indexMappedFbin(Path, Config)` | CAGRA and the flat-vector writer share a read-only mapping, avoiding per-row decode and ingest copies; the flat vectors are still written into the index | Self-contained; the FBIN can be removed after a successful build |
| Immutable external-FBIN bulk | Register the FBIN, then call `indexImmutableFbin(Registration, Config, ExternalFbinOptions)` | CAGRA reads the mapping and the index stores a content-addressed descriptor instead of duplicating the flat-vector payload | Not self-contained; exact scoring reads the registered FBIN |

The streaming/native-buffered mode has three entry shapes:

- The direct constructor is the manual, single-segment form. The caller creates every document,
  promises the exact document count, and calls `addDocument` exactly that many times.
- `build(VectorSource, Config)` owns document creation for a caller-owned, forward-only source. It
  can build multiple segments sequentially, but does not close the source and does not support the
  overlapped pipeline.
- `indexFbin(Path, Config)` owns the FBIN reader. It supports sequential segment builds and an
  overlapped multi-segment pipeline in which host ingest can overlap a serialized GPU build.
  `FieldCallback` can add non-vector fields in either one-shot form.

Both mapped modes currently require `segments(1, false)`. They also require exactly one dense
`FLOAT32` vector field, one vector per document, and row `i` to correspond to Lucene document and
vector ordinal `i`.

### Ownership and sharing

All bulk forms own their internal `IndexWriter`, disable index sorting, automatic flushes, compound
files, and merges, and commit only after the promised vector count has been written. Closing a
manual writer with too few vectors rolls it back; adding too many vectors is rejected. A failed
segment is rolled back. A sequential multi-segment build can already have committed earlier
segments when a later segment fails, so callers must treat and replace that target as a failed
build.

The generic, streaming, and mapped self-contained results can be moved by copying the Lucene index
directory. The immutable external-FBIN result must be shipped as two artifacts: the Lucene index
directory and the exact FBIN identified by the descriptor's complete-file SHA-256. The descriptor
does not persist a host path, so another process or node may place the FBIN at a different local
path. Before opening the index, that process must register the path and digest:

```java
try (ExternalFbinFileRegistry.Registration lease =
        ExternalFbinFileRegistry.register(localFbin, sha256Hex);
    Directory directory = FSDirectory.open(indexPath);
    DirectoryReader reader = DirectoryReader.open(directory)) {
  // Search while both the reader and registration lease are open.
}
```

Registrations are process-local and reference counted. Keep at least one lease open for the entire
lifetime of every reader using that content ID. Do not modify, replace, relocate, or delete the
registered FBIN while a build or reader is alive. An index directory copied without its FBIN, or a
process that has not registered the local FBIN, cannot be opened. Registering a different path for
the same content ID while an existing registration is live is rejected. External-FBIN indexes also
require a cuvs-lucene runtime that understands their external-vector descriptor; they are not
readable by a stock Lucene runtime alone.

### Validation

Every FBIN path is structurally checked for a positive shape and an exact
`8 + rows * dimensions * 4` byte length. `Config` also rejects a mismatch between the Lucene
similarity and the cuVS graph metric:

- `EUCLIDEAN` requires `L2Expanded`.
- `DOT_PRODUCT` and `MAXIMUM_INNER_PRODUCT` require `InnerProduct`.
- `COSINE` requires `CosineExpanded`.

The manual bulk form additionally validates that every document has exactly one vector field with
the configured name, dimension, `FLOAT32` encoding, and similarity. Borrowed mapped data is not
scanned component by component, so the caller must ensure that every source float is finite.

Immutable external-FBIN builds require the caller to supply a previously established SHA-256 for
the complete file. `ExternalFbinFileRegistry.register` validates and allowlists the real path, but
does not compute the digest. `ExternalFbinBuildValidation` controls the build-time scan:

| Validation | Build-time behavior |
| --- | --- |
| `TRUSTED_IMMUTABLE` | Checks the registration, reference metadata, header, range, and file length; does not scan the payload or verify the digest. Use only with storage that already enforces the content identity. |
| `PREFETCH` | Sequentially scans the referenced payload as read-ahead while graph construction runs; trusts the supplied digest. |
| `VERIFY_SHA256` | Hashes the complete FBIN and compares it with the supplied digest while graph construction runs. A mismatch prevents the commit. |

`ExternalFbinOptions.scanHeadStartBytes()` can delay graph construction until a requested amount of
the selected payload has been scanned. It is a scheduling control, not a reduction in validation:
the selected scan still completes before commit. It must be zero for `TRUSTED_IMMUTABLE` and no
larger than the referenced payload.

Opening an external-FBIN index verifies the descriptor checksum, field metadata, registered path,
header, range, and file length, but does not hash the full file. Lucene's `checkIntegrity()` hashes
the complete external FBIN and compares it with the persisted SHA-256.

### Metrics

Supply one `CagraHnswBuildMetrics` per build through `Config.Builder.metrics`. After the build,
`snapshot()` returns a stable `Map<String, Number>`, and `appendTo(target, prefix)` adds that
snapshot to a benchmark result map.

Metric keys use three namespaces:

- `stage/<name>/seconds`, `stage/<name>/count`, and optional `stage/<name>/bytes` report aggregated
  graph build, graph conversion and output, mapped flat output, external scan/overlap, and bulk
  writer commit/close stages. The commit wall includes the flush and its nested GPU/output work;
  nested stage durations must not be added to it.
- `counter/<name>` reports values such as logical adjacency bytes, mapped flat chunks, and external
  scan progress at CAGRA start and end.
- `gauge/<name>` records effective graph-build parameters, including graph degrees, writer threads,
  NN-descent iterations, and IVF-PQ dimensions, lists, probes, and k-means iterations when used.

### Why these controls are bulk-only

Native buffering and borrowed FBIN storage depend on guarantees that a general Lucene codec cannot
make: one controlled flush, an exact dense vector count and order, no index sort, no merge during
the build, and an external file whose immutability and reader lifetime are managed by the
application. `CagraHnswBulkIndexWriter` owns those conditions. Storage and external-file controls
therefore remain on the bulk API (`Config`, `ExternalFbinOptions`, and
`ExternalFbinFileRegistry`) rather than `AcceleratedHNSWParams` or public codec constructors. The
generic codec remains suitable for application-owned Lucene, Solr, Elasticsearch, and OpenSearch
indexing lifecycles without adding an external-file contract they cannot enforce.

## Contributing

If you are interested in contributing to cuvs-lucene, please read the cuVS [Contributing guide](https://docs.nvidia.com/cuvs/developer-guide/contributing).

> [!NOTE]
> The code style format is enforced using the [Spotless maven plugin](https://github.com/diffplug/spotless/tree/main/plugin-maven), which runs as a `pre-commit` hook. Run `pre-commit run --all-files`, or `mvn spotless:apply` in this directory, to format the sources.

## References

- [Bring Massive-Scale Vector Search to the GPU with Apache Lucene](https://www.nvidia.com/en-us/on-demand/session/gtc25-S71286/) — NVIDIA GTC 2025 session video
- [cuVS and Lucene: GPU-based Vector Search](https://www.youtube.com/watch?v=qiW7iIDFJC0) — Berlin Buzzwords 2024 session video
- [Exploring GPU-accelerated vector search in Elasticsearch with NVIDIA](https://www.elastic.co/search-labs/blog/gpu-accelerated-vector-search-elasticsearch-nvidia) — Elasticsearch Blog
- [Apache Lucene Accelerated with the NVIDIA cuVS 25.06 Release](https://searchscale.com/blog/apache-lucene-accelerated-with-nvidia-cuvs-25.06-release/) — SearchScale Blog
