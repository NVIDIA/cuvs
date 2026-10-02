---
slug: user-guide/lucene
---

# Lucene Integration

NVIDIA cuVS Lucene (`cuvs-lucene`) is a pluggable [KnnVectorsFormat](https://lucene.apache.org/core/10_2_0/core/org/apache/lucene/codecs/KnnVectorsFormat.html) that offloads vector index build, and optionally search, from the CPU to NVIDIA GPUs. Because it plugs in through a standard Lucene codec, an existing Lucene application adopts it by setting a codec on its `IndexWriterConfig`; the rest of the indexing and query code is unchanged.

`cuvs-lucene` supports Lucene 10.2 through 10.5, with one artifact per Lucene minor release, and is built on the [NVIDIA cuVS Java APIs](/api-reference/java-api-documentation). For installation and build instructions, see [cuVS Lucene](/installation/java#cuvs-lucene) in the Java installation guide. For class-level details, see the [Lucene API Documentation](/api-reference/lucene-api-documentation).

## When To Use It

Use `cuvs-lucene` when vector index construction is the bottleneck. Building an HNSW graph on the CPU is expensive, and it dominates ingest and reindex time for large vector collections. The accelerated HNSW codecs build the graph on the GPU with CAGRA and write a standard Lucene HNSW index, so query serving, replication, and existing operational tooling continue to work unchanged.

Use the GPU search codec when queries should also run on the GPU, which suits high-throughput search over collections that stay resident on GPU-equipped nodes.

If you are running Apache Solr, you do not need to integrate `cuvs-lucene` directly. Solr exposes it through its own `cuvs` module; see [Solr](/getting-started/integrations#solr) on the Integrations page.

## Choosing a Codec

Four codecs are available, created through [`CuVSCodecs`](/api-reference/lucene-api-com-nvidia-cuvs-lucene-cuvscodecs). All four build the vector index on the GPU; they differ in where search runs and in what is written to disk.

| Codec | Search | On-disk format | Use when |
| --- | --- | --- | --- |
| `CuVSCodecs.acceleratedHNSW` | CPU | Standard Lucene HNSW | You want faster index builds without changing the search path or index format. |
| `CuVSCodecs.acceleratedHNSWScalarQuantized` | CPU | Standard Lucene HNSW, scalar-quantized | As above, with a smaller index footprint and some loss of precision. |
| `CuVSCodecs.acceleratedHNSWBinaryQuantized` | CPU | Standard Lucene HNSW, full-precision vectors | As above, with the graph built over binary-quantized vectors. The vectors themselves are stored and scored in full precision. |
| `CuVSCodecs.gpuSearch` | GPU | cuVS format | Search throughput matters and GPUs are available on the query path. |

The three accelerated HNSW codecs write the stock Lucene HNSW vector format and read it back through the standard Lucene reader. GPU acceleration therefore applies to index build only: the query API and the read path are unchanged, and the read path touches neither a GPU nor the native cuVS library. The graph itself is built by CAGRA rather than Lucene's HNSW builder, so benchmark recall if you are migrating an existing index.

Deployment does change, though. Lucene resolves codecs by name through its service loader, so search nodes need the same `cuvs-lucene` and `cuvs-java` jars as indexing nodes, or the segments cannot be opened. Searching the accelerated HNSW codecs' indexes needs no GPU, so search nodes can run any JDK 21 or newer.

The GPU search codec writes a cuVS-specific format instead. That format is experimental and backward compatibility across releases is not guaranteed, so plan on being able to rebuild indexes when upgrading.

The three accelerated HNSW codecs fall back to CPU index construction when cuVS resources cannot be created, which happens when no GPU is present or the native library cannot be loaded. They log a warning and hand the segment to the stock Lucene HNSW writer, so the same application runs on GPU and non-GPU hosts and produces the same index format either way.

The GPU search codec has no such fallback. Its on-disk format has no CPU reader, so both indexing and search throw `UnsupportedOperationException` when cuVS is unavailable.

## Lucene Versions and Upgrades

Lucene changes its codec APIs between minor releases, so `cuvs-lucene` is published as one artifact per Lucene minor release, `cuvs-lucene-10.2` through `cuvs-lucene-10.5`; see [cuVS Lucene](/installation/java#cuvs-lucene). Use the one that matches your application's Lucene version. An artifact refuses to load against another Lucene minor release.

Each codec also wraps the default codec of one Lucene release, and Lucene records the codec's name in every segment it writes. The codec that writes therefore changes with Lucene:

| Lucene | Accelerated HNSW codec | GPU search codec |
| --- | --- | --- |
| 10.2 | `Lucene101AcceleratedHNSWCodec` | `CuVS2510GPUSearchCodec` |
| 10.3 | `Lucene103AcceleratedHNSWCodec` | `Lucene103CuVSGPUSearchCodec` |
| 10.4, 10.5 | `Lucene104AcceleratedHNSWCodec` | `Lucene104CuVSGPUSearchCodec` |

The quantized codecs follow the same pattern, for example `Lucene104AcceleratedHNSWScalarQuantizedCodec`. `CuVSCodecs` always returns the codecs that write on the Lucene release in use, so code that creates its codecs through it does not change when the application moves to another Lucene release.

To upgrade Lucene, switch to the matching `cuvs-lucene` artifact. Every artifact also includes the codecs of the earlier Lucene releases it supports, so existing segments stay readable. Those codecs are read-only, like Lucene's own backward codecs: new segments, including those produced by merges, are written with the current codec, and writing with an older one throws `UnsupportedOperationException`. Configuration that names a codec, rather than creating it through `CuVSCodecs`, has to be updated to the new name. The same applies to the per-field vectors formats: from Lucene 10.4 on, `Lucene99AcceleratedHNSWScalarQuantizedVectorsFormat` can only read, because Lucene 10.4 can no longer write the `Lucene99ScalarQuantizedVectorsFormat` it stores its vectors with. Configuration that names it, such as a per-field `knnAlgorithm`, has to switch to `Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat`, which `CuVSCodecs.acceleratedHNSWScalarQuantizedFormat()` returns.

## Quickstart

Setting one of the codecs on an `IndexWriterConfig` is the only change an existing Lucene application needs. Indexing then proceeds as it would with the default codec, and search uses the stock `KnnFloatVectorQuery`. The program below writes an index into a new temporary directory and does not delete it, so clean up between runs.

```java
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import com.nvidia.cuvs.lucene.AcceleratedHNSWParams;
import com.nvidia.cuvs.lucene.CuVSCodecs;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

public class AcceleratedHnswQuickstart {

  private static final String ID_FIELD = "id";
  private static final String VECTOR_FIELD = "vector_field";

  public static void main(String[] args) throws Exception {
    int numDocs = 2000;
    int dimension = 32;
    int topK = 5;

    Random random = new Random(222);
    float[][] dataset = generateDataset(random, numDocs, dimension);
    Path indexDirPath = Files.createTempDirectory("cuvs-lucene-quickstart");

    AcceleratedHNSWParams params = new AcceleratedHNSWParams.Builder().build();
    Codec codec = CuVSCodecs.acceleratedHNSW(params);
    IndexWriterConfig config = new IndexWriterConfig().setCodec(codec);

    // Indexing
    try (Directory indexDirectory = FSDirectory.open(indexDirPath);
        IndexWriter indexWriter = new IndexWriter(indexDirectory, config)) {
      for (int i = 0; i < numDocs; i++) {
        Document document = new Document();
        document.add(new StringField(ID_FIELD, Integer.toString(i), Field.Store.YES));
        document.add(new KnnFloatVectorField(VECTOR_FIELD, dataset[i], EUCLIDEAN));
        indexWriter.addDocument(document);
      }
    }

    // Searching
    try (Directory indexDirectory = FSDirectory.open(indexDirPath);
        DirectoryReader reader = DirectoryReader.open(indexDirectory)) {
      IndexSearcher searcher = new IndexSearcher(reader);
      float[] queryVector = generateDataset(random, 1, dimension)[0];
      KnnFloatVectorQuery query = new KnnFloatVectorQuery(VECTOR_FIELD, queryVector, topK);
      TopDocs results = searcher.search(query, topK);

      for (ScoreDoc scoreDoc : results.scoreDocs) {
        Document document = searcher.storedFields().document(scoreDoc.doc);
        System.out.println("doc " + document.get(ID_FIELD) + " score=" + scoreDoc.score);
      }
    }
  }

  private static float[][] generateDataset(Random random, int size, int dimensions) {
    float[][] dataset = new float[size][dimensions];
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < dimensions; j++) {
        dataset[i][j] = random.nextFloat() * 100;
      }
    }
    return dataset;
  }
}
```

Substituting `CuVSCodecs.acceleratedHNSWScalarQuantized` or `CuVSCodecs.acceleratedHNSWBinaryQuantized` is the only change needed to use a quantized variant; both take the same [`AcceleratedHNSWParams`](/api-reference/lucene-api-com-nvidia-cuvs-lucene-acceleratedhnswparams).

This program is a condensed version of `AcceleratedHnswExample`. That example and a GPU search one live in the [`examples/java/cuvs-lucene`](https://github.com/NVIDIA/cuvs/tree/main/examples/java/cuvs-lucene) directory of the NVIDIA cuVS repository, which the build compiles through `./build.sh lucene --build-java-examples`.

## Searching on the GPU

The GPU search codec, from `CuVSCodecs.gpuSearch`, runs both index build and search on the GPU. Configure it with [`GPUSearchParams`](/api-reference/lucene-api-com-nvidia-cuvs-lucene-gpusearchparams). A stock `KnnFloatVectorQuery` already searches on the GPU against this codec, so existing query code keeps working; [`GPUKnnFloatVectorQuery`](/api-reference/lucene-api-com-nvidia-cuvs-lucene-gpuknnfloatvectorquery) adds control over the CAGRA search parameters and lets cuVS serve all segments in a single request, which is usually faster. This program is the GPU counterpart of the quickstart, with a filter applied to the search:

```java
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import com.nvidia.cuvs.lucene.CuVSCodecs;
import com.nvidia.cuvs.lucene.GPUKnnFloatVectorQuery;
import com.nvidia.cuvs.lucene.GPUSearchParams;
import com.nvidia.cuvs.spi.CuVSProvider;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

public class GpuSearchQuickstart {

  private static final String ID_FIELD = "id";
  private static final String VECTOR_FIELD = "vector_field";
  private static final String STATUS_FIELD = "status";

  public static void main(String[] args) throws Exception {
    // Select the process-wide RMM allocator before creating any cuVS resources or codecs.
    CuVSProvider.provider().enableRMMAsyncMemory();

    int numDocs = 2000;
    int dimension = 32;
    int topK = 5;

    Random random = new Random(222);
    float[][] dataset = generateDataset(random, numDocs, dimension);
    Path indexDirPath = Files.createTempDirectory("cuvs-lucene-gpu-search");

    GPUSearchParams params = new GPUSearchParams.Builder().build();
    Codec codec = CuVSCodecs.gpuSearch(params);
    IndexWriterConfig config = new IndexWriterConfig().setCodec(codec);

    // Indexing
    try (Directory indexDirectory = FSDirectory.open(indexDirPath);
        IndexWriter indexWriter = new IndexWriter(indexDirectory, config)) {
      for (int i = 0; i < numDocs; i++) {
        Document document = new Document();
        document.add(new StringField(ID_FIELD, Integer.toString(i), Field.Store.YES));
        document.add(new KnnFloatVectorField(VECTOR_FIELD, dataset[i], EUCLIDEAN));
        document.add(
            new StringField(STATUS_FIELD, i % 2 == 0 ? "active" : "in-active", Field.Store.YES));
        indexWriter.addDocument(document);
      }
    }

    // Searching, restricted to the documents matching the filter
    try (Directory indexDirectory = FSDirectory.open(indexDirPath);
        DirectoryReader reader = DirectoryReader.open(indexDirectory)) {
      IndexSearcher searcher = new IndexSearcher(reader);
      float[] queryVector = generateDataset(random, 1, dimension)[0];
      Query filter = new TermQuery(new Term(STATUS_FIELD, "active"));
      KnnFloatVectorQuery query =
          new GPUKnnFloatVectorQuery(VECTOR_FIELD, queryVector, topK, filter, topK, 1);
      TopDocs results = searcher.search(query, topK);

      for (ScoreDoc scoreDoc : results.scoreDocs) {
        Document document = searcher.storedFields().document(scoreDoc.doc);
        System.out.println("doc " + document.get(ID_FIELD) + " score=" + scoreDoc.score);
      }
    }
  }

  private static float[][] generateDataset(Random random, int size, int dimensions) {
    float[][] dataset = new float[size][dimensions];
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < dimensions; j++) {
        dataset[i][j] = random.nextFloat() * 100;
      }
    }
    return dataset;
  }
}
```

With this codec both index build and search execute on the GPU, including queries that carry a filter and segments that contain deleted documents. Unlike the accelerated HNSW codecs it has no CPU path, so a GPU must be available wherever the index is written or queried.

## Tuning Index Builds

Both `AcceleratedHNSWParams` and `GPUSearchParams` default to a `HEURISTIC` strategy that lets cuVS derive the CAGRA build parameters, and this is the recommended setting. Under `HEURISTIC`, express the quality and cost trade-off in familiar terms and let cuVS translate it:

- For the accelerated HNSW codecs, set `maxConn` and `beamWidth`, the HNSW parameters you would tune on the CPU. cuVS derives graph degrees and the build algorithm from them. These two values also configure the CPU fallback writer, so one setting covers both paths.
- For the GPU search codec, set `buildQuality`. Higher values spend more build time for a higher-quality graph. This codec also passes `graphDegree` into the heuristic, so leave it at its default unless you intend to cap the graph.

Increasing `writerThreads` raises index build concurrency. The accelerated HNSW codecs default to a single writer thread, while the GPU search codec defaults to 32.

Switching either class to the `CUSTOM` strategy exposes the underlying CAGRA parameters directly, including `graphDegree`, `intermediateGraphDegree`, the graph build algorithm, and its parameters. Use `CUSTOM` only when you have measurements that justify specific values; the defaults derived by cuVS are a better starting point. For background on the parameters themselves, see the [CAGRA indexing guide](/user-guide/api-guides/indexing-guide/cagra) and the [tuning guide](/getting-started/introduction/tuning-indexes).

## Managing GPU Resources

A few settings matter once `cuvs-lucene` runs in a long-lived, multi-threaded application such as a search server.

**Device memory allocator.** Applications using the GPU search codec can opt into RMM's stream-ordered asynchronous device allocator during startup:

```java
CuVSProvider.provider().enableRMMAsyncMemory();
```

Enable the allocator before creating any cuVS resources, codecs, writers, or readers. It applies to the whole process on the current CUDA device, so it is an application-level choice rather than a codec setting. Enabling it changes performance only, never results: it helps workloads that allocate device memory repeatedly, such as a server running many searches at once. Without it, cuVS uses RMM's default allocator.

**Search workspace memory.** Each thread that issues searches gets its own GPU workspace. Set the `com.nvidia.cuvs.workspacePoolSize` system property, in bytes, to give those threads a dedicated memory pool. There is no default pool: when the property is unset or `0`, threads use cuVS's default workspace resource instead. The pool is per thread and index-writer threads use the same resources, so total device memory scales with the number of threads searching and building concurrently, including `writerThreads`.

Applications that need more control can install their own resources handle rather than relying on the property:

```java
long poolBytes = 512L * 1024 * 1024;
try {
  CuVSResources resources = CuVSResources.create();
  resources.setWorkspacePool(poolBytes);
  ThreadLocalCuVSResourcesProvider.setCuVSResourcesInstance(resources);
} catch (Throwable t) {
  // CuVSResources.create() declares throws Throwable; cuVS may be unavailable here.
}
```

Install the handle on each search thread before that thread runs its first search. A later call still replaces the handle, but it does not close the one the thread already holds, which strands that handle and its pinned host buffer for the life of the process. [`ThreadLocalCuVSResourcesProvider`](/api-reference/lucene-api-com-nvidia-cuvs-lucene-threadlocalcuvsresourcesprovider) hands out one [`CuVSResources`](/api-reference/java-api-com-nvidia-cuvs-cuvsresources) per thread, and an application that supplies its own owns the lifecycle, closing it with `closeCuVSResourcesInstance()`.

**Filter bitsets.** Filtered GPU searches upload an acceptance mask per segment. Two independent settings control this:

- [`FilterBitsetCacheConfig`](/api-reference/lucene-api-com-nvidia-cuvs-lucene-filterbitsetcacheconfig) caches the packed host-side bitsets for one vectors-format instance, with a 128 MiB budget by default. Pass a custom configuration to `CuVSCodecs.gpuSearch` to change or disable it. The budget is a retention cap, not a preallocation.
- The `com.nvidia.cuvs.filterBitsetPoolSize` system property sizes the process-wide RMM device pool that backs filter bitset uploads, defaulting to a 4 MiB initial reservation that can grow. Set it to `0` to disable pooling.

## Troubleshooting

**Index builds run on the CPU on a GPU host.** The accelerated HNSW codecs fall back to CPU construction whenever cuVS resources cannot be created. The fallback is logged at `WARNING` level, starting with `GPU based indexing not supported, falling back to using the`; the rest of the message names the stock Lucene format that took over, which differs per codec. Check the application log first, then confirm that the directory containing `libcuvs.so` and `libcuvs_c.so` is on `LD_LIBRARY_PATH`.

**An index written with these codecs will not open.** Lucene looks codecs up by name through its service loader, so a reader without `cuvs-lucene` on its classpath fails with an SPI lookup error naming the missing codec. Deploy the same `cuvs-lucene` and `cuvs-java` jars to search nodes as to indexing nodes.

**`UnsupportedOperationException: cuVS is not supported`.** The GPU search codec raises this when cuVS resources cannot be created, on both the indexing and the search path. The causes are the same as above; unlike the accelerated HNSW codecs, this codec cannot continue without a GPU.

**`IllegalStateException: cuvs-lucene-10.X is built for Lucene 10.X.x, but Lucene 10.Y.z is in use`.** The `cuvs-lucene` artifact on the classpath was built for another Lucene minor release than the one the application uses. Lucene itself keeps working, and so do indexes that do not use the cuvs-lucene codecs; creating a cuvs-lucene codec, or reading or writing an index that uses one, fails with this message. The same problem is logged at `SEVERE` the first time Lucene looks up its codecs, usually at startup. Applications that would rather refuse to start can call `CuVSCodecs.checkLuceneVersion()` during startup, which throws this exception. Replace the artifact with the one the message names, if it exists; see [Lucene Versions and Upgrades](#lucene-versions-and-upgrades).

**`UnsupportedOperationException: The ... codec can only read indexes`.** The application writes with the codec of an earlier Lucene release, which the current release can only read. Create the codec through `CuVSCodecs`, or switch to the codec for the current release.

**`UnsupportedOperationException: Lucene99AcceleratedHNSWScalarQuantizedVectorsFormat can only read indexes`.** On Lucene 10.4 and later, the application writes with the scalar-quantized vectors format of earlier releases. Use the format the message names, which `CuVSCodecs.acceleratedHNSWScalarQuantizedFormat()` returns; see [Lucene Versions and Upgrades](#lucene-versions-and-upgrades).

**Native libraries do not match.** A missing or mismatched native cuVS library is caught during resource creation and logged at `WARNING` rather than thrown, so it surfaces as one of the two symptoms above rather than as a load error. Make sure the installed native libraries match the `cuvs-lucene` version; see [cuVS Lucene](/installation/java#cuvs-lucene).

**Errors about native access, or indexing never uses the GPU.** `cuvs-lucene` is compiled for Java 21, but `cuvs-java` reaches the GPU through the Panama FFI APIs, which need a JDK 22 or newer runtime on Linux (`amd64` or `aarch64`). On JDK 21, cuVS reports that it is not supported: the accelerated HNSW codecs build graphs on the CPU, and the GPU search codec throws `UnsupportedOperationException`. Run the JVM with `--enable-native-access=ALL-UNNAMED` to suppress native-access warnings, or `--enable-native-access=com.nvidia.cuvs` if you put `cuvs-java` on the module path.

**GPU search is slower than expected.** A stock `KnnFloatVectorQuery` searches each segment separately; switching to `GPUKnnFloatVectorQuery` lets cuVS serve all segments in one request. Indexes with many small segments also benefit from force-merging to fewer, more evenly sized segments.
