# What is a Vector Database?

Vector search indexes and vector databases solve related but different problems. A vector search index is the algorithmic structure used to find nearest neighbors. A vector database is a production system that stores vectors, manages writes, handles failures, scales across machines, and uses one or more vector search indexes internally.

For guidance on choosing an index, see [What is Vector Search?](/getting-started/introduction/vector-search).

If you are looking for NVIDIA cuVS-powered vector databases and search engines, see the [Databases section of Integrations](/getting-started/integrations#databases).

This page explains how vector databases use indexes internally, why local and global partitioning behave differently, and how hybrid designs combine both approaches. By the end, you should understand which database architecture you are tuning and why that changes the search, ingestion, and compaction tradeoffs.

## Quick comparison

| Capability | Vector search index | Vector database |
| --- | --- | --- |
| Primary job | Find nearest neighbors quickly | Store, query, update, and operate vector data in production |
| Examples | NVIDIA cuVS, FAISS, HNSW libraries | Milvus, Vespa, Elasticsearch, MongoDB, OpenSearch |
| Scope | Build and search an index | Durability, metadata, filtering, replication, sharding, APIs, operations |
| Tuning focus | Recall, latency, throughput, memory, build time | Index tuning plus ingestion, consistency, scaling, and operational constraints |
| Best fit | Embedded search, custom systems, benchmarks, controlled pipelines | Applications that need database features around vector search |

Standalone index libraries are often closer to machine learning libraries than databases. They expose build and search parameters, can often serialize indexes to disk, and may support filtering or multi-GPU execution. They usually do not provide durability, replication, distributed query planning, or operational management by themselves.

Vector databases use vector search indexes as one component inside a larger system. They may also support metadata filters, hybrid text and vector search, access control, background compaction, and horizontal scaling.

## How databases use vector indexes

Vector databases usually organize data in one of two ways. The distinction matters because it changes how you choose index parameters.

### Locally partitioned indexes

Most vector databases split data into local partitions, often with hash partitioning, also called blind sharding. This is the most trivial and naive way to scale out because each partition can keep its own vector index without understanding the global vector space. The cost is paid during search: each query usually needs to visit every partition, then merge the partial results.

<img alt="A locally partitioned vector database fans each query out to every local partition and merges the partial top-k results." src="/assets/images/locally_partitioned_index.svg" />

This pattern is common in segment- or LSM-style systems: Lucene-backed search engines create many immutable searchable segments, and Milvus builds indexes per sealed segment.

For tuning, think about the size of each local partition, not only the total database size. For example, if a database stores 100M vectors as ten partitions of about 10M vectors each, IVF parameters such as `n_lists` should usually be chosen for a 10M-vector local index.

### Globally partitioned indexes

Some systems train a global partitioning structure across the whole dataset. This is often called an inverted file index, or IVF, and can also be described as semantic partitioning. The system pays an upfront cost to preserve spatial locality by grouping similar vectors into partitions, usually with centroid-based clustering such as k-means.

<img alt="A globally partitioned vector database routes each query through a global partitioning layer, searches only relevant partitions, and merges the partial top-k results." src="/assets/images/globally_partitioned_index.svg" />

The search benefit can be large: a query can visit only the most relevant partitions instead of every local shard. This allows larger-scale indexes and can make hot partitions easier to cache. Some systems organize centroids in a tree, graph, or ANN structure so new vectors can be assigned quickly and partitions can be rebalanced as data distributions change.

Turbopuffer's SPFresh index and LanceDB's IVF-based indexes are examples of systems designed around centroid or global partitioning.

### Hybrid Local & Global Partitioned

A database can also implement global partitioning inside a locally partitioned architecture. In many local designs, partitions are allocated as data arrives. These partitions are often called segments. A working segment grows until it reaches a size or time threshold, then background compaction merges several segments into larger segments to reduce the number of files on disk and the number of segments searched per query.

In a hybrid design, the long-lived segments are allocated up front, one per global cluster or partition. New data can still land in one or a few local queue segments for fast ingestion. During compaction, vectors in those queue segments are assigned to global clusters and moved into the corresponding global segments. This keeps the operational benefits of segment-based ingestion while giving search the locality benefits of global partitioning.

In an IVF-style design, a partition does not need to be a flat list of vectors. Each partition can contain another ANN index, including graph-based indexes such as CAGRA, HNSW, or Vamana.

[KIOXIA AiSAQ](/getting-started/integrations#kioxia-aisaq) is an example of this hybrid model: it can use globally assigned partitions inside a segment-oriented database architecture, while using Vamana/DiskANN-style graph ANN inside each partition.

## Tuning implication

Vector indexes are often tuned like machine learning models: sample representative data, compute exact ground truth, and compare recall, latency, throughput, memory, and build time across parameter sweeps. Vector databases add system-level tuning questions: worker threads for ingest and search, minimum and maximum segment size, compaction frequency, flush frequency for freshness, and memory or disk budgets for cost targets. The best choices depend heavily on the database architecture.

Locally partitioned indexes are usually easier to tune for index quality and latency because each segment can be treated as a smaller local problem. Globally partitioned indexes are harder to tune at scale because parameters such as `n_lists` and `n_probes` depend on the full dataset size and vector distribution, so the problem cannot be broken into independent segments as easily.

Refer to [Tuning Indexes](/getting-started/introduction/tuning-indexes) for more detail.

## Conclusion

A vector database is more than a vector search index. It combines indexing with storage, ingestion, filtering, consistency, compaction, scaling, and operational controls.

The most important architectural question is how the database partitions vectors. Local partitioning is straightforward to operate, but each query may need to search every partition. Global partitioning pays more cost up front to preserve vector locality, which can reduce query work and improve cache behavior. Hybrid designs combine segment-oriented ingestion with globally assigned partitions to balance operational simplicity with better search efficiency.

When choosing or tuning a vector database, consider both algorithmic metrics such as recall and latency and system-level constraints such as freshness, memory, disk, compaction, and scale-out behavior.
