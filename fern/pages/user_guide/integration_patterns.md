---
slug: user-guide/field-guide/integration-patterns
---

# Integration Patterns

NVIDIA cuVS is used in several different ways across vector databases, search engines, data platforms, and application libraries. Some products call NVIDIA cuVS directly inside the same process. Others offload expensive index builds to a separate service, container, or serverless worker, then load the resulting index back into the serving system.

The right pattern depends on where the product wants to spend GPU time, how it manages upgrades, and whether search should run on GPU, CPU, or both. For a list of specific products, see the [Integrations](/getting-started/integrations) page.

## Direct library integration

In a direct integration, the product links to NVIDIA cuVS and calls the NVIDIA cuVS APIs from the same process that owns indexing or query execution. This gives the product the most control over memory, resources, batching, and index lifecycle.

Direct integrations work well when the host application already controls GPU resources or when the integration is library-oriented. [Faiss](/getting-started/integrations#faiss) can use NVIDIA cuVS-backed GPU indexes while preserving familiar Faiss APIs. [Milvus](/getting-started/integrations#milvus) exposes NVIDIA cuVS-backed GPU indexes such as CAGRA, IVF-Flat, IVF-PQ, and brute-force through database configuration. [NVIDIA cuVS Lucene](/getting-started/integrations#lucene) integrates NVIDIA cuVS with Lucene-style vector formats so Lucene-based systems can use GPU-accelerated indexing paths.

This pattern usually gives the lowest integration overhead, but it also means the product must manage GPU availability, NVIDIA cuVS runtime packaging, memory limits, and compatibility with the rest of its process.

## Offloaded index builds

In an offloaded build pattern, the database or search engine keeps its normal serving path, but sends expensive vector index construction to a separate GPU-enabled process. The build worker creates or accelerates the index, writes an artifact, and returns that artifact to the serving system.

This is a good fit when indexing is expensive, but query serving should remain in the product's existing runtime. [Oracle AI Database 26ai](/getting-started/integrations#oracle-ai-database-26ai) uses a Vector Index Service with GPU-enabled containers to build vector indexes outside the database, then returns the result to Oracle AI Database. [OpenSearch](/getting-started/integrations#opensearch) describes remote GPU index build workers that can build NVIDIA cuVS-backed CAGRA graphs and convert them for CPU search. [Amazon OpenSearch Service](/getting-started/integrations#amazon-opensearch-service) provides managed GPU acceleration for supported vector indexing workflows, including OpenSearch Serverless vector collections.

This pattern separates GPU build capacity from CPU-oriented serving capacity. It can simplify operations for managed services and serverless deployments because the serving fleet does not need to keep GPUs attached for every query.

## Hybrid GPU-build and CPU-search

Some integrations use GPUs for the part of the workflow where NVIDIA cuVS is most valuable, then serve queries with a CPU-native index format. A common example is building a CAGRA graph quickly on GPU, converting it to an HNSW-compatible graph, and serving with an existing CPU search stack.

This pattern is useful when ingest or index refresh time is the bottleneck, but the product already has mature CPU search infrastructure. It also lets a product adopt GPU acceleration incrementally without replacing its full query-serving path.

## ABI-stable C API integration

Products that integrate at the binary level should prefer the NVIDIA cuVS C APIs when they need ABI stability. The C shared library provides a stable runtime contract across compatible NVIDIA cuVS releases, which helps downstream applications load newer compatible NVIDIA cuVS runtimes without being rebuilt.

ABI stability is especially useful for databases, search engines, language bindings, and packaged applications. It allows vendors to build against one compatible NVIDIA cuVS release while giving users or package managers flexibility to install a newer runtime from the same ABI compatibility window.

For compatibility rules, release windows, and shared library naming, see [Compatibility](/user-guide/field-guide/compatibility).

## Choosing a pattern

| Pattern | Best fit | Examples |
| --- | --- | --- |
| Direct library integration | Products that own GPU resources and want tight control over index build or search | Faiss, Milvus, NVIDIA cuVS Lucene |
| Offloaded index builds | Databases or services that want GPU acceleration for indexing while keeping serving separate | Oracle AI Database 26ai, OpenSearch, Amazon OpenSearch Service |
| Hybrid GPU-build and CPU-search | Systems that need faster index construction but want to keep CPU search infrastructure | CAGRA-to-HNSW workflows |
| ABI-stable C API integration | Products that need binary compatibility across compatible NVIDIA cuVS runtime versions | Databases, search engines, language bindings, packaged applications |
