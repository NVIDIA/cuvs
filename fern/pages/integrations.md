# Integrations

NVIDIA cuVS can be adopted at several layers of a vector-search stack. Applications can call NVIDIA cuVS directly, use a database or search engine with NVIDIA cuVS-backed indexing, rely on data platforms that bring NVIDIA accelerated computing closer to enterprise data, or use libraries that expose NVIDIA cuVS algorithms through familiar APIs.

This page summarizes where each integration fits and links to vendor documentation for setup, supported configurations, and operational details.

## Databases

Use these integrations when vector search should be managed by a database or search engine. The system owns ingestion, indexing, query APIs, and operations, while NVIDIA cuVS-backed paths accelerate supported index build or search workflows.

### Amazon OpenSearch Service

[Amazon OpenSearch Service](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/gpu-acceleration-vector-index.html) provides managed GPU acceleration for vector indexing on OpenSearch 3.1+ domains and OpenSearch Serverless vector collections. OpenSearch Service detects supported Faiss vector index builds, offloads the build work to managed GPU capacity, and applies acceleration to indexing and force-merge operations without requiring users to manage GPU instances. See the AWS docs for [enabling GPU acceleration](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/gpu-acceleration-enabling.html), [creating GPU-accelerated vector indexes](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/gpu-acceleration-creating-indexes.html), and [indexing vector data](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/gpu-acceleration-index-force-merge.html).

### CyborgDB

[CyborgDB](https://docs.cyborg.co/) is an encrypted vector database proxy for confidential vector search across backing stores such as PostgreSQL, Redis, S3, and in-memory storage. Cyborg and NVIDIA demonstrated a proof of concept accelerated by NVIDIA cuVS that speeds encrypted vector index build and retrieval while preserving CyborgDB's confidentiality model. See the CyborgDB [overview](https://docs.cyborg.co/versions/v0.16.x/intro/about), [quickstart](https://docs.cyborg.co/versions/v0.16.x/intro/quickstart), [embedded deployment docs](https://docs.cyborg.co/embedded), and NVIDIA's [technical blog about Cyborg and NVIDIA cuVS](https://developer.nvidia.com/blog/bringing-confidentiality-to-vector-search-with-cyborg-and-nvidia-cuvs/).

### Elasticsearch

[Elasticsearch GPU accelerated vector indexing](https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/gpu-vector-indexing) uses NVIDIA cuVS to accelerate dense-vector HNSW index construction. This path is intended for ingestion-heavy workloads where CPU HNSW graph construction is a bottleneck. Elastic's implementation builds graph structures on the GPU and converts them for Elasticsearch search workflows. See Elastic's [GPU vector indexing reference](https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/gpu-vector-indexing), plus the GPU acceleration writeups for [chapter 1](https://www.elastic.co/search-labs/blog/gpu-accelerated-vector-search-elasticsearch-nvidia/) and [chapter 2](https://www.elastic.co/search-labs/blog/elasticsearch-gpu-accelerated-vector-indexing-nvidia).

### Kinetica

[Kinetica](https://docs.kinetica.com/7.2/vector_search/) is a GPU-accelerated database with native vector columns, SQL vector operators, and Python APIs for vector search. Kinetica supports CAGRA indexes on vector columns for GPU-oriented approximate nearest-neighbor search, while HNSW remains the automatically maintained option for mutable vector data. See Kinetica's [CAGRA index docs](https://docs.kinetica.com/7.2/concepts/indexes/#cagra-index), [NVIDIA partner page](https://www.kinetica.com/partner/nvidia/), [vector I/O example](https://github.com/kineticadb/examples/blob/master/python_dev_guide/python_vector_io.ipynb), and [vector search example](https://github.com/kineticadb/examples/blob/master/python_dev_guide/python_vector_search.ipynb).

### KX / KDB.AI

[KDB.AI](https://code.kx.com/kdbai/integrations/nvidia-cuvs-cagra.html) is KX's vector database for AI applications and similarity search workflows. The NVIDIA cuVS-enabled KDB.AI Server image, `kdbai-db-cuvs`, bundles the GPU dependencies needed to build and search CAGRA indexes with NVIDIA cuVS while keeping the standard KDB.AI client APIs. KDB.AI also integrates with [kdb+](https://code.kx.com/kdbai/integrations/kdb.html), allowing existing q and kdb+ datasets to participate in vector search workflows. See KX's [kdb+ product page](https://kx.com/products/kdb/) and [NVIDIA partner page](https://kx.com/partners/nvidia/) for broader KX and NVIDIA integration context.

### Milvus

[Milvus GPU indexes](https://milvus.io/docs/gpu_index.md) provide GPU-accelerated options for high-throughput and high-recall vector search. Supported index types include [`GPU_CAGRA`](https://milvus.io/docs/gpu-cagra.md), [`GPU_IVF_FLAT`](https://milvus.io/docs/gpu-ivf-flat.md), [`GPU_IVF_PQ`](https://milvus.io/docs/gpu-ivf-pq.md), and [`GPU_BRUTE_FORCE`](https://milvus.io/docs/gpu-brute-force.md). `GPU_CAGRA` also supports hybrid deployments where a GPU-built graph can be adapted at load time for CPU search in Milvus 2.6.4 and newer. See the Milvus docs for [building indexes](https://milvus.io/docs/build_index.md) and [installing Milvus with GPU support](https://milvus.io/docs/install_standalone-docker-compose-gpu.md).

### OpenSearch

[OpenSearch Vector Engine](https://opensearch.org/platform/vector-search/) supports vector search, hybrid search, and RAG-oriented retrieval in OpenSearch. The OpenSearch GPU indexing design offloads vector index builds to GPU workers, uses NVIDIA cuVS through Faiss to build CAGRA graphs, converts those graphs into an HNSW-compatible form for CPU search, and falls back to CPU index building when needed. See the OpenSearch [vector search docs](https://docs.opensearch.org/latest/vector-search/), [GPU-accelerated vector search blog](https://opensearch.org/blog/GPU-Accelerated-Vector-Search-OpenSearch-New-Frontier/), k-NN RFCs for [GPU acceleration](https://github.com/opensearch-project/k-NN/issues/2293) and [remote vector index build](https://github.com/opensearch-project/k-NN/issues/2294), and Amazon OpenSearch Service [GPU acceleration docs](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/gpu-acceleration-vector-index.html).

### Oracle AI Database 26ai

[Oracle AI Database 26ai](https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/) includes Oracle AI Vector Search for storing embeddings, creating vector indexes, and running similarity search inside Oracle Database. Oracle's [Vector Index Service](https://www.oracle.com/database/private_ai_services_container/vector-indexes/) uses GPU-enabled containers and NVIDIA cuVS to generate Oracle In-Memory Neighbor Graph vector indexes, then returns the index to Oracle AI Database for query execution. See Oracle's [NVIDIA collaboration announcement](https://blogs.oracle.com/database/oracle-ai-database-nvidia-collaboration-advances-enterprise-ai-at-nvidia-gtc-2026), [Vector Index Service introduction](https://blogs.oracle.com/database/introducing-the-vector-index-service), and [getting started guide](https://blogs.oracle.com/database/getting-started-with-the-vector-index-service).

### Solr

[Apache Solr dense vector search](https://solr.apache.org/guide/solr/latest/query-guide/dense-vector-search.html#gpu-acceleration) can use NVIDIA cuVS through the `cuvs-lucene` vector format. Solr exposes this through `CuVSCodecFactory` and the [`CuVSCodec`](https://solr.apache.org/docs/10_0_0/modules/cuvs/org/apache/solr/cuvs/CuVSCodec.html) module, allowing CAGRA graph construction on GPU and serialization into an HNSW graph for search. The Solr guide covers setup steps, including the `cuvs` module jars, `knnAlgorithm="cagra_hnsw"` on `DenseVectorField`, and codec configuration in `solrconfig.xml`.

## Data Platforms

Use these integrations when data locality, storage throughput, and RAG pipeline architecture are central to the deployment. These platforms usually do not expose NVIDIA cuVS APIs directly; they bring NVIDIA accelerated computing, networking, and AI software closer to enterprise data through the [NVIDIA AI Data Platform](https://nvidianews.nvidia.com/news/nvidia-and-storage-industry-leaders-unveil-new-class-of-enterprise-infrastructure-for-the-age-of-ai) reference design.

### Cloudian

[Cloudian HyperScale AI Data Platform](https://cloudian.com/nvidia/) combines HyperStore S3-compatible object storage with NVIDIA accelerated computing and NVIDIA AI Enterprise software for on-premises AI factories. Cloudian positions the platform for agentic RAG, semantic search, and enterprise knowledge retrieval, with integrated vector database capabilities for ingesting, embedding, and indexing multimodal content. See Cloudian's [platform launch](https://cloudian.com/press/cloudian-launches-aidp-ai-platform), [AI inferencing overview](https://cloudian.com/blog/cloudian-ai-inferencing-platform/), and [NVIDIA integration page](https://cloudian.com/nvidia/).

### DDN

[DDN Infinia](https://www.ddn.com/products/infinia/) is DDN's data platform for AI workflows across core, cloud, and edge environments. In DDN's NVIDIA AI Data Platform example, Infinia is paired with NVIDIA NIM microservices, NVIDIA Spectrum-X, BlueField DPUs, and Milvus to support RAG, vector search, and inference-serving pipelines. See DDN's [RAG workflow writeup](https://www.ddn.com/blog/ddn-and-nvidia-power-the-ai-data-platform-for-fast-accurate-and-scalable-enterprise-rag/) and [Data Intelligence Platform overview](https://www.ddn.com/products/data-intelligence-platform/).

### Dell AI Data Platform

[Dell AI Data Platform with NVIDIA](https://www.dell.com/en-us/shop/artificial-intelligence/sc/ai-data-platform) combines Dell storage, modular data engines, NVIDIA accelerated computing, networking, and NVIDIA AI Enterprise software for enterprise AI data pipelines. Dell describes a GPU Accelerated Data Search Engine that applies NVIDIA cuVS to vector indexing and search over unstructured data, alongside NVIDIA cuDF for data processing and Dell data orchestration for preparing governed AI-ready datasets. See Dell's [platform launch blog](https://www.dell.com/en-us/blog/ai-at-scale-starts-with-your-data-introducing-the-supercharged-dell-ai-data-platform-with-nvidia/), [press release](https://www.dell.com/en-us/dt/corporate/newsroom/announcements/detailpage.press-releases~usa~2026~03~dell-ai-data-platform-with-nvidia-supercharges-enterprise-ai-with-breakthrough-data-orchestration-and-storage-innovations.htm), and [GPU-fed AI data platform overview](https://www.dell.com/en-us/blog/how-dell-s-ai-data-platform-keeps-your-gpus-fed/).

### MinIO

[MinIO AIStor](https://www.min.io/partners/nvidia) is an S3-compatible object data platform for NVIDIA AI Factory and NVIDIA STX deployments. In vector-search and RAG pipelines, AIStor provides the durable object layer for embeddings, segment objects, index artifacts, and retrieval data while GPU-accelerated services such as Milvus and NVIDIA cuVS perform index construction and search. MinIO has published a [Milvus and NVIDIA cuVS benchmark](https://www.min.io/blog/accelerating-vector-indexing-with-minio-aistor-milvus-and-nvidia-cuvs) showing how AIStor, NVIDIA GPUDirect RDMA for S3-compatible storage, and NVIDIA cuVS fit together in large-scale vector index creation. See also MinIO's [NVIDIA STX announcement](https://www.min.io/press/minio-aistor-nvidia-stx-ai-factory) and [GPUDirect RDMA overview](https://www.min.io/blog/minio-aistor-with-nvidia-gpudirect-r-rdma-for-s3-compatible-storage-unlocking-performance-for-ai-factory-workloads).

### NetApp

[NetApp AIPod](https://www.netapp.com/nvidia/) supports NVIDIA AI Data Platform deployments with NetApp ONTAP data management, scalable storage, and NVIDIA accelerated computing. The integration is designed for governed RAG and inference pipelines that scan, index, classify, and retrieve enterprise documents for AI agents. See NetApp's [AI Data Platform announcement](https://www.netapp.com/newsroom/press-releases/news-rel-20250518-700517/) and [AI infrastructure documentation](https://docs.netapp.com/us-en/netapp-solutions-ai/ai-infrastructures/).

### Everpure

[Everpure FlashBlade](https://www.purestorage.com/company/newsroom/press-releases/pure-storage-integrates-nvidia-ai-data-platform-into-flashblade.html?Social+Account=Pure+Storage+%28Default%29) integrates with the NVIDIA AI Data Platform reference design for agentic AI, RAG, and inference workflows. Everpure positions FlashBlade as a high-throughput storage layer for NVIDIA accelerated compute and NVIDIA AI Enterprise software, while [FlashBlade//EXA](https://www.pure.ai/flashblade-exa.html) targets large-scale AI and HPC workloads that need high metadata performance and low-latency access to multimodal data.

### WEKA

[WEKA Data Platform](https://www.weka.io/company/weka-newsroom/press-releases/weka-unveils-nvidia-integrations-and-certifications-at-gtc-2025/) integrates with the NVIDIA AI Data Platform reference design to provide a high-performance storage foundation for agentic AI reasoning and inference. WEKA's NVIDIA work includes Augmented Memory Grid, NVIDIA Cloud Partner certification, NVIDIA-Certified Systems Storage validation, and DGX reference architectures. See WEKA's [NVIDIA Cloud Partner certification](https://www.weka.io/company/weka-newsroom/press-releases/weka-achieves-nvidia-cloud-network-partner-certification/) and [DGX BasePOD reference architecture](https://www.weka.io/resources/reference-architecture/scaling-deep-learning-with-weka-and-dgx-a100-basepod/).

## Libraries

Use these integrations when you want library-level control inside an application or service. These options expose familiar APIs while letting developers use NVIDIA cuVS-backed indexing or search paths where supported.

### Faiss

[Faiss](https://github.com/facebookresearch/faiss) integrates NVIDIA cuVS as an optional backend for GPU vector indexes. The NVIDIA cuVS-backed path keeps the Faiss API model while accelerating supported GPU indexes such as `GpuIndexFlat`, `GpuIndexIVFFlat`, `GpuIndexIVFPQ`, and `GpuIndexCagra`. Faiss can also move between CPU and GPU indexes, including converting GPU-built CAGRA indexes into HNSW-compatible CPU indexes with `IndexHNSWCagra`. See the [Faiss installation guide](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md), [GPU Faiss with NVIDIA cuVS](https://github.com/facebookresearch/faiss/wiki/GPU-Faiss-with-cuVS), [NVIDIA cuVS usage guide](https://github.com/facebookresearch/faiss/wiki/GPU-Faiss-with-cuVS-usage), and [example notebook](https://github.com/facebookresearch/faiss/blob/main/tutorial/python/10-FaissCuvsExample.ipynb).

### KIOXIA AiSAQ

[KIOXIA AiSAQ](https://github.com/kioxia-jp/aisaq-diskann) is an SSD-oriented ANN library based on DiskANN and Vamana for vector collections that are too large to keep entirely in DRAM. AiSAQ can use NVIDIA cuVS to accelerate index build, including Vamana graph construction and k-means workflows. It is also available as a disk-based vector index in [Milvus 2.6.4 and newer](https://milvus.io/docs/aisaq.md). See KIOXIA's [4.8B-vector scaling demo](https://americas.kioxia.com/en-us/business/news/2026/ssd-20260316-2.html), [technical blog](https://blog-us.kioxia.com/post/2026/03/16/kioxia-aisaq-achieves-4-8-billion-high-dimensional-vector-search-on-a-single-server-with-7-8x-index-build-time-acceleration-via-gpus), and the NVIDIA cuVS [Vamana indexing guide](/user-guide/api-guides/indexing-guide/vamana).

### Lucene

[NVIDIA cuVS Lucene](https://github.com/rapidsai/cuvs-lucene) provides a Lucene `KnnVectorFormat` that lets Java search applications use NVIDIA cuVS through Lucene codecs. The package is published as [`com.nvidia.cuvs.lucene:cuvs-lucene`](https://central.sonatype.com/artifact/com.nvidia.cuvs.lucene/cuvs-lucene) and builds on the NVIDIA cuVS Java APIs. The integration targets GPU-accelerated vector indexing and search paths for Lucene-based systems, including CAGRA graph construction, filtering, index merge support, and off-heap data movement. See the SearchScale and NVIDIA writeup on [Apache Lucene accelerated with NVIDIA cuVS](https://searchscale.com/blog/apache-lucene-accelerated-with-nvidia-cuvs-25.06-release/).
