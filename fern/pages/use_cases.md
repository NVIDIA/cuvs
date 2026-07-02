# Vector Search Use-Cases

Vector search is a building block for finding nearby items in high-dimensional data. The vectors usually come from embedding models, feature extractors, simulations, sensors, logs, or other numeric representations. NVIDIA cuVS focuses on the accelerated algorithms behind those workflows: building indexes, searching indexes, clustering vectors, compressing vectors, and constructing neighborhood graphs.

Use a [vector database](/getting-started/introduction/vector-database) when the application needs ingestion, updates, metadata filtering, service APIs, durability, replication, and operations around vector search. Use NVIDIA cuVS directly when you need control over index construction, tuning, batch analysis, or integration into a custom system.

## Retrieval and search

*Semantic search* is the most common vector search use-case. Text, images, audio, video, molecules, products, or user profiles are embedded into vectors, then a query is embedded and matched against nearby items. This supports natural-language document search, image search, code search, product search, and cross-modal search where the query and result use different media types.

*Retrieval augmented generation*, or *RAG*, uses vector search to retrieve relevant context before generating an answer. A vector database is often useful here because production RAG systems need document ingestion, metadata filters, access control, index refreshes, and query APIs. NVIDIA cuVS can accelerate the index build and search paths used inside those systems or in custom RAG pipelines.

*Agentic AI systems* use vector search as memory, context retrieval, and tool-selection infrastructure. An agent can search prior conversations, task traces, documents, examples, code snippets, and tool descriptions to decide what context to use next. These workflows often need low-latency lookup, metadata filters, and frequent index refreshes as new memories or artifacts are created.

*Recommender systems* and *personalization workflows* use nearest-neighbor search to find similar users, products, sessions, or content. Vector search can power related-item recommendations, candidate generation before ranking, content-based recommendations, and personalization features that combine learned embeddings with business filters.

*Semantic deduplication* and *entity resolution* use vector search to find records that are nearly the same but not exact string matches. This is useful for catalog cleanup, repeated media detection, similar document grouping, customer record matching, and fraud or abuse workflows where related items may not share exact identifiers.

*Semantic caching* uses vector search to decide whether a new query is close enough to a previous query, prompt, response, or intermediate result to reuse cached work. This can reduce latency and cost in AI applications, especially when many users ask similar questions with different wording.

## Databases and search engines

Vector databases and search engines add operational features around vector search. Common use-cases include *enterprise knowledge retrieval*, *hybrid keyword and vector search*, *semantic product catalogs*, *multimodal asset search*, *agent memory*, *chatbot context retrieval*, *semantic caching*, and *real-time application search*.

These systems usually combine vector similarity with metadata filters, ranking rules, tenant boundaries, index lifecycle management, and distributed execution. In that setting, NVIDIA cuVS is usually the accelerated indexing and search layer, while the database owns serving, storage, writes, and query planning. See [Integrations](/getting-started/integrations) for systems that expose NVIDIA cuVS-backed workflows.

## Batch analysis and exploration

Vector search is also useful when search is not the final product. In *batch exploratory data analysis*, nearest-neighbor graphs and clustering algorithms help users understand the structure of a dataset before building an application.

[All-neighbors](/user-guide/api-guides/indexing-guide/all-neighbors) can build neighborhood graphs for *graph analytics*, *clustering*, *visualization*, and *manifold learning workflows*. A KNN graph often becomes the input to algorithms that need local connectivity rather than point lookups.

[K-Means](/user-guide/api-guides/clustering-guide/k-means) groups vectors around representative centroids. It is useful for segmentation, coarse partitioning, vector quantization, sampling, and summarizing large datasets. [Single-linkage](/user-guide/api-guides/clustering-guide/single-linkage) builds a hierarchy of merges, which is useful when the tree structure or dendrogram is part of the analysis. [Spectral clustering](/user-guide/api-guides/clustering-guide/spectral-clustering) works through graph structure and can separate clusters that are not well described by compact centroids.

*Manifold learning* and *dimensionality reduction workflows* use neighborhood structure to reveal lower-dimensional patterns. These workflows are common for *visualization*, *exploratory analysis*, *quality checks*, *dataset debugging*, and *feature inspection*. [PCA](/user-guide/api-guides/preprocessing-guide/pca) provides a fast linear reduction path, while neighborhood graphs can feed nonlinear manifold methods in adjacent RAPIDS workflows.

## Compression and preprocessing

Large vector datasets often need compression before they are practical to store or search. [Scalar Quantizer](/user-guide/api-guides/preprocessing-guide/scalar-quantizer), [Binary Quantizer](/user-guide/api-guides/preprocessing-guide/binary-quantizer), and [Product Quantization](/user-guide/api-guides/preprocessing-guide/product-quantization) reduce memory footprint and memory bandwidth. Compression is useful for *approximate scoring*, *storage reduction*, *index construction*, and *large-scale retrieval systems* where full-precision vectors are too expensive.

*Quantization* often pairs with *refinement* or *reranking*. A compressed index quickly finds candidates, then a second stage recomputes distances for a smaller candidate set using more accurate vectors. This preserves much of the speed and memory benefit while recovering recall.

## Model and data quality workflows

Vector search can be used to inspect *embedding quality*. Nearest-neighbor examples make it easier to see whether a model groups related items, separates unrelated items, or encodes unwanted shortcuts. Batch searches can expose *mislabeled examples*, *duplicated samples*, *outliers*, *sparse regions*, and *hard negative examples* for training.

*Clustering* and *graph analysis* can also help track *dataset drift*. If new embeddings form new clusters, move away from older clusters, or change neighbor relationships, that can signal a change in data distribution or model behavior.

## Choosing a starting point

Start with [Vector Search](/getting-started/introduction/vector-search) when you need to choose an index. Start with [Vector Database](/getting-started/introduction/vector-database) when you are deciding between a standalone index library and a managed database or search engine. Start with the [Clustering Guide](/user-guide/api-guides/clustering-guide/k-means) or [Preprocessing Guide](/user-guide/api-guides/preprocessing-guide/product-quantization) when the goal is batch analysis, compression, or dataset preparation rather than online retrieval.
