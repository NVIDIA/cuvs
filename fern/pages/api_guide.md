---
slug: user-guide/api-guides
---

# API Guide

Use these pages to find task-focused NVIDIA cuVS API examples for clustering, vector indexing, preprocessing, and supporting routines.

NVIDIA cuVS is written in C++ at its core and wrapped by a stable C API layer. The Python, Java, Rust, and Go bindings use that C layer so they can share the same ABI boundary; see [Compatibility](/user-guide/field-guide/compatibility) for why that matters. These API guides are intended for general use and include examples for supported programming languages where possible, but some guides document C++ concepts explicitly because all NVIDIA cuVS algorithm implementations are C++ at the core.

## Common Types

- [Array Types](/user-guide/api-guides/core-types/array-types): choose between dense arrays and sparse arrays for NVIDIA cuVS APIs.
- [Dense Arrays](/user-guide/api-guides/core-types/array-types/dense-arrays): pass dense vectors, matrices, and outputs into NVIDIA cuVS APIs across supported languages.
- [Memory Management](/user-guide/api-guides/core-types/memory-management): configure RMM device, pool, pinned host, host, and managed memory resources for NVIDIA cuVS workflows.
- [Multi-GPU](/user-guide/api-guides/core-types/multi-gpu): initialize multi-GPU resources and understand RAFT/NCCL communication setup.
- [Resources](/user-guide/api-guides/core-types/resources): reuse CUDA streams, library handles, stream pools, and workspace resources across NVIDIA cuVS calls.
- [Sparse Arrays](/user-guide/api-guides/core-types/array-types/sparse-arrays): use CSR and COO sparse matrix views with NVIDIA cuVS C++ APIs that accept sparse inputs.

## Clustering Guide

- [K-Means](/user-guide/api-guides/clustering-guide/k-means): partition vectors into a fixed number of clusters, often as part of scalable vector-search systems.
- [Single-linkage](/user-guide/api-guides/clustering-guide/single-linkage): build hierarchical clusters from nearest-neighbor relationships.
- [Spectral Clustering](/user-guide/api-guides/clustering-guide/spectral-clustering): use graph structure and spectral methods to identify clusters with more complex shapes.

## Indexing Guide

- [Brute-force](/user-guide/api-guides/indexing-guide/brute-force): run exact nearest-neighbor search by comparing each query with every vector.
- [CAGRA](/user-guide/api-guides/indexing-guide/cagra): build and search GPU-optimized graph indexes for high-throughput ANN search.
- [NN-Descent](/user-guide/api-guides/indexing-guide/nn-descent): build approximate nearest-neighbor graphs with an iterative algorithm.
- [IVF-Flat](/user-guide/api-guides/indexing-guide/ivf-flat): partition vectors into inverted-file lists while storing full-precision vectors.
- [IVF-PQ](/user-guide/api-guides/indexing-guide/ivf-pq): combine inverted-file partitioning with product quantization for compact indexes.
- [ScaNN](/user-guide/api-guides/indexing-guide/sca-nn): combine partitioning, quantization, and refinement for high-quality approximate search.
- [Vamana](/user-guide/api-guides/indexing-guide/vamana): build graph indexes for large-scale and disk-backed search workflows.
- [All-neighbors](/user-guide/api-guides/indexing-guide/all-neighbors): compute all-neighbors graph structures.

## Preprocessing Guide

- [Binary Quantizer](/user-guide/api-guides/preprocessing-guide/binary-quantizer): compress vectors into binary representations for compact storage and fast comparisons.
- [PCA](/user-guide/api-guides/preprocessing-guide/pca): reduce dimensionality with a linear projection while preserving as much variance as possible.
- [Product Quantization](/user-guide/api-guides/preprocessing-guide/product-quantization): split vectors into subvectors and encode each part with compact codebooks.
- [Scalar Quantizer](/user-guide/api-guides/preprocessing-guide/scalar-quantizer): compress each vector dimension independently with scalar quantization.
- [Spectral Embedding](/user-guide/api-guides/preprocessing-guide/spectral-embedding): create lower-dimensional embeddings from graph structure.

## Other APIs

- [Dynamic Batching](/user-guide/api-guides/other-ap-is/dynamic-batching): collect many concurrent small ANN searches into larger GPU search batches.
- [K-selection](/user-guide/api-guides/other-ap-is/k-selection): select the top `k` values or nearest candidates from larger result sets.
- [Pairwise Distances](/user-guide/api-guides/other-ap-is/pairwise-distances): compute distances between vectors for analysis, validation, or algorithm building blocks.
