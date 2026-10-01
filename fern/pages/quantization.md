# Vector Compression

Vector compression reduces the cost of storing, moving, and comparing high-dimensional vectors. In vector search, that usually means using fewer bytes per vector, fewer dimensions per vector, or compact codes that approximate the original data well enough for the workload.

Compression is useful because large vector datasets are often limited by memory capacity and memory bandwidth. A smaller representation can make an index cheaper to store, faster to scan, easier to cache, and practical at larger scale. The tradeoff is that compression is usually lossy, so recall and distance quality need to be validated.

This page explains the two main families of vector compression: quantization and dimensionality reduction. It also compares common NVIDIA cuVS-adjacent methods, including scalar quantization, binary quantization, vector quantization, product quantization, RaBitQ-style quantization, PCA, and Spectral Embedding.

## Quick comparison

| Method | Compression family | What changes | Compute cost to prepare | Good fit |
| --- | --- | --- | --- | --- |
| [Scalar quantization](/user-guide/api-guides/preprocessing-guide/scalar-quantizer) | Quantization | Each value is mapped to a smaller numeric type | Low: learn value ranges or thresholds | Simple memory and bandwidth reduction |
| [Binary quantization](/user-guide/api-guides/preprocessing-guide/binary-quantizer) | Quantization | Each value becomes one bit | Low to medium: fixed thresholds are cheap; learned thresholds require a pass over data | Very compact bitwise search |
| Vector quantization | Quantization | Each vector is assigned to a learned centroid | Medium to high: usually requires k-means or another clustering step | Coarse partitioning and residual compression |
| [Product quantization](/user-guide/api-guides/preprocessing-guide/product-quantization) | Quantization | Vector subspaces are represented by compact code IDs | Medium to high: trains multiple codebooks, often with k-means | Large-scale ANN indexes where memory is the bottleneck |
| [RaBitQ quantization](https://arxiv.org/abs/2405.12497) | Quantization | Vectors become randomized binary codes with distance-error correction | Medium: specialized randomization and correction terms | Compact high-dimensional ANN distance estimation |
| [PCA](/user-guide/api-guides/preprocessing-guide/pca) | Dimensionality reduction | Vectors are projected into fewer linear components | Medium to high: computes covariance and eigensolver or SVD-style work | Denoising and reducing distance-computation cost |
| [Spectral Embedding](/user-guide/api-guides/preprocessing-guide/spectral-embedding) | Dimensionality reduction | Vectors are replaced with graph-derived coordinates | High: builds or consumes a graph and computes eigenvectors | Preserving local neighborhood or manifold structure |

## Quantization

Quantization compresses a vector by changing how its values are represented. The vector may keep the same number of dimensions, but each value uses fewer bits, or the vector may be represented by one or more learned code IDs.

Scalar quantization is the simplest form. It maps each floating-point value to a smaller type, such as `int8`, using a learned range. It is usually cheap to train and easy to apply, which makes it a good first compression option when memory bandwidth is the bottleneck. The main risk is clipping or losing precision in dimensions whose values do not fit the learned range well. See the [Scalar Quantizer](/user-guide/api-guides/preprocessing-guide/scalar-quantizer) API guide for NVIDIA cuVS examples.

Binary quantization is more aggressive. It turns each value into one bit, often by comparing against zero or a learned threshold. This can make vectors extremely compact and enables fast bitwise comparisons, but it discards most magnitude information. It is useful when bitwise distance is a good match for the workload or when storage size matters more than fine-grained distance estimates. See the [Binary Quantizer](/user-guide/api-guides/preprocessing-guide/binary-quantizer) API guide for NVIDIA cuVS examples.

Vector quantization, or VQ, assigns each full vector to a representative centroid in a learned codebook. The codebook is often trained with k-means, so preparation is more expensive than scalar or simple binary quantization. VQ is useful for coarse partitioning, IVF-style search, and residual compression because each vector can be represented by a centroid ID plus optional leftover information.

Product quantization, or PQ, splits each vector into subspaces and quantizes each subspace separately. It usually compresses more strongly than scalar quantization while preserving more structure than a single whole-vector code. PQ requires training one or more codebooks, so build cost is higher, but it is widely used when full-precision vectors are too large to store or scan efficiently. NVIDIA cuVS exposes standalone PQ through the [Product Quantization](/user-guide/api-guides/preprocessing-guide/product-quantization) guide and uses PQ-style compression in indexes such as [IVF-PQ](/user-guide/api-guides/indexing-guide/ivf-pq).

RaBitQ-style quantization maps high-dimensional vectors into randomized binary codes and uses correction terms to estimate distances with controlled error. This is more specialized than the standard NVIDIA cuVS preprocessing quantizers, but it is useful to understand when comparing newer ANN systems that rely on compact binary distance estimation.

## Dimensionality reduction

Dimensionality reduction compresses a vector by changing the coordinate space. Instead of storing all original dimensions, it stores fewer derived dimensions. This can reduce memory footprint, reduce distance-computation cost, remove noise, or create an embedding that is easier for later algorithms to use.

[PCA](/user-guide/api-guides/preprocessing-guide/pca) is a linear dimensionality reduction method. It learns directions that explain the most variance, then projects vectors into a smaller space. PCA is useful when many dimensions are redundant or noisy and a linear projection preserves the signal needed by the downstream algorithm. The preparation cost can be significant because PCA needs covariance and eigensolver or SVD-style work, but applying the learned projection is straightforward.

[Spectral Embedding](/user-guide/api-guides/preprocessing-guide/spectral-embedding) is graph-based. It builds or consumes a connectivity graph, computes a graph Laplacian, and uses eigenvectors from that graph as lower-dimensional coordinates. It is useful when local neighborhood or manifold structure matters more than global linear variance. Its preparation cost is usually higher than PCA because it depends on graph construction and eigensolver work.

## Quantization vs. dimensionality reduction

Quantization changes the representation of values or codes. It is usually the right starting point when the original vector dimensions are still meaningful, but full precision is too expensive. Use quantization when you want smaller storage, lower memory bandwidth, faster approximate scoring, or compact codes inside an ANN index.

Dimensionality reduction changes the feature space itself. It is usually the right starting point when the vector has redundant dimensions, noisy dimensions, or a lower-dimensional structure that should be preserved. Use dimensionality reduction when fewer coordinates can represent the signal well enough for search, clustering, visualization, or preprocessing.

The two approaches can be combined. A workflow might use PCA to reduce dimensionality, then scalar quantization or PQ to reduce memory bandwidth further. Another workflow might use vector quantization for coarse partitioning, then PQ on residuals inside each partition.

The preparation cost is part of the tradeoff. Scalar and simple binary quantization are usually cheap. VQ and PQ require codebook training, often with k-means. PCA requires linear algebra over the dataset. Spectral Embedding requires graph construction and eigenvectors. A compression method is only worth the extra build cost when it pays back in index size, query speed, cacheability, or recall at scale.

## Refinement and reranking

Compressed distances are usually approximate. Refinement, also called reranking, helps recover accuracy by using the compressed representation to find candidates, then recomputing distances for those candidates with a more accurate representation.

This is especially common with PQ and scalar-quantized indexes. The compressed representation keeps the first stage fast and compact, while the second stage uses full-precision vectors or a less compressed representation to choose final neighbors. If recall matters, tune compression and refinement together instead of looking only at compression ratio.

## Choosing a starting point

Use scalar quantization when you want a simple memory and bandwidth reduction while keeping the same vector shape.

Use binary quantization when compact bit vectors and bitwise distance are a good match for the workload.

Use vector quantization when you need coarse assignments, partitioning, or residual compression.

Use product quantization when the dataset is too large for full-precision storage and approximate distances are acceptable.

Use RaBitQ-style quantization when evaluating ANN systems that need very compact high-dimensional codes with distance-error correction.

Use PCA when reducing the number of dimensions is acceptable and a linear projection preserves the signal needed by the downstream algorithm.

Use Spectral Embedding when reducing dimensions should preserve local graph or manifold structure rather than only linear variance.

## Conclusion

Vector compression is a practical scaling tool for vector search. Quantization makes each vector cheaper to represent, while dimensionality reduction makes each vector shorter or moves it into a more useful coordinate space.

The best method depends on what cost you are trying to reduce and what error you can tolerate. Always validate compressed workflows with the same metrics used for index tuning: recall, latency, throughput, memory, and build time. Compression is only useful when the resulting system still meets the quality target for the workload.
