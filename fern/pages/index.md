# NVIDIA cuVS: A library for vector search on the GPU

NVIDIA cuVS is a GPU-accelerated library for vector search on the GPU. Vector search includes nearest neighbors, vector compression and clustering. It provides both core building blocks for constructing new algorithms and end-to-end algorithms that can be used directly or through a growing list of [integrations](/getting-started/integrations).

## Useful Resources

[cuvs_reference]: https://docs.rapids.ai/api/cuvs/stable/

- [Example Notebooks](https://github.com/rapidsai/cuvs/tree/HEAD/notebooks): Example notebooks
- [Code Examples](https://github.com/rapidsai/cuvs/tree/HEAD/examples): Self-contained code examples
- [RAPIDS Community](https://rapids.ai/community.html): Get help, contribute, and collaborate.
- [GitHub repository](https://github.com/rapidsai/cuvs): Download the NVIDIA cuVS source code.
- [Issue tracker](https://github.com/rapidsai/cuvs/issues): Report issues or request features.

## What is NVIDIA cuVS?

NVIDIA cuVS contains state-of-the-art implementations of several algorithms for running approximate and exact nearest neighbors, vector compression, and clustering on the GPU. It can be used directly or through the various databases and other libraries that have integrated it. The primary goal of NVIDIA cuVS is to simplify the use of GPUs for vector similarity search, preprocessing, and clustering. For a broader introduction, start with the [introductory materials](/getting-started/introduction) in Getting Started.

Vector search is an information retrieval method for finding semantically similar items in embedding spaces, especially when working with multimedia embeddings created from unstructured data. It is also used in *data mining and machine learning* tasks and comprises an important step in many *clustering* and *visualization* algorithms like [UMAP](https://arxiv.org/abs/2008.00325), [t-SNE](https://lvdmaaten.github.io/tsne/), K-means, and [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html).

Finally, faster vector search enables interactions between dense vectors and graphs. Converting a pile of dense vectors into nearest neighbors graphs unlocks the entire world of graph analysis algorithms, such as those found in [GraphBLAS](https://graphblas.org/) and [cuGraph](https://github.com/rapidsai/cugraph).

## Where is NVIDIA cuVS used?

These are common places where vector search appears. For more examples, see [Use-cases](/getting-started/use-cases) and [Integrations](/getting-started/integrations).

### Semantic search
- Generative AI: RAG & Agentic AI
- Recommender systems
- Computer vision
- Image search
- Text search
- Audio search
- Molecular search
- Model Training: LLMs & Transformers

### Data mining
- Clustering algorithms
- Visualization algorithms
- Sampling algorithms
- Class balancing
- Ensemble methods
- k-NN graph construction

## Why NVIDIA cuVS?

There are several benefits to using NVIDIA cuVS and GPUs for vector search, including

1. Fast index build
2. Latency critical and high throughput search
3. [Parameter tuning](/getting-started/introduction/tuning-indexes)
4. Cost savings
5. Interoperability (build on GPU, deploy on CPU)
6. Multiple language support
7. Building blocks for composing new or accelerating existing algorithms

In addition to the items above, NVIDIA cuVS shoulders the responsibility of keeping non-trivial accelerated code up to date as new NVIDIA architectures and CUDA versions are released. This provides a delightful development experience, guaranteeing that any libraries, databases, or applications built on top of it will always be receiving the best performance and scale.

## NVIDIA cuVS Technology Stack

NVIDIA cuVS is built on top of the [RAPIDS RAFT](https://github.com/rapidsai/raft) library of high performance machine learning primitives and provides all the necessary routines for vector search and clustering on the GPU.

<img alt="NVIDIA cuVS is built on top of low-level CUDA libraries and provides many important routines that enable vector search and clustering on the GPU" src="/assets/images/tech_stack.png" />
