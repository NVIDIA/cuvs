# Retrieval Service Notes

The service stores 8 million text embeddings with 768 float32 dimensions.
Queries ask for top-50 nearest neighbors and the product team wants recall at
0.95 or better versus the current CPU reference.

Current state:

- CPU service uses hnswlib.
- Offline experiments also include a FAISS IVF-PQ prototype.
- Batch indexing is acceptable overnight.
- Query latency matters more than index build time.
- Some teams may want a CPU-serving export after building on GPU.

A teammate pasted a broad GPU optimization checklist that says:

- replace NumPy loops with CuPy;
- use Numba CUDA for custom kernels;
- use custom CUDA when the operation is performance critical.

The service owner wants a library-level recommendation for this vector-search
service, not a generic array or kernel rewrite.
