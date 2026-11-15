# Retrieval Service Notes

The service owns approximately 8 million text embeddings with 768 float32
dimensions. It returns top-50 nearest neighbors for each query and the product
team wants recall of at least 0.95 against the current CPU reference.

Current implementation:

- CPU serving uses hnswlib.
- Offline experiments include a FAISS IVF-PQ prototype.
- Batch index building is acceptable overnight.
- Query latency matters more than index build time.
- The deployment team may want an optional CPU-serving export after building on
  GPU, but online GPU serving is also acceptable if it meets latency targets.

The service owner wants a library-level recommendation for the embedding
retrieval system, plus the validation plan needed before migration.
