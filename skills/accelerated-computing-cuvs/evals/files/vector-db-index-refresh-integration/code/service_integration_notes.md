# Retrieval Service Integration Notes

The service is a database-backed retrieval product with an existing CPU serving
path. Query serving is reliable, but index refresh is becoming the bottleneck as
the corpus and embedding dimensions grow.

Current state:

- The serving path keeps a CPU index in each database shard.
- The refresh job rebuilds indexes after each embedding backfill.
- The refresh SLA is four hours, and the largest tenants now miss it.
- Query traffic has a strict p95 latency budget and already has production
  dashboards.
- Recall@20 must stay at or above 0.97 versus the approved exact-search sample.
- GPU workers are available in a batch pool; not every serving shard has a GPU.
- Some deployments load vector-search support through a native plugin boundary
  owned by another service team.

The team wants a practical integration recommendation, including what to
prototype first, which measurements decide whether to change the serving path,
and how to keep the comparison fair against the current CPU index.
