# Tuning Indexes

Tuning means choosing parameters that meet recall, latency, throughput, memory, freshness, and build-time goals. For NVIDIA cuVS workflows, there are two related problems: tuning a vector search index and tuning a vector database that uses one or more indexes internally.

For background on index choices, see [What is Vector Search?](/getting-started/introduction/vector-search). For the difference between an index and a vector database, see [Vector Database](/getting-started/introduction/vector-database).

This page explains how to tune standalone indexes, how that maps to common hyperparameter optimization workflows, and how tuning changes when the index is embedded inside a locally or globally partitioned vector database.

## Tuning vector search indexes

Start with the default parameters for a first test. Tune when you need higher recall, lower latency, more throughput, lower memory usage, faster builds, or a better balance across those goals.

Index parameters are workload-specific. For IVF indexes, important knobs include the number of lists, the number of probes, quantization, and refinement. For graph indexes, important knobs include graph degree, construction quality, and search breadth. For compressed indexes, tune compression together with reranking because stronger compression often needs a larger candidate set to recover recall.

Use a representative dataset split. Sample training vectors, test queries, and held-out evaluation queries. Compute exact brute-force neighbors for the query sets so recall can be measured against ground truth. Tune against the test queries, then validate the best candidates against held-out queries to avoid overfitting one benchmark run.

## Automated tuning

Vector search indexes are tuned much like machine learning models. You define an objective, choose a training and validation split, search over candidate hyperparameters, and validate the best candidates on held-out queries. The objective is usually multi-objective: maximize recall while staying within latency, throughput, memory, and build-time limits.

Hyperparameter optimization tools such as [Ray Tune](https://medium.com/rapids-ai/30x-faster-hyperparameter-search-with-raytune-and-rapids-403013fbefc5), [Optuna](https://docs.rapids.ai/deployment/stable/examples/rapids-optuna-hpo/notebook/), and similar HPO frameworks can automate random search, Bayesian search, early stopping, and trial tracking. These tools are useful when the search space is large or when recall and performance must be optimized together.

[cuVS Bench](/user-guide/benchmarking-guide/cu-vs-bench-tool/introduction) also provides a `tune` mode that can perform hyperparameter optimization for benchmark configurations. This is useful when you want the same tool to run reproducible benchmark trials and search for parameter settings that satisfy a recall, latency, throughput, build-time, or memory target.

This is the same idea as k-fold cross-validation for machine learning models: repeat the experiment across representative splits, tune on one portion of the data, and confirm that the selected parameters generalize to data that was not used during tuning.

## Tuning vector databases

Vector databases add system-level tuning questions on top of index tuning. You may also need to choose worker counts for ingest and search, minimum and maximum segment sizes, compaction frequency, flush frequency for freshness, memory and disk budgets, caching behavior, and replication or sharding strategy. The right workflow depends on whether the database uses locally partitioned or globally partitioned indexes.

### Locally partitioned indexes

Locally partitioned databases split the corpus into independent local indexes, often called shards or segments, and merge partial results at query time. This is common in segment-oriented and hash-partitioned systems. The tuning problem is usually smaller and more modular because each local index can be treated like a representative standalone index.

For the architecture background, see [Locally partitioned indexes](/getting-started/introduction/vector-database#locally-partitioned-indexes) in the Vector Database guide.

Tune on a sample that looks like one production shard or segment. If each local partition is expected to hold about 10M vectors, tune on a sample with the same scale and distribution rather than tuning against the full database size. This makes the selected parameters more likely to transfer to production segments.

The following workflow is most useful for locally partitioned databases:

1. Define the target constraints: minimum recall, maximum latency, throughput goal, memory budget, freshness target, and acceptable build or compaction time.

1. Choose candidate index types and parameter ranges. Keep the search space focused on settings that could realistically satisfy the target constraints.

1. Randomly sample training vectors from the full dataset. A good starting point is the number of vectors expected in one production shard, segment, or local index.

1. Randomly sample test and evaluation query sets. These are often 1-10% of the training sample size.

1. Compute exact ground truth for the test and evaluation queries against the training vectors.

1. Run the tuning search. Optimize for recall and the performance metric that matters most for the workload, such as latency, throughput, memory, build time, or compaction cost.

1. Validate the best candidates on the held-out evaluation queries. Prefer parameters that satisfy the target constraints and generalize beyond the test queries used during tuning.

1. Repeat the process on several random samples when you need more confidence. Compare the selected parameters across runs and choose stable settings.

1. Build production segments with the selected parameters and verify recall and performance on production-like traffic.

Also tune database operations around the local index. Segment size controls how many local indexes must be searched and merged. Flush frequency controls freshness. Compaction frequency controls how quickly many small segments are merged into fewer larger segments. These choices can matter as much as the index parameters themselves.

### Globally partitioned indexes

Globally partitioned databases train a partitioning structure across the corpus, such as an IVF-style centroid model. Search can visit only the most relevant partitions, but tuning is less separable because parameters such as `n_lists`, `n_probes`, partition balance, and cache behavior depend on the full dataset distribution.

For the architecture background, see [Globally partitioned indexes](/getting-started/introduction/vector-database#globally-partitioned-indexes) in the Vector Database guide.

For global partitioning, tune on data that preserves the global vector distribution. A small local shard is usually not enough because it may not show how clusters balance, how queries map to partitions, or how often hot partitions are reused. When full-scale tuning is too expensive, use the largest representative sample that preserves the same embedding model, data mix, and query distribution.

Tune the global partitioning and the per-partition search together. Increasing the number of global partitions can reduce work per query, but it can also increase training cost, metadata, assignment cost, and the number of probes needed to maintain recall. Increasing the number of probes usually improves recall, but it also touches more partitions and can reduce cache locality.

If each global partition contains another ANN index, such as a graph index or compressed index, tune that inner index at the expected partition size. Then validate the full database path end to end, including partition assignment, partition probing, per-partition search, result merging, caching, and reranking.

Hybrid designs combine both approaches. For example, a database may ingest into local queue segments, then compact vectors into globally assigned partitions. Tune the queue segments for freshness and ingest throughput, then tune the long-lived global partitions for recall, latency, cacheability, and compaction cost.

## Practical tips

- Treat tuning as a multi-objective problem. The highest-recall configuration is not always the best choice if it is too slow, too large, or too expensive to build.
- Tune parameter ranges before fine-tuning individual values. Broad sweeps usually reveal the useful region of the search space faster.
- Keep exact ground truth and held-out query sets separate from the queries used during tuning.
- Re-tune after major changes to the dataset, embedding model, index type, hardware, query pattern, or database partitioning strategy.
- For locally partitioned databases, prefer parameters that work well across representative shards instead of overfitting one sample.
- For globally partitioned databases, validate at a scale large enough to expose partition balance, hot partitions, and probe-count behavior.

## Summary

Index tuning follows the same experimental pattern as machine learning hyperparameter optimization: define metrics, search parameter ranges, validate on held-out data, and avoid overfitting one run. Vector database tuning adds partitioning and operations. Local partitioning can usually be tuned shard by shard, while global partitioning must account for the full corpus distribution and the cost of assigning, probing, caching, and merging partitions.
