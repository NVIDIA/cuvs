# cuVS Bench

Use this when a result has to be compared or reproduced. The recall snippet in `SKILL.md` checks one index you just built. cuVS Bench is the sweep across indexes, machines, and parameter settings.

## When to run it

Run cuVS Bench for any of these:

- A GPU index against another GPU index, including single-GPU against multi-GPU.
- A GPU index against a CPU index, including a GPU CAGRA build served as CPU HNSW.
- A parameter sweep whose useful settings must land in a recall bucket.
- The same configuration on another machine. The tool records the index configuration with the result.
- `tune` mode, when the cross product of build and search parameters is too large and the goal is a recall, latency, throughput, build-time, or memory target.

Leave it for later when the task is still "which index family fits." Pick the family from `references/index-selection.md`, confirm one build with the recall snippet, then sweep.

## Fix the comparison first

Write these down before either run: scope, dataset, distance metric, `k`, batch size, filters, hardware, and concurrency.

| Scope | What to benchmark |
|---|---|
| Standalone index | That index, on the production-sized sample |
| Locally partitioned or hash-sharded database | One production segment. A 100M corpus in ten 10M segments is a 10M-vector benchmark. |
| Globally partitioned index | A sample that keeps the full embedding mix and query mix. A single local shard hides partition balance. |
| Full database | The path the query actually takes: assign, probe, search, merge, and any rerank |

Ground truth is the exact top-k from `SKILL.md`. Built-in datasets ship it for 100 neighbors, so benchmark `k` at 100 or below. Generate a missing file with `python -m cuvs_bench.generate_groundtruth` before the search run. A subset of a base file needs its own ground truth.

Group every run into one recall bucket, then compare inside that bucket:

| Recall | The comparison is about |
|---|---|
| 80–89% | Fast exploratory search |
| 90–94% | Lower-latency approximate search |
| 95–98% | High-quality approximate search |
| 99%+ | Near-exact search |

A run is on the Pareto curve for its bucket when no other run is better on the metric being compared without being worse on another metric. For the bucket's build time, average the build times of those Pareto points. Report recall, latency, throughput, build time, and memory together, with `k`, batch size, and hardware on the same line. A migration is ready when that bucket meets the recall target and the serving latency budget.

Search latency in the result table is the batch latency. `items_per_second` is query throughput. `GPU` is device time. Build `Time` is wall time to train the index. `end_to_end` is the full search loop, and it is only approximately `Time * Iterations`.

## GPU run against a CPU run

Change the algorithm name and, on a machine with no GPU, the package. Keep the dataset and ground truth from the section above.

The `cuvs-bench` image includes the GPU benchmarks and the CPU benchmarks, so a same-machine comparison runs there. The `cuvs-bench-cpu` image, and the `cuvs-bench-cpu` conda package, currently run HNSW. Use that image on a machine with no GPU.

| Side | Algorithm names |
|---|---|
| GPU cuVS | `cuvs_cagra`, `cuvs_ivf_flat`, `cuvs_ivf_pq`, `cuvs_brute_force`, `cuvs_vamana`, `cuvs_mg_cagra`, `cuvs_mg_ivf_flat`, `cuvs_mg_ivf_pq` |
| GPU build, CPU HNSW search | `cuvs_cagra_hnswlib` |
| CPU | `hnswlib` on the CPU package. The GPU package also runs `faiss_cpu_flat`, `faiss_cpu_ivf_flat`, `faiss_cpu_ivf_pq`, `faiss_cpu_hnsw_flat`, `diskann_memory`, and `diskann_ssd`. |

An algorithm whose config sets `requires_gpu: true` runs on a GPU node. Keep that algorithm off the CPU-only image.

Million-scale data, up through about 10M vectors, can use a built-in dataset. Above about 10M vectors, prepare the dataset first and expect a large-memory GPU such as A100 or H100. `--normalize` on an angular dataset rewrites it to the inner-product name: prepare `deep-image-96-angular`, then run `deep-image-96-inner`.

```bash
# Inside the cuvs-bench image: CAGRA on GPU and HNSW on CPU, one dataset.
python -m cuvs_bench.get_dataset --dataset deep-image-96-angular --normalize
python -m cuvs_bench.run \
  --dataset deep-image-96-inner \
  --algorithms cuvs_cagra,hnswlib \
  --batch-size 10 -k 10 --build --search
python -m cuvs_bench.run --data-export --dataset deep-image-96-inner
python -m cuvs_bench.plot --dataset deep-image-96-inner
```

```bash
# GPU image. The mounted folder holds the dataset and the results for both runs.
export DATA_FOLDER=path/to/store/datasets/and/results
docker run --gpus all --rm -it -u "$(id -u)" \
  -v "$DATA_FOLDER":/data/benchmarks \
  rapidsai/cuvs-bench:26.06a-cuda12-py3.13 \
  "--dataset deep-image-96-angular" \
  "--normalize" \
  "--algorithms cuvs_cagra,hnswlib --batch-size 10 -k 10"
```

```bash
# CPU-only image. HNSW is the algorithm this package runs.
docker run --rm -it -u "$(id -u)" \
  -v "$DATA_FOLDER":/data/benchmarks \
  rapidsai/cuvs-bench-cpu:26.06a-py3.13 \
  "--dataset deep-image-96-angular" \
  "--normalize" \
  "--algorithms hnswlib --batch-size 10 -k 10"
```

Match the image tag to the installed CUDA and Python. CUDA 12 and 13, and Python 3.11 through 3.14, are the documented ranges. Conda packages are `cuvs-bench` for the GPU package and `cuvs-bench-cpu` for the CPU package.

For a custom corpus, provide `base`, `query`, and `groundtruth.neighbors` files plus a dataset YAML, and pass `--dataset-configuration`. Vector files are little-endian, with `num_vectors` and `num_dimensions` as `uint32` in the first 8 bytes. Suffixes are `.fbin` (`float32`), `.f16bin`, `.u8bin`, `.i8bin`, and `.ibin`.

Algorithm YAML groups are the sweep. A `base` group is the cross product of its `build` and `search` lists. Add a custom YAML with a `base` group to change that sweep. `tune` mode searches that space against a target instead of running the full cross product.
