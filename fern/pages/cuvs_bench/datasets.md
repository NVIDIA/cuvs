# Benchmark Datasets

cuVS Bench datasets provide the vectors to index, the queries to search, and the exact nearest neighbors used to measure recall. This page explains the expected file layout, supported binary formats, built-in dataset helpers, ground-truth generation, and the YAML descriptors used for custom datasets.

## Dataset files

Most datasets contain four binary files:

| File | Purpose |
| --- | --- |
| `base.fbin` | Database vectors used to build the index |
| `query.fbin` | Query vectors used during search |
| `groundtruth.neighbors.ibin` | Exact nearest-neighbor ids |
| `groundtruth.distances.fbin` | Exact nearest-neighbor distances |

The vector files are used for build and search. Ground-truth files are tied to a distance metric and are used only for evaluation.

## Binary format

Dataset suffixes describe the stored type:

| Suffix | Type |
| --- | --- |
| `.fbin` | `float32` |
| `.f16bin` | `float16` |
| `.ibin` | `int32` |
| `.u8bin` | `uint8` |
| `.i8bin` | `int8` |

All binary files are little-endian. The first 8 bytes store `num_vectors` and `num_dimensions` as `uint32_t` values. The remaining bytes store `num_vectors * num_dimensions` values in row-major order.

Some implementations can use `float16` vectors for better performance. Convert `float32` files with:

```bash
python/cuvs_bench/cuvs_bench/get_dataset/fbin_to_f16bin.py input.fbin output.f16bin
```

## Built-in datasets

Use `cuvs_bench.get_dataset` to download and prepare common benchmark datasets. Files are stored under `RAPIDS_DATASET_ROOT_DIR` when set, or under a local `datasets` directory otherwise.

```bash
python -m cuvs_bench.get_dataset --dataset deep-image-96-angular --normalize
```

Common built-in datasets include:

| Dataset name | Train rows | Columns | Test rows | Distance |
| --- | --- | --- | --- | --- |
| `deep-image-96-angular` | 10M | 96 | 10K | Angular |
| `fashion-mnist-784-euclidean` | 60K | 784 | 10K | Euclidean |
| `glove-50-angular` | 1.1M | 50 | 10K | Angular |
| `glove-100-angular` | 1.1M | 100 | 10K | Angular |
| `mnist-784-euclidean` | 60K | 784 | 10K | Euclidean |
| `nytimes-256-angular` | 290K | 256 | 10K | Angular |
| `sift-128-euclidean` | 1M | 128 | 10K | Euclidean |

These datasets include ground truth for 100 neighbors, so benchmark `k` must be 100 or smaller.

## Dataset sources

Million-scale datasets are available from [ann-benchmarks](https://github.com/erikbern/ann-benchmarks#data-sets). Convert the HDF5 files to cuVS Bench binaries with:

```bash
python/cuvs_bench/cuvs_bench/get_dataset/hdf5_to_fbin.py [-n] <input>.hdf5
```

Use `-n` to normalize base and query vectors, which is useful when measuring angular datasets as inner-product search.

Billion-scale datasets are available from [big-ann-benchmarks](http://big-ann-benchmarks.com). Split their combined ground-truth files before benchmarking:

```bash
python -m cuvs_bench.split_groundtruth --groundtruth deep_new_groundtruth.public.10K.bin
```

This produces `groundtruth.neighbors.ibin` and `groundtruth.distances.fbin`.

The `wiki-all` dataset contains 88M 768-dimensional vectors, plus 1M and 10M subsets, for realistic RAG/LLM-scale benchmarking. See the [Wiki-all Dataset Guide](/user-guide/benchmarking-guide/cu-vs-bench-tool/wiki-all-dataset) to download it.

## Generate ground truth

If a dataset does not include ground truth, generate it with `cuvs_bench.generate_groundtruth`:

```bash
# With an existing query file
python -m cuvs_bench.generate_groundtruth /dataset/base.fbin --output=groundtruth_dir --queries=/dataset/query.public.10K.fbin

# With randomly generated queries
python -m cuvs_bench.generate_groundtruth /dataset/base.fbin --output=groundtruth_dir --queries=random --n_queries=10000

# With random queries selected from a subset of the dataset
python -m cuvs_bench.generate_groundtruth /dataset/base.fbin --rows=2000000 --output=groundtruth_dir --queries=random-choice --n_queries=10000
```

For billion-scale sources that provide ground truth for only the first 10M or 100M base vectors, use `subset_size` in the dataset configuration so the benchmark uses the matching prefix of the base file.

### Prefiltered ground truth

To benchmark filtered search, generate ground truth against the same filter the benchmark will apply. `--filter_reject_rate` builds a random bitset, uses it while computing the neighbors, and saves it next to the ground truth as `groundtruth.filter.bin`:

```bash
python -m cuvs_bench.generate_groundtruth /dataset/base.fbin --output=groundtruth_dir --queries=/dataset/query.fbin --filter_reject_rate=0.9
```

`--bitset` reuses an existing one instead, so several ground-truth sets can share a filter. The two options are mutually exclusive:

```bash
python -m cuvs_bench.generate_groundtruth /dataset/base.fbin --output=groundtruth_dir --queries=/dataset/query.fbin --bitset=groundtruth_dir/groundtruth.filter.bin
```

The alternative is `filtering_rate` (below), which filters unfiltered ground truth on the fly. That is simpler to configure, but at high reject rates it discards most ground-truth neighbors and leaves fewer than `k` valid entries per query, so recall becomes unreliable. Prefiltered ground truth keeps `k` valid neighbors regardless of the reject rate.

The bitset is indexed by absolute row id, so it must cover the whole dataset and match the ground truth it was generated with. Both the generator and the benchmark reject a bitset that is too short; a mismatched one of the right size is reported as a warning during the run.

## Dataset configurations

Each benchmark dataset needs a YAML descriptor with file names and basic properties. Common descriptors are available in [datasets.yaml](https://github.com/rapidsai/cuvs/blob/branch-25.04/python/cuvs_bench/cuvs_bench/config/datasets/datasets.yaml).

The default `${CUVS_HOME}/python/cuvs_bench/cuvs_bench/config/datasets/datasets.yaml` includes entries like:

```yaml
- name: sift-128-euclidean
  base_file: sift-128-euclidean/base.fbin
  query_file: sift-128-euclidean/query.fbin
  groundtruth_neighbors_file: sift-128-euclidean/groundtruth.neighbors.ibin
  dims: 128
  distance: euclidean
```

For a new dataset, create a descriptor such as `mydataset.yaml`:

```yaml
- name: mydata-1M
  base_file: mydata-1M/base.100M.u8bin
  subset_size: 1000000
  dims: 128
  query_file: mydata-10M/queries.u8bin
  groundtruth_neighbors_file: mydata-1M/groundtruth.neighbors.ibin
  distance: euclidean
```

Choose any `name` and pass it as `--dataset`. File paths are relative to `--dataset-path`. The optional `subset_size` uses the first `subset_size` vectors, which lets you benchmark subsets without duplicating base files. Generate separate ground truth for each subset.

Two optional keys enable filtered search, and at most one may be set:

| Key | Purpose |
| --- | --- |
| `filtering_rate` | Fraction of vectors to reject. Generates a random bitset at startup and filters the ground truth on the fly. Needs no extra files, but see the caveat above at high reject rates. |
| `filter_bitset_file` | Path to a bitset written by `generate_groundtruth`, applied during search. Pair it with the prefiltered `groundtruth_neighbors_file` produced by the same run. |

```yaml
- name: mydata-1M-filtered
  base_file: mydata-1M/base.100M.u8bin
  subset_size: 1000000
  dims: 128
  query_file: mydata-10M/queries.u8bin
  groundtruth_neighbors_file: mydata-1M/groundtruth.neighbors.ibin
  filter_bitset_file: mydata-1M/groundtruth.filter.bin
  distance: euclidean
```

Run the custom dataset with:

```bash
python -m cuvs_bench.run --dataset mydata-1M --dataset-path=/path/to/data/folder --dataset-configuration=mydataset.yaml --algorithms=cuvs_cagra
```

## Summary

cuVS Bench expects a small set of binary vector and ground-truth files plus a YAML descriptor that tells the benchmark runner where those files live. Built-in helpers can download common datasets, convert HDF5 sources, split ground truth, and generate exact neighbors when needed. For custom datasets, prepare the files, write a descriptor, and pass it with `--dataset-configuration`.
