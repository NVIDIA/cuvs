---
slug: user-guide/benchmarking-guide/cu-vs-bench-tool/lucene-backend
---

# Lucene Backend

The optional `lucene` backend runs cuVS Bench against a local Lucene index. It
embeds the JVM through PyLucene. The CPU HNSW control uses stock Lucene through
PyLucene, while the GPU CAGRA path uses the cuVS-Lucene thin JAR. PyLucene is an
implementation detail; the public cuVS Bench backend name is `lucene`.

Selecting this backend is explicit. Commands that do not select it with
`--backend lucene` continue to use the `cpp_gbench` backend and its existing
`cuvs_cagra` default.

## Algorithms

| Algorithm | Codec | Build and search path |
| --- | --- | --- |
| `lucene_cuvs_cagra` | `CuVS2510GPUSearchCodec` | cuVS CAGRA on a supported NVIDIA GPU |
| `lucene_cpu_hnsw` | `Lucene101` | Lucene CPU HNSW control |

`lucene_cuvs_cagra` is selected when `--backend lucene` is used without an
explicit `--algorithms` value. Select `lucene_cpu_hnsw` explicitly when a CPU
control is needed.

```bash
python -m cuvs_bench.run \
    --backend lucene \
    --dataset test-data \
    --dataset-path /absolute/path/to/datasets \
    --algorithms lucene_cuvs_cagra \
    --groups test \
    --batch-size 10 \
    -k 10 \
    --build --search
```

## Runtime requirements

The Lucene backend is opt-in because its runtime is not provisioned by the
ordinary cuVS Bench installation. Provisioning the required custom PyLucene
build is currently external to cuVS Bench. Every algorithm requires PyLucene
10.2.0 and JDK 22. CAGRA additionally requires matching `cuvs-java` and thin
`cuvs-lucene` JARs plus compatible native cuVS, CUDA, and NVIDIA GPU support.
The backend discovers artifacts built by the current checkout or installed in
the local Maven repository. For nonstandard locations, set both
`CUVS_LUCENE_CUVS_JAVA_JAR` and `CUVS_LUCENE_JAR`; use `JAVA_LIBRARY_PATH` for
native libraries. An explicit native-library path must contain an unversioned
`libcuvs_c.so`.

Do not use the cuVS-Lucene JAR assembled with dependencies: PyLucene already
provides Lucene classes, and loading a second copy can make the embedded JVM
classpath inconsistent.

The initial backend accepts nonempty, finite, `float32` Euclidean/L2 vectors
and supports latency-mode sweeps. The CPU HNSW algorithm accepts at most 1024
dimensions; the CAGRA algorithm accepts at most 4096. CAGRA uses the codec's
fixed defaults and supports `k <= 1024`. The backend validates the physical
segment codec and every persisted vector field before searching, and fails if
CAGRA construction silently produced a brute-force index. The CPU control
supports an explicit `num_candidates` value greater than or equal to `k`.

## Integration tests

Run the live CPU/GPU suite from the repository root after providing the same
runtime prerequisites:

```bash
python -m pytest -q -s \
    python/cuvs_bench/cuvs_bench/tests/test_lucene_integration.py \
    --run-lucene-e2e
```

Without `--run-lucene-e2e`, ordinary cuVS Bench test runs skip these live
cases. Once selected, every case requires PyLucene and Java. CAGRA cases also
require the cuVS Java artifacts, native libraries, CUDA, and a supported GPU;
missing prerequisites fail rather than skip.
