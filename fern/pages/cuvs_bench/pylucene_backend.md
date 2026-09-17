# PyLucene Backend

The optional `pylucene` backend runs cuVS Bench through PyLucene's embedded JVM and cuVS-Lucene. It does not use the `*_ANN_BENCH` executables or `--executable-dir`. Use this guide to prepare its external dependencies, configure the runtime, and run representative HNSW and CAGRA benchmarks.

## Supported benchmark paths

| Algorithm | Codec | Index construction | Search |
| --- | --- | --- | --- |
| `pylucene_cuvs_hnsw` | `Lucene101AcceleratedHNSWCodec` | cuVS-accelerated HNSW construction when GPU support is available, with an intentional Lucene CPU-writer fallback | Lucene CPU HNSW |
| `pylucene_cuvs_cagra` | `CuVS2510GPUSearchCodec` | GPU CAGRA | GPU CAGRA |

The backend accepts only FLOAT32 datasets with Euclidean distance, at most 4096 dimensions, and at least two indexed vectors. CAGRA requires a supported NVIDIA GPU for construction and search. HNSW can fall back to Lucene's CPU writer, with a warning, when cuVS GPU support is unavailable. Select the CAGRA path with `--algorithms pylucene_cuvs_cagra`.

## Prerequisites

cuVS Bench does not install the PyLucene runtime automatically. The backend requires:

- JDK 22, including `javac`;
- a custom PyLucene wrapper generated against Lucene 10.2.0;
- [Maven 3.9.6 or newer](https://maven.apache.org/download.cgi) to build cuVS-Lucene; and
- the base `cuvs-java` JAR, the standard `cuvs-lucene` JAR, and matching native cuVS libraries for GPU execution.

Apache does not publish PyLucene 10.2.0, and the official PyLucene 10.0.0 distribution is incompatible with the Lucene 10.2 APIs used here. The [PyLucene source-build instructions](https://lucene.apache.org/pylucene/install.html) describe the general build mechanics, but neither Apache nor cuVS currently provides a ready-to-use 10.2.0 wrapper.

### Build matched cuVS dependencies

cuVS-Lucene lives in this repository under `java/cuvs-lucene`. Build the base `cuvs-java` JAR, the standard `cuvs-lucene` JAR, and the native cuVS libraries from the same checkout as cuVS Bench:

```bash
export CUVS_ROOT="/absolute/path/to/cuvs-checkout"
(
    cd "$CUVS_ROOT"
    ./build.sh libcuvs java lucene
)
export CUVS_VERSION="$(
    mvn -q -f "$CUVS_ROOT/java/cuvs-lucene/pom.xml" \
        help:evaluate -Dexpression=project.version -DforceStdout
)"
```

If matching native cuVS libraries are already built and installed, running `./build.sh java lucene` from `$CUVS_ROOT` is sufficient. The Java build installs the base and native-classifier JARs into the local Maven repository; see the [cuVS Java build guide](https://github.com/NVIDIA/cuvs/blob/main/java/README.md).

The conventional JAR paths are:

```text
$HOME/.m2/repository/com/nvidia/cuvs/cuvs-java/$CUVS_VERSION/cuvs-java-$CUVS_VERSION.jar
$CUVS_ROOT/java/cuvs-lucene/target/cuvs-lucene-$CUVS_VERSION.jar
```

Use the base `cuvs-java` JAR, not a native-classifier JAR. Use the standard cuVS-Lucene JAR, not its `-jar-with-dependencies`, sources, or Javadoc variants. Native-library paths must resolve `libcuvs.so`, `libcuvs_c.so`, their dependencies, and the CUDA runtime libraries from the matching cuVS build.

Use a clean environment without another cuVS native installation on its library path; otherwise, the JVM can load the other `libcuvs_c.so` first and reject the Java/native version mismatch.

### Install cuVS Bench

Install the cuVS Bench development checkout into the activated PyLucene environment:

```bash
cd "$CUVS_ROOT"
python -m pip install -e ./python/cuvs_bench
```

### Validate the dependency build

Before starting the process-wide JVM, the backend checks that `lucene.VERSION` is exactly `10.2.0`, verifies the Maven identity and release version of the configured base `cuvs-java` and thin `cuvs-lucene` JARs, checks their required classes and Lucene SPI providers, rejects a native-classifier cuVS-Java JAR, and rejects either JAR if it bundles Lucene classes. It also compiles its configured-codec adapter against the selected JARs, which fails early when the HNSW heuristic API is missing.

Validate the matching cuVS artifacts with the opt-in Bench PyLucene suite. Pytest owns the scenarios, assertions, parameterization, and reporting under `cuvs_bench/tests/pylucene`. Test-only Java adapters live under `tests/java` (repository path `python/cuvs_bench/tests/java`); the shared session fixture compiles them with `javac` into one temporary classes directory before the process-wide JVM starts. They are excluded from the cuVS Bench wheel and the production cuVS-Lucene JAR.

```bash
python -m pip install pytest
cd "$CUVS_ROOT/python/cuvs_bench"

# Non-extended live suite, including one case for each execution path.
python -m pytest -q -s cuvs_bench/tests/pylucene \
    --run-pylucene -m "pylucene and not pylucene_extended"

# Full GPU E2E suite, including the extended execution-path matrix.
python -m pytest -q -s cuvs_bench/tests/pylucene \
    --run-pylucene -m pylucene
```

Run the codec-path matrix directly with:

```bash
python -m pytest -q -s \
    cuvs_bench/tests/pylucene/test_pylucene_execution_paths.py \
    --run-pylucene
```

The matrix distinguishes CPU HNSW, GPU/CAGRA-built HNSW, and GPU CAGRA search. Extended cases cover segment and force-merge layouts, one- and three-layer HNSW, CAGRA `searchWidth` values 16 and 32, deletion, filtering, and a 2,000-result HNSW search over a CAGRA-built graph. They also check deterministic brute-force recall, duplicate hits, and rank-one self matches. `searchWidth` and `iTopK` are cuVS CAGRA query settings exercised through a test-only query bridge; they are not the backend's HNSW `num_candidates` setting or cuVS Bench `--batch-size`.

## Runtime discovery and advanced overrides

`--backend pylucene` discovers a complete JAR pair from the current source tree or the exact-version local Maven repository. It discovers native libraries from `CUVS_HOME`, the current source build, or one active Python or conda environment. Existing JAR and native-library environment variables remain supported for CI and nonstandard installations. Override both JAR paths together to keep artifact selection explicit, and build both JARs from the same checkout.

The local Maven repository defaults to `$HOME/.m2/repository`; set `MAVEN_LOCAL_REPO` when the matched Java artifacts were installed elsewhere.

Once those prerequisites are installed, `--backend pylucene` is the only backend-specific option required. If `--algorithms` is omitted, this backend selects `pylucene_cuvs_cagra`; specify `pylucene_cuvs_hnsw` to benchmark HNSW, using cuVS-accelerated construction when available and the documented CPU-writer fallback otherwise.

For an advanced layout, create `pylucene-backend.yaml` with explicit paths. Replace `<cuvs-version>` with the `CUVS_VERSION` value obtained above; YAML does not expand shell variables.

```yaml
backend: pylucene
cuvs_java_jar: /home/user/.m2/repository/com/nvidia/cuvs/cuvs-java/<cuvs-version>/cuvs-java-<cuvs-version>.jar
cuvs_lucene_jar: /work/cuvs-pylucene-deps/java/cuvs-lucene/target/cuvs-lucene-<cuvs-version>.jar
java_library_path: /work/cuvs-pylucene-deps/cpp/build:/work/cuvs-pylucene-deps/cpp/build/c:/usr/local/cuda/lib64
```

Pass that file instead of the simple backend selector:

```bash
python -m cuvs_bench.run --backend-config pylucene-backend.yaml <options>
```

Configuration sources are:

| Backend configuration key | Environment variable |
| --- | --- |
| `cuvs_java_jar` | `CUVS_LUCENE_CUVS_JAVA_JAR` |
| `cuvs_lucene_jar` | `CUVS_LUCENE_JAR` |
| `java_library_path` | `JAVA_LIBRARY_PATH` or `LD_LIBRARY_PATH` |
| `jvm_args` | No environment alias; YAML list of additional JVM arguments. |

PyLucene's JVM is process-global and can be initialized only once. When using overrides, set the JAR paths, native-library locations, and `jvm_args` before the first PyLucene benchmark, and start a new Python process to change any of them.

## Run a functional smoke test

Use one explicit dataset root for both preparation and execution:

```bash
export DATASET_ROOT=/absolute/path/to/cuvs-bench-data

python -m cuvs_bench.get_dataset \
    --dataset test-data \
    --dataset-path "$DATASET_ROOT" \
    --test-data-n-train 512 \
    --test-data-n-test 4 \
    --test-data-k 5

python -m cuvs_bench.run \
    --backend pylucene \
    --dataset test-data \
    --dataset-path "$DATASET_ROOT" \
    --algorithms pylucene_cuvs_hnsw \
    --groups test \
    --batch-size 2 -k 5 \
    -m latency \
    --build --search --force
```

This small synthetic run verifies the workflow; do not use it as performance or quality evidence. For a representative recall run, prepare `deep-image-96-angular` with `--normalize` into the same `DATASET_ROOT`, then run the backend against `deep-image-96-inner` with the same `--dataset-path`.

## Configure HNSW benchmarks

The HNSW algorithm maps benchmark parameters to cuVS-Lucene as follows:

| Parameter | Scope | Default | Accepted values | Effect |
| --- | --- | --- | --- | --- |
| `codec` | Build | `Lucene101AcceleratedHNSWCodec` | That codec name | Selects the cuVS-accelerated HNSW codec. |
| `m` | Build | `32` | Integer from 1 through 512 | Sets `AcceleratedHNSWParams.maxConn`. |
| `ef_construction` | Build | `32` | Integer from 1 through 512 | Sets `AcceleratedHNSWParams.beamWidth`. |
| `direct_single_segment` | Build | `false` | Boolean | Requests one direct segment, as described below. |
| `num_candidates` | Search | `top_k` | Integer greater than or equal to `top_k` | Sets the candidate count passed to `KnnFloatVectorQuery`, capped at the index size; the backend returns `top_k` neighbors. |

`m` and `ef_construction` are HNSW-equivalent inputs to cuVS-Lucene's `SAME_GRAPH_FOOTPRINT` heuristic. cuVS-Lucene derives the underlying CAGRA build parameters from them. `num_candidates` is Lucene's candidate budget, not a direct cuVS `ef_search` setting.

Automatic tune mode samples `m` and `ef_construction` from 1 through 512 and `num_candidates` from `top_k` through 2500. Explicit YAML sweeps may use larger candidate counts.

Elasticsearch shard, replica, and field-name settings do not apply to this local Lucene index. Its HNSW index type maps to `codec`, and the backend validates Euclidean similarity from the dataset rather than accepting Elasticsearch's type, quantization, or similarity options.

### Run a representative sweep

The following configuration creates three indexes and searches each one with three candidate counts. Save it as `pylucene-deep1m.yaml`:

```yaml
name: pylucene_cuvs_hnsw
groups:
  deep1m:
    build:
      codec: ["Lucene101AcceleratedHNSWCodec"]
      m: [16, 24, 32]
      ef_construction: [32]
      direct_single_segment: [true]
    search:
      num_candidates: [150, 200, 300]
```

Run it with `top_k=150`, substituting the dataset name and paths from its dataset configuration:

```bash
python -m cuvs_bench.run \
    --backend pylucene \
    --configuration pylucene-deep1m.yaml \
    --dataset deep1b-1M \
    --dataset-configuration /absolute/path/to/deep1b-1M.yaml \
    --dataset-path /absolute/path/to/dataset-root \
    --algorithms pylucene_cuvs_hnsw \
    --groups deep1m \
    --batch-size 100 -k 150 \
    -m latency \
    --build --search --force
```

## Index topology, provenance, and reuse

When `direct_single_segment` is true, the backend disables ordinary RAM-triggered flushes and merging, buffers the requested vectors for one flush, and fails unless the committed index has exactly one segment. Lucene's per-indexing-thread hard RAM limit remains in force: 1945 MiB by default and less than 2048 MiB through the public API. If that limit forces an earlier flush, the build fails instead of merging the segments.

`direct_single_segment` does not call Lucene `forceMerge`. The cuVS Bench `--force` option requests a rebuild; it does not control Lucene's segment merging. A larger JVM heap does not disable the per-thread hard limit.

The current `--force` implementation removes an existing index at the configured path before it constructs and verifies the replacement. If that forced build fails, the partial replacement is removed and the previous index is no longer available; rerun the build to recreate it.

New HNSW and CAGRA builds atomically write commit-bound provenance manifests named `.cuvs-bench-pylucene-hnsw.json` and `.cuvs-bench-pylucene-cagra.json`, respectively. Reuse and search fail if the applicable manifest is missing, malformed, stale, or names different build parameters, writer policy, or compound-file policy. Indexes created outside this backend without the applicable manifest must be rebuilt with `--force`.

For HNSW, the writer policy permits cuVS-Lucene's production CPU fallback. For CAGRA, the backend fails closed: it verifies the committed vector segments and checksums and rejects an index that is not CAGRA-only.

For CAGRA, the backend disables compound files for both flushed and merged segments so the verifier can inspect the codec's `.vemc` and `.vcag` files directly. HNSW retains Lucene's default compound-file policy and its default index-writer scheduling unless `direct_single_segment` is enabled.

## Runtime and measurement limits

- The backend currently supports latency mode with one search thread.
- `--batch-size` groups queries for measurement; reported latency percentiles are milliseconds per batch.
- `num_candidates` must be greater than or equal to `top_k` for HNSW.
- The production backend's stock `KnnFloatVectorQuery` path currently requires requested `k <= 1024` for direct CAGRA indexes. This is a cuVS-Lucene reader-dispatch boundary, not a native CAGRA limit: larger per-leaf collectors are routed to a brute-force index that a CAGRA-only benchmark does not contain. CAGRA-built HNSW is unaffected and is covered at `top_k=2000` with `num_candidates=2500`.
- Throughput mode and multiple search threads are not implemented.
- PyLucene requires a wrapper generated against Lucene 10.2.0, and its process-wide JVM configuration cannot change after initialization.
