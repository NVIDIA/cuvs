# OpenSearch kNN Benchmark

Docker Compose benchmark comparing CPU and GPU kNN index builds in OpenSearch using `cuvs-bench`. Supports both local CPU builds and [GPU-accelerated remote index builds](https://docs.opensearch.org/latest/vector-search/remote-index-build/) via the `REMOTE_INDEX_BUILD` environment variable.

## How it works

OpenSearch's kNN plugin can offload Faiss HNSW index construction to a dedicated GPU service. Rather than building the index in-process on the OpenSearch node, the workflow is:

```
OpenSearch flushes a segment
  → uploads raw vectors + doc-IDs to S3
  → POSTs /_build to the remote-index-builder service
      → service downloads vectors from S3
      → builds Faiss HNSW index on GPU
      → uploads finished index back to S3
  → OpenSearch downloads the GPU-built index and merges it into the shard
```

## Services

| Service | Image | Purpose |
|---|---|---|
| `opensearch` | custom build of `opensearchproject/opensearch` | OpenSearch node with kNN plugin and `repository-s3` plugin |
| `remote-index-builder` | `opensearchproject/remote-vector-index-builder:api-latest` | FastAPI service that builds Faiss indexes on the GPU |
| `bench` | custom Python | Downloads dataset, registers repo + cluster settings, runs cuvs-bench build/search benchmark, exports results, generates plots |

## Requirements

- **Docker Compose v2**
- **ANN benchmark dataset** in binary format (`.fbin`) — see [Dataset format](#dataset-format)
- **GPU mode only** (`--profile gpu`, `REMOTE_INDEX_BUILD=true`):
  - NVIDIA GPU with CUDA support
  - NVIDIA Container Toolkit — [installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
  - AWS S3 bucket for staging vectors and built indexes

## Usage

Set the host kernel parameter required by OpenSearch (once per reboot):

```bash
sudo sysctl -w vm.max_map_count=262144
```

Set required environment variables:

```bash
export DATASET_PATH="$(pwd)/ann-benchmark-datasets"   # directory containing dataset files
```

GPU mode also requires an S3 bucket. Static AWS keys are supported, but optional
when the containers can use another AWS default credential provider such as an
EC2 instance role:

```bash
export S3_BUCKET=opensearch-cuvs-bench            # S3 bucket name
```

If you are using static credentials instead of a default provider, also export:

```bash
export AWS_ACCESS_KEY_ID=<access-key-id>
export AWS_SECRET_ACCESS_KEY=<secret-access-key>
export AWS_SESSION_TOKEN=<session-token>   # required when using temporary (STS) credentials
```

Optionally configure the benchmark:

```bash
export AWS_DEFAULT_REGION=us-west-2        # AWS region for the S3 bucket (default: us-west-2)
export DATASET=sift-128-euclidean          # default
export BENCH_GROUPS=test                   # test | base (default: test)
export K=10                                # number of neighbors (default: 10)
export BATCH_SIZE=                         # optional query batch size override
export BUILD_BATCH_SIZE=                   # optional bulk ingest batch size override
export REMOTE_BUILD_TIMEOUT=1800           # seconds to wait for remote builds (default: 1800)
```

Start all services:

```bash
# CPU build (no GPU required)
docker compose up --build

# GPU build
docker compose --profile gpu up --build
```

The `bench` container logs its progress through each phase. When complete you'll see a results table followed by the paths to the generated plot PNGs under `$DATASET_PATH`.

To tear everything down:

```bash
docker compose down -v
```

## What the bench container does

1. Downloads the dataset (skipped if already present in `$DATASET_PATH`)
2. **GPU mode only**: Registers the S3 bucket as an OpenSearch snapshot repository
3. **GPU mode only**: Applies cluster settings to enable remote index build and point OpenSearch at the builder service
4. Runs `cuvs-bench` build phase (handled entirely by the OpenSearch backend):
   - Creates the kNN index and bulk-ingests dataset vectors
   - **GPU mode**: Flushes segments, waits for all submitted remote GPU builds to complete, and polls the kNN stats API every 5 s until the build is confirmed complete
   - **CPU mode**: Flushes and refreshes the local OpenSearch index
   - Uses the backend's automatic OpenSearch bulk-ingest batch sizing by default; set `BUILD_BATCH_SIZE` to override it
   - Records total build time in the result
5. Runs `cuvs-bench` search phase and prints a recall/QPS/latency table
6. Exports benchmark JSON results to CSV
7. Generates recall vs. latency/throughput plots as PNGs in `$DATASET_PATH` (`cuvs_bench.plot`)

## Dataset format

cuvs-bench reads binary vector files with a simple header:

```
[4 bytes: n_rows as uint32]
[4 bytes: n_cols as uint32]
[n_rows × n_cols × itemsize bytes: vector data]
```

Supported extensions: `.fbin` (float32), `.f16bin` (float16), `.u8bin` (uint8), `.i8bin` (int8).

`DATASET_PATH` should be a directory where each dataset lives in its own subdirectory named after the dataset, e.g.:

```
$DATASET_PATH/
  sift-128-euclidean/
    base.fbin
    query.fbin
    groundtruth.neighbors.ibin
```

## Key configuration

**Cluster settings** (applied by `bench/run.py`):

```json
{
  "persistent": {
    "knn.remote_index_build.enabled": true,
    "knn.remote_index_build.repository": "vector-repo",
    "knn.remote_index_build.service.endpoint": "http://remote-index-builder:1025"
  }
}
```

**Parameter groups** (`BENCH_GROUPS`):

| Group | Build params | Search params | Use case |
|---|---|---|---|
| `test` | 1 combo (m=16) | ef_search: 10, 20 | Quick smoke test |
| `base` | 4 combos (m=[16,32,48,64]) | ef_search: 10, 20, 40, 60, 80, 120, 200, 400, 600, 800 | Standard benchmark |

## GPU build verification

The cuvs-bench OpenSearch backend snapshots remote-build stats before ingest, then polls the kNN stats API every 5 seconds until `index_build_success_count` catches up with `build_request_success_count` and all in-flight flush and merge operations reach zero.

The build raises a `TimeoutError` (causing the `bench` container to exit with code 1) if the expected successful builds are not confirmed within `REMOTE_BUILD_TIMEOUT` seconds. If no remote build is observed shortly after ingest, the backend raises an error that suggests lowering `REMOTE_BUILD_SIZE_MIN`; leave it unset to use OpenSearch's default threshold, or set it explicitly to override that value.

## CPU vs GPU comparison

To compare CPU and GPU builds on the same dataset, run the benchmark twice — once in each mode — clearing the OpenSearch volume between runs so the index is rebuilt from scratch each time:

```bash
# GPU build (starts the remote-index-builder via --profile gpu)
docker compose --profile gpu up --build
docker compose --profile gpu down -v

# CPU build (no GPU or S3 required)
docker compose up --build
docker compose down -v
```

## Running tests

The cuvs-bench OpenSearch backend has three tiers of tests. All run inside the `bench` container so no local Python environment is needed.

**Build the bench image first** (or after any code changes):

```bash
docker compose build --no-cache bench
```

### Unit tests (no server required)

```bash
docker compose run --rm --no-deps bench \
    pytest /opt/cuvs/python/cuvs_bench/cuvs_bench/tests/test_opensearch.py -v
```

### Integration tests (live OpenSearch node only)

Requires a running OpenSearch node. S3 credentials and the GPU profile are not required for these tests.

```bash
docker compose up -d --wait opensearch
docker compose run --rm --no-deps \
    -e OPENSEARCH_URL=http://opensearch:9200 \
    bench \
    pytest /opt/cuvs/python/cuvs_bench/cuvs_bench/tests/test_opensearch.py -v -m integration
```

### Remote index build integration tests (full GPU stack)

Requires the full stack (OpenSearch and the remote index builder), S3 access, and a GPU-capable host. Export the S3 bucket and region before starting OpenSearch. If you are using static AWS keys, export them before startup so OpenSearch can populate its S3 keystore; otherwise the containers can use the AWS default credential provider chain. Use `--profile gpu` when starting the services so Docker Compose includes `remote-index-builder`. The pytest command itself does not need the profile flag because it runs against the already-started services.

When using static AWS keys, map them into the test fixture's `S3_*` variable names:

```bash
export AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-west-2}
docker compose --profile gpu up -d --wait opensearch remote-index-builder
docker compose run --rm --no-deps \
    -e OPENSEARCH_URL=http://opensearch:9200 \
    -e BUILDER_URL=http://remote-index-builder:1025 \
    -e S3_BUCKET=${S3_BUCKET} \
    -e AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION} \
    bench \
    pytest /opt/cuvs/python/cuvs_bench/cuvs_bench/tests/test_opensearch.py -v -m integration
```

This lets the pytest `integration` marker decide which tests run. With only OpenSearch running, remote-build tests skip because the GPU builder and S3 environment are unavailable. With the GPU stack running, the same marker includes the remote-build coverage.

## Ports

| Port | Service |
|---|---|
| `9200` | OpenSearch REST API |
| `1025` | Remote index builder API |
