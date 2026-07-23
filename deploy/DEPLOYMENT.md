# OpenSearch GPU Remote Index Build — Deployment Guide

This guide walks through running OpenSearch with GPU-accelerated vector index construction using the [remote index build service](https://docs.opensearch.org/latest/vector-search/remote-index-build/). When enabled, OpenSearch offloads Faiss HNSW index building to a dedicated GPU service rather than building indexes in-process on the data nodes.

## How it works

The GPU build is triggered automatically during normal ingest — no changes to your indexing workflow are required beyond the one-time cluster and index configuration described below.

```mermaid
sequenceDiagram
    participant Client
    participant OpenSearch
    participant S3
    participant Builder as Remote Index Builder (GPU)

    Client->>OpenSearch: Bulk ingest vectors
    note over OpenSearch: segment flush
    OpenSearch->>S3: Upload raw vectors + doc IDs
    OpenSearch->>Builder: POST /_build (S3 paths)
    Builder->>S3: Download vectors
    note over Builder: Build Faiss HNSW on GPU
    Builder->>S3: Upload .faiss index
    OpenSearch->>S3: Download .faiss index
    note over OpenSearch: Merge into shard
    Client->>OpenSearch: kNN search query
    OpenSearch->>Client: Top-k results
```

## Services

| Service | Image | Purpose |
|---|---|---|
| `opensearch` | custom build of `opensearchproject/opensearch:3.6.0` | OpenSearch node with kNN plugin and `repository-s3` plugin |
| `remote-index-builder` | `opensearchproject/remote-vector-index-builder:api-latest` | GPU-accelerated Faiss HNSW index builder |

The custom OpenSearch image adds the `repository-s3` plugin (required for S3-backed vector staging). When static AWS keys are provided, the image populates the S3 keystore at startup so credentials are never baked into image layers. Without static keys, OpenSearch can fall back to the AWS default credential provider chain, such as an EC2 instance role.

## Requirements

- **Docker Compose v2**
- **NVIDIA GPU** with CUDA support
- **NVIDIA Container Toolkit** — [installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- **AWS S3 bucket** for staging vectors during the build

## Setup

Set the host kernel parameter required by OpenSearch (once per reboot):

```bash
sudo sysctl -w vm.max_map_count=262144
```

Set the required bucket name:

```bash
export S3_BUCKET=opensearch-cuvs-bench
```

If you are using static credentials instead of a default AWS credential provider, also export:

```bash
export AWS_ACCESS_KEY_ID=<access-key-id>
export AWS_SECRET_ACCESS_KEY=<secret-access-key>
export AWS_SESSION_TOKEN=<session-token>   # required for temporary (STS) credentials
```

Optionally configure the region:

```bash
export AWS_DEFAULT_REGION=us-west-2        # default: us-west-2
```

Start OpenSearch and the GPU builder:

```bash
docker compose --profile gpu up --build -d --wait opensearch remote-index-builder
```

## Connecting OpenSearch to the GPU builder

Before any index can use GPU builds, you need to register an S3-backed snapshot repository and apply the cluster settings that point OpenSearch at the builder service. Run these once against a live cluster.

**Register S3 repository:**

```bash
curl -X PUT http://localhost:9200/_snapshot/vector-repo \
  -H "Content-Type: application/json" \
  -d '{
    "type": "s3",
    "settings": {
      "bucket": "opensearch-cuvs-bench",
      "base_path": "knn-indexes",
      "region": "us-west-2"
    }
  }'
```

**Apply cluster settings:**

```bash
curl -X PUT http://localhost:9200/_cluster/settings \
  -H "Content-Type: application/json" \
  -d '{
    "persistent": {
      "knn.remote_index_build.enabled": true,
      "knn.remote_index_build.repository": "vector-repo",
      "knn.remote_index_build.service.endpoint": "http://remote-index-builder:1025"
    }
  }'
```

> **Note:** `remote-index-builder` resolves inside the Docker network. If OpenSearch and the builder are not on the same Docker network, replace this with a reachable hostname or IP.

## Creating an index with GPU builds enabled

Add `"index.knn.remote_index_build.enabled": true` to your index settings alongside the standard kNN configuration:

```bash
curl -X PUT http://localhost:9200/my-vectors \
  -H "Content-Type: application/json" \
  -d '{
    "settings": {
      "index.knn": true,
      "index.knn.remote_index_build.enabled": true,
      "index.knn.remote_index_build.size.min": "1kb",
      "number_of_shards": 1,
      "number_of_replicas": 1
    },
    "mappings": {
      "properties": {
        "vector": {
          "type": "knn_vector",
          "dimension": 256,
          "method": {
            "name": "hnsw",
            "engine": "faiss",
            "space_type": "l2",
            "parameters": {
              "m": 32,
              "ef_construction": 512
            }
          }
        }
      }
    }
  }'
```

GPU builds are only available with the `faiss` engine. The `lucene` engine always builds locally. The low `size.min` value above is useful for demos because it forces small flushed segments onto the remote path; use OpenSearch's default or a larger production threshold for real workloads.

## Verifying the GPU build

The `remote-index-build/` directory contains an end-to-end demo script that ingests 200,000 random vectors, flushes the index to trigger remote builds, and confirms the GPU build completed by polling S3 for the resulting `.faiss` file.

Run it inside a temporary container on the same Docker network:

```bash
docker compose run --rm \
  -e OPENSEARCH_URL=http://opensearch:9200 \
  -e BUILDER_URL=http://remote-index-builder:1025 \
  -e S3_BUCKET=${S3_BUCKET} \
  -e AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-west-2} \
  -e REMOTE_BUILD_SIZE_MIN=${REMOTE_BUILD_SIZE_MIN:-} \
  -e REMOTE_BUILD_TIMEOUT=${REMOTE_BUILD_TIMEOUT:-1800} \
  -v $(pwd)/remote-index-build:/app/remote-index-build \
  --no-deps bench \
  python remote-index-build/run.py
```

Static AWS credential environment variables are passed through by the `bench` service when they are exported on the host.

Or run it directly if you have Python and the dependencies installed locally (`boto3`, `numpy`, `requests`), pointing `OPENSEARCH_URL` at `http://localhost:9200`.

A successful run prints a `.faiss` file path in S3 and returns top-10 nearest-neighbor results.

## Tearing down

```bash
docker compose --profile gpu down -v
```

The `-v` flag removes the OpenSearch data volume. Omit it to preserve indexed data across restarts.

## Production considerations

This setup is a working demonstration, not a production-hardened deployment. Key differences to address before running in production:

- **Security plugin**: `opensearch.yml` has `plugins.security.disabled: true`. Re-enable it and configure TLS and authentication for any non-local deployment.
- **Single-node cluster**: `discovery.type: single-node` bypasses multi-node bootstrap checks. Replace with a properly configured multi-node cluster for production.
- **Replicas**: The demo uses `number_of_replicas: 0`. Set this to at least `1` for production workloads.
- **S3 permissions**: The IAM principal used by OpenSearch and the builder needs `s3:GetObject`, `s3:PutObject`, `s3:ListBucket`, and `s3:DeleteObject` on the staging bucket.

## Ports

| Port | Service |
|---|---|
| `9200` | OpenSearch REST API |
| `1025` | Remote index builder API |
