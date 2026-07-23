#!/usr/bin/env python3
"""
OpenSearch GPU Remote Index Build — End-to-End Demo
====================================================
Steps:
  1. Register an S3 snapshot repository with OpenSearch
  2. Configure cluster settings to enable GPU-based remote index building
  3. Create a kNN index (Faiss HNSW / L2) with remote build enabled
  4. Ingest 200,000 random 256-dimensional float vectors via the bulk API (8 parallel workers)
  5. Flush segments to trigger the GPU build
  6. Poll S3 for a .faiss file — hard-fail if the GPU build never completes
  7. Execute a kNN search and print the top-10 nearest neighbors
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import numpy as np
import requests

OPENSEARCH_URL = os.environ.get("OPENSEARCH_URL", "http://opensearch:9200")
BUILDER_URL    = os.environ.get("BUILDER_URL",    "http://remote-index-builder:1025")

S3_BUCKET = os.environ.get("S3_BUCKET", "").strip()
S3_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
# boto3 uses AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY when set, otherwise it
# falls through to the default credential provider chain.

INDEX_NAME = "gpu-demo"
DIMENSION  = 256    # matches common embedding model output sizes
NUM_DOCS   = 200_000
REPO_NAME  = "vector-repo"
REMOTE_BUILD_SIZE_MIN = os.environ.get("REMOTE_BUILD_SIZE_MIN", "").strip()
REMOTE_BUILD_TIMEOUT = int(os.environ.get("REMOTE_BUILD_TIMEOUT", "1800"))

session = requests.Session()
session.headers.update({"Content-Type": "application/json"})


def banner(msg: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {msg}")
    print(f"{'─'*60}")


# ── configuration ─────────────────────────────────────────────────────────────

def register_repository() -> None:
    banner(f"Registering S3 repository '{REPO_NAME}'")
    r = session.put(
        f"{OPENSEARCH_URL}/_snapshot/{REPO_NAME}",
        json={
            "type": "s3",
            "settings": {
                "bucket":    S3_BUCKET,
                "base_path": "knn-indexes",
                "region":    S3_REGION,
            },
        },
    )
    r.raise_for_status()
    print(f"  {r.json()}")


def configure_cluster() -> None:
    banner("Enabling remote GPU index build (cluster settings)")
    r = session.put(
        f"{OPENSEARCH_URL}/_cluster/settings",
        json={
            "persistent": {
                "knn.remote_index_build.enabled":          True,
                "knn.remote_index_build.repository":       REPO_NAME,
                "knn.remote_index_build.service.endpoint": BUILDER_URL,
            }
        },
    )
    r.raise_for_status()
    print(f"  {r.json()}")


# ── index ─────────────────────────────────────────────────────────────────────

def create_index() -> None:
    banner(f"Creating kNN index '{INDEX_NAME}'")

    resp = session.delete(f"{OPENSEARCH_URL}/{INDEX_NAME}")
    if resp.status_code == 200:
        print("  Deleted existing index")

    index_settings = {
        "index.knn": True,
        "index.knn.remote_index_build.enabled": True,
        "number_of_shards": 1,
        "number_of_replicas": 0,
    }
    if REMOTE_BUILD_SIZE_MIN:
        index_settings["index.knn.remote_index_build.size.min"] = (
            REMOTE_BUILD_SIZE_MIN
        )

    r = session.put(
        f"{OPENSEARCH_URL}/{INDEX_NAME}",
        json={
            "settings": index_settings,
            "mappings": {
                "properties": {
                    "vector": {
                        "type":      "knn_vector",
                        "dimension": DIMENSION,
                        "method": {
                            "name":       "hnsw",
                            "engine":     "faiss",
                            "space_type": "l2",
                            "parameters": {"m": 32, "ef_construction": 512},
                        },
                    },
                    "doc_id": {"type": "integer"},
                    "label":  {"type": "keyword"},
                }
            },
        },
    )
    r.raise_for_status()
    print(f"  {r.json()}")


# ── ingest ────────────────────────────────────────────────────────────────────

def ingest_vectors() -> None:
    batch_size = 500
    banner(
        f"Ingesting {NUM_DOCS:,} random {DIMENSION}-dim vectors "
        "(bulk API, 8 workers)"
    )

    def send_batch(start: int) -> int:
        end  = min(start + batch_size, NUM_DOCS)
        vecs = np.random.randn(end - start, DIMENSION).astype(np.float32)
        lines = []
        for i, vec in enumerate(vecs, start):
            lines.append(
                json.dumps({"index": {"_index": INDEX_NAME, "_id": str(i)}})
            )
            lines.append(
                json.dumps(
                    {
                        "vector": vec.tolist(),
                        "doc_id": i,
                        "label": f"item-{i:04d}",
                    }
                )
            )
        payload = ("\n".join(lines) + "\n").encode("utf-8")
        r = session.post(
            f"{OPENSEARCH_URL}/_bulk",
            data=payload,
            headers={"Content-Type": "application/x-ndjson"},
        )
        r.raise_for_status()
        body = r.json()
        if body.get("errors"):
            failed = [
                item["index"]["error"]
                for item in body["items"]
                if "error" in item.get("index", {})
            ]
            print(
                f"  Warning: {len(failed)} error(s) in batch "
                f"{start}–{end}: {failed[0]}"
            )
            return (end - start) - len(failed)
        return end - start

    ingested = 0
    starts = list(range(0, NUM_DOCS, batch_size))
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(send_batch, s): s for s in starts}
        for future in as_completed(futures):
            ingested += future.result()
            if ingested % 10_000 == 0 or ingested >= NUM_DOCS:
                print(f"  Ingested {ingested:,}/{NUM_DOCS:,}")

    session.post(f"{OPENSEARCH_URL}/{INDEX_NAME}/_refresh")
    r = session.get(f"{OPENSEARCH_URL}/{INDEX_NAME}/_count")
    print(f"  Document count after ingest: {r.json()['count']:,}")


# ── GPU build ─────────────────────────────────────────────────────────────────

def trigger_gpu_build() -> None:
    banner("Triggering GPU index build via flush")
    print(
        "  OpenSearch will upload eligible flushed segments to S3, "
        "then call the GPU builder."
    )
    r = session.post(f"{OPENSEARCH_URL}/{INDEX_NAME}/_flush", timeout=300)
    r.raise_for_status()
    print(f"  Flush complete: {r.json()}")


def verify_gpu_build(timeout: int = REMOTE_BUILD_TIMEOUT) -> None:
    """Confirm the GPU builder uploaded a .faiss index file to S3.

    The remote-index-builder is the *only* component that writes .faiss files
    back to the S3 bucket, so their presence is definitive proof that the GPU
    build completed.  The kNN stats API does not expose remote build counters
    in OpenSearch 3.x, so we poll S3 directly via boto3 instead.

    Exits with code 1 if no .faiss file appears within `timeout` seconds.
    """
    banner("Verifying GPU index build (polling S3 for .faiss files)")
    print(f"  Bucket  : s3://{S3_BUCKET}/knn-indexes/")
    print(f"  Timeout : {timeout}s  (poll interval: 5s)\n")

    # boto3 uses static AWS env vars when set, otherwise it falls through to
    # the default credential provider chain.
    s3 = boto3.client("s3", region_name=S3_REGION)

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix="knn-indexes/")
            faiss_files = [
                obj["Key"]
                for obj in resp.get("Contents", [])
                if obj["Key"].endswith(".faiss")
            ]
            if faiss_files:
                print(
                    "  PASS: GPU build confirmed — "
                    f"{len(faiss_files)} .faiss file(s) in S3:"
                )
                for f in faiss_files:
                    print(f"    s3://{S3_BUCKET}/{f}")
                return

            remaining = int(deadline - time.time())
            all_keys = [obj["Key"] for obj in resp.get("Contents", [])]
            print(
                f"  Waiting for .faiss file...  objects={all_keys}  "
                f"({remaining}s left)"
            )
        except Exception as e:
            print(f"  S3 check error: {e}")
        time.sleep(5)

    print(f"\n  FAIL: no GPU-built .faiss index appeared in S3 after {timeout}s")
    print("\n  Possible causes:")
    print("  1. remote-index-builder is unreachable from the OpenSearch container.")
    print(
        "     Verify the container is running and "
        f"BUILDER_URL={BUILDER_URL} is correct."
    )
    print("  2. Segment size never exceeded index.knn.remote_index_build.size.min.")
    print("     Try increasing NUM_DOCS or lowering the size.min threshold.")
    print("  3. No GPU is available inside the remote-index-builder container.")
    print("     Check: docker compose logs remote-index-builder")
    print("     Ensure the NVIDIA Container Toolkit is installed on the host.")
    sys.exit(1)


# ── search ────────────────────────────────────────────────────────────────────

def search_vectors() -> None:
    banner("kNN test search (top-10 nearest neighbors)")
    query_vec = np.random.randn(DIMENSION).astype(np.float32).tolist()

    r = session.post(
        f"{OPENSEARCH_URL}/{INDEX_NAME}/_search",
        json={
            "size": 10,
            "query": {"knn": {"vector": {"vector": query_vec, "k": 10}}},
            "_source": ["doc_id", "label"],
        },
    )
    r.raise_for_status()
    hits  = r.json()["hits"]["hits"]
    total = r.json()["hits"]["total"]["value"]

    print(f"  Index contains {total} documents")
    print(f"  Top {len(hits)} results:")
    for rank, hit in enumerate(hits, 1):
        src = hit["_source"]
        print(f"    #{rank:>2}  id={hit['_id']:>6}  score={hit['_score']:.6f}  label={src['label']}")


# ── entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    if not S3_BUCKET:
        print(
            "ERROR: S3_BUCKET is not set. The remote index build demo requires "
            "an S3 bucket for vector and index staging. Set it before running, "
            "for example: export S3_BUCKET=<your-s3-bucket>",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\n" + "═" * 60)
    print("  OpenSearch GPU Remote Index Build — End-to-End Demo")
    print("═" * 60)
    print(f"  OpenSearch : {OPENSEARCH_URL}")
    print(f"  GPU builder: {BUILDER_URL}")
    print(f"  S3 bucket  : s3://{S3_BUCKET}/knn-indexes/  (region: {S3_REGION})")
    print(
        f"  Vectors    : {NUM_DOCS} × dim={DIMENSION}  "
        "engine=faiss  method=hnsw  space=l2"
    )
    print(f"  Build size minimum: {REMOTE_BUILD_SIZE_MIN or 'OpenSearch default'}")
    print(f"  Build timeout: {REMOTE_BUILD_TIMEOUT}s")

    register_repository()
    configure_cluster()
    create_index()
    ingest_vectors()
    trigger_gpu_build()
    verify_gpu_build(timeout=REMOTE_BUILD_TIMEOUT)
    search_vectors()

    print("\n" + "═" * 60)
    print("  Demo complete!")
    print("═" * 60)
    print(f"\n  OpenSearch is still running at {OPENSEARCH_URL}")
    print(f"  GPU builder   : {BUILDER_URL}")
    print()


if __name__ == "__main__":
    main()
