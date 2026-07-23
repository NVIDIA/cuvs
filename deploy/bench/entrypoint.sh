#!/bin/bash
set -e

DATASET="${DATASET:-sift-128-euclidean}"
BENCH_GROUPS="${BENCH_GROUPS:-test}"
K="${K:-10}"
ALGORITHM="opensearch_faiss_hnsw"

wait_for_builder() {
    builder_url="${BUILDER_URL:-http://remote-index-builder:1025}"
    echo "Remote index build enabled — waiting for builder at ${builder_url}..."
    until BUILDER_URL="${builder_url}" python3 -c 'import os, socket; from urllib.parse import urlparse; url = urlparse(os.environ["BUILDER_URL"]); socket.create_connection((url.hostname, url.port or 1025), 2).close()' 2>/dev/null; do
        sleep 5
    done
    echo "Remote index builder is ready."
}

if [ -n "${REMOTE_INDEX_BUILD:-}" ]; then
    case "${REMOTE_INDEX_BUILD,,}" in
        true|1|yes)
            wait_for_builder
            export REMOTE_INDEX_BUILD=true
            ;;
        false|0|no)
            echo "REMOTE_INDEX_BUILD=false — using CPU build mode."
            export REMOTE_INDEX_BUILD=false
            ;;
        *)
            echo "ERROR: REMOTE_INDEX_BUILD must be true or false when set (got '${REMOTE_INDEX_BUILD}')" >&2
            exit 1
            ;;
    esac
else
    # Auto-detect GPU mode: remote-index-builder only appears in Docker DNS when
    # started via --profile gpu. DNS entries are registered at network setup time
    # (before containers run), so this check is reliable by the time entrypoint
    # executes (OpenSearch healthy check alone takes 30+ seconds).
    if getent hosts remote-index-builder > /dev/null 2>&1; then
        wait_for_builder
        export REMOTE_INDEX_BUILD=true
    else
        echo "remote-index-builder not available — using CPU build mode."
        export REMOTE_INDEX_BUILD=false
    fi
fi

# Step 1: Download dataset (skipped automatically if already present)
python -m cuvs_bench.get_dataset \
    --dataset "$DATASET" \
    --dataset-path /data/datasets

# Step 2: Run benchmark (build + search + writes result JSON files)
python -u run.py

# Step 3: Export JSON → CSV (required by cuvs_bench.plot)
# --batch-size is ignored when --data-export is set, but Click prompts for it
# before entering main(), so pass a dummy value to keep the container non-interactive.
python -m cuvs_bench.run --data-export \
    --dataset "$DATASET" \
    --dataset-path /data/datasets \
    --algorithms "$ALGORITHM" \
    --groups "$BENCH_GROUPS" \
    --count "$K" \
    --batch-size 1 \
    --search-mode latency

# Step 4: Plot — PNGs written to /data/datasets (mounted from host $DATASET_PATH)
python -m cuvs_bench.plot \
    --dataset "$DATASET" \
    --dataset-path /data/datasets \
    --algorithms "$ALGORITHM" \
    --groups "$BENCH_GROUPS" \
    --count "$K" \
    --output-filepath /data/datasets
