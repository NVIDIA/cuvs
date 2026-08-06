#!/bin/bash
set -e

DATASET="${DATASET:-sift-128-euclidean}"
CUSTOM_DATASET="miracl-en-5m-1024d-fp32"
BENCH_GROUPS="${BENCH_GROUPS:-test}"
K="${K:-10}"
BATCH_SIZE="${BATCH_SIZE:-10000}"
ALGORITHM="opensearch_faiss_hnsw"
BACKEND_CONFIG="/tmp/opensearch-backend.yaml"

export DATASET
if [ "$DATASET" = "$CUSTOM_DATASET" ]; then
    export DATASET_CONFIGURATION="/data/datasets/${DATASET}/config.yaml"
else
    unset DATASET_CONFIGURATION
fi

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

# Step 1: Prepare the selected dataset (skipped when files already exist).
if [ "$DATASET" = "$CUSTOM_DATASET" ]; then
    python -u prepare_custom_dataset.py
else
    python -m cuvs_bench.get_dataset \
        --dataset "$DATASET" \
        --dataset-path /data/datasets
fi

# Step 2: Configure OpenSearch and write the backend configuration.
python -u configure_opensearch.py "$BACKEND_CONFIG"

# Step 3: Run the standard cuvs-bench CLI. Python backends write plotting CSV
# files automatically.
run_args=(
    python -m cuvs_bench.run
    --backend-config "$BACKEND_CONFIG"
    --dataset "$DATASET"
    --dataset-path /data/datasets
    --algorithms "$ALGORITHM"
    --groups "$BENCH_GROUPS"
    --count "$K"
    --batch-size "$BATCH_SIZE"
    --search-mode latency
    --build
    --search
    --force
)
if [ -n "${DATASET_CONFIGURATION:-}" ]; then
    run_args+=(--dataset-configuration "$DATASET_CONFIGURATION")
fi
"${run_args[@]}"

# Step 4: Print a compact overview of the generated results.
python -u print_results.py \
    --dataset-path /data/datasets \
    --dataset "$DATASET" \
    --algorithm "$ALGORITHM" \
    --groups "$BENCH_GROUPS" \
    --count "$K" \
    --batch-size "$BATCH_SIZE"

# Step 5: Plot — PNGs written to /data/datasets (mounted from host $DATASET_PATH)
python -m cuvs_bench.plot \
    --dataset "$DATASET" \
    --dataset-path /data/datasets \
    --algorithms "$ALGORITHM" \
    --groups "$BENCH_GROUPS" \
    --count "$K" \
    --batch-size "$BATCH_SIZE" \
    --raw \
    --output-filepath /data/datasets
