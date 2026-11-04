#!/bin/bash
set -x
set -e

# Run a cuvs-bench cadence and write the results and run metadata to
# ${RESULTS_DIR}. Uploading is a separate step; see upload-results.sh.
#
# Usage: run-benchmarks.sh [nightly|weekly|monthly] [options]

CADENCE="${1:-nightly}"
shift || true

SEARCH_MODE="latency"
BATCH_SIZE="1"
DATASETS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --search-mode)
            SEARCH_MODE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --datasets)
            DATASETS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# The three cadences form a ladder. Nightly narrows the parameter sweep so a
# regression shows up within minutes. Weekly runs the full sweep so the whole
# recall/throughput curve is covered. Monthly adds the large datasets and the
# second k value, which are too slow to run more often.
SMALL_DATASETS="fashion-mnist-784-euclidean sift-128-euclidean glove-100-angular
gist-960-euclidean nytimes-256-angular deep-image-96-inner"
LARGE_DATASETS="wiki_all_1M wiki_all_10M deep-100M bigann-100M"

# Not named GROUPS: that is a bash builtin holding the caller's group IDs, and
# assigning to it silently leaves it as "0".
case "${CADENCE}" in
    nightly)
        CONFIG_GROUPS="nightly"
        CADENCE_DATASETS="fashion-mnist-784-euclidean sift-128-euclidean"
        K_VALUES="10"
        ;;
    weekly)
        CONFIG_GROUPS="base"
        CADENCE_DATASETS="${SMALL_DATASETS}"
        K_VALUES="10"
        ;;
    monthly)
        CONFIG_GROUPS="base"
        CADENCE_DATASETS="${SMALL_DATASETS} ${LARGE_DATASETS}"
        K_VALUES="10 100"
        ;;
    *)
        echo "Unknown cadence: ${CADENCE} (expected nightly, weekly, or monthly)"
        exit 1
        ;;
esac

DATASETS="${DATASETS:-${CADENCE_DATASETS}}"
DATASET_PATH="${DATASET_PATH:-${RESULTS_DIR}/datasets}"
ALL_ALGORITHMS="cuvs_cagra,cuvs_ivf_pq,cuvs_ivf_flat,cuvs_brute_force"
# Exact search over 10M or more vectors is not a useful comparison point.
NO_BRUTE_FORCE_ALGORITHMS="cuvs_cagra,cuvs_ivf_pq,cuvs_ivf_flat"

source /opt/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-base}"
# Executable discovery reads CONDA_PREFIX. Without it every algorithm is
# skipped with "Executable not found".
export CONDA_PREFIX="${CONDA_PREFIX}"

echo "Gather Metadata..."

git rev-parse HEAD > "$RESULTS_DIR/cuvs-hash.txt"

DRIVER_VERSION=`nvidia-smi --query-gpu=driver_version --format=csv,noheader --id=0`
echo $DRIVER_VERSION > "$RESULTS_DIR/driver-version.txt"

conda env export --json > "$RESULTS_DIR/conda-env.json"

nvidia-smi > "$RESULTS_DIR/nvidia-smi.txt"

for dataset in ${DATASETS}; do
    if [[ " ${LARGE_DATASETS} " == *" ${dataset} "* ]]; then
        algorithms="${NO_BRUTE_FORCE_ALGORITHMS}"
    else
        algorithms="${ALL_ALGORITHMS}"
    fi

    echo "[Running ${CADENCE} ${dataset}]"

    # Build once per dataset. An index does not depend on k, batch size, or
    # search mode. --force so a cached index cannot skip the build, which
    # would leave no build time to trend.
    python -m cuvs_bench.run \
        --dataset "${dataset}" \
        --dataset-path "${DATASET_PATH}" \
        --algorithms "${algorithms}" \
        --groups "${CONFIG_GROUPS}" \
        --search-mode "${SEARCH_MODE}" \
        --build --force

    for k in ${K_VALUES}; do
        # --search-mode is explicit because the option prompts when it is
        # missing, which would hang a scheduled job.
        python -m cuvs_bench.run \
            --dataset "${dataset}" \
            --dataset-path "${DATASET_PATH}" \
            --algorithms "${algorithms}" \
            --groups "${CONFIG_GROUPS}" \
            -k "${k}" \
            --batch-size "${BATCH_SIZE}" \
            --search-mode "${SEARCH_MODE}" \
            --search --force
    done
done

echo "Finished running ${CADENCE} cuvs-bench"
