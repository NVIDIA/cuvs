#!/usr/bin/env bash
# Upload a cuvs-bench result directory to accel-etl.
#
# Usage:
#   ./upload-results.sh <path/to/dataset/result> [extra args to post_cuvs_bench.py...]
#
# Required env:
#   BENCHMARK_API_KEY  Sourced from ~/.config/benchmark/env if not already set.
#   SKU_NAME                    Hardware SKU registered with the API.
#   STORAGE_CONFIGURATION_NAME  Storage configuration registered with the API.
#
# Optional env (with defaults shown):
#   CACHE_STATE                 warm
#   SEARCH_MODE                 latency
#   ENGINE_NAME                 cuvs-bench
#   CUVS_VERSION                read from the installed cuvs package
#   COMMIT_HASH                 read from cuvs-hash.txt in the result directory
#   POST_SCRIPT                 $(dirname $0)/post_cuvs_bench.py
#
# The benchmark definition name is derived per result file from the dataset, k,
# and batch size, so it is not passed here.
#
# Examples:
#   ./upload-results.sh /data/datasets/sift-128-euclidean/result
#   SEARCH_MODE=throughput ./upload-results.sh /data/.../result
#   ./upload-results.sh /data/.../result --dry-run        # passes through

set -euo pipefail

usage() {
    cat <<EOF
Upload a cuvs-bench result directory to accel-etl.

Usage:
  $0 <path/to/dataset/result> [extra args to post_cuvs_bench.py]
  $0 --help

Required env:
  BENCHMARK_API_KEY            Sourced from ~/.config/benchmark/env if unset.
  SKU_NAME                     Hardware SKU registered with the API.
  STORAGE_CONFIGURATION_NAME   Storage configuration registered with the API.

Optional env (with defaults):
  CACHE_STATE                  warm
  SEARCH_MODE                  latency
  ENGINE_NAME                  cuvs-bench
  CUVS_VERSION                 from the installed cuvs package
  COMMIT_HASH                  from cuvs-hash.txt in the result directory
  POST_SCRIPT                  <script_dir>/post_cuvs_bench.py

Examples:
  $0 /data/datasets/sift-128-euclidean/result
  $0 /data/datasets/sift-128-euclidean/result --dry-run
  SEARCH_MODE=throughput $0 /data/datasets/sift-128-euclidean/result
EOF
}

if [ "$#" -lt 1 ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    usage
    [ "$#" -lt 1 ] && exit 2 || exit 0
fi

result_dir="$1"
shift   # remaining args ($@) are passed through to post_cuvs_bench.py

if [ ! -d "$result_dir" ]; then
    echo "error: result directory not found: $result_dir" >&2
    echo "Run '$0 --help' for usage." >&2
    exit 2
fi

# Source API key from the standard location if not in env
if [ -z "${BENCHMARK_API_KEY:-}" ] && [ -f ~/.config/benchmark/env ]; then
    # shellcheck disable=SC1090
    source ~/.config/benchmark/env
fi
: "${BENCHMARK_API_KEY:?Set BENCHMARK_API_KEY (or add it to ~/.config/benchmark/env)}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
post_script="${POST_SCRIPT:-$script_dir/post_cuvs_bench.py}"

if [ ! -f "$post_script" ]; then
    echo "error: post_cuvs_bench.py not found at: $post_script" >&2
    exit 2
fi

sku_name="${SKU_NAME:?Set SKU_NAME to the hardware SKU registered with the API}"
storage_name="${STORAGE_CONFIGURATION_NAME:?Set STORAGE_CONFIGURATION_NAME to the storage configuration registered with the API}"
cache_state="${CACHE_STATE:-warm}"
search_mode="${SEARCH_MODE:-latency}"
engine_name="${ENGINE_NAME:-cuvs-bench}"

# The result files carry the Google Benchmark version, not the cuVS one.
cuvs_version="${CUVS_VERSION:-$(python -c 'import cuvs; print(cuvs.__version__)' 2>/dev/null || echo unknown)}"
# run-benchmarks.sh writes this next to the results.
commit_hash="${COMMIT_HASH:-$(cat "$result_dir/cuvs-hash.txt" 2>/dev/null || echo unknown)}"

echo "Uploading $result_dir"
echo "  sku-name:                   $sku_name"
echo "  storage-configuration-name: $storage_name"
echo "  cache-state:                $cache_state"
echo "  search-mode:                $search_mode"
echo "  engine-name:                $engine_name"
echo "  cuvs-version:               $cuvs_version"
echo "  commit-hash:                $commit_hash"
[ "$#" -gt 0 ] && echo "  extra args:                 $*"

post_args=(
    python "$post_script" "$result_dir"
    --api-key "$BENCHMARK_API_KEY"
    --sku-name "$sku_name"
    --storage-configuration-name "$storage_name"
    --cache-state "$cache_state"
    --search-mode "$search_mode"
    --engine-name "$engine_name"
    --cuvs-version "$cuvs_version"
    --commit-hash "$commit_hash"
    # Search mode is the one axis absent from the result file name, so the two
    # modes are otherwise indistinguishable once submitted.
    --labels "search-mode:$search_mode"
)
post_args+=("$@")

exec "${post_args[@]}"
