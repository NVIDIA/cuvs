#!/usr/bin/env python3
"""
OpenSearch GPU Remote Index Build Benchmark
===========================================
Steps:
  1. Register S3 snapshot repository with OpenSearch
  2. Configure cluster settings for GPU remote index build
  3. Build kNN index via cuvs-bench:
       a. Bulk-ingest dataset vectors
       b. Flush segments to kick off remote GPU builds when enabled
       c. Poll kNN stats until every submitted remote build completes
  4. Run cuvs-bench search benchmarks and print results
  5. Write gbench-compatible JSON result files for CSV export and plotting
"""

import json
import os
import sys

import requests
from cuvs_bench.orchestrator import BenchmarkOrchestrator
from cuvs_bench.backends.base import BuildResult, SearchResult


OPENSEARCH_URL  = os.environ.get("OPENSEARCH_URL",  "http://opensearch:9200")
OPENSEARCH_HOST = os.environ.get("OPENSEARCH_HOST", "opensearch")
OPENSEARCH_PORT = int(os.environ.get("OPENSEARCH_PORT", "9200"))
BUILDER_URL     = os.environ.get("BUILDER_URL",     "http://remote-index-builder:1025")

REMOTE_INDEX_BUILD = os.environ.get("REMOTE_INDEX_BUILD", "false").lower() == "true"
REMOTE_BUILD_SIZE_MIN = os.environ.get("REMOTE_BUILD_SIZE_MIN", "").strip()
REMOTE_BUILD_TIMEOUT = int(os.environ.get("REMOTE_BUILD_TIMEOUT", "1800"))

S3_BUCKET  = os.environ.get("S3_BUCKET", "").strip()
S3_PREFIX  = os.environ.get("S3_PREFIX", "knn-indexes").strip() or "knn-indexes"
S3_REGION  = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")

DATASET      = os.environ.get("DATASET",       "sift-128-euclidean")
DATASET_PATH = os.environ.get("DATASET_PATH",  "/data/datasets")
BENCH_GROUPS = os.environ.get("BENCH_GROUPS",  "test")
K            = int(os.environ.get("K", "10"))
BATCH_SIZE = os.environ.get("BATCH_SIZE", "").strip()
BATCH_SIZE = int(BATCH_SIZE) if BATCH_SIZE else None
BUILD_BATCH_SIZE = os.environ.get("BUILD_BATCH_SIZE", "").strip()
BUILD_BATCH_SIZE = int(BUILD_BATCH_SIZE) if BUILD_BATCH_SIZE else None

ALGORITHM = "opensearch_faiss_hnsw"
REPO_NAME = os.environ.get("REMOTE_VECTOR_REPOSITORY", "vector-repo").strip()
REPO_NAME = REPO_NAME or "vector-repo"

session = requests.Session()
session.headers.update({"Content-Type": "application/json"})


# ── helpers ───────────────────────────────────────────────────────────────────

def banner(msg: str) -> None:
    print(f"\n{'─'*60}\n  {msg}\n{'─'*60}")


def _recall_for_entry(
    result: SearchResult, entry_index: int, entry_count: int
) -> float | None:
    # cuVS computes recall in the orchestrator from SearchResult.neighbors. The
    # OpenSearch backend returns neighbors for the final search-parameter run.
    if entry_index == entry_count - 1:
        return float(result.recall)
    return None


def _get_search_batch_size(search_results: list) -> int | None:
    if BATCH_SIZE is not None:
        return BATCH_SIZE
    for result in search_results:
        if not isinstance(result, SearchResult):
            continue
        batch_size = (result.metadata or {}).get("batch_size")
        if batch_size is not None:
            return int(batch_size)
    return None


# ── OpenSearch setup ──────────────────────────────────────────────────────────

def register_repository() -> None:
    banner(f"Registering S3 repository '{REPO_NAME}'")
    r = session.put(
        f"{OPENSEARCH_URL}/_snapshot/{REPO_NAME}",
        json={
            "type": "s3",
            "settings": {
                "bucket":    S3_BUCKET,
                "base_path": S3_PREFIX,
                "region":    S3_REGION,
            },
        },
    )
    r.raise_for_status()
    print(f"  {r.json()}")


def configure_cluster() -> None:
    if REMOTE_INDEX_BUILD:
        banner("Enabling GPU remote index build (cluster settings)")
        settings = {
            "knn.remote_index_build.enabled":          True,
            "knn.remote_index_build.repository":       REPO_NAME,
            "knn.remote_index_build.service.endpoint": BUILDER_URL,
        }
    else:
        banner("Disabling GPU remote index build (cluster settings)")
        settings = {
            "knn.remote_index_build.enabled": False,
        }
    r = session.put(
        f"{OPENSEARCH_URL}/_cluster/settings",
        json={"persistent": settings},
    )
    r.raise_for_status()
    print(f"  {r.json()}")


# ── result files ─────────────────────────────────────────────────────────────

def write_result_files(
    build_results: list,
    search_results: list,
    dataset: str,
    dataset_path: str,
    algo: str,
    groups: str,
    k: int,
    batch_size: int,
) -> None:
    """Write gbench-compatible JSON result files.

    Creates files under <dataset_path>/<dataset>/result/{build,search}/ in the
    same format the C++ backend produces, so the cuvs-bench CSV exporters and
    ``cuvs_bench.plot`` work without modification.
    """
    build_dir = os.path.join(dataset_path, dataset, "result", "build")
    search_dir = os.path.join(dataset_path, dataset, "result", "search")
    os.makedirs(build_dir, exist_ok=True)
    os.makedirs(search_dir, exist_ok=True)

    # Build JSON – one record per successfully built index.
    # data_export.py assumes the build CSV has columns:
    #   [algo_name, index_name, time, threads, cpu_time, ...]
    # so "threads" and "cpu_time" must be present in the JSON (they are not in
    # skip_build_cols and therefore get included as columns 3 and 4).
    build_benchmarks = [
        {
            "name": r.index_path,
            "real_time": r.build_time_seconds,
            "time_unit": "s",
            "threads": 1,
            "cpu_time": r.build_time_seconds,
        }
        for r in build_results
        if isinstance(r, BuildResult) and r.success and r.index_path
    ]

    # Search JSON – zip build + search results to recover the index name, then
    # expand per-search-param entries so each (index, ef_search) is one record.
    build_list  = [r for r in build_results  if isinstance(r, BuildResult)]
    search_list = [r for r in search_results if isinstance(r, SearchResult)]
    search_benchmarks = []
    skipped_without_recall = 0
    for build_r, search_r in zip(build_list, search_list):
        if not search_r.success or not build_r.index_path:
            continue
        per_param = (search_r.metadata or {}).get("per_search_param_results", [])
        for entry_index, entry in enumerate(per_param):
            recall = _recall_for_entry(search_r, entry_index, len(per_param))
            if recall is None:
                skipped_without_recall += 1
                continue
            latency_ms = float(entry["search_time_ms"])
            search_benchmarks.append(
                {
                    "name": build_r.index_path,
                    "real_time": latency_ms,
                    "time_unit": "ms",
                    "Recall": recall,
                    "items_per_second": float(entry["queries_per_second"]),
                    # Latency field expected by data_export in seconds
                    "Latency": latency_ms / 1000.0,
                }
            )

    build_file  = os.path.join(build_dir,  f"{algo},{groups}.json")
    search_file = os.path.join(
        search_dir, f"{algo},{groups},k{k},bs{batch_size}.json"
    )

    with open(build_file, "w") as fh:
        json.dump({"benchmarks": build_benchmarks}, fh, indent=2)
    with open(search_file, "w") as fh:
        json.dump({"benchmarks": search_benchmarks}, fh, indent=2)

    print(f"\n  Result files written:")
    print(f"    {build_file}")
    print(f"    {search_file}")
    if skipped_without_recall:
        print(
            "    skipped "
            f"{skipped_without_recall} search rows without per-parameter recall"
        )


# ── results ───────────────────────────────────────────────────────────────────

def _print_result_row(
    params: dict, recall: float | None, qps: float | None, latency_ms: float | None
) -> None:
    params_str = ", ".join(f"{k}={v}" for k, v in params.items())
    recall_str = "n/a" if recall is None else f"{recall:.4f}"
    qps_str = "n/a" if qps is None else f"{qps:.1f}"
    latency_str = "n/a" if latency_ms is None else f"{latency_ms:.2f}"
    print(f"  {params_str:<40}  {recall_str:<12}  {qps_str:>8}  {latency_str:>12}")


def print_results(results: list) -> None:
    banner("Benchmark Results")
    search_results = [r for r in results if isinstance(r, SearchResult)]
    if not search_results:
        print("  No search results returned.")
        return

    header = f"  {'params':<40}  {'recall@'+str(K):<12}  {'QPS':>8}  {'latency (ms)':>12}"
    print(header)
    print("  " + "─" * (len(header) - 2))
    missing_recall_rows = 0
    for r in search_results:
        per_param = (r.metadata or {}).get("per_search_param_results", [])
        entry_count = len(per_param)
        for entry_index, entry in enumerate(per_param):
            recall = _recall_for_entry(r, entry_index, entry_count)
            if recall is None:
                missing_recall_rows += 1
                continue
            _print_result_row(
                entry["search_params"],
                recall,
                float(entry["queries_per_second"]),
                float(entry["search_time_ms"]),
            )
    if missing_recall_rows:
        print(
            "\n  Omitted "
            f"{missing_recall_rows} timing-only search rows without recall."
        )


# ── entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    if REMOTE_INDEX_BUILD and not S3_BUCKET:
        print(
            "ERROR: S3_BUCKET is not set. Remote index build requires an S3 "
            "bucket for vector and index staging. Set it before starting the "
            "stack, for example: export S3_BUCKET=<your-s3-bucket>",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\n" + "═" * 60)
    print("  OpenSearch kNN Benchmark")
    print("═" * 60)
    print(f"  OpenSearch         : {OPENSEARCH_URL}")
    print(f"  Remote index build : {REMOTE_INDEX_BUILD}")
    if REMOTE_INDEX_BUILD:
        print(f"  GPU builder        : {BUILDER_URL}")
        print(f"  S3 bucket          : s3://{S3_BUCKET}/{S3_PREFIX}/  (region: {S3_REGION})")
        print(f"  Repository         : {REPO_NAME}")
        print(f"  Build size minimum : {REMOTE_BUILD_SIZE_MIN or 'OpenSearch default'}")
        print(f"  Build timeout      : {REMOTE_BUILD_TIMEOUT}s")
    print(f"  Dataset            : {DATASET}  (path: {DATASET_PATH})")
    print(f"  Algorithm          : {ALGORITHM}")
    print(f"  Groups             : {BENCH_GROUPS}  k={K}")
    print(
        "  Search batch size  : "
        f"{BATCH_SIZE if BATCH_SIZE is not None else 'backend default'}"
    )
    print(
        "  Build batch size   : "
        f"{BUILD_BATCH_SIZE if BUILD_BATCH_SIZE is not None else 'backend auto'}"
    )

    if REMOTE_INDEX_BUILD:
        register_repository()
    configure_cluster()

    orchestrator = BenchmarkOrchestrator(backend_type="opensearch")

    # Shared kwargs for both build and search phases.
    common_kwargs = dict(
        dataset=DATASET,
        dataset_path=DATASET_PATH,
        algorithms=ALGORITHM,
        groups=BENCH_GROUPS,
        host=OPENSEARCH_HOST,
        port=OPENSEARCH_PORT,
        use_ssl=False,
        verify_certs=False,
    )

    build_kwargs = dict(
        common_kwargs,
        remote_index_build=REMOTE_INDEX_BUILD,
    )
    if BUILD_BATCH_SIZE is not None:
        build_kwargs["build_batch_size"] = BUILD_BATCH_SIZE
    if REMOTE_INDEX_BUILD:
        build_kwargs["remote_build_timeout"] = REMOTE_BUILD_TIMEOUT
        if REMOTE_BUILD_SIZE_MIN:
            build_kwargs["remote_build_size_min"] = REMOTE_BUILD_SIZE_MIN

    # ── Build phase ───────────────────────────────────────────────────────────
    mode = "GPU remote build" if REMOTE_INDEX_BUILD else "CPU"
    banner(f"Building index ({mode} via cuvs-bench)")
    build_results = orchestrator.run_benchmark(
        build=True,
        search=False,
        force=True,
        **build_kwargs,
    )

    index_names = [
        r.index_path
        for r in build_results
        if isinstance(r, BuildResult) and r.success and r.index_path
    ]
    if not index_names:
        print("  ERROR: no indexes were successfully built")
        sys.exit(1)
    build_times = {
        r.index_path: r.build_time_seconds
        for r in build_results
        if isinstance(r, BuildResult) and r.success and r.index_path
    }
    for name, t in build_times.items():
        print(f"  {name}: built in {t:.1f}s")

    # ── Search phase ──────────────────────────────────────────────────────────
    banner("Running search benchmarks (via cuvs-bench)")
    search_run_kwargs = dict(
        build=False,
        search=True,
        count=K,
        **common_kwargs,
    )
    if BATCH_SIZE is not None:
        search_run_kwargs["batch_size"] = BATCH_SIZE
    search_results = orchestrator.run_benchmark(**search_run_kwargs)

    print_results(search_results)

    batch_size = _get_search_batch_size(search_results)
    if batch_size is None:
        print("  ERROR: no successful search result reported a batch size")
        sys.exit(1)

    write_result_files(
        build_results=build_results,
        search_results=search_results,
        dataset=DATASET,
        dataset_path=DATASET_PATH,
        algo=common_kwargs["algorithms"],
        groups=BENCH_GROUPS,
        k=K,
        batch_size=batch_size,
    )

    print("\n" + "═" * 60)
    print("  Benchmark complete!")
    print("═" * 60)
    print(f"\n  OpenSearch         : {OPENSEARCH_URL}")
    if REMOTE_INDEX_BUILD:
        print(f"  GPU builder        : {BUILDER_URL}")
    print()


if __name__ == "__main__":
    main()
