#!/usr/bin/env python3
"""
CLI for loading cuvs-bench results and posting them to the API Gateway.

This script parses the result tree written by `python -m cuvs_bench.run` and
converts it to the BenchmarkSubmission format for the Lambda API.

Usage:
    python scripts/post_cuvs_bench.py /path/to/dataset/result \
        --sku-name <registered-sku> \
        --storage-configuration-name <registered-storage-config> \
        --cache-state warm \
        --cuvs-version 25.10.00 \
        --commit-hash abc1234

    # Check the payload without sending it
    python scripts/post_cuvs_bench.py /path/to/dataset/result \
        --sku-name <registered-sku> \
        --storage-configuration-name <registered-storage-config> \
        --cache-state warm --dry-run

Environment variables:
    BENCHMARK_API_URL: Default API URL (get via: ./scripts/print-api-url)
    BENCHMARK_API_KEY: Default API key (get via: ./scripts/print-api-key)

Result tree:
    result/build/<algo>,<group>.{csv,json}
    result/search/<algo>,<group>,k<k>,bs<batch_size>,raw.csv

One submission is posted per (dataset, k, batch_size), carrying the query logs of
every algorithm run against that combination. That matches the benchmark
definition naming, cuvs-bench-<dataset>-k<k>-bs<batch_size>, where each algorithm
is one named query log and each swept configuration is one iteration of it.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import httpx

# Columns the harness always writes. Everything else in a search CSV is an
# algorithm input from the YAML config.
NON_PARAMETER_COLUMNS = {
    "algo_name",
    "index_name",
    "recall",
    "throughput",
    "latency",
    "threads",
    "cpu_time",
    "GPU",
    "end_to_end",
    "k",
    "n_queries",
    "total_queries",
    "build time",
    "build threads",
    "build cpu_time",
    "build GPU",
    "label",
    "search_label",
    "build_label",
}

# String-valued inputs get no column of their own. They are packed into a label
# column, '#'-separated, and which label column appears depends on whether the
# build or the search phase contributed them.
LABEL_COLUMNS = ("label", "search_label", "build_label")

# Inputs declared on the benchmark definition. Undeclared keys are rejected by
# the API, so anything new in the YAML config has to be registered first.
STRING_PARAMETERS = {
    "graph_build_algo",
    "internalDistanceDtype",
    "smemLutDtype",
}
INTEGER_PARAMETERS = {
    "graph_degree",
    "intermediate_graph_degree",
    "itopk",
    "search_width",
    "pq_dim",
    "pq_bits",
    "refine_ratio",
    "nlist",
    "nprobe",
    "ratio",
    "niter",
}
DECLARED_PARAMETERS = STRING_PARAMETERS | INTEGER_PARAMETERS


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Load cuvs-bench results and post to the API Gateway.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to a dataset result directory or a single search CSV",
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get("BENCHMARK_API_URL"),
        help="Lambda API URL (default: from BENCHMARK_API_URL env var)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("BENCHMARK_API_KEY"),
        help="API key for authentication (default: from BENCHMARK_API_KEY env var)",
    )
    parser.add_argument(
        "--sku-name",
        required=True,
        help="Compute hardware SKU name registered with the API",
    )
    parser.add_argument(
        "--storage-configuration-name",
        required=True,
        help="Storage configuration name",
    )
    parser.add_argument(
        "--cache-state",
        required=True,
        choices=["cold", "warm", "hot"],
        help="Cache state for the benchmark run",
    )
    parser.add_argument(
        "--engine-name",
        default="cuvs-bench",
        help="Query engine name (default: 'cuvs-bench')",
    )
    parser.add_argument(
        "--cuvs-version",
        default="",
        help="cuVS version. The result files only carry the Google Benchmark version.",
    )
    parser.add_argument(
        "--commit-hash",
        default="",
        help="cuVS commit hash. Not recorded by the harness.",
    )
    parser.add_argument(
        "--search-mode",
        default="latency",
        choices=["latency", "throughput"],
        help="Search mode the results were produced with (default: 'latency')",
    )
    parser.add_argument(
        "--labels",
        action="append",
        default=None,
        help="Label for the benchmark run (repeatable). Optional; defaults to [].",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the request payload without sending it",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--is-official",
        action="store_true",
        help="Mark the benchmark run as official",
    )
    return parser.parse_args()


def find_search_files(input_path: str) -> list[Path]:
    """Find the search result CSVs to process.

    If input_path is a file, return it as a single-item list. If it is a
    directory, return every raw search CSV beneath it.
    """
    path = Path(input_path)

    if path.is_file():
        return [path]

    if path.is_dir():
        return sorted(path.glob("**/search/*,raw.csv"))

    raise ValueError(f"Path does not exist: {input_path}")


def parse_search_filename(csv_path: Path) -> tuple[str, str, int, int]:
    """Split a search CSV name into algorithm, group, k, and batch size.

    'cuvs_cagra,test,k10,bs1,raw.csv' -> ('cuvs_cagra', 'test', 10, 1)
    """
    parts = csv_path.name.removesuffix(".csv").split(",")
    if len(parts) != 5 or not parts[2].startswith("k") or not parts[3].startswith("bs"):
        raise ValueError(f"Unexpected search file name: {csv_path.name}")
    return parts[0], parts[1], int(parts[2][1:]), int(parts[3][2:])


def get_benchmark_definition_name(dataset: str, k: int, batch_size: int) -> str:
    """Build the benchmark definition name for one dataset / k / batch size."""
    return f"cuvs-bench-{dataset}-k{k}-bs{batch_size}"


def generate_identifier_hash(cuvs_version: str, commit_hash: str, gpu_name: str) -> str:
    """Generate a unique identifier hash from version info.

    Combines the cuVS version, commit hash, and GPU name into a SHA256 hash for
    use as the query engine identifier.
    """
    combined = f"{cuvs_version}:{commit_hash}:{gpu_name}"
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def parse_csv_file(csv_path: Path) -> list[dict]:
    """Parse a CSV file and return list of row dicts."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def read_search_context(csv_path: Path) -> dict:
    """Read the Google Benchmark context block beside a search CSV."""
    json_path = csv_path.with_name(csv_path.name.replace(",raw.csv", ".json"))
    if not json_path.exists():
        return {}
    with open(json_path) as f:
        return json.load(f).get("context", {})


def read_index_sizes(csv_path: Path, algo: str, group: str) -> dict[str, int]:
    """Map index name to index size, which only the build JSON records.

    The build JSON names each benchmark '<index_name>/process_time/real_time',
    so the index name is the leading path segment.
    """
    json_path = csv_path.parents[1] / "build" / f"{algo},{group}.json"
    if not json_path.exists():
        return {}

    with open(json_path) as f:
        benchmarks = json.load(f).get("benchmarks", [])

    sizes = {}
    for benchmark in benchmarks:
        index_size = benchmark.get("index_size")
        if index_size is not None:
            sizes[benchmark["name"].split("/")[0]] = int(float(index_size))
    return sizes


def parse_labels(row: dict) -> dict[str, str]:
    """Parse the label columns, which carry the string-valued inputs.

    'internalDistanceDtype="float"#smemLutDtype="half"'
        -> {'internalDistanceDtype': 'float', 'smemLutDtype': 'half'}
    """
    values = {}
    for column in LABEL_COLUMNS:
        for item in (row.get(column) or "").split("#"):
            key, separator, value = item.partition("=")
            if separator:
                values[key.strip()] = value.strip().strip('"')
    return values


def build_metrics(row: dict, index_size: int | None) -> tuple[dict, set[str]]:
    """Build the metrics map for one search row.

    Returns the metrics and the names of any input columns that are not
    declared on the benchmark definition, which the caller should report
    rather than submit.
    """
    metrics: dict[str, float | int | str] = {
        "recall": float(row["recall"]),
        "throughput": float(row["throughput"]),
    }
    # Both are absent when the index already existed, since the harness then
    # writes a build JSON with no benchmarks in it and no build CSV at all.
    if row.get("build time"):
        metrics["build_time"] = float(row["build time"])
    if index_size is not None:
        metrics["index_size"] = index_size

    candidates = {
        key: value for key, value in row.items() if key not in NON_PARAMETER_COLUMNS and key
    }
    candidates.update(parse_labels(row))

    undeclared = set()
    for key, value in candidates.items():
        if value is None or value == "":
            continue
        if key not in DECLARED_PARAMETERS:
            undeclared.add(key)
            continue
        # The harness writes every numeric input as a float, including the
        # ones declared as integers.
        metrics[key] = value if key in STRING_PARAMETERS else int(float(value))

    return metrics, undeclared


def group_files_by_run(files: list[Path]) -> dict[tuple[str, int, int], list[Path]]:
    """Group search CSVs into submissions, keyed by dataset, k, and batch size."""
    runs = defaultdict(list)
    for csv_path in files:
        _, _, k, batch_size = parse_search_filename(csv_path)
        context = read_search_context(csv_path)
        # The harness records the dataset it was given; the result tree is
        # laid out as <dataset>/result/search/<file>.
        dataset = context.get("dataset") or csv_path.parents[2].name
        runs[(dataset, k, batch_size)].append(csv_path)
    return dict(runs)


def build_submission_payload(
    files: list[Path],
    dataset: str,
    k: int,
    batch_size: int,
    sku_name: str,
    storage_configuration_name: str,
    cache_state: str,
    engine_name: str,
    cuvs_version: str,
    commit_hash: str,
    search_mode: str,
    labels: list[str] | None,
    is_official: bool,
) -> tuple[dict, set[str]]:
    """Build a BenchmarkSubmission payload for one dataset / k / batch size.

    Every algorithm run against that combination contributes query logs, named
    after the algorithm, one per swept configuration.
    """
    if not files:
        raise ValueError("No files to process")

    context = read_search_context(files[0])
    gpu_name = context.get("gpu_name", "")

    query_logs = []
    undeclared: set[str] = set()
    for csv_path in files:
        algo, group, _, _ = parse_search_filename(csv_path)
        index_sizes = read_index_sizes(csv_path, algo, group)

        for row in parse_csv_file(csv_path):
            metrics, row_undeclared = build_metrics(row, index_sizes.get(row["index_name"]))
            undeclared |= row_undeclared

            # The harness reports latency in seconds.
            latency_seconds = float(row["latency"])

            query_logs.append(
                {
                    "query_name": algo,
                    "execution_order": len(query_logs),
                    "runtime_ms": latency_seconds * 1000,
                    "status": "success",
                    "metrics": metrics,
                    "extra_info": {
                        "index_name": row["index_name"],
                        "config_group": group,
                    },
                }
            )

    payload = {
        "sku_name": sku_name,
        "storage_configuration_name": storage_configuration_name,
        "benchmark_definition_name": get_benchmark_definition_name(dataset, k, batch_size),
        "cache_state": cache_state,
        "query_engine": {
            "engine_name": engine_name,
            "identifier_hash": generate_identifier_hash(cuvs_version, commit_hash, gpu_name),
            "version": cuvs_version,
            "commit_hash": commit_hash,
        },
        "run_at": context.get("date", ""),
        "node_count": 1,
        "gpu_count": 1,
        "query_logs": query_logs,
        "concurrency_streams": 1,
        "engine_config": {
            "k": k,
            "batch_size": batch_size,
            "search_mode": search_mode,
        },
        "labels": labels if labels is not None else [],
        "extra_info": {
            "dataset": dataset,
            "dim": context.get("dim"),
            "distance": context.get("distance"),
            "gpu_name": gpu_name,
            "gpu_driver_version": context.get("gpu_driver_version", ""),
            "gpu_sm_count": context.get("gpu_sm_count"),
            "host_name": context.get("host_name", ""),
            "command_line": context.get("command_line", ""),
        },
        "is_official": is_official,
    }
    return payload, undeclared


def post_submission(api_url: str, api_key: str, payload: dict, timeout: float) -> tuple[int, str]:
    """Post a benchmark submission to the API.

    Returns:
        Tuple of (status_code, response_text)
    """
    response = httpx.post(
        api_url,
        json=payload,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )
    return response.status_code, response.text


def process_run(
    key: tuple[str, int, int],
    files: list[Path],
    args: argparse.Namespace,
) -> int:
    """Build and submit one benchmark run.

    Returns:
        0 on success, 1 on failure
    """
    dataset, k, batch_size = key
    print(f"\n  Run: {dataset} k={k} bs={batch_size}", file=sys.stderr)
    print(f"    Files: {len(files)}", file=sys.stderr)

    try:
        payload, undeclared = build_submission_payload(
            files=files,
            dataset=dataset,
            k=k,
            batch_size=batch_size,
            sku_name=args.sku_name,
            storage_configuration_name=args.storage_configuration_name,
            cache_state=args.cache_state,
            engine_name=args.engine_name,
            cuvs_version=args.cuvs_version,
            commit_hash=args.commit_hash,
            search_mode=args.search_mode,
            labels=args.labels,
            is_official=args.is_official,
        )
    except Exception as e:
        print(f"    Error building payload: {e}", file=sys.stderr)
        return 1

    if undeclared:
        print(
            f"    Warning: dropped inputs not declared on the benchmark definition: "
            f"{', '.join(sorted(undeclared))}",
            file=sys.stderr,
        )

    algorithms = sorted({log["query_name"] for log in payload["query_logs"]})
    print(f"    Benchmark definition: {payload['benchmark_definition_name']}", file=sys.stderr)
    print(f"    Labels: {', '.join(payload['labels'])}", file=sys.stderr)
    print(f"    Algorithms: {', '.join(algorithms)}", file=sys.stderr)
    print(f"    Query logs: {len(payload['query_logs'])}", file=sys.stderr)

    if args.dry_run:
        print("\n    [DRY RUN] Payload:", file=sys.stderr)
        print(json.dumps(payload, indent=2))
        return 0

    try:
        status_code, response_text = post_submission(
            args.api_url, args.api_key, payload, args.timeout
        )
        print(f"    Status: {status_code}", file=sys.stderr)
        if status_code >= 400:
            print(f"    Response: {response_text}", file=sys.stderr)
            return 1
        print(f"    Success: {response_text}", file=sys.stderr)
    except httpx.RequestError as e:
        print(f"    Error posting: {e}", file=sys.stderr)
        return 1

    return 0


def main() -> int:
    args = parse_args()

    # Validate required arguments
    if not args.api_url and not args.dry_run:
        print(
            "Error: --api-url or BENCHMARK_API_URL environment variable required", file=sys.stderr
        )
        return 1

    if not args.api_key and not args.dry_run:
        print(
            "Error: --api-key or BENCHMARK_API_KEY environment variable required", file=sys.stderr
        )
        return 1

    try:
        search_files = find_search_files(args.input_path)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not search_files:
        print(f"No search result CSVs found in: {args.input_path}", file=sys.stderr)
        return 1

    print(f"Found {len(search_files)} search result file(s)", file=sys.stderr)

    runs = group_files_by_run(search_files)
    print(f"Benchmark runs: {len(runs)}", file=sys.stderr)

    errors = 0
    for key in sorted(runs):
        if process_run(key, runs[key], args) != 0:
            errors += 1

    print(f"\nProcessed {len(runs)} run(s), {errors} with errors", file=sys.stderr)
    return 1 if errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
