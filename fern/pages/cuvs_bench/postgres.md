# PostgreSQL: pgvector and pg_cuvs

This optional backend runs vector searches through PostgreSQL using the same
corpus, zero-based row IDs, ground truth, and cuvs-bench recall calculation.

| Algorithm | Build | Search setting | Search execution |
| --- | --- | --- | --- |
| `pgvector_hnsw` | pgvector HNSW | `ef_search` | CPU |
| `pgvector_ivfflat` | pgvector IVFFlat | `probes` | CPU |
| `pgcuvs_cagra` | pg_cuvs CAGRA | `candidates` (`cuvs.k`) | GPU sidecar |
| `pgcuvs_hnsw_import` | CAGRA plus `USING pg_cuvs_hnsw` export | `ef_search` | pgvector CPU |

Only Euclidean (`<->`, `vector_l2_ops`) and single-client latency mode are
implemented. `batch_size` is retained in the orchestration/output interface;
it does not create SQL batching, concurrency, or GPU saturation. QPS is the
inverse of mean serial query latency. Compare PostgreSQL rows to each other
at matched recall. Native C++ cuVS rows measure a different boundary and must
not be presented as equivalent SQL application timings.

## Install

Use Python 3.12 or newer. From this cuVS checkout:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e './python/cuvs_bench[postgres]' -Crapidsai.disable-cuda=true
```

The optional extra installs psycopg 3 and pgvector's Python adapter. It does
not install a PostgreSQL server, the pgvector server extension, pg_cuvs, or
its daemon. The RAPIDS build option above avoids requiring `nvcc` on the
SQL benchmark client; GPU work runs in the PostgreSQL server's daemon.
CPU pgvector algorithms do not require pg_cuvs or a GPU on the server.

On a PostgreSQL server with pgvector installed, create a dedicated database:

```bash
createdb cuvs_bench
psql -X -v ON_ERROR_STOP=1 -d cuvs_bench -c 'CREATE EXTENSION vector;'
export PGDATABASE=cuvs_bench
# Remote server: set PGHOST/PGPORT/PGUSER; use .pgpass or PGSERVICE for credentials.
```

The role needs CREATE privilege on this database. CAGRA measurements also need
SELECT privilege on `pg_stat_gpu_search` for the VRAM and GPU-dispatch checks.
An administrator must grant this to an ordinary benchmark role, for example
`GRANT SELECT ON pg_stat_gpu_search TO benchmark_role;` (replace the role name).
Database CREATE privilege alone does not grant access to this view. The GPU
verification below used the bootstrap's administrative role.
Every build loads its own
corpus into a fresh `cuvs_bench_<uuid>` schema. Normal cleanup drops only that
schema, including its indexes. It does not replace public tables or existing
indexes. Abrupt client termination can leave a schema behind; inspect those
schemas before removing abandoned benchmark data. Run build and search in the
same orchestrator call. Separate-process search-only and persistent index
reuse are not supported; `force` does not change that lifetime.
The optional `index_dir` connection override sets a PostgreSQL `PGC_SUSET`
parameter and needs the corresponding administrative privilege. Omit it to
use the server's configured directory when connecting as an ordinary role.

## GPU server setup and build

The pgcuvs algorithms require PostgreSQL 16, pgvector >= 0.7.0, pg_cuvs >= 0.8.0,
CUDA 12.4 and cuVS 26.04. PostgreSQL must preload pg_cuvs and its GPU daemon
must be running. The client may connect remotely and needs no GPU.

On a fresh Ubuntu 22.04 A100 host with a working NVIDIA driver, systemd and
passwordless sudo, the public pg_cuvs bootstrap installs the toolchain, builds
the extension and daemon, configures PostgreSQL 16 and downloads wiki_all_1M.
It changes system packages and PostgreSQL configuration; use a dedicated host.
The fresh-host setup below was exercised at this pinned public source revision;
the verification environment and GPU test scope are recorded below:

```bash
git clone https://github.com/pg-cuvs/pg_cuvs.git "$HOME/pg_cuvs"
git -C "$HOME/pg_cuvs" checkout 8b73d65fb8a3d6f66cdfe69eec4969c1f2fc753c
bash "$HOME/pg_cuvs/infra/brev/bootstrap.sh"
# Keep the server build toolchain out of the benchmark client's Python PATH.
(
. "$HOME/.pg_cuvs_env"
cd "$HOME/pg_cuvs"
# Explicit rebuild commands; bootstrap already performs these steps.
make -j2
sudo -E env PATH="$PATH" CONDA_PREFIX="$CONDA_PREFIX" make install install-server
)
systemctl is-active postgresql@16-main pg-cuvs-server
createdb cuvs_bench
psql -X -v ON_ERROR_STOP=1 -d cuvs_bench \
  -c 'CREATE EXTENSION IF NOT EXISTS vector;' \
  -c 'CREATE EXTENSION pg_cuvs;'
psql -X -d cuvs_bench -c 'SHOW shared_preload_libraries;'
nvidia-smi
```

The bootstrap's [public source and assumptions](https://github.com/pg-cuvs/pg_cuvs/blob/8b73d65fb8a3d6f66cdfe69eec4969c1f2fc753c/infra/brev/bootstrap.sh)
are the authority for server setup. Use the same `cuvs.index_dir` as the daemon
(`index_dir` in backend options overrides it for the connection). Never add a
Conda environment's entire library directory to system ldconfig. This backend
does not change system configuration or restart the daemon.

## Obtain data and run

From the cuVS checkout, download the standard million-row binary dataset:

```bash
. .venv/bin/activate
mkdir -p datasets/wiki_all_1M
curl -fL https://data.rapids.ai/raft/datasets/wiki_all_1M/wiki_all_1M.tar \
  -o datasets/wiki_all_1M/wiki_all_1M.tar
tar -xf datasets/wiki_all_1M/wiki_all_1M.tar -C datasets/wiki_all_1M
export PGDATABASE=cuvs_bench
```

The supplied ground truth describes the full corpus. Using `subset_size`
without recomputing ground truth invalidates recall. Run the small parameter
group first (it still loads the complete corpus):

```python
from cuvs_bench.orchestrator import BenchmarkOrchestrator
from cuvs_bench.run.data_export import write_results_to_csv

results = BenchmarkOrchestrator(backend_type="postgres").run_benchmark(
    dataset="wiki_all_1M",
    dataset_path="datasets",
    algorithms="pgvector_hnsw,pgvector_ivfflat",
    groups="test",
    count=10,
    search_mode="latency",
)
for result in results:
    if not result.success:
        raise RuntimeError(result.error_message)
write_results_to_csv(results, "wiki_all_1M", "datasets", 10, 10000)
```

After GPU server setup, set `algorithms` to
`"pgvector_hnsw,pgvector_ivfflat,pgcuvs_cagra,pgcuvs_hnsw_import"`.
Set `groups="base"` for the larger build/search parameter grids. For a custom
sweep, copy YAML from `python/cuvs_bench/cuvs_bench/config/algos/`, edit the
`build` and `search` lists, and pass
`algorithm_configuration="/absolute/path/to/custom.yaml"`.
One index is built per build cell and reused for every search point in that
cell. Keep `ef_search` or `candidates` >= `count`. The HNSW export uses
`USING pg_cuvs_hnsw`, rather than deprecated `pg_cuvs_build_hnsw()`.

CSV files appear under `datasets/wiki_all_1M/result/{build,search}/`. Standard
cuvs-bench plotting tools can consume them. Retain every build cell and search
point when selecting matched-recall Pareto points. Apply a recall threshold
before selecting the fastest point.

### A bounded wiki_all_1M CLI validation sweep

The following grid uses one build cell and two search settings per algorithm.
It searches all 10,000 supplied queries, with `k=10`. The resource settings
below were used on a 14-vCPU, 100-GiB RAM A100 host. They are session settings,
not changes to the PostgreSQL configuration file.

```bash
mkdir -p pg-validation-grids
python - <<'PY'
from pathlib import Path
import yaml

graph = {"graph_degree": [32], "intermediate_graph_degree": [128],
         "build_algo": ["ivf_pq"]}
grids = {
    "pgcuvs_cagra": (graph, {"candidates": [64, 200]}),
    "pgcuvs_hnsw_import": ({**graph, "mode": ["nsw"]}, {"ef_search": [64, 200]}),
    "pgvector_hnsw": ({"m": [16], "ef_construction": [64]}, {"ef_search": [64, 200]}),
    "pgvector_ivfflat": ({"lists": [1000]}, {"probes": [4, 16]}),
}
for algo, (build, search) in grids.items():
    Path(f"pg-validation-grids/{algo}.yaml").write_text(yaml.safe_dump(
        {"name": algo, "groups": {"validation": {"build": build, "search": search}}}
    ))
Path("pg-backend.yaml").write_text("backend: postgres\nwarmup_queries: 10\n")
PY
export PGDATABASE=cuvs_bench
export PGOPTIONS='-c maintenance_work_mem=8GB -c max_parallel_maintenance_workers=7'
for algo in pgcuvs_cagra pgcuvs_hnsw_import pgvector_hnsw pgvector_ivfflat; do
  python -m cuvs_bench.run --backend-config pg-backend.yaml \
    --dataset wiki_all_1M --dataset-path datasets \
    --configuration pg-validation-grids --algorithms "$algo" \
    --groups validation -k 10 --batch-size 10000 --search-mode latency --force
done
```

Check for one build row and two raw search rows per algorithm. CAGRA search
rows must include `observed_search_mode=gpu_cagra` and `gpu_queries=10000`.
Build CSVs preserve `index_size_bytes` and `index_size_source`. These runs
validate execution and serialization; they do not establish a speedup over
another engine, machine, or measurement boundary.
To exercise the GPU build axis as well, set `graph_degree` to `[32, 64]` in
the two GPU grids. Each then produces two build rows and four raw search rows.

## Measurement contract

- File reading, binary COPY and ANALYZE are outside the build timer.
- Build time covers CREATE INDEX. HNSW export includes the CAGRA build and
  conversion. YAML records the source graph and export parameters.
- Query time includes client adaptation, SQL planning/execution, IPC/GPU work
  for CAGRA, heap fetch and result transfer. Warmup and EXPLAIN are excluded.
  The backend checks for an Index Scan before and after each search point; a
  sequential-scan fallback becomes a failed result. CAGRA connections explicitly
  disable CPU HNSW fallback. Outside the timer, daemon observations must report
  resident `gpu_cagra` mode and exactly one additional GPU search per measured
  query. Missing/reset counters, CPU mode or an unavailable daemon fail the
  point instead of producing a GPU-labeled measurement.
- IDs are corpus offsets. Missing neighbors stay `-1` with infinite distance.
  The orchestrator calculates recall using actual IDs and ground truth.
- CPU index size is `pg_relation_size`. CAGRA size comes from the daemon's
  `pg_stat_gpu_search.vram_bytes`; missing accounting fails the build. Disk
  bytes and daemon VRAM bytes are different resources, labeled in metadata.

## Validate

CPU contracts require no database. Live tests need a disposable database with
the vector extension installed:

```bash
python -m pip install pytest
python -m pytest python/cuvs_bench/cuvs_bench/tests/test_postgres.py -m 'not postgres'
CUVS_BENCH_POSTGRES_DSN='dbname=cuvs_bench' \
  python -m pytest python/cuvs_bench/cuvs_bench/tests/test_postgres.py
CUVS_BENCH_POSTGRES_GPU_DSN='dbname=cuvs_bench' \
  python -m pytest python/cuvs_bench/cuvs_bench/tests/test_postgres.py -m postgres_gpu
```

Set the GPU DSN only for a server built with real CUDA/cuVS, not a CPU
reference implementation. The opt-in tests build CAGRA and export HNSW from
a 4096-by-32 corpus, check real GPU mode/counters, and check that CPU HNSW
search does not increment its source CAGRA's search counter.

Fresh-host verification on 2026-10-16 used Ubuntu 22.04.5, A100-SXM4-80GB,
driver 580.126.09, PostgreSQL 16.15, pgvector 0.8.7, pg_cuvs 0.8.0 at the source
revision above, and CUDA 12.4/cuVS 26.04. The client was installed from this
cuvs-bench source with Python 3.12.15. Both real GPU cases passed, along with
the 28 CPU and telemetry-contract cases. CPU-only test success by itself is
not evidence that GPU kernels ran.

That initial GPU run predates the client-side connection-error, group-selection,
thread-option, and test-marker fixes. Those later changes were checked with CPU
integration and wheel tests; GPU execution of the final revision has not been
repeated.

Always provide ground truth for recall comparisons. Without it, the current
result interface leaves recall at zero rather than marking it unmeasured.
Python API dry-run search results must not be passed to the generic CSV
exporter as measurements: it currently retains zero-valued skipped rows.
The CLI skips export for dry runs.

This backend adapts the public
[pg_cuvs prototype](https://github.com/pg-cuvs/pg_cuvs/tree/53a8ca0ff82f3bb2073bcd4c7ee8a7d8dc49cdd0/bench/cuvs_bench_backend)
to current cuvs-bench interfaces. It changes no GPU kernels and makes no
performance improvement claim.
