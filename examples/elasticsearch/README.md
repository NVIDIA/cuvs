# Elasticsearch GPU + cuVS — Install Guide

**Path:** Elasticsearch from GitHub (build from source) + cuVS from public tarball (no commercial licenses — basic license only)

**Automation:** The [Makefile](./Makefile) automates deployment. Run `make help` for targets.

---

## Prerequisites

| Item | Source |
|------|--------|
| Linux host with NVIDIA GPU | Cloud VM, bare metal, or local workstation |
| Elasticsearch 9.3.1 source | https://github.com/elastic/elasticsearch/releases/tag/v9.3.1 |
| cuVS v26.02.00 (pre-built) | NVIDIA cuVS redistributable tarball (CUDA 12) |
| CUDA Toolkit 12.2+ (13.0 OK) | Pre-installed on GPU hosts |
| Java 21+ | Gradle auto-provisions JDK 21 for build |

**License:** Basic (free). No trial or Enterprise. GPU patch bypasses license check.

---

## Quick Start

```bash
make full
```

Runs: kill ES → download → install-cuvs-deps → install-cuvs → patch-gpu-license → patch-gradle-license → ldconfig-cuvs → start ES (bg) → add-es-config → restart ES → index 100k vectors.

**Variables:** `ES_DIR`, `CUVS_PREFIX`, `ES_URL`, `ES_USER`, `ES_PASS`, `CUDA_HOME`, `CUVS_TARBALL_URL` — override with `make VAR=value`.

---

## Manual Flow (step-by-step)

If not using `make full`, run in order:

1. **Verify environment:** `nvidia-smi`, `ls /usr/local/cuda`, `df -h /` (~100GB free)
2. **CUDA (if needed):** `make install-cuda13 CUDA_DEB=/path/to/cuda-repo-ubuntu2204-13-0-local_*.deb`
3. **Java (optional):** `make install-java` — Gradle provisions JDK 21
4. **Download & setup:** `make download`, `make install-cuvs-deps`, `make install-cuvs`, `make ldconfig-cuvs`
5. **Patches:** `make patch-gpu-license`, `make patch-gradle-license`
6. **Run ES:** `make run-es` (foreground). On first run, stop with Ctrl+C once config dir exists.
7. **Config:** `make add-es-config` — adds GPU + basic license to `elasticsearch.yml`
8. **Restart:** `make run-es` again

---

## Index and Test

| Target | Action |
|--------|--------|
| `make test-vectors` | create-index + index-doc + knn-search (quick smoke test) |
| `make index-100k-gpu` | Delete, create, bulk index 100k vectors (GPU) |
| `make create-index` | Create test-vectors index only |
| `make index-doc` | Index one sample doc |
| `make index-100k` | Bulk index (assumes index exists) |
| `make knn-search` | Run KNN search |
| `make verify` | Check cluster |

---

## Pre-installed Elasticsearch

```bash
make full ES_PREINSTALLED=1 ES_DIR=/path/to/elasticsearch [ES_CONFIG=/path/to/elasticsearch.yml]
```

Skips download and patches. Set `ES_CONFIG` if config path differs.

---

## Clean

| Target | Action |
|--------|--------|
| `make kill-es` | Stop ES and Gradle |
| `make clean-es` | Clean ES build (fixes "module not found: org.elasticsearch.xcore") |
| `make clean-all` | Remove cuVS, ES dir, ldconfig entries |

---

## Config Paths

| Context | Config Path |
|---------|-------------|
| `./gradlew :run` | `build/testclusters/runTask-0/config/elasticsearch.yml` |
| Pre-installed | `config/elasticsearch.yml` (in install dir) |

---

## Troubleshooting

- **Address already in use:** `make kill-es`
- **libcuvs_c.so not found:** `make ldconfig-cuvs`
- **module not found: org.elasticsearch.xcore:** `make clean-es` then `make run-es`
- **unknown setting [index.vectors.indexing.use_gpu]:** Use cluster-level only (handled by `make add-es-config`)
- **no handler for /_knn_search:** Use `/_search` with `knn` query (handled by `make knn-search`)
- **Config not found:** Run `make run-es` once, stop, then `make add-es-config`
