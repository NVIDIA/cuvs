# Elasticsearch GPU + cuVS — Brev / GPU Host Deployment
#
# Run this Makefile on a Linux host with NVIDIA GPU (e.g. Brev instance).
# No trial license required; patch-gpu-license bypasses Enterprise license for GPU indexing.
#
# Usage:
#   make help              - Show available targets
#   make full              - Run full flow: patch, install deps, cuVS, config, run ES (bg), index 100k
#   make test-vectors      - Create index, index doc, run KNN search
#
# Full flow: download -> patch-gpu-license -> patch-gradle-license -> install-cuvs-deps -> install-cuvs -> ldconfig-cuvs -> add-es-config -> run-es -> index-100k-gpu
# Pre-installed: make full ES_PREINSTALLED=1 ES_DIR=/path/to/elasticsearch [ES_CONFIG=/path/to/elasticsearch.yml]
.PHONY: help download install-cuda13 install-java install-cuvs-deps install-cuvs ldconfig-cuvs patch-gpu-license patch-gradle-license \
	add-es-config run-es create-index delete-index index-doc index-100k index-100k-gpu knn-search test-vectors verify kill-es \
	full clean-all clean-es build-es
# Config
HOME ?= $(shell echo $$HOME)
ES_VERSION ?= 9.3.1
CUVS_VERSION ?= 26.02.00
ES_DIR ?= /raid/elasticsearch-$(ES_VERSION)
# Set ES_PREINSTALLED=1 when using a pre-installed binary (e.g. from apt or tar). Skips download + patch-gpu-license.
ES_PREINSTALLED ?=
ES_CONFIG ?= $(if $(ES_PREINSTALLED),$(ES_DIR)/config/elasticsearch.yml,$(ES_DIR)/build/testclusters/runTask-0/config/elasticsearch.yml)
ES_URL ?= http://localhost:9200
ES_USER ?= elastic
ES_PASS ?= password
CUDA_HOME ?= /usr/local/cuda
CUVS_PREFIX ?= /raid/libcuvs
# NVIDIA cuVS redistributable (CUDA 12). Override for other builds.
CUVS_TARBALL_URL ?= https://developer.download.nvidia.com/compute/cuvs/redist/libcuvs/linux-x86_64/libcuvs-linux-x86_64-26.02.00.189485_cuda12-archive.tar.xz
# Or set path to local tarball (e.g. libcuvs_c.tar.gz from build_libcuvs_tarball.sh)
CUVS_TARBALL_LOCAL ?=
GPU_WATCH_INTERVAL ?= 1
INDEX_100K_COUNT ?= 100000
INDEX_100K_BATCH ?= 1000
help:
	@echo "Elasticsearch GPU + cuVS — Brev Deployment"
	@echo ""
	@echo "Prerequisites: Linux host with NVIDIA GPU, CUDA 13.0 (Toolkit), Java 21+ (Gradle auto-provisions JDK 21)"
	@echo ""
	@echo "Targets (run in order for fresh install):"
	@echo "  make download          - Download Elasticsearch source"
	@echo "  make install-cuda13    - Install CUDA Toolkit 13.0 (Ubuntu 22.04 deb_local, sudo)"
	@echo "  make install-java      - Install OpenJDK 25 (sudo)"
	@echo "  make install-cuvs-deps - Install cuVS runtime deps (NCCL, OpenMP)"
	@echo "  make install-cuvs      - Download & extract libcuvs tarball to $(CUVS_PREFIX)"
	@echo "  make ldconfig-cuvs     - Add libcuvs to ldconfig (sudo)"
	@echo "  make patch-gpu-license   - Bypass GPU Enterprise license check (for local builds)"
	@echo "  make patch-gradle-license - Use basic not trial in Gradle test clusters"
	@echo "  make clean-es         - Clean ES build (run if 'module not found: org.elasticsearch.xcore')"
	@echo "  make build-es         - Clean + full build (use before run-es if build fails)"
	@echo "  make add-es-config     - Add GPU config to elasticsearch.yml"
	@echo "  make run-es            - Start Elasticsearch (foreground)"
	@echo "  make create-index      - Create test-vectors index"
	@echo "  make delete-index      - Delete test-vectors index (needed before GPU re-index)"
	@echo "  make index-doc         - Index a sample document"
	@echo "  make index-100k        - Bulk index 100k vectors"
	@echo "  make index-100k-gpu    - Full GPU flow: add-es-config, delete-index, create-index, index-100k"
	@echo "  make knn-search        - Run KNN search"
	@echo "  make test-vectors      - create-index + index-doc + knn-search"
	@echo "  make full              - Full flow from scratch (kill, patch, ldconfig, add-config, run-es bg, index-100k-gpu)"
	@echo "  make full ES_PREINSTALLED=1 - Same but use pre-installed ES (skip download, patch); set ES_DIR, ES_CONFIG"
	@echo "  make clean-all         - Remove everything: kill ES, delete libcuvs, ES dir, ldconfig entries (sudo)"
	@echo "  make verify            - Check cluster"
	@echo "  make kill-es           - Kill existing Elasticsearch process"
	@echo ""
	@echo "Variables (override with make VAR=value):"
	@echo "  ES_VERSION=$(ES_VERSION)"
	@echo "  ES_DIR=$(ES_DIR) ES_PREINSTALLED=$(ES_PREINSTALLED)"
	@echo "  CUDA_HOME=$(CUDA_HOME)"
	@echo "  CUVS_VERSION=$(CUVS_VERSION)"
	@echo "  CUVS_PREFIX=$(CUVS_PREFIX)"
	@echo "  CUVS_TARBALL_URL=$(CUVS_TARBALL_URL)"
	@echo "  ES_USER=$(ES_USER) ES_PASS=$(ES_PASS)"
clean-all:
	@echo "Removing everything installed..."
	@(pkill -f elasticsearch 2>/dev/null; pkill -f gradlew 2>/dev/null) || true; sleep 2
	@echo "Removing $(CUVS_PREFIX)..."
	@rm -rf $(CUVS_PREFIX)
	@if [ -z "$(ES_PREINSTALLED)" ]; then echo "Removing $(ES_DIR)..."; rm -rf $(ES_DIR); else echo "Skipping ES_DIR (ES_PREINSTALLED)"; fi
	@echo "Removing ldconfig entries..."
	@sudo rm -f /etc/ld.so.conf.d/cuvs.conf /etc/ld.so.conf.d/cuda-lib.conf 2>/dev/null || true
	@sudo ldconfig 2>/dev/null || true
	@echo "clean-all done. Run 'make download' then full flow to start again."
download:
	@echo "Downloading Elasticsearch $(ES_VERSION)..."
	@mkdir -p $(dir $(ES_DIR))
	@curl -L -sSf -o /tmp/elasticsearch-$(ES_VERSION).tar.gz \
		https://github.com/elastic/elasticsearch/archive/refs/tags/v$(ES_VERSION).tar.gz
	@tar -xzf /tmp/elasticsearch-$(ES_VERSION).tar.gz -C $(dir $(ES_DIR))
	@rm -f /tmp/elasticsearch-$(ES_VERSION).tar.gz
	@echo "Elasticsearch source downloaded to $(ES_DIR)"
	@ls -la $(ES_DIR) 2>/dev/null || true
install-cuda13:
	@echo "Installing CUDA Toolkit 13.0 (Ubuntu 22.04 deb_local)..."
	@echo "NOTE: This follows the NVIDIA deb_local flow. If you already have CUDA 13.0 installed, you can skip."
	@echo "1) Download the Ubuntu 22.04 x86_64 deb_local installer from:"
	@echo "   https://developer.nvidia.com/cuda-13-0-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local"
	@echo "2) Set CUDA_DEB to the downloaded .deb path, then run:"
	@echo "   make install-cuda13 CUDA_DEB=/path/to/cuda-repo-ubuntu2204-13-0-local_*.deb"
	@[ -n "$$CUDA_DEB" ] || (echo "ERROR: CUDA_DEB not set"; exit 2)
	@sudo mkdir -p /tmp/cuda13 && sudo cp "$$CUDA_DEB" /tmp/cuda13/cuda13.deb
	@sudo dpkg -i /tmp/cuda13/cuda13.deb
	@sudo cp /var/cuda-repo-ubuntu2204-13-0-local/cuda-*-keyring.gpg /usr/share/keyrings/ 2>/dev/null || true
	@sudo apt-get update
	@sudo apt-get install -y cuda-toolkit-13-0
	@echo "CUDA Toolkit installed. Expect $(CUDA_HOME) to exist."
	@ls -la $(CUDA_HOME) 2>/dev/null || true
install-java:
	@echo "Installing OpenJDK 25..."
	@sudo apt-get update
	@sudo apt-get install -y openjdk-25-jdk
	@java -version
install-cuvs-deps:
	@echo "Installing cuVS runtime dependencies..."
	@echo " - NCCL: follow https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html"
	@echo " - libopenmp"
	@echo " - CUDA Toolkit Runtime 12.2+ (CUDA 13.0 is OK)"
	@sudo apt-get update
	@sudo apt-get install -y libomp5 || sudo apt-get install -y libgomp1
	@echo "OpenMP runtime installed."
install-cuvs:
	@echo "Installing libcuvs from tarball into $(CUVS_PREFIX)..."
	@mkdir -p $(CUVS_PREFIX)
	@if [ -n "$(CUVS_TARBALL_LOCAL)" ]; then \
		echo "Using local tarball: $(CUVS_TARBALL_LOCAL)"; \
		tar -xzvf "$(CUVS_TARBALL_LOCAL)" -C $(CUVS_PREFIX); \
	elif [ -n "$(CUVS_TARBALL_URL)" ]; then \
		curl -L -sSf -o /tmp/libcuvs.tar.xz "$(CUVS_TARBALL_URL)"; \
		rm -rf /tmp/cuvs_extract && mkdir -p /tmp/cuvs_extract && tar --no-same-owner -xJf /tmp/libcuvs.tar.xz -C /tmp/cuvs_extract; \
		d=$$(ls -d /tmp/cuvs_extract/libcuvs-* 2>/dev/null | head -1); \
		if [ -n "$$d" ]; then cp -a "$$d"/* $(CUVS_PREFIX)/; else cp -a /tmp/cuvs_extract/* $(CUVS_PREFIX)/; fi; \
		rm -rf /tmp/cuvs_extract /tmp/libcuvs.tar.xz; \
	else \
		echo "ERROR: Set CUVS_TARBALL_URL or CUVS_TARBALL_LOCAL"; exit 2; \
	fi
	@echo "libcuvs extracted to $(CUVS_PREFIX)"
	@find $(CUVS_PREFIX) -maxdepth 3 -type f \( -name "libcuvs*.so*" -o -name "libcuvs*.dylib" \) 2>/dev/null || true
	@echo ""
	@echo "IMPORTANT: Add cuVS to your library load path (use lib64 for standalone tarball):"
	@echo "  export LD_LIBRARY_PATH=\$$LD_LIBRARY_PATH:$(CUVS_PREFIX)/lib64:$(CUVS_PREFIX)/lib"
GPU_PLUGIN_JAVA := $(ES_DIR)/x-pack/plugin/gpu/src/main/java/org/elasticsearch/xpack/gpu/GPUPlugin.java
AUTH_TESTCLUSTERS_GRADLE := $(ES_DIR)/build-tools-internal/src/main/groovy/elasticsearch.authenticated-testclusters.gradle
patch-gpu-license:
	@echo "Patching GPUPlugin to bypass Enterprise license check..."
	@if [ ! -f "$(GPU_PLUGIN_JAVA)" ]; then \
		echo "GPUPlugin.java not found. Run 'make download' first."; exit 1; \
	fi
	@if ! grep -q 'GPU_INDEXING_FEATURE.check' "$(GPU_PLUGIN_JAVA)" 2>/dev/null; then \
		echo "Already patched."; \
	else \
		perl -i -0pe 's/var licenseState = XPackPlugin\.getSharedLicenseState\(\);\s*return licenseState != null && GPU_INDEXING_FEATURE\.check\(licenseState\);/return true; \/\* bypass: allow GPU indexing without Enterprise license \*\//s' \
			"$(GPU_PLUGIN_JAVA)" && echo "Patch applied." || (echo "Patch failed."; exit 1); \
	fi
patch-gradle-license:
	@echo "Patching Gradle to use basic instead of trial..."
	@if [ ! -f "$(AUTH_TESTCLUSTERS_GRADLE)" ]; then \
		echo "$(AUTH_TESTCLUSTERS_GRADLE) not found. Run 'make download' first."; exit 1; \
	fi
	@if grep -q "self_generated.type.*basic" "$(AUTH_TESTCLUSTERS_GRADLE)" 2>/dev/null; then \
		echo "Already patched."; \
	else \
		sed -i "s/setting 'xpack.license.self_generated.type', 'trial'/setting 'xpack.license.self_generated.type', 'basic'/g" "$(AUTH_TESTCLUSTERS_GRADLE)" && echo "Patched." || (echo "Patch failed."; exit 1); \
	fi
ldconfig-cuvs:
	@echo "Adding libcuvs and CUDA (NCCL) to system library path..."
	@(for d in $(CUVS_PREFIX)/lib64 $(CUVS_PREFIX)/lib; do [ -d "$$d" ] && echo "$$d"; done) | sudo tee /etc/ld.so.conf.d/cuvs.conf
	@(for d in $(CUDA_HOME)/lib $(CUDA_HOME)/lib64; do [ -d "$$d" ] && echo "$$d"; done) | sudo tee /etc/ld.so.conf.d/cuda-lib.conf
	@sudo ldconfig
	@echo "ldconfig updated."
add-es-config:
	@if [ ! -f "$(ES_CONFIG)" ]; then \
		echo "Config not found: $(ES_CONFIG)"; \
		echo "   Run 'make run-es' once to create the cluster, stop with Ctrl+C, then run this again."; \
		exit 1; \
	fi
	@grep -q "vectors.indexing.use_gpu" "$(ES_CONFIG)" || echo 'vectors.indexing.use_gpu: "true"' >> "$(ES_CONFIG)"
	@grep -q "index_buffer_size" "$(ES_CONFIG)" || echo 'indices.memory.index_buffer_size: "30%"' >> "$(ES_CONFIG)"
	@grep -q "xpack.license.self_generated.type" "$(ES_CONFIG)" 2>/dev/null && \
		sed -i 's/xpack.license.self_generated.type:.*/xpack.license.self_generated.type: "basic"/' "$(ES_CONFIG)" || \
		echo 'xpack.license.self_generated.type: "basic"' >> "$(ES_CONFIG)"
	@echo "Config added to $(ES_CONFIG)"
clean-es:
	@echo "Cleaning Elasticsearch build (fixes 'module not found: org.elasticsearch.xcore')..."
	@cd $(ES_DIR) && ./gradlew --no-daemon --no-configuration-cache clean
	@echo "Clean done. Run 'make run-es' to rebuild and start."
build-es: clean-es
run-es:
	@echo "Starting Elasticsearch (foreground). Press Ctrl+C to stop."
	@export LD_LIBRARY_PATH="$(CUVS_PREFIX)/lib64:$(CUVS_PREFIX)/lib:$(CUDA_HOME)/lib64:$(CUDA_HOME)/lib:/usr/lib64:/usr/lib:$$LD_LIBRARY_PATH"; \
	if [ -n "$(ES_PREINSTALLED)" ]; then \
		cd $(ES_DIR) && ./bin/elasticsearch; \
	else \
		cd $(ES_DIR) && export ES_JAVA_OPTS="-Xms4g -Xmx4g" && ./gradlew --no-daemon --no-configuration-cache :run; \
	fi
delete-index:
	@echo "Deleting test-vectors index..."
	@curl -s -X DELETE "$(ES_URL)/test-vectors" -u $(ES_USER):$(ES_PASS)
	@echo ""
create-index:
	@echo "Creating test-vectors index..."
	@curl -s -X PUT "$(ES_URL)/test-vectors" \
		-u $(ES_USER):$(ES_PASS) \
		-H "Content-Type: application/json" \
		-d '{"settings":{"index":{"number_of_shards":1,"number_of_replicas":0}},"mappings":{"properties":{"embedding":{"type":"dense_vector","dims":128,"index":true,"similarity":"cosine","index_options":{"type":"hnsw","m":32,"ef_construction":100}},"text":{"type":"text"}}}}'
	@echo ""
index-doc:
	@echo "Indexing sample document..."
	@VEC=$$(python3 -c "import random; print(','.join(str(round(random.random(),4)) for _ in range(128)))"); \
		curl -s -X POST "$(ES_URL)/test-vectors/_doc" \
			-u $(ES_USER):$(ES_PASS) \
			-H "Content-Type: application/json" \
			-d "{\"embedding\": [$$VEC], \"text\": \"test document\"}"
	@echo ""
define INDEX_100K_PY
import random,json,urllib.request,urllib.error,base64,sys
n=int(sys.argv[4]) if len(sys.argv)>4 else 100_000
batch=int(sys.argv[5]) if len(sys.argv)>5 else 1000
url=f"{sys.argv[1]}/test-vectors/_bulk"
auth=base64.b64encode(f"{sys.argv[2]}:{sys.argv[3]}".encode()).decode()
for i in range(0,n,batch):
  body=[]
  for j in range(batch):
    if i+j>=n: break
    vec=[round(random.random(),4) for _ in range(128)]
    body.append(json.dumps({"index":{}}))
    body.append(json.dumps({"embedding":vec,"text":f"doc {i+j}"}))
  req=urllib.request.Request(url,data=("\n".join(body)+"\n").encode(),headers={"Content-Type":"application/x-ndjson"})
  req.add_header("Authorization","Basic "+auth)
  try: urllib.request.urlopen(req,timeout=120)
  except urllib.error.HTTPError as e: print(e.read().decode(),file=sys.stderr); sys.exit(1)
  if (i+batch)%10000==0 or i+batch>=n: print(f"  indexed {min(i+batch,n)}/{n}")
endef
export INDEX_100K_PY
index-100k:
	@echo "Bulk indexing $(INDEX_100K_COUNT) vectors (batch $(INDEX_100K_BATCH))..."
	@echo "$$INDEX_100K_PY" | python3 - "$(ES_URL)" "$(ES_USER)" "$(ES_PASS)" $(INDEX_100K_COUNT) $(INDEX_100K_BATCH)
	@echo "Indexed $(INDEX_100K_COUNT) vectors."
index-100k-gpu: add-es-config
	@echo "Deleting index (format chosen at creation; must recreate for GPU)..."
	@curl -s -X DELETE "$(ES_URL)/test-vectors" -u $(ES_USER):$(ES_PASS)
	@echo ""
	@echo "Creating fresh index (will use GPU format if patch-gpu-license + vectors.indexing.use_gpu)..."
	@$(MAKE) create-index
	@echo "Bulk indexing $(INDEX_100K_COUNT) vectors..."
	@echo "$$INDEX_100K_PY" | python3 - "$(ES_URL)" "$(ES_USER)" "$(ES_PASS)" $(INDEX_100K_COUNT) $(INDEX_100K_BATCH)
	@echo "Done."
knn-search:
	@echo "Running KNN search..."
	@VEC=$$(python3 -c "import random; print(','.join(str(round(random.random(),4)) for _ in range(128)))"); \
		curl -s -X POST "$(ES_URL)/test-vectors/_search" \
			-u $(ES_USER):$(ES_PASS) \
			-H "Content-Type: application/json" \
			-d "{\"knn\": {\"field\": \"embedding\", \"query_vector\": [$$VEC], \"k\": 5, \"num_candidates\": 50}, \"size\": 5}"
	@echo ""
test-vectors: create-index index-doc knn-search
	@echo "Vector index test complete"
full:
	@echo "=== Full flow from scratch (no trial license) ==="
	@$(MAKE) kill-es
	@if [ -z "$(ES_PREINSTALLED)" ]; then $(MAKE) download; fi
	@$(MAKE) install-cuvs-deps
	@$(MAKE) install-cuvs
	@if [ -z "$(ES_PREINSTALLED)" ]; then $(MAKE) patch-gpu-license; $(MAKE) patch-gradle-license; fi
	@$(MAKE) ldconfig-cuvs
	@if [ -n "$(ES_PREINSTALLED)" ]; then \
		[ -f "$(ES_CONFIG)" ] || (echo "Config not found: $(ES_CONFIG). Set ES_CONFIG for pre-installed ES."; exit 1); \
		$(MAKE) add-es-config; \
		echo "Starting ES (pre-installed) in background..."; \
		nohup sudo -u ubuntu bash -c "cd $(ES_DIR) && export LD_LIBRARY_PATH=\"$(CUVS_PREFIX)/lib64:$(CUVS_PREFIX)/lib:$(CUDA_HOME)/lib64:$(CUDA_HOME)/lib\" && ./bin/elasticsearch" > /tmp/es-run.log 2>&1 & \
	else \
		echo "Removing old test cluster (ensures fresh config with basic license, no trial)..."; \
		rm -rf $(ES_DIR)/build/testclusters/runTask-0; \
		echo "Starting ES in background (creates config on first run)..."; \
		nohup sudo -u ubuntu bash -c "cd $(ES_DIR) && export LD_LIBRARY_PATH=\"$(CUVS_PREFIX)/lib64:$(CUVS_PREFIX)/lib:$(CUDA_HOME)/lib64:$(CUDA_HOME)/lib\" && export ES_JAVA_OPTS=\"-Xms4g -Xmx4g\" && ./gradlew --no-daemon :run" > /tmp/es-run.log 2>&1 & \
		echo "Waiting for config (for add-es-config)..."; \
		for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120; do [ -f "$(ES_CONFIG)" ] && echo "Config ready" && break || (echo "Waiting for config... ($$i)"; sleep 15); done; \
		$(MAKE) add-es-config; \
		echo "Restarting ES to apply GPU config..."; \
		-pkill -f elasticsearch 2>/dev/null || true; \
		-pkill -f gradlew 2>/dev/null || true; \
		sleep 5; \
		nohup sudo -u ubuntu bash -c "cd $(ES_DIR) && export LD_LIBRARY_PATH=\"$(CUVS_PREFIX)/lib64:$(CUVS_PREFIX)/lib:$(CUDA_HOME)/lib64:$(CUDA_HOME)/lib\" && export ES_JAVA_OPTS=\"-Xms4g -Xmx4g\" && ./gradlew --no-daemon :run" > /tmp/es-run.log 2>&1 & \
	fi
	@echo "Waiting for ES (up to 3 min)..."
	@for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18; do curl -s -u $(ES_USER):$(ES_PASS) $(ES_URL) 2>/dev/null | grep -q cluster_name && echo "ES ready" && break || sleep 10; done
	@$(MAKE) index-100k-gpu
	@echo "Full flow complete."
verify:
	@echo "Cluster info:"
	@curl -s -u $(ES_USER):$(ES_PASS) "$(ES_URL)" | head -20
kill-es:
	@-pkill -f elasticsearch 2>/dev/null || true
	@-pkill -f gradlew 2>/dev/null || true
	@sleep 2
	@echo "ES processes stopped (or none running)"
