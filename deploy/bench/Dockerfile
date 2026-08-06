FROM python:3.11-slim
WORKDIR /app

ARG CUVS_REPOSITORY=https://github.com/jrbourbeau/cuvs.git
ARG CUVS_BRANCH=deploy-opensearch-tmp

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Sparse-clone just the cuvs_bench Python package.
# Installing via pip is not possible without CUDA (rapids-build-backend requires
# nvcc, and cuvs_bench declares a hard dep on the `cuvs` CUDA package).
# The opensearch backend is pure Python and needs neither, so we add the
# package directly to PYTHONPATH and install only the actual runtime deps.
RUN git clone --depth=1 --filter=blob:none --sparse \
      --branch "${CUVS_BRANCH}" "${CUVS_REPOSITORY}" /opt/cuvs \
    && cd /opt/cuvs \
    && git sparse-checkout set python/cuvs_bench

ENV PYTHONPATH=/opt/cuvs/python/cuvs_bench

# Runtime dependencies from cuvs_bench/pyproject.toml (excluding `cuvs` itself)
# plus extras needed by the opensearch backend and this benchmark script.
RUN pip install --no-cache-dir \
    boto3 \
    pytest \
    botocore \
    click \
    h5py \
    matplotlib \
    "numpy<2" \
    "opensearch-py>=2.4.0" \
    pandas \
    pyyaml \
    requests \
    scikit-learn \
    scipy

COPY --chmod=755 configure_opensearch.py entrypoint.sh prepare_custom_dataset.py print_results.py .
COPY opensearch_m64.yaml /app/opensearch_m64.yaml
CMD ["/app/entrypoint.sh"]
