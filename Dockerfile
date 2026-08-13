# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# syntax=docker/dockerfile:1.5

# ==============================================================================
# RAPIDS cuVS Docker Image - Vector Search and Clustering on GPU
#
# Usage:
#   docker build -t cuvs:latest .
#   docker run --gpus all -it cuvs:latest
#
# Customizable Build Arguments:
#   docker build --build-arg CUDA_VER=12.8 --build-arg RAPIDS_VER=25.06 -t cuvs:custom .
# ==============================================================================

# Configurable build arguments - override with --build-arg during build
ARG CUDA_VER=12.9.1

# Python version for the conda environment (supported: 3.11, 3.12, 3.13)
ARG PYTHON_VER=3.12

# RAPIDS/cuVS version
ARG RAPIDS_VER=25.06

FROM nvidia/cuda:${CUDA_VER}-devel-ubuntu24.04

# Display build configuration for verification
RUN echo "  Building cuVS Docker image with:" && \
    echo "   CUDA Version: ${CUDA_VER}" && \
    echo "   Python Version: ${PYTHON_VER}" && \
    echo "   RAPIDS Version: ${RAPIDS_VER}"

# Container metadata
LABEL maintainer="RAPIDS cuVS Team"
LABEL description="RAPIDS cuVS - Vector Search and Clustering on GPU"
LABEL org.opencontainers.image.source="https://github.com/nvidia/cuvs"
LABEL org.opencontainers.image.usage="docker run --gpus all -it <image>"

# Environment setup
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge (lightweight conda with conda-forge defaults)
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    /bin/bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh && \
    ${CONDA_DIR}/bin/conda clean -afy

# Create conda environment and install cuVS packages with pinned dependencies
RUN conda create -n cuvs python=${PYTHON_VER} -y && \
    conda run -n cuvs conda install -c rapidsai -c conda-forge \
        cuvs=${RAPIDS_VER} \
        libcuvs=${RAPIDS_VER} \
        cuda-version=${CUDA_VER} \
        "numpy>=1.23,<3.0a0" \
        "cupy>=12.0.0" \
        jupyter \
        ipython \
        -y && \
    conda clean -afy

# Create non-root user
RUN useradd -m -s /bin/bash cuvs
USER cuvs
WORKDIR /home/cuvs

# Initialize conda for the user and configure environment activation
RUN ${CONDA_DIR}/bin/conda init bash && \
    echo "conda activate cuvs" >> ~/.bashrc && \
    echo "conda activate cuvs" >> ~/.bash_profile

# Set environment variables for automatic cuvs environment activation
ENV PATH=${CONDA_DIR}/envs/cuvs/bin:$PATH
ENV CONDA_DEFAULT_ENV=cuvs
ENV CONDA_PREFIX=${CONDA_DIR}/envs/cuvs

# Verify installation and show version info
RUN python -c "import cuvs; print('cuVS version:', cuvs.__version__)"

# Default command
CMD ["bash"]
