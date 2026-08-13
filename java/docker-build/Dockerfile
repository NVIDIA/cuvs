#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

###
# Build the image for cuVS Java API development environment.
#
# Arguments: CUDA_VERSION=[12.X.Y], OS_RELEASE=[8, 9], TARGETPLATFORM=[linux/amd64, linux/arm64]
#
###
ARG CUDA_VERSION=12.9.1
ARG OS_RELEASE=9
ARG TARGETPLATFORM=linux/amd64

# multi-platform build with: docker buildx build --platform linux/arm64,linux/amd64 <ARGS> on either amd64 or arm64 host
# check available official arm-based docker images at https://hub.docker.com/r/nvidia/cuda/tags (OS/ARCH)
FROM --platform=$TARGETPLATFORM nvidia/cuda:$CUDA_VERSION-devel-rockylinux$OS_RELEASE

# Re-declare ARGs after FROM since they don't persist across FROM boundary
ARG CUDA_VERSION=12.9.1
ARG OS_RELEASE=9
ARG TARGETPLATFORM=linux/amd64

# If DEV_BUILD is ON, the gcc-toolset will be enabled by default for bash shell
ARG DEV_BUILD=OFF

# Dependency versions
# Act as default GCC toolset in the image
ARG TOOLSET_VERSION=14
ARG CMAKE_VERSION=4.0
ARG CCACHE_VERSION=4.11.2

# Default x86_64 from x86 build, aarch64 cmake for arm build
ARG CMAKE_ARCH=x86_64

### Install basic requirements
RUN if [ "$OS_RELEASE" = "9" ]; then \
    dnf --enablerepo=crb install -y scl-utils gcc-toolset-${TOOLSET_VERSION} python39 \
      zlib-devel maven tar wget patch ninja-build git zip; \
  elif [ "$OS_RELEASE" = "8" ]; then \
    dnf --enablerepo=devel install -y scl-utils gcc-toolset-${TOOLSET_VERSION} python39 \
      zlib-devel maven tar wget patch ninja-build git zip; \
  else \
    echo "Unsupported OS version: $OS_RELEASE"; \
  fi

# Enable the gcc-toolset by default for bash shell if DEV_BUILD is ON
RUN if [ "$DEV_BUILD" = "ON" ]; then \
        echo "source scl_source enable gcc-toolset-${TOOLSET_VERSION}" >> /etc/bashrc; \
    fi

# Execute every time a new non-interactive bash shell is started
ENV BASH_ENV=/etc/bashrc

## pre-create the CMAKE_INSTALL_PREFIX folder, set writable by any user for Jenkins
RUN mkdir -m 777 /usr/local/rapids /rapids

# Fetch and install CMake.
RUN cd /usr/local && wget --quiet https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-${CMAKE_ARCH}.tar.gz && \
   tar zxf cmake-${CMAKE_VERSION}-linux-${CMAKE_ARCH}.tar.gz && \
   rm cmake-${CMAKE_VERSION}-linux-${CMAKE_ARCH}.tar.gz

# Make version-less alias for external reference such as when cmake is called by an IDE outside of the container
RUN ln -s /usr/local/cmake-${CMAKE_VERSION}-linux-${CMAKE_ARCH}/bin/cmake /usr/local/bin/cmake

# ccache for interactive builds
RUN cd /tmp && wget --quiet https://github.com/ccache/ccache/releases/download/v${CCACHE_VERSION}/ccache-${CCACHE_VERSION}.tar.gz && \
   tar zxf ccache-${CCACHE_VERSION}.tar.gz && \
   rm ccache-${CCACHE_VERSION}.tar.gz && \
   cd ccache-${CCACHE_VERSION} && \
   mkdir build && \
   cd build && \
   scl enable gcc-toolset-${TOOLSET_VERSION} \
      "cmake .. \
         -DCMAKE_BUILD_TYPE=Release \
         -DZSTD_FROM_INTERNET=ON \
         -DREDIS_STORAGE_BACKEND=OFF && \
      cmake --build . --parallel ${PARALLEL_LEVEL} --target install" && \
   cd ../.. && \
   rm -rf ccache-${CCACHE_VERSION}

# Fetch and install JDK 22.
RUN cd /usr/local && wget --quiet https://download.oracle.com/java/22/archive/jdk-22.0.2_linux-x64_bin.tar.gz && \
   tar zxf jdk-22.0.2_linux-x64_bin.tar.gz && \
   rm jdk-22.0.2_linux-x64_bin.tar.gz

# Make Java22 available in path.
RUN ln -sf /usr/local/jdk-22.0.2/bin/java /usr/local/bin/java
ENV JAVA_HOME="/usr/local/jdk-22.0.2"

# disable cuda container constraints to allow running w/ older drivers on datacenter GPUs
ENV NVIDIA_DISABLE_REQUIRE="true"
