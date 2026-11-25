#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# TEMPORARY: build and test against conda packages from an unmerged RAFT PR.
# Do not merge. https://github.com/NVIDIA/raft/pull/3208

LIBRAFT_CHANNEL=$(rapids-get-pr-artifact NVIDIA/raft 3208 cpp conda)
# The 'raft' python artifact holds pylibraft (and raft-dask).
RAFT_CHANNEL=$(rapids-get-pr-artifact NVIDIA/raft 3208 python conda)

# For rattler builds: 'rapids-rattler-channel-string' prepends these, so with
# strict channel priority they win over the nightly channel.
RAPIDS_PREPENDED_CONDA_CHANNELS=(
  "${LIBRAFT_CHANNEL}"
  "${RAFT_CHANNEL}"
)
export RAPIDS_PREPENDED_CONDA_CHANNELS

# For any other conda installs: prepend to the system-wide channel list.
for _channel in "${RAPIDS_PREPENDED_CONDA_CHANNELS[@]}"; do
  conda config --system --add channels "${_channel}"
done

# For environments created from 'rapids-dependency-file-generator' output: the
# env file's own channel list (including rapidsai-nightly) can outrank the
# system config, so these channels are also passed in explicitly with
#   rapids-dependency-file-generator ... "${RAFT_PR_PREPEND_CHANNEL_ARGS[@]}"
RAFT_PR_PREPEND_CHANNEL_ARGS=()
for _channel in "${RAPIDS_PREPENDED_CONDA_CHANNELS[@]}"; do
  RAFT_PR_PREPEND_CHANNEL_ARGS+=(--prepend-channel "${_channel}")
done
