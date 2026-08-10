/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace cuvs::neighbors::ivf_rabitq::detail {

using PID = uint32_t;

struct Candidate {
  PID id;
  float distance;

  Candidate() = default;
  Candidate(PID id, float distance) : id(id), distance(distance) {}

  bool operator<(const Candidate& other) const { return distance < other.distance; }

  bool operator>(const Candidate& other) const { return !(*this < other); }
};

struct ExFactor {
  float xipnorm;
};

}  // namespace cuvs::neighbors::ivf_rabitq::detail
