/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * The minimal viable dataset API: the owning dataset, its view, and the one spec that needs no
 * storage backend at all. A dataset that does keep data around takes the spec that describes how -
 * `cuvs/core/dataset/dense.hpp`, `sparse.hpp`, `vpq.hpp`, `sq.hpp` - so that including this header
 * costs nothing beyond the types themselves.
 */

#include <cuvs/core/dataset/empty.hpp>
#include <cuvs/core/dataset/type.hpp>
#include <cuvs/core/dataset/view.hpp>
