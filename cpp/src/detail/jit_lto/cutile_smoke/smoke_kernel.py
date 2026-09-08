# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

import cuda.tile as ct


TILE_SIZE = 256


@ct.kernel
def cutile_smoke_add(lhs, rhs, output):
    block = ct.bid(0)
    lhs_tile = ct.load(lhs, block, TILE_SIZE)
    rhs_tile = ct.load(rhs, block, TILE_SIZE)
    ct.store(output, block, lhs_tile + rhs_tile)
