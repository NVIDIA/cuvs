#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3


cdef extern from "cuvs/distance/distance.h" nogil:
    ctypedef enum cuvsDistanceType:
        L2Expanded
        L2SqrtExpanded
        CosineExpanded
        L1
        L2Unexpanded
        L2SqrtUnexpanded
        InnerProduct
        Linf
        Canberra
        LpUnexpanded
        CorrelationExpanded
        JaccardExpanded
        HellingerExpanded
        Haversine
        BrayCurtis
        JensenShannon
        HammingUnexpanded
        KLDivergence
        RusselRaoExpanded
        DiceExpanded
        BitwiseHamming
        BitwiseJaccard
        Precomputed
