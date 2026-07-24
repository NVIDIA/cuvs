#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


###############################################################################
#                                 Utilities                                   #
###############################################################################

dtype_sizes = {
    "float": 4,
    "fp8": 1,
    "half": 2,
}


###############################################################################
#                              cuVS constraints                               #
###############################################################################


def cuvs_cagra_build(params, dims):
    valid = True
    if "graph_degree" in params and "intermediate_graph_degree" in params:
        valid = params["graph_degree"] <= params["intermediate_graph_degree"]
    if params.get("merge_algo") == "FASTENER":
        levels = params.get("fastener_levels", 2)
        root_fanout = params.get("fastener_root_fanout", 2)
        lower_fanout = params.get("fastener_lower_fanout", 3)
        leader_fraction = params.get("fastener_leader_fraction", 0.02)
        max_leaders = params.get("fastener_max_leaders", 1000)
        leaf_size = params.get("fastener_leaf_size", 256)
        leaf_degree = params.get("fastener_leaf_degree", 4)
        valid = valid and levels > 0
        valid = valid and 1 <= root_fanout <= 32
        valid = valid and 1 <= lower_fanout <= 32
        valid = valid and 0.0 < leader_fraction <= 1.0
        valid = valid and max(root_fanout, lower_fanout) <= max_leaders <= 8192
        valid = valid and 1 <= leaf_size <= 256
        valid = valid and 1 <= leaf_degree <= 8
        if valid:
            max_spill = 255 // leaf_degree
            spill = root_fanout
            if lower_fanout > 1:
                for _ in range(1, levels):
                    if spill > max_spill // lower_fanout:
                        valid = False
                        break
                    spill *= lower_fanout
            valid = valid and spill <= max_spill
    return valid


def cuvs_ivf_pq_build(params, dims):
    if "pq_dim" in params:
        return params["pq_dim"] <= dims
    return True


def cuvs_ivf_pq_search(params, build_params, k, batch_size):
    ret = True
    if "internalDistanceDtype" in params and "smemLutDtype" in params:
        ret = (
            dtype_sizes[params["smemLutDtype"]]
            <= dtype_sizes[params["internalDistanceDtype"]]
        )

    if "nlist" in build_params and "nprobe" in params:
        ret = ret and build_params["nlist"] >= params["nprobe"]
    return ret


def cuvs_cagra_search(params, build_params, k, batch_size):
    if "itopk" in params:
        return params["itopk"] >= k
    return True


def cuvs_ivf_sq_search(params, build_params, k, batch_size):
    if "nlist" in build_params and "nprobe" in params:
        return build_params["nlist"] >= params["nprobe"]
    return True


###############################################################################
#                              FAISS constraints                              #
###############################################################################


def faiss_gpu_ivf_sq_search(params, build_params, k, batch_size):
    if "nlist" in build_params and "nprobe" in params:
        return build_params["nlist"] >= params["nprobe"]
    return True


def faiss_cpu_ivf_sq_search(params, build_params, k, batch_size):
    if "nlist" in build_params and "nprobe" in params:
        return build_params["nlist"] >= params["nprobe"]
    return True


def faiss_gpu_ivf_pq_build(params, dims):
    ret = True
    # M must be defined
    ret = params["M"] <= dims and dims % params["M"] == 0
    if "use_cuvs" in params and params["use_cuvs"]:
        return ret
    pq_bits = params.get("bitsPerCode", 8)
    lookup_table_size = 4
    if "useFloat16" in params and params["useFloat16"]:
        lookup_table_size = 2
    # FAISS constraint to check if lookup table fits in shared memory
    # for now hard code maximum shared memory per block to 49 kB
    # (the value for A100 and V100)
    return ret and lookup_table_size * params["M"] * (2**pq_bits) <= 49152


def faiss_gpu_ivf_pq_search(params, build_params, k, batch_size):
    ret = True
    if "nlist" in build_params and "nprobe" in params:
        ret = ret and build_params["nlist"] >= params["nprobe"]
    return ret


###############################################################################
#                              hnswlib constraints                            #
###############################################################################


def hnswlib_search(params, build_params, k, batch_size):
    if "ef" in params:
        return params["ef"] >= k


###############################################################################
#                              DiskANN constraints                            #
###############################################################################


def diskann_memory_build(params, dim):
    ret = True
    if "R" in params and "L_build" in params:
        ret = params["R"] <= params["L_build"]
    return ret


def diskann_ssd_build(params, dim):
    ret = True
    if "R" in params and "L_build" in params:
        ret = params["R"] <= params["L_build"]
    if "QD" in params:
        ret = params["QD"] <= dim
    return ret
