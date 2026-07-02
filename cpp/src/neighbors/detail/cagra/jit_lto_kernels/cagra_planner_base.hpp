/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/detail/jit_lto/AlgorithmPlanner.hpp>
#include <cuvs/detail/jit_lto/cagra/cagra_fragments.hpp>
#include <cuvs/detail/jit_lto/common_fragments.hpp>
#include <cuvs/distance/distance.hpp>
#include <raft/core/error.hpp>
#include <raft/core/logger.hpp>

#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

namespace cuvs::neighbors::cagra::detail {

template <typename DataTag_,
          typename IndexTag_,
          typename DistanceTag_,
          typename QueryTag_,
          typename CodebookTag_,
          typename SampleFilterJitTag_ = tag_cagra_jit_sample_filter_link_absent>
struct CagraPlannerBase : AlgorithmPlanner {
  using DataTag            = DataTag_;
  using IndexTag           = IndexTag_;
  using DistanceTag        = DistanceTag_;
  using QueryTag           = QueryTag_;
  using CodebookTag        = CodebookTag_;
  using SampleFilterJitTag = SampleFilterJitTag_;

  explicit CagraPlannerBase(std::string entrypoint, LauncherJitCache& jit_cache)
    : AlgorithmPlanner(std::move(entrypoint), jit_cache)
  {
  }

  /// Standard codebook: workspace fragments always use `pq_bits=0`, `pq_len=0`.
  template <typename CB                                                  = CodebookTag,
            std::enable_if_t<std::is_same_v<CB, tag_codebook_none>, int> = 0>
  void add_setup_workspace_device_function(uint32_t team_size, uint32_t dataset_block_dim)
  {
    auto add = [&]<uint32_t TeamSz, uint32_t Dim, uint32_t PqBitsV, uint32_t PqLenV>() {
      this->add_static_fragment<fragment_tag_setup_workspace<DataTag,
                                                             IndexTag,
                                                             DistanceTag,
                                                             QueryTag,
                                                             CodebookTag,
                                                             TeamSz,
                                                             Dim,
                                                             PqBitsV,
                                                             PqLenV>>();
    };
    dispatch_cagra_team_dim(team_size, dataset_block_dim, [&add]<uint32_t TeamSz, uint32_t Dim>() {
      add.template operator()<TeamSz, Dim, 0u, 0u>();
    });
  }

  /// VPQ (`tag_codebook_half`): JIT matrix fixes `pq_bits=8`; only `pq_len` is selected at runtime.
  template <typename CB                                                  = CodebookTag,
            std::enable_if_t<std::is_same_v<CB, tag_codebook_half>, int> = 0>
  void add_setup_workspace_device_function(uint32_t team_size,
                                           uint32_t dataset_block_dim,
                                           uint32_t pq_len)
  {
    if (pq_len != 2 && pq_len != 4) {
      RAFT_FAIL("CAGRA JIT VPQ setup_workspace expects pq_len in {2,4} (matrix uses pq_bits=8)");
    }
    auto add = [&]<uint32_t TeamSz, uint32_t Dim, uint32_t PqBitsV, uint32_t PqLenV>() {
      this->add_static_fragment<fragment_tag_setup_workspace<DataTag,
                                                             IndexTag,
                                                             DistanceTag,
                                                             QueryTag,
                                                             CodebookTag,
                                                             TeamSz,
                                                             Dim,
                                                             PqBitsV,
                                                             PqLenV>>();
    };
    dispatch_cagra_team_dim(
      team_size, dataset_block_dim, [&add, pq_len]<uint32_t TeamSz, uint32_t Dim>() {
        if (pq_len == 2) {
          add.template operator()<TeamSz, Dim, 8u, 2u>();
        } else {
          add.template operator()<TeamSz, Dim, 8u, 4u>();
        }
      });
  }

  /// Registers dist_op + normalization + `compute_distance` for standard layout.
  template <typename CB                                                  = CodebookTag,
            std::enable_if_t<std::is_same_v<CB, tag_codebook_none>, int> = 0>
  void add_compute_distance_device_function(cuvs::distance::DistanceType metric,
                                            uint32_t team_size,
                                            uint32_t dataset_block_dim)
  {
    add_dist_op_device_function(metric);
    add_normalization_device_function(metric, team_size, dataset_block_dim);
    auto add = [&]<uint32_t TeamSz, uint32_t Dim, uint32_t PqBitsV, uint32_t PqLenV>() {
      this->add_static_fragment<fragment_tag_compute_distance<DataTag,
                                                              IndexTag,
                                                              DistanceTag,
                                                              QueryTag,
                                                              CodebookTag,
                                                              TeamSz,
                                                              Dim,
                                                              PqBitsV,
                                                              PqLenV>>();
    };
    dispatch_cagra_team_dim(team_size, dataset_block_dim, [&add]<uint32_t TeamSz, uint32_t Dim>() {
      add.template operator()<TeamSz, Dim, 0u, 0u>();
    });
  }

  /// VPQ: only the `compute_distance` fragment (no standard dist_op / normalization in this path).
  template <typename CB                                                  = CodebookTag,
            std::enable_if_t<std::is_same_v<CB, tag_codebook_half>, int> = 0>
  void add_compute_distance_device_function(uint32_t team_size,
                                            uint32_t dataset_block_dim,
                                            uint32_t pq_len)
  {
    if (pq_len != 2 && pq_len != 4) {
      RAFT_FAIL("CAGRA JIT VPQ compute_distance expects pq_len in {2,4} (matrix uses pq_bits=8)");
    }
    auto add = [&]<uint32_t TeamSz, uint32_t Dim, uint32_t PqBitsV, uint32_t PqLenV>() {
      this->add_static_fragment<fragment_tag_compute_distance<DataTag,
                                                              IndexTag,
                                                              DistanceTag,
                                                              QueryTag,
                                                              CodebookTag,
                                                              TeamSz,
                                                              Dim,
                                                              PqBitsV,
                                                              PqLenV>>();
    };
    dispatch_cagra_team_dim(
      team_size, dataset_block_dim, [&add, pq_len]<uint32_t TeamSz, uint32_t Dim>() {
        if (pq_len == 2) {
          add.template operator()<TeamSz, Dim, 8u, 2u>();
        } else {
          add.template operator()<TeamSz, Dim, 8u, 4u>();
        }
      });
  }

 private:
  void add_dist_op_device_function(cuvs::distance::DistanceType metric)
  {
    // dist_op_matrix.json pairs tag_metric_hamming with uint8 query (tag_u8) only; L2/IP/L1 use
    // float query (tag_f). A single switch over metric would still instantiate every case for each
    // QueryTag, pulling in fragment types that have no fatbin (e.g. tag_u8 + L2).
    if constexpr (std::is_same_v<QueryTag, cuvs::neighbors::detail::tag_u8>) {
      if (metric != cuvs::distance::DistanceType::BitwiseHamming) {
        RAFT_FAIL(
          "CAGRA JIT uint8 query layout (tag_u8) only supports BitwiseHamming for dist_op "
          "fragments");
      }
      this->add_static_fragment<fragment_tag_dist_op<QueryTag, DistanceTag, tag_metric_hamming>>();
    } else {
      switch (metric) {
        case cuvs::distance::DistanceType::L2Expanded:
        case cuvs::distance::DistanceType::L2Unexpanded:
          this->add_static_fragment<fragment_tag_dist_op<QueryTag, DistanceTag, tag_metric_l2>>();
          break;
        case cuvs::distance::DistanceType::InnerProduct:
        case cuvs::distance::DistanceType::CosineExpanded:
          // CosineExpanded reuses the InnerProduct dist_op; the cosine normalization is
          // layered on by add_normalization_device_function below.
          this->add_static_fragment<
            fragment_tag_dist_op<QueryTag, DistanceTag, tag_metric_inner_product>>();
          break;
        case cuvs::distance::DistanceType::BitwiseHamming:
          // Matrix only emits hamming dist_op for tag_u8; float-query layout is not built.
          RAFT_FAIL(
            "CAGRA JIT BitwiseHamming dist_op is only registered for uint8_t data / tag_u8 query "
            "layout");
          break;
        case cuvs::distance::DistanceType::L1:
          this->add_static_fragment<fragment_tag_dist_op<QueryTag, DistanceTag, tag_metric_l1>>();
          break;
        default: RAFT_FAIL("Unsupported metric for CAGRA JIT dist_op");
      }
    }
  }

  void add_normalization_device_function(cuvs::distance::DistanceType metric,
                                         uint32_t team_size,
                                         uint32_t dataset_block_dim)
  {
    auto go = [&]<typename NormT>() {
      dispatch_cagra_team_dim(team_size, dataset_block_dim, [&]<uint32_t TeamSz, uint32_t Dim>() {
        this->add_static_fragment<fragment_tag_apply_normalization_standard<DataTag,
                                                                            IndexTag,
                                                                            DistanceTag,
                                                                            QueryTag,
                                                                            TeamSz,
                                                                            Dim,
                                                                            NormT>>();
      });
    };
    // tag_u8 is only used for BitwiseHamming query layout; cosine norm fragments are built for
    // float query tag. Use if constexpr so we do not instantiate tag_norm_cosine with tag_u8
    // (a runtime metric check would still pull in those template specializations).
    if constexpr (std::is_same_v<QueryTag, cuvs::neighbors::detail::tag_u8>) {
      go.template operator()<tag_norm_noop>();
    } else if (metric == cuvs::distance::DistanceType::CosineExpanded) {
      go.template operator()<tag_norm_cosine>();
    } else {
      go.template operator()<tag_norm_noop>();
    }
  }

 public:
  // Maps runtime dataset layout (same grid as the JIT matrix) to uint32_t team / block-dim
  // template parameters; CAGRA reads team_size / dataset_block_dim from the host descriptor at
  // planning time.
  template <typename Lambda>
  static void dispatch_cagra_team_dim(uint32_t team_size, uint32_t dataset_block_dim, Lambda&& l)
  {
    switch (team_size) {
      case 8:
        switch (dataset_block_dim) {
          case 128: std::forward<Lambda>(l).template operator()<8u, 128u>(); return;
          case 256: std::forward<Lambda>(l).template operator()<8u, 256u>(); return;
          case 512: std::forward<Lambda>(l).template operator()<8u, 512u>(); return;
          default: break;
        }
        break;
      case 16:
        switch (dataset_block_dim) {
          case 128: std::forward<Lambda>(l).template operator()<16u, 128u>(); return;
          case 256: std::forward<Lambda>(l).template operator()<16u, 256u>(); return;
          case 512: std::forward<Lambda>(l).template operator()<16u, 512u>(); return;
          default: break;
        }
        break;
      case 32:
        switch (dataset_block_dim) {
          case 128: std::forward<Lambda>(l).template operator()<32u, 128u>(); return;
          case 256: std::forward<Lambda>(l).template operator()<32u, 256u>(); return;
          case 512: std::forward<Lambda>(l).template operator()<32u, 512u>(); return;
          default: break;
        }
        break;
      default: break;
    }
    RAFT_FAIL("Unsupported team_size / dataset_block_dim for CAGRA JIT: team=%u dim=%u",
              static_cast<unsigned>(team_size),
              static_cast<unsigned>(dataset_block_dim));
  }

  void add_sample_filter_device_function(std::unique_ptr<UDFFatbinFragment> udf_fragment = nullptr)
  {
    if constexpr (std::is_same_v<SampleFilterJitTag_, tag_cagra_jit_sample_filter_link_absent>) {
      RAFT_EXPECTS(udf_fragment == nullptr, "Unexpected CAGRA sample-filter UDF fragment");
    } else if constexpr (std::is_same_v<SampleFilterJitTag_,
                                        cuvs::neighbors::detail::tag_filter_udf>) {
      RAFT_EXPECTS(udf_fragment != nullptr, "CAGRA UDF filter requires a JIT-LTO fragment");
      this->add_fragment(std::move(udf_fragment));
    } else {
      RAFT_EXPECTS(udf_fragment == nullptr, "Built-in CAGRA sample filters use static fragments");
      this->add_static_fragment<fragment_tag_sample_filter<cuvs::neighbors::detail::tag_bitset_u32,
                                                           cuvs::neighbors::detail::tag_index_u32,
                                                           SampleFilterJitTag_>>();
    }
  }
};

}  // namespace cuvs::neighbors::cagra::detail
