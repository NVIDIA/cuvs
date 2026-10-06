/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../common/ann_types.hpp"

#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexBinaryIVF.h>
#include <faiss/index_io.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace cuvs::bench {

/** CPU Faiss Binary IVF benchmark wrapper for byte-packed binary vectors. */
class faiss_cpu_binary_ivf final : public algo<uint8_t> {
 public:
  using search_param_base = typename algo<uint8_t>::search_param;

  struct build_param {
    uint32_t nlist                      = 1;
    int niter                           = 25;
    int seed                            = 42;
    int max_points_per_centroid         = 64;
    std::size_t coarse_query_batch_size = 512;
  };

  struct search_param : public search_param_base {
    std::size_t nprobe = 1;
  };

  faiss_cpu_binary_ivf(Metric metric, int packed_dim, const build_param& params)
    : algo<uint8_t>(metric, packed_dim),
      build_params_(params),
      binary_dimension_(to_binary_dimension(packed_dim))
  {
    if (metric != Metric::kBitwiseHamming) {
      throw std::invalid_argument("Faiss Binary IVF requires bitwise_hamming distance");
    }
    validate_build_params();
  }

  faiss_cpu_binary_ivf(const faiss_cpu_binary_ivf& other)
    : algo<uint8_t>(other.metric_, other.dim_),
      build_params_(other.build_params_),
      binary_dimension_(other.binary_dimension_),
      quantizer_(other.quantizer_),
      index_(other.index_),
      search_params_(other.search_params_),
      distance_scratch_(other.distance_scratch_.size())
  {
  }

  void build(const uint8_t* dataset, std::size_t nrow) override
  {
    if (dataset == nullptr) { throw std::invalid_argument("Dataset must not be null"); }
    if (nrow < build_params_.nlist) {
      throw std::invalid_argument("Faiss Binary IVF requires at least nlist training vectors");
    }
    if (nrow > static_cast<std::size_t>(std::numeric_limits<faiss::idx_t>::max())) {
      throw std::overflow_error("Faiss Binary IVF dataset size exceeds faiss::idx_t");
    }

    initialize_index();
    index_->train(static_cast<faiss::idx_t>(nrow), dataset);
    if (!index_->is_trained) { throw std::runtime_error("Faiss Binary IVF training failed"); }
    index_->add(static_cast<faiss::idx_t>(nrow), dataset);
  }

  void set_search_param(const search_param_base& param, const void* filter_bitset) override
  {
    if (filter_bitset != nullptr) { throw std::runtime_error("Filtering is not supported yet."); }

    const auto& binary_param = dynamic_cast<const search_param&>(param);
    const auto nlist         = index_ ? index_->nlist : build_params_.nlist;
    if (binary_param.nprobe == 0 || binary_param.nprobe > nlist) {
      throw std::invalid_argument("nprobe must be in [1, nlist]");
    }
    search_params_.nprobe    = binary_param.nprobe;
    search_params_.max_codes = 0;
  }

  void search(const uint8_t* queries,
              int batch_size,
              int k,
              algo_base::index_type* neighbors,
              float* distances) const override
  {
    if (!index_) { throw std::runtime_error("Faiss Binary IVF index is not initialized"); }
    if (batch_size < 0 || k <= 0) { throw std::invalid_argument("Invalid search shape"); }

    const auto result_count = checked_result_count(batch_size, k);
    if (distance_scratch_.size() < result_count) { distance_scratch_.resize(result_count); }

    static_assert(sizeof(algo_base::index_type) == sizeof(faiss::idx_t));

    index_->search(batch_size,
                   queries,
                   k,
                   distance_scratch_.data(),
                   reinterpret_cast<faiss::idx_t*>(neighbors),
                   &search_params_);

    std::transform(distance_scratch_.begin(),
                   distance_scratch_.begin() + result_count,
                   distances,
                   [](int32_t distance) { return static_cast<float>(distance); });
  }

  void save(const std::string& file) const override
  {
    if (!index_) { throw std::runtime_error("Faiss Binary IVF index is not initialized"); }
    faiss::write_index_binary(index_.get(), file.c_str());
  }

  void load(const std::string& file) override
  {
    std::unique_ptr<faiss::IndexBinary> loaded{faiss::read_index_binary(file.c_str())};
    auto* loaded_ivf = dynamic_cast<faiss::IndexBinaryIVF*>(loaded.get());
    if (loaded_ivf == nullptr) {
      throw std::runtime_error("Serialized Faiss index is not an IndexBinaryIVF");
    }
    if (loaded_ivf->d != binary_dimension_) {
      throw std::runtime_error("Serialized Faiss Binary IVF dimension does not match the dataset");
    }

    if (!loaded_ivf->own_fields) {
      throw std::runtime_error("Loaded Faiss Binary IVF does not own its coarse quantizer");
    }
    configure_search(*loaded_ivf);
    configure_coarse_quantizer(loaded_ivf->quantizer);

    auto new_index = std::shared_ptr<faiss::IndexBinaryIVF>(loaded_ivf);
    loaded.release();

    index_ = std::move(new_index);
    // read_index_binary makes the loaded IVF index own its deserialized quantizer.
    quantizer_.reset();
  }

  [[nodiscard]] auto get_preference() const -> algo_property override
  {
    return {.dataset_memory_type = MemoryType::kHost, .query_memory_type = MemoryType::kHost};
  }

  auto copy() -> std::unique_ptr<algo<uint8_t>> override
  {
    // The index and quantizer are read-only during search and can be shared. Search parameters and
    // the integer-distance scratch buffer remain private to each benchmark thread.
    return std::make_unique<faiss_cpu_binary_ivf>(*this);
  }

 private:
  static auto to_binary_dimension(int packed_dim) -> faiss::idx_t
  {
    if (packed_dim <= 0) {
      throw std::invalid_argument("Packed binary dimension must be positive");
    }
    if (packed_dim > std::numeric_limits<faiss::idx_t>::max() / 8) {
      throw std::overflow_error("Packed binary dimension exceeds faiss::idx_t");
    }
    return static_cast<faiss::idx_t>(packed_dim) * 8;
  }

  template <typename BatchSize, typename Depth>
  static auto checked_result_count(BatchSize batch_size, Depth k) -> std::size_t
  {
    if constexpr (std::is_signed_v<BatchSize>) {
      if (batch_size < 0) { throw std::invalid_argument("Batch size must not be negative"); }
    }
    if constexpr (std::is_signed_v<Depth>) {
      if (k < 0) { throw std::invalid_argument("Search depth must not be negative"); }
    }
    const auto batch = static_cast<std::size_t>(batch_size);
    const auto depth = static_cast<std::size_t>(k);
    if (batch != 0 && depth > std::numeric_limits<std::size_t>::max() / batch) {
      throw std::overflow_error("Faiss Binary IVF result size overflow");
    }
    return batch * depth;
  }

  void validate_build_params() const
  {
    if (build_params_.nlist == 0) { throw std::invalid_argument("nlist must be positive"); }
    if (build_params_.niter <= 0) { throw std::invalid_argument("niter must be positive"); }
    if (build_params_.max_points_per_centroid <= 0) {
      throw std::invalid_argument("max_points_per_centroid must be positive");
    }
    if (build_params_.coarse_query_batch_size == 0) {
      throw std::invalid_argument("coarse_query_batch_size must be positive");
    }
  }

  void initialize_index()
  {
    auto new_quantizer              = std::make_shared<faiss::IndexBinaryFlat>(binary_dimension_);
    new_quantizer->query_batch_size = build_params_.coarse_query_batch_size;

    auto new_index = std::make_shared<faiss::IndexBinaryIVF>(
      new_quantizer.get(), binary_dimension_, build_params_.nlist);
    new_index->cp.niter                   = build_params_.niter;
    new_index->cp.seed                    = build_params_.seed;
    new_index->cp.max_points_per_centroid = build_params_.max_points_per_centroid;
    configure_search(*new_index);

    // Destroy an old borrowed-quantizer index before releasing the quantizer it points to.
    index_     = std::move(new_index);
    quantizer_ = std::move(new_quantizer);
  }

  static void configure_search(faiss::IndexBinaryIVF& index)
  {
    index.use_heap           = true;
    index.per_invlist_search = false;
    index.max_codes          = 0;
  }

  void configure_coarse_quantizer(faiss::IndexBinary* quantizer) const
  {
    auto* flat_quantizer = dynamic_cast<faiss::IndexBinaryFlat*>(quantizer);
    if (flat_quantizer == nullptr) {
      throw std::runtime_error("Faiss Binary IVF coarse quantizer is not IndexBinaryFlat");
    }
    flat_quantizer->query_batch_size = build_params_.coarse_query_batch_size;
  }

  build_param build_params_;
  faiss::idx_t binary_dimension_;
  std::shared_ptr<faiss::IndexBinaryFlat> quantizer_;
  std::shared_ptr<faiss::IndexBinaryIVF> index_;
  faiss::SearchParametersIVF search_params_;
  mutable std::vector<int32_t> distance_scratch_;
};

}  // namespace cuvs::bench
