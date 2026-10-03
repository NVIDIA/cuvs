/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../common/ann_types.hpp"
#include "../common/util.hpp"
#include "../cuvs/cuvs_ann_bench_utils.h"
#include <cuvs/neighbors/refine.hpp>

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexRefine.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/gpu/GpuIndexCagra.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/GpuIndexIVFScalarQuantizer.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/index_io.h>
#include <omp.h>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>

#include <cassert>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

auto parse_metric_faiss(cuvs::bench::Metric metric) -> faiss::MetricType
{
  if (metric == cuvs::bench::Metric::kInnerProduct) {
    return faiss::METRIC_INNER_PRODUCT;
  } else if (metric == cuvs::bench::Metric::kEuclidean) {
    return faiss::METRIC_L2;
  } else {
    throw std::runtime_error("faiss supports only metric type of inner product and L2");
  }
}

// note BLAS library can still use multi-threading, and
// setting environment variable like OPENBLAS_NUM_THREADS can control it
class omp_single_thread_scope {
 public:
  omp_single_thread_scope()
  {
    max_threads_ = omp_get_max_threads();
    omp_set_num_threads(1);
  }
  ~omp_single_thread_scope()
  {
    // the best we can do
    omp_set_num_threads(max_threads_);
  }

 private:
  int max_threads_;
};

}  // namespace

namespace cuvs::bench {

template <typename T>
class faiss_gpu : public algo<T>, public algo_gpu {
 public:
  using search_param_base = typename algo<T>::search_param;
  struct search_param : public search_param_base {
    int nprobe         = 1;
    float refine_ratio = 1.0;
    [[nodiscard]] auto needs_dataset() const -> bool override { return refine_ratio > 1.0f; }
  };

  struct build_param {
    int nlist = 1;
    int ratio = 2;
  };

  faiss_gpu(Metric metric, int dim, const build_param& param)
    : algo<T>(metric, dim),
      gpu_resource_{std::make_shared<faiss::gpu::StandardGpuResources>()},
      metric_type_(parse_metric_faiss(metric)),
      nlist_{param.nlist},
      training_sample_fraction_{1.0 / double(param.ratio)}
  {
    static_assert(std::is_same_v<T, float>, "faiss support only float type");
    cudaGetDevice(&device_);
    // Disable Faiss' generic temporary memory reservation. All such allocations happen through the
    // pool memory resource.
    gpu_resource_->noTempMemory();
  }

  virtual void build(const T* dataset, size_t nrow);

  virtual void set_search_param(const search_param_base& param, const void* filter_bitset) {}

  void set_search_dataset(const T* dataset, size_t nrow) override { dataset_ = dataset; }

  // TODO(snanditale): if the number of results is less than k, the remaining elements of
  // 'neighbors' will be filled with (size_t)-1
  virtual void search(const T* queries,
                      int batch_size,
                      int k,
                      algo_base::index_type* neighbors,
                      float* distances) const;

  [[nodiscard]] auto get_sync_stream() const noexcept -> cudaStream_t override
  {
    return gpu_resource_->getDefaultStream(device_);
  }

  [[nodiscard]] auto get_preference() const -> algo_property override
  {
    algo_property property;
    // to enable building big dataset which is larger than GPU memory
    property.dataset_memory_type = MemoryType::kHost;
    property.query_memory_type   = MemoryType::kHost;
    return property;
  }

 protected:
  template <typename GpuIndex, typename CpuIndex>
  void save_(const std::string& file) const;  // NOLINT

  template <typename GpuIndex, typename CpuIndex>
  void load_(const std::string& file);  // NOLINT

  /** [NOTE Multithreading]
   *
   * `gpu_resource_` is a shared resource:
   *   1. It uses a shared_ptr under the hood, so the copies of it refer to the same
   *      resource implementation instance
   *   2. GpuIndex is probably keeping a reference to it, as it's passed to the constructor
   *
   * To avoid copying the index (database) in each thread, we make both the index and
   * the gpu_resource shared.
   * This means faiss GPU streams are possibly shared among the CPU threads;
   * the throughput search mode may be inaccurate.
   *
   * WARNING: we haven't investigated whether faiss::gpu::GpuIndex or
   * faiss::gpu::StandardGpuResources are thread-safe.
   *
   */

  // simply owning a configured_raft_resource object takes care of setting the pool memory resource
  configured_raft_resources handle_{};
  mutable std::shared_ptr<faiss::gpu::StandardGpuResources> gpu_resource_;
  std::shared_ptr<faiss::gpu::GpuIndex> index_;
  std::shared_ptr<faiss::IndexRefineFlat> index_refine_{nullptr};
  faiss::MetricType metric_type_;
  int nlist_;
  int device_ = 0;
  double training_sample_fraction_;
  std::shared_ptr<faiss::SearchParameters> search_params_;
  std::shared_ptr<faiss::IndexRefineSearchParameters> refine_search_params_{nullptr};
  const T* dataset_;
  float refine_ratio_ = 1.0;
};

template <typename T>
void faiss_gpu<T>::build(const T* dataset, size_t nrow)
{
  omp_single_thread_scope omp_single_thread;
  auto index_ivf = dynamic_cast<faiss::gpu::GpuIndexIVF*>(index_.get());
  if (index_ivf != nullptr) {
    // set the min/max training size for clustering to use the whole provided training set.
    double trainset_size       = training_sample_fraction_ * static_cast<double>(nrow);
    double points_per_centroid = trainset_size / static_cast<double>(nlist_);
    int max_ppc                = std::ceil(points_per_centroid);
    int min_ppc                = std::floor(points_per_centroid);
    if (min_ppc < index_ivf->cp.min_points_per_centroid) {
      log_warn(
        "The suggested training set size %zu (data size %zu, training sample ratio %f) yields %d "
        "points per cluster (n_lists = %d). This is smaller than the FAISS default "
        "min_points_per_centroid = %d.",
        static_cast<size_t>(trainset_size),
        nrow,
        training_sample_fraction_,
        min_ppc,
        nlist_,
        index_ivf->cp.min_points_per_centroid);
    }
    index_ivf->cp.max_points_per_centroid = max_ppc;
    index_ivf->cp.min_points_per_centroid = min_ppc;
  }
  index_->train(nrow, dataset);  // faiss::gpu::GpuIndexFlat::train() will do nothing
  assert(index_->is_trained);
  auto index_cagra = dynamic_cast<faiss::gpu::GpuIndexCagra*>(index_.get());
  if (index_cagra == nullptr) { index_->add(nrow, dataset); }
}

template <typename T>
void faiss_gpu<T>::search(
  const T* queries, int batch_size, int k, algo_base::index_type* neighbors, float* distances) const
{
  ASSERT(
    cuvs::bench::benchmark_n_threads == 1,
    "Throughput mode disabled. Underlying StandardGpuResources object might not be thread-safe.");
  using IdxT = faiss::idx_t;
  static_assert(sizeof(size_t) == sizeof(faiss::idx_t),
                "sizes of size_t and faiss::idx_t are different");

  if (refine_ratio_ > 1.0) {
    if (raft::get_device_for_address(queries) >= 0) {
      uint32_t k0        = static_cast<uint32_t>(refine_ratio_ * k);
      auto distances_tmp = raft::make_device_matrix<float, IdxT>(
        gpu_resource_->getRaftHandle(device_), batch_size, k0);
      auto candidates =
        raft::make_device_matrix<IdxT, IdxT>(gpu_resource_->getRaftHandle(device_), batch_size, k0);
      index_->search(batch_size,
                     queries,
                     k0,
                     distances_tmp.data_handle(),
                     candidates.data_handle(),
                     this->search_params_.get());
      gpu_resource_->getRaftHandle(device_).sync_stream();

      auto queries_host    = raft::make_host_matrix<T, IdxT>(batch_size, index_->d);
      auto candidates_host = raft::make_host_matrix<IdxT, IdxT>(batch_size, k0);
      auto neighbors_host  = raft::make_host_matrix<IdxT, IdxT>(batch_size, k);
      auto distances_host  = raft::make_host_matrix<float, IdxT>(batch_size, k);
      auto dataset_v       = raft::make_host_matrix_view<const T, faiss::idx_t>(
        this->dataset_, index_->ntotal, index_->d);

      raft::device_resources handle_ = gpu_resource_->getRaftHandle(device_);

      raft::copy(queries_host.data_handle(), queries, queries_host.size(), handle_.get_stream());
      raft::copy(candidates_host.data_handle(),
                 candidates.data_handle(),
                 candidates_host.size(),
                 handle_.get_stream());

      // wait for the queries to copy to host in 'stream`
      handle_.sync_stream();

      cuvs::neighbors::refine(handle_,
                              dataset_v,
                              queries_host.view(),
                              candidates_host.view(),
                              neighbors_host.view(),
                              distances_host.view(),
                              parse_metric_type(this->metric_));
      handle_.sync_stream();

      raft::copy(
        neighbors, neighbors_host.data_handle(), neighbors_host.size(), handle_.get_stream());
      raft::copy(
        distances, distances_host.data_handle(), distances_host.size(), handle_.get_stream());
    } else {
      index_refine_->search(batch_size,
                            queries,
                            k,
                            distances,
                            reinterpret_cast<faiss::idx_t*>(neighbors),
                            this->refine_search_params_.get());
    }
  } else {
    index_->search(batch_size,
                   queries,
                   k,
                   distances,
                   reinterpret_cast<faiss::idx_t*>(neighbors),
                   this->search_params_.get());
  }
}

template <typename T>
template <typename GpuIndex, typename CpuIndex>
void faiss_gpu<T>::save_(const std::string& file) const
{
  omp_single_thread_scope omp_single_thread;

  auto cpu_index  = std::make_unique<CpuIndex>();
  auto hnsw_index = dynamic_cast<faiss::IndexHNSWCagra*>(cpu_index.get());
  if (hnsw_index) { hnsw_index->base_level_only = true; }
  static_cast<GpuIndex*>(index_.get())->copyTo(cpu_index.get());
  faiss::write_index(cpu_index.get(), file.c_str());
}

template <typename T>
template <typename GpuIndex, typename CpuIndex>
void faiss_gpu<T>::load_(const std::string& file)
{
  omp_single_thread_scope omp_single_thread;

  std::unique_ptr<CpuIndex> cpu_index(dynamic_cast<CpuIndex*>(faiss::read_index(file.c_str())));
  assert(cpu_index);

  try {
    dynamic_cast<GpuIndex*>(index_.get())->copyFrom(cpu_index.get());

  } catch (const std::exception& e) {
    std::cout << "Error loading index file: " << std::string(e.what()) << std::endl;
  }
}

template <typename T>
class faiss_gpu_ivf_flat : public faiss_gpu<T> {
 public:
  struct build_param : public faiss_gpu<T>::build_param {
    bool use_cuvs;
  };
  using typename faiss_gpu<T>::search_param_base;

  faiss_gpu_ivf_flat(Metric metric, int dim, const build_param& param)
    : faiss_gpu<T>(metric, dim, param)
  {
    faiss::gpu::GpuIndexIVFFlatConfig config;
    config.device   = this->device_;
    config.use_cuvs = param.use_cuvs;
    this->index_    = std::make_shared<faiss::gpu::GpuIndexIVFFlat>(
      this->gpu_resource_.get(), dim, param.nlist, this->metric_type_, config);
  }

  void set_search_param(const search_param_base& param, const void* filter_bitset) override
  {
    if (filter_bitset != nullptr) { throw std::runtime_error("Filtering is not supported yet."); }
    auto sp    = dynamic_cast<const typename faiss_gpu<T>::search_param&>(param);
    int nprobe = sp.nprobe;
    assert(nprobe <= this->nlist_);

    faiss::IVFSearchParameters faiss_search_params;
    faiss_search_params.nprobe = nprobe;
    this->search_params_       = std::make_shared<faiss::IVFSearchParameters>(faiss_search_params);
    this->refine_ratio_        = sp.refine_ratio;
  }

  void save(const std::string& file) const override
  {
    this->template save_<faiss::gpu::GpuIndexIVFFlat, faiss::IndexIVFFlat>(file);
  }
  void load(const std::string& file) override
  {
    this->template load_<faiss::gpu::GpuIndexIVFFlat, faiss::IndexIVFFlat>(file);
  }
  std::unique_ptr<algo<T>> copy() override
  {
    return std::make_unique<faiss_gpu_ivf_flat<T>>(*this);
  }
};

template <typename T>
class faiss_gpu_ivfpq : public faiss_gpu<T> {
 public:
  struct build_param : public faiss_gpu<T>::build_param {
    int m;
    bool use_float16;
    bool use_precomputed;
    bool use_cuvs;
    int bitsPerCode;
  };
  using typename faiss_gpu<T>::search_param_base;

  faiss_gpu_ivfpq(Metric metric, int dim, const build_param& param)
    : faiss_gpu<T>(metric, dim, param)
  {
    faiss::gpu::GpuIndexIVFPQConfig config;
    config.useFloat16LookupTables = param.use_float16;
    config.usePrecomputedTables   = param.use_precomputed;
    config.use_cuvs               = param.use_cuvs;
    if (param.use_cuvs) { config.interleavedLayout = param.use_cuvs; }
    config.device = this->device_;

    this->index_ = std::make_shared<faiss::gpu::GpuIndexIVFPQ>(this->gpu_resource_.get(),
                                                               dim,
                                                               param.nlist,
                                                               param.m,
                                                               param.bitsPerCode,
                                                               this->metric_type_,
                                                               config);
  }

  void set_search_param(const search_param_base& param, const void* filter_bitset) override
  {
    if (filter_bitset != nullptr) { throw std::runtime_error("Filtering is not supported yet."); }
    auto sp    = dynamic_cast<const typename faiss_gpu<T>::search_param&>(param);
    int nprobe = sp.nprobe;
    assert(nprobe <= this->nlist_);
    this->refine_ratio_ = sp.refine_ratio;
    faiss::IVFPQSearchParameters faiss_search_params;
    faiss_search_params.nprobe = nprobe;

    this->search_params_ = std::make_shared<faiss::IVFPQSearchParameters>(faiss_search_params);

    if (sp.refine_ratio > 1.0) {
      this->index_refine_ =
        std::make_shared<faiss::IndexRefineFlat>(this->index_.get(), this->dataset_);
      this->index_refine_.get()->k_factor = sp.refine_ratio;
      faiss::IndexRefineSearchParameters faiss_refine_search_params;
      faiss_refine_search_params.k_factor          = this->index_refine_.get()->k_factor;
      faiss_refine_search_params.base_index_params = this->search_params_.get();
      this->refine_search_params_ =
        std::make_unique<faiss::IndexRefineSearchParameters>(faiss_refine_search_params);
    }
  }

  void save(const std::string& file) const override
  {
    this->template save_<faiss::gpu::GpuIndexIVFPQ, faiss::IndexIVFPQ>(file);
  }
  void load(const std::string& file) override
  {
    this->template load_<faiss::gpu::GpuIndexIVFPQ, faiss::IndexIVFPQ>(file);
  }
  std::unique_ptr<algo<T>> copy() override { return std::make_unique<faiss_gpu_ivfpq<T>>(*this); };
};

// TODO(snanditale): Enable this in cmake
//  ref: https://github.com/NVIDIA/raft/issues/1876
template <typename T>
class faiss_gpu_ivfsq : public faiss_gpu<T> {
 public:
  struct build_param : public faiss_gpu<T>::build_param {
    std::string quantizer_type;
  };
  using typename faiss_gpu<T>::search_param_base;

  faiss_gpu_ivfsq(Metric metric, int dim, const build_param& param)
    : faiss_gpu<T>(metric, dim, param)
  {
    faiss::ScalarQuantizer::QuantizerType qtype;
    if (param.quantizer_type == "fp16") {
      qtype = faiss::ScalarQuantizer::QT_fp16;
    } else if (param.quantizer_type == "int8") {
      qtype = faiss::ScalarQuantizer::QT_8bit;
    } else {
      throw std::runtime_error("faiss_gpu_ivfsq supports only fp16 and int8 but got " +
                               param.quantizer_type);
    }

    faiss::gpu::GpuIndexIVFScalarQuantizerConfig config;
    config.device = this->device_;
    this->index_  = std::make_shared<faiss::gpu::GpuIndexIVFScalarQuantizer>(
      this->gpu_resource_.get(), dim, param.nlist, qtype, this->metric_type_, true, config);
  }

  void set_search_param(const search_param_base& param, const void* filter_bitset) override
  {
    if (filter_bitset != nullptr) { throw std::runtime_error("Filtering is not supported yet."); }
    auto sp    = dynamic_cast<const typename faiss_gpu<T>::search_param&>(param);
    int nprobe = sp.nprobe;
    assert(nprobe <= this->nlist_);

    faiss::IVFSearchParameters faiss_search_params;
    faiss_search_params.nprobe = nprobe;

    this->search_params_ = std::make_shared<faiss::IVFSearchParameters>(faiss_search_params);
    this->refine_ratio_  = sp.refine_ratio;
    if (sp.refine_ratio > 1.0) {
      this->index_refine_ =
        std::make_shared<faiss::IndexRefineFlat>(this->index_.get(), this->dataset_);
      this->index_refine_.get()->k_factor = sp.refine_ratio;
    }
  }

  void save(const std::string& file) const override
  {
    this->template save_<faiss::gpu::GpuIndexIVFScalarQuantizer, faiss::IndexIVFScalarQuantizer>(
      file);
  }
  void load(const std::string& file) override
  {
    this->template load_<faiss::gpu::GpuIndexIVFScalarQuantizer, faiss::IndexIVFScalarQuantizer>(
      file);
  }
  std::unique_ptr<algo<T>> copy() override { return std::make_unique<faiss_gpu_ivfsq<T>>(*this); };
};

template <typename T>
class faiss_gpu_flat : public faiss_gpu<T> {
 public:
  using typename faiss_gpu<T>::search_param_base;

  faiss_gpu_flat(Metric metric, int dim)
    : faiss_gpu<T>(metric, dim, typename faiss_gpu<T>::build_param{})
  {
    faiss::gpu::GpuIndexFlatConfig config;
    config.device = this->device_;
    this->index_  = std::make_shared<faiss::gpu::GpuIndexFlat>(
      this->gpu_resource_.get(), dim, this->metric_type_, config);
  }
  void set_search_param(const search_param_base& param, const void* filter_bitset) override
  {
    if (filter_bitset != nullptr) { throw std::runtime_error("Filtering is not supported yet."); }
    auto sp    = dynamic_cast<const typename faiss_gpu<T>::search_param&>(param);
    int nprobe = sp.nprobe;
    assert(nprobe <= this->nlist_);

    this->search_params_ = std::make_shared<faiss::SearchParameters>();
  }

  void save(const std::string& file) const override
  {
    this->template save_<faiss::gpu::GpuIndexFlat, faiss::IndexFlat>(file);
  }
  void load(const std::string& file) override
  {
    this->template load_<faiss::gpu::GpuIndexFlat, faiss::IndexFlat>(file);
  }
  std::unique_ptr<algo<T>> copy() override { return std::make_unique<faiss_gpu_flat<T>>(*this); };
};

template <typename T>
class faiss_gpu_cagra : public faiss_gpu<T> {
 public:
  struct build_param : public faiss_gpu<T>::build_param {
    size_t intermediate_graph_degree;
    /// Degree of output graph.
    size_t graph_degree;
    /// ANN algorithm to build knn graph.
    std::string cagra_build_algo;
    /// Number of Iterations to run if building with NN_DESCENT
    size_t nn_descent_niter;

    std::shared_ptr<faiss::gpu::IVFPQBuildCagraConfig> ivf_pq_build_params = nullptr;

    std::shared_ptr<faiss::gpu::IVFPQSearchCagraConfig> ivf_pq_search_params = nullptr;
  };
  using typename faiss_gpu<T>::search_param_base;
  struct search_param : public faiss_gpu<T>::search_param {
    faiss::gpu::SearchParametersCagra p;
  };

  faiss_gpu_cagra(Metric metric, int dim, const build_param& param)
    : faiss_gpu<T>(metric, dim, param)
  {
    faiss::gpu::GpuIndexCagraConfig config;
    config.graph_degree              = param.graph_degree;
    config.intermediate_graph_degree = param.intermediate_graph_degree;
    config.device                    = this->device_;
    config.store_dataset             = true;
    if (param.cagra_build_algo == "IVF_PQ") {
      config.build_algo           = faiss::gpu::graph_build_algo::IVF_PQ;
      this->ivf_pq_build_params_  = param.ivf_pq_build_params;
      config.ivf_pq_params        = this->ivf_pq_build_params_;
      this->ivf_pq_search_params_ = param.ivf_pq_search_params;
      config.ivf_pq_search_params = this->ivf_pq_search_params_;
      config.refine_rate          = 1.0;
    } else {
      config.build_algo = faiss::gpu::graph_build_algo::NN_DESCENT;
    }
    config.nn_descent_niter = param.nn_descent_niter;

    this->index_ = std::make_shared<faiss::gpu::GpuIndexCagra>(
      this->gpu_resource_.get(), dim, parse_metric_faiss(this->metric_), config);
  }

  void set_search_param(const search_param_base& param, const void* filter_bitset) override
  {
    if (filter_bitset != nullptr) { throw std::runtime_error("Filtering is not supported yet."); }
    auto sp              = static_cast<const typename faiss_gpu_cagra<T>::search_param&>(param);
    this->search_params_ = std::make_shared<faiss::gpu::SearchParametersCagra>(sp.p);
  }

  void save(const std::string& file) const override
  {
    omp_single_thread_scope omp_single_thread;

    auto cpu_hnsw_index = std::make_unique<faiss::IndexHNSWCagra>();
    // Only add the base HNSW layer to serialize the CAGRA index.
    cpu_hnsw_index->base_level_only = true;
    static_cast<faiss::gpu::GpuIndexCagra*>(this->index_.get())->copyTo(cpu_hnsw_index.get());
    faiss::write_index(cpu_hnsw_index.get(), file.c_str());
  }
  void load(const std::string& file) override
  {
    this->template load_<faiss::gpu::GpuIndexCagra, faiss::IndexHNSWCagra>(file);
  }
  std::unique_ptr<algo<T>> copy() override { return std::make_unique<faiss_gpu_cagra<T>>(*this); };

  std::shared_ptr<faiss::gpu::GpuIndex> faiss_index() { return this->index_; }

 private:
  std::shared_ptr<faiss::gpu::IVFPQBuildCagraConfig> ivf_pq_build_params_;
  std::shared_ptr<faiss::gpu::IVFPQSearchCagraConfig> ivf_pq_search_params_;
};

template <typename T>
class faiss_gpu_cagra_hnsw : public faiss_gpu<T> {
 public:
  struct build_param : public faiss_gpu<T>::build_param {
    typename faiss_gpu_cagra<T>::build_param p;
    bool base_level_only = true;
  };
  using typename faiss_gpu<T>::search_param_base;
  struct search_param : public faiss_gpu<T>::search_param {
    faiss::SearchParametersHNSW p;
  };

  faiss_gpu_cagra_hnsw(Metric metric, int dim, const build_param& param)
    : faiss_gpu<T>(metric, dim, param)
  {
    this->build_index_  = std::make_shared<faiss_gpu_cagra<T>>(metric, dim, param.p);
    this->search_index_ = std::make_shared<faiss::IndexHNSWCagra>(
      dim, int(param.p.graph_degree / 2), this->metric_type_);
    this->search_index_->base_level_only = param.base_level_only;
  }

  void build(const T* dataset, size_t nrow) override
  {
    this->build_index_->build(dataset, nrow);
    static_cast<faiss::gpu::GpuIndexCagra*>((build_index_->faiss_index()).get())
      ->copyTo(search_index_.get());
  }

  void set_search_param(const search_param_base& param, const void* filter_bitset) override
  {
    if (filter_bitset != nullptr) { throw std::runtime_error("Filtering is not supported yet."); }
    auto sp = static_cast<const typename faiss_gpu_cagra_hnsw<T>::search_param&>(param);
    this->search_params_ = std::make_shared<faiss::SearchParametersHNSW>(sp.p);
  }

  void search(const T* queries,
              int batch_size,
              int k,
              algo_base::index_type* neighbors,
              float* distances) const override
  {
    search_index_->search(batch_size,
                          queries,
                          k,
                          distances,
                          reinterpret_cast<faiss::idx_t*>(neighbors),
                          this->search_params_.get());
  }

  void save(const std::string& file) const override
  {
    faiss::write_index(search_index_.get(), file.c_str());
  }
  void load(const std::string& file) override
  {
    omp_single_thread_scope omp_single_thread;
    this->search_index_.reset(static_cast<faiss::IndexHNSWCagra*>(faiss::read_index(file.c_str())));
  }

  [[nodiscard]] auto get_sync_stream() const noexcept -> cudaStream_t override
  {
    if (this->gpu_resource_ == nullptr) { return 0; }
    return this->gpu_resource_->getDefaultStream(this->device_);
  }
  std::unique_ptr<algo<T>> copy() override
  {
    auto new_instance          = std::make_unique<faiss_gpu_cagra_hnsw<T>>(*this);
    new_instance->build_index_ = nullptr;
    return new_instance;
  };

 private:
  std::shared_ptr<faiss_gpu_cagra<T>> build_index_;
  std::shared_ptr<faiss::IndexHNSWCagra> search_index_;
};
}  // namespace cuvs::bench
