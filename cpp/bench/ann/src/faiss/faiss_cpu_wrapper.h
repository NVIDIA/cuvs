/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../common/ann_types.hpp"
#include "../common/util.hpp"

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexIVFRaBitQ.h>
#include <faiss/IndexRefine.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/index_io.h>

#include <cassert>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

auto parse_metric_type(cuvs::bench::Metric metric) -> faiss::MetricType
{
  if (metric == cuvs::bench::Metric::kInnerProduct) {
    return faiss::METRIC_INNER_PRODUCT;
  } else if (metric == cuvs::bench::Metric::kEuclidean) {
    return faiss::METRIC_L2;
  } else {
    throw std::runtime_error("faiss supports only metric type of inner product and L2");
  }
}
}  // namespace

namespace cuvs::bench {

template <typename T>
class faiss_cpu : public algo<T> {
 public:
  using search_param_base = typename algo<T>::search_param;
  struct search_param : public search_param_base {
    int nprobe;
    float refine_ratio = 1.0;
    /**
     * parallel_mode determines how queries are parallelized with OpenMP.
     *
     * Options:
     * 0: Split over queries.
     * 1: Parallelize over inverted lists.
     * 2: Parallelize over both queries and inverted lists.
     * 3: Split over queries with finer granularity.
     */
    int parallel_mode = 0;
  };

  struct build_param {
    int nlist = 1;
    int ratio = 2;
  };

  faiss_cpu(Metric metric, int dim, const build_param& param)
    : algo<T>(metric, dim),
      metric_type_(parse_metric_type(metric)),
      nlist_{param.nlist},
      training_sample_fraction_{1.0 / double(param.ratio)}
  {
    static_assert(std::is_same_v<T, float>, "faiss support only float type");
  }

  void build(const T* dataset, size_t nrow) final;

  void set_search_param(const search_param_base& param, const void* filter_bitset) override;

  void init_quantizer(int dim)
  {
    if (this->metric_type_ == faiss::MetricType::METRIC_L2) {
      this->quantizer_ = std::make_shared<faiss::IndexFlatL2>(dim);
    } else if (this->metric_type_ == faiss::MetricType::METRIC_INNER_PRODUCT) {
      this->quantizer_ = std::make_shared<faiss::IndexFlatIP>(dim);
    }
  }

  // TODO(snanditale): if the number of results is less than k, the remaining elements of
  // 'neighbors' will be filled with (size_t)-1
  virtual void search(const T* queries,
                      int batch_size,
                      int k,
                      algo_base::index_type* neighbors,
                      float* distances) const;

  [[nodiscard]] auto get_preference() const -> algo_property override
  {
    algo_property property;
    // to enable building big dataset which is larger than  memory
    property.dataset_memory_type = MemoryType::kHost;
    property.query_memory_type   = MemoryType::kHost;
    return property;
  }

 protected:
  template <typename Index>
  void save_(const std::string& file) const;  // NOLINT

  template <typename Index>
  void load_(const std::string& file);  // NOLINT

  std::shared_ptr<faiss::Index> index_;
  std::shared_ptr<faiss::Index> quantizer_;
  std::shared_ptr<faiss::IndexRefineFlat> index_refine_;
  faiss::MetricType metric_type_;
  int nlist_;
  double training_sample_fraction_;
};

template <typename T>
void faiss_cpu<T>::build(const T* dataset, size_t nrow)
{
  auto index_ivf = dynamic_cast<faiss::IndexIVF*>(index_.get());
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
  faiss::IndexHNSWFlat* hnsw_index = dynamic_cast<faiss::IndexHNSWFlat*>(index_.get());
  index_->train(nrow, dataset);  // faiss::IndexFlat::train() will do nothing
  assert(index_->is_trained);
  index_->add(nrow, dataset);
  index_refine_ = std::make_shared<faiss::IndexRefineFlat>(this->index_.get(), dataset);
}

template <typename T>
void faiss_cpu<T>::set_search_param(const search_param_base& param, const void* filter_bitset)
{
  if (filter_bitset != nullptr) { throw std::runtime_error("Filtering is not supported yet."); }
  auto sp    = dynamic_cast<const search_param&>(param);
  int nprobe = sp.nprobe;
  assert(nprobe <= nlist_);
  auto index_ivf = dynamic_cast<faiss::IndexIVF*>(index_.get());
  if (index_ivf != nullptr) {
    index_ivf->nprobe        = nprobe;
    index_ivf->parallel_mode = sp.parallel_mode;
  }

  if (sp.refine_ratio > 1.0) { this->index_refine_.get()->k_factor = sp.refine_ratio; }
}

template <typename T>
void faiss_cpu<T>::search(
  const T* queries, int batch_size, int k, algo_base::index_type* neighbors, float* distances) const
{
  static_assert(sizeof(size_t) == sizeof(faiss::idx_t),
                "sizes of size_t and faiss::idx_t are different");

  index_->search(batch_size, queries, k, distances, reinterpret_cast<faiss::idx_t*>(neighbors));
}

template <typename T>
template <typename Index>
void faiss_cpu<T>::save_(const std::string& file) const
{
  faiss::write_index(index_.get(), file.c_str());
}

template <typename T>
template <typename Index>
void faiss_cpu<T>::load_(const std::string& file)
{
  index_ = std::shared_ptr<Index>(dynamic_cast<Index*>(faiss::read_index(file.c_str())));
}

template <typename T>
class faiss_cpu_ivf_flat : public faiss_cpu<T> {
 public:
  using typename faiss_cpu<T>::build_param;

  faiss_cpu_ivf_flat(Metric metric, int dim, const build_param& param)
    : faiss_cpu<T>(metric, dim, param)
  {
    this->init_quantizer(dim);
    this->index_ = std::make_shared<faiss::IndexIVFFlat>(
      this->quantizer_.get(), dim, param.nlist, this->metric_type_);
  }

  void save(const std::string& file) const override
  {
    this->template save_<faiss::IndexIVFFlat>(file);
  }
  void load(const std::string& file) override { this->template load_<faiss::IndexIVFFlat>(file); }

  std::unique_ptr<algo<T>> copy()
  {
    return std::make_unique<faiss_cpu_ivf_flat<T>>(*this);  // use copy constructor
  }
};

template <typename T>
class faiss_cpu_ivfpq : public faiss_cpu<T> {
 public:
  struct build_param : public faiss_cpu<T>::build_param {
    int m;
    int bits_per_code;
    bool use_precomputed;
  };

  faiss_cpu_ivfpq(Metric metric, int dim, const build_param& param)
    : faiss_cpu<T>(metric, dim, param)
  {
    this->init_quantizer(dim);
    this->index_ = std::make_shared<faiss::IndexIVFPQ>(
      this->quantizer_.get(), dim, param.nlist, param.m, param.bits_per_code, this->metric_type_);
  }

  void save(const std::string& file) const override
  {
    this->template save_<faiss::IndexIVFPQ>(file);
  }
  void load(const std::string& file) override { this->template load_<faiss::IndexIVFPQ>(file); }

  std::unique_ptr<algo<T>> copy()
  {
    return std::make_unique<faiss_cpu_ivfpq<T>>(*this);  // use copy constructor
  }
};

template <typename T>
class faiss_cpu_ivfrabitq : public faiss_cpu<T> {
 public:
  struct build_param : public faiss_cpu<T>::build_param {};

  faiss_cpu_ivfrabitq(Metric metric, int dim, const build_param& param)
    : faiss_cpu<T>(metric, dim, param)
  {
    this->init_quantizer(dim);
    this->index_ = std::make_shared<faiss::IndexIVFRaBitQ>(
      this->quantizer_.get(), dim, param.nlist, this->metric_type_);
  }

  void save(const std::string& file) const override
  {
    this->template save_<faiss::IndexIVFRaBitQ>(file);
  }
  void load(const std::string& file) override { this->template load_<faiss::IndexIVFRaBitQ>(file); }

  std::unique_ptr<algo<T>> copy()
  {
    return std::make_unique<faiss_cpu_ivfrabitq<T>>(*this);  // use copy constructor
  }
};

// TODO(snanditale): Enable this in cmake
//  ref: https://github.com/rapidsai/raft/issues/1876
template <typename T>
class faiss_cpu_ivfsq : public faiss_cpu<T> {
 public:
  struct build_param : public faiss_cpu<T>::build_param {
    std::string quantizer_type;
  };

  faiss_cpu_ivfsq(Metric metric, int dim, const build_param& param)
    : faiss_cpu<T>(metric, dim, param)
  {
    faiss::ScalarQuantizer::QuantizerType qtype;
    if (param.quantizer_type == "fp16") {
      qtype = faiss::ScalarQuantizer::QT_fp16;
    } else if (param.quantizer_type == "int8") {
      qtype = faiss::ScalarQuantizer::QT_8bit;
    } else {
      throw std::runtime_error("faiss_cpu_ivfsq supports only fp16 and int8 but got " +
                               param.quantizer_type);
    }

    this->init_quantizer(dim);
    this->index_ = std::make_shared<faiss::IndexIVFScalarQuantizer>(
      this->quantizer_.get(), dim, param.nlist, qtype, this->metric_type_, true);
  }

  void save(const std::string& file) const override
  {
    this->template save_<faiss::IndexIVFScalarQuantizer>(file);
  }
  void load(const std::string& file) override
  {
    this->template load_<faiss::IndexIVFScalarQuantizer>(file);
  }

  std::unique_ptr<algo<T>> copy()
  {
    return std::make_unique<faiss_cpu_ivfsq<T>>(*this);  // use copy constructor
  }
};

template <typename T>
class faiss_cpu_flat : public faiss_cpu<T> {
 public:
  faiss_cpu_flat(Metric metric, int dim)
    : faiss_cpu<T>(metric, dim, typename faiss_cpu<T>::build_param{})
  {
    this->index_ = std::make_shared<faiss::IndexFlat>(dim, this->metric_type_);
  }

  // class faiss_cpu is more like a IVF class, so need special treating here
  void set_search_param(const typename algo<T>::search_param& param,
                        const void* filter_bitset) override
  {
    if (filter_bitset != nullptr) { throw std::runtime_error("Filtering is not supported yet."); }
    auto search_param = dynamic_cast<const typename faiss_cpu<T>::search_param&>(param);
  }

  void save(const std::string& file) const override
  {
    this->template save_<faiss::IndexFlat>(file);
  }
  void load(const std::string& file) override { this->template load_<faiss::IndexFlat>(file); }

  std::unique_ptr<algo<T>> copy()
  {
    return std::make_unique<faiss_cpu_flat<T>>(*this);  // use copy constructor
  }
};

template <typename T>
class faiss_cpu_hnsw_flat : public faiss_cpu<T> {
 public:
  struct build_param : public faiss_cpu<T>::build_param {
    int M;
    int efConstruction;
  };
  struct search_param : public faiss_cpu<T>::search_param {
    faiss::SearchParametersHNSW p;
  };
  faiss_cpu_hnsw_flat(Metric metric, int dim, const build_param& param)
    : faiss_cpu<T>(metric, dim, param)
  {
    this->index_ = std::make_shared<faiss::IndexHNSWFlat>(dim, param.M, this->metric_type_);
    faiss::IndexHNSWFlat* hnsw_index = static_cast<faiss::IndexHNSWFlat*>(this->index_.get());
    hnsw_index->hnsw.efConstruction  = param.efConstruction;
  }

  void set_search_param(const typename algo<T>::search_param& param,
                        const void* filter_bitset) override
  {
    if (filter_bitset != nullptr) { throw std::runtime_error("Filtering is not supported yet."); }
    auto sp              = static_cast<const typename faiss_cpu_hnsw_flat<T>::search_param&>(param);
    this->search_params_ = std::make_shared<faiss::SearchParametersHNSW>(sp.p);
  };

  void save(const std::string& file) const override
  {
    this->template save_<faiss::IndexHNSWFlat>(file);
  }
  void load(const std::string& file) override { this->template load_<faiss::IndexHNSWFlat>(file); }

  std::unique_ptr<algo<T>> copy()
  {
    return std::make_unique<faiss_cpu_hnsw_flat<T>>(*this);  // use copy constructor
  }

  void search(const T* queries,
              int batch_size,
              int k,
              algo_base::index_type* neighbors,
              float* distances) const override
  {
    static_assert(sizeof(size_t) == sizeof(faiss::idx_t),
                  "sizes of size_t and faiss::idx_t are different");

    this->index_->search(batch_size,
                         queries,
                         k,
                         distances,
                         reinterpret_cast<faiss::idx_t*>(neighbors),
                         search_params_.get());
  }

 private:
  std::shared_ptr<faiss::SearchParameters> search_params_;
};

}  // namespace cuvs::bench
