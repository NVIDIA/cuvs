/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cub/device/device_histogram.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/linalg/detail/cusolver_wrappers.hpp>
#include <raft/linalg/dot.cuh>
#include <raft/linalg/linalg_types.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/matrix_vector.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/multiply.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/power.cuh>
#include <raft/linalg/reduce.cuh>
#include <raft/linalg/transpose.cuh>
#include <raft/matrix/copy.cuh>
#include <raft/matrix/diagonal.cuh>
#include <raft/matrix/init.cuh>

#include "scann_common.cuh"

namespace cuvs::neighbors::experimental::scann::detail {

namespace {

template <typename LabelT, typename IdxT>
__global__ void build_clusters(
  LabelT const* node_to_cluster, LabelT* clusters, IdxT* cluster_start, int nodes, int n_clusters)
{
  size_t node = blockDim.x * blockIdx.x + threadIdx.x;

  if (node < nodes) {
    LabelT cluster        = node_to_cluster[node];
    LabelT cluster_ptr    = atomicAdd(&cluster_start[cluster], 1);
    clusters[cluster_ptr] = node;
  }
}
}  // namespace

// Compute cluster sizes/offsets
template <typename LabelT, typename IdxT>
void compute_cluster_offsets(raft::resources const& dev_resources,
                             raft::device_vector_view<const LabelT, IdxT> clusters,
                             raft::device_vector_view<LabelT, int64_t> cluster_sizes,
                             int64_t& max_cluster_size)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(dev_resources);
  rmm::device_async_resource_ref device_memory =
    raft::resource::get_workspace_resource_ref(dev_resources);

  // Histrogram to compute cluster sizes
  int num_levels  = cluster_sizes.extent(0) + 1;
  int lower_level = 0;
  int upper_level = cluster_sizes.extent(0);

  size_t temp_storage_bytes = 0;

  cub::DeviceHistogram::HistogramEven(nullptr,
                                      temp_storage_bytes,
                                      clusters.data_handle(),
                                      cluster_sizes.data_handle(),
                                      num_levels,
                                      lower_level,
                                      upper_level,
                                      clusters.extent(0),
                                      stream);

  rmm::device_uvector<char> temp_storage_hist(temp_storage_bytes, stream, device_memory);

  cub::DeviceHistogram::HistogramEven(temp_storage_hist.data(),
                                      temp_storage_bytes,
                                      clusters.data_handle(),
                                      cluster_sizes.data_handle(),
                                      num_levels,
                                      lower_level,
                                      upper_level,
                                      clusters.extent(0),
                                      stream);

  int num_items = cluster_sizes.extent(0);

  // Compute max cluster size
  // auto d_max_cluster_size = rmmo::make_device_scalar<int64_t>(dev_resources, 0);
  rmm::device_scalar<int64_t> d_max_cluster_size(stream);

  temp_storage_bytes = 0;

  cub::DeviceReduce::Max(
    nullptr, temp_storage_bytes, cluster_sizes.data_handle(), d_max_cluster_size.data(), num_items);

  rmm::device_uvector<int64_t> temp_storage_max(temp_storage_bytes, stream, device_memory);

  cub::DeviceReduce::Max(temp_storage_max.data(),
                         temp_storage_bytes,
                         cluster_sizes.data_handle(),
                         d_max_cluster_size.data(),
                         num_items);

  max_cluster_size = d_max_cluster_size.value(stream);

  // Scan to sum cluster sizes and get cluster start ptrs in flat array
  // Done in place
  temp_storage_bytes = 0;

  cub::DeviceScan::ExclusiveSum(nullptr,
                                temp_storage_bytes,
                                cluster_sizes.data_handle(),
                                cluster_sizes.data_handle(),
                                num_items);

  rmm::device_uvector<char> temp_storage_sum(temp_storage_bytes, stream, device_memory);

  cub::DeviceScan::ExclusiveSum(temp_storage_sum.data(),
                                temp_storage_bytes,
                                cluster_sizes.data_handle(),
                                cluster_sizes.data_handle(),
                                num_items);
}

// Sum elements of device vector into device scalar
template <typename T>
void sum_reduce_vector(raft::resources const& dev_resources,
                       raft::device_vector_view<T, int64_t> v,
                       raft::device_scalar_view<T> s)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(dev_resources);
  rmm::device_async_resource_ref device_memory =
    raft::resource::get_workspace_resource_ref(dev_resources);

  size_t temp_storage_bytes = 0;

  cub::DeviceReduce::Sum(
    nullptr, temp_storage_bytes, v.data_handle(), s.data_handle(), v.extent(0), stream);

  rmm::device_uvector<char> temp_storage(temp_storage_bytes, stream, device_memory);
  // cudaMalloc(&d_temp_storage, temp_storage_bytes);

  cub::DeviceReduce::Sum(
    temp_storage.data(), temp_storage_bytes, v.data_handle(), s.data_handle(), v.extent(0), stream);

  // raft::resource::sync_stream(dev_resources, stream);

  // cudaFree(d_temp_storage);
}

// Solve Ax = b for a symmetric, pos-def matrix A via cholesky factorization
template <typename T>
void cholesky_solver(raft::resources const& dev_resources,
                     raft::device_matrix_view<T, int64_t, raft::col_major> A,
                     raft::device_vector_view<T, int64_t> b,
                     raft::device_vector_view<T, int64_t> x)
{
  cudaStream_t stream          = raft::resource::get_cuda_stream(dev_resources);
  cusolverDnHandle_t cusolverH = raft::resource::get_cusolver_dn_handle(dev_resources);
  rmm::device_async_resource_ref device_memory =
    raft::resource::get_workspace_resource_ref(dev_resources);

  // RAFT_CUSOLVER_TRY(cusolverDnSetStream(cusolverH, stream));

  int n                 = A.extent(0);
  int lda               = n;
  int lwork             = 0;
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

  // compute bufferszie for potrf
  RAFT_CUSOLVER_TRY(raft::linalg::detail::cusolverDnpotrf_bufferSize(
    cusolverH, uplo, n, A.data_handle(), lda, &lwork));
  // compute cholesky factorization w/ potrf
  auto devInfo = raft::make_device_scalar(dev_resources, 0);
  rmm::device_uvector<T> d_work(lwork, stream, device_memory);

  RAFT_CUSOLVER_TRY(raft::linalg::detail::cusolverDnpotrf(
    cusolverH, uplo, n, A.data_handle(), lda, d_work.data(), lwork, devInfo.data_handle(), stream));

  // solve Ax = b
  int ldb  = b.extent(0);
  int nrhs = 1;

  RAFT_CUSOLVER_TRY(raft::linalg::detail::cusolverDnpotrs(cusolverH,
                                                          uplo,
                                                          n,
                                                          nrhs,
                                                          A.data_handle(),
                                                          lda,
                                                          b.data_handle(),
                                                          ldb,
                                                          devInfo.data_handle(),
                                                          stream));
}

// Apply Anisotropic Vector Quantization to a single cluster
// via Theorem 4.2 in https://arxiv.org/abs/1908.10396 with
//   h_i_parallel = eta * || x_i || ^ (eta - 1)
//   h_i_orthogonal = ||x _i || ^ (eta -1)
// as taken from the open source ScaNN implementation
template <typename T>
void compute_avq_centroid(raft::resources const& dev_resources,
                          raft::device_matrix_view<T, int64_t> x,
                          raft::device_vector_view<T, int64_t> avq_centroid,
                          raft::device_scalar_view<T> rescale_num,
                          raft::device_scalar_view<T> rescale_denom,
                          float eta)
{
  // Compute and scale norms
  auto norms = raft::make_device_vector<float, int64_t>(dev_resources, x.extent(0));

  raft::linalg::norm<raft::linalg::NormType::L2Norm, raft::Apply::ALONG_ROWS>(
    dev_resources, raft::make_const_mdspan(x), norms.view());

  // Compute || x_i || ^ 0.5 * (eta -3)
  auto norms_eta_3 = raft::make_device_vector<float, int64_t>(dev_resources, x.extent(0));
  auto eta_3       = raft::make_host_scalar<float>((eta - 3) / 2);

  raft::linalg::power_scalar(dev_resources,
                             raft::make_const_mdspan(norms.view()),
                             norms_eta_3.view(),
                             raft::make_const_mdspan(eta_3.view()));

  // Compute || x_i || ^ (eta - 1)
  auto norms_eta_1 = raft::make_device_vector<float, int64_t>(dev_resources, x.extent(0));
  auto eta_1       = raft::make_host_scalar<float>(eta - 1);

  raft::linalg::power_scalar(dev_resources,
                             raft::make_const_mdspan(norms.view()),
                             norms_eta_1.view(),
                             raft::make_const_mdspan(eta_1.view()));

  auto sum_norms_eta_1 = raft::make_device_scalar<float>(dev_resources, 0.0);

  sum_reduce_vector(dev_resources, norms_eta_1.view(), sum_norms_eta_1.view());

  auto x_eta_1 = raft::make_device_matrix<float, int64_t>(dev_resources, x.extent(0), x.extent(1));

  raft::linalg::matrix_vector_op<raft::Apply::ALONG_COLUMNS>(
    dev_resources,
    raft::make_const_mdspan(x),
    raft::make_const_mdspan(norms_eta_1.view()),
    x_eta_1.view(),
    raft::mul_op());

  raft::linalg::reduce<raft::Apply::ALONG_COLUMNS>(
    dev_resources,
    raft::make_const_mdspan(x_eta_1.view()),
    raft::make_device_vector_view<float, int64_t>(avq_centroid.data_handle(), x_eta_1.extent(1)),
    0.0f);

  // scale x
  // skipping zero elements in the vector should be ok, since they are norms
  // of dataset vectors
  raft::linalg::binary_mult_skip_zero<raft::Apply::ALONG_COLUMNS>(
    dev_resources, x, raft::make_const_mdspan(norms_eta_3.view()));

  // x^T x
  auto x_trans = raft::make_device_matrix<float, int64_t>(dev_resources, x.extent(1), x.extent(0));

  raft::linalg::transpose(dev_resources, x, x_trans.view());

  // Solving a system of linear equations via QR decomp only supports col_major
  // for some reason
  auto x_trans_x = raft::make_device_matrix<float, int64_t, raft::col_major>(
    dev_resources, x.extent(1), x.extent(1));

  raft::matrix::eye(dev_resources, x_trans_x.view());

  auto eta_m_1       = raft::make_device_scalar<float>(dev_resources, eta - 1);
  auto cublas_handle = raft::resource::get_cublas_handle(dev_resources);

  raft::linalg::detail::cublas_device_pointer_mode<true> pm(cublas_handle);

  RAFT_CUBLAS_TRY(cublasSetStream(cublas_handle, raft::resource::get_cuda_stream(dev_resources)));

  RAFT_CUBLAS_TRY(cublasSgemm(cublas_handle,
                              cublasOperation_t::CUBLAS_OP_T,
                              cublasOperation_t::CUBLAS_OP_T,
                              x.extent(1),
                              x.extent(1),
                              x.extent(0),
                              eta_m_1.data_handle(),
                              x_trans.view().data_handle(),
                              x.extent(0),
                              x.data_handle(),
                              x.extent(1),
                              sum_norms_eta_1.data_handle(),
                              x_trans_x.view().data_handle(),
                              x.extent(1)));

  cholesky_solver(dev_resources, x_trans_x.view(), avq_centroid, avq_centroid);

  auto h_eta = raft::make_host_scalar<float>(eta);

  raft::linalg::multiply_scalar(dev_resources,
                                raft::make_const_mdspan(avq_centroid),
                                avq_centroid,
                                raft::make_const_mdspan(h_eta.view()));

  auto dots = raft::make_device_vector<float, int64_t>(dev_resources, x.extent(1));

  raft::linalg::reduce<raft::Apply::ALONG_COLUMNS>(
    dev_resources, raft::make_const_mdspan(x), dots.view(), 0.0f);

  raft::linalg::dot(dev_resources,
                    raft::make_const_mdspan(dots.view()),
                    raft::make_const_mdspan(avq_centroid),
                    rescale_num);

  raft::linalg::dot(dev_resources,
                    raft::make_const_mdspan(avq_centroid),
                    raft::make_const_mdspan(avq_centroid),
                    rescale_denom);
}

template <typename T>
void rescale_avq_centroids(raft::resources const& dev_resources,
                           raft::device_matrix_view<T, int64_t> centroids,
                           raft::device_vector_view<T, int64_t> rescale_num_v,
                           raft::device_vector_view<T, int64_t> rescale_denom_v,
                           raft::device_vector_view<uint32_t, int64_t> cluster_sizes,
                           uint32_t dataset_size)
{
  auto rescale_num   = raft::make_device_scalar<float>(dev_resources, 0);
  auto rescale_denom = raft::make_device_scalar<float>(dev_resources, 0);

  sum_reduce_vector(dev_resources, rescale_num_v, rescale_num.view());

  raft::linalg::map_offset(
    dev_resources,
    rescale_denom_v,
    [cluster_sizes, dataset_size] __device__(size_t i, float x) {
      uint32_t cluster_size = i + 1 < cluster_sizes.extent(0)
                                ? cluster_sizes[i + 1] - cluster_sizes[i]
                                : dataset_size - cluster_sizes[i];

      return x * cluster_size;
    },
    raft::make_const_mdspan(rescale_denom_v));

  sum_reduce_vector(dev_resources, rescale_denom_v, rescale_denom.view());

  auto rescale_num_ptr   = rescale_num.data_handle();
  auto rescale_denom_ptr = rescale_denom.data_handle();

  raft::linalg::map_offset(
    dev_resources,
    centroids,
    [rescale_num_ptr, rescale_denom_ptr] __device__(size_t i, float x) {
      // should probably check the denominator is nonzero
      float rescale = (*rescale_num_ptr) / (*rescale_denom_ptr);

      return x * rescale;
    },
    raft::make_const_mdspan(centroids));
}

/**
 * A class for loading clusters into a compact matrix (sparse gather)
 * for use in AVQ.
 *
 * There are two possible scenarios:
 * 1. Dataset is stored in device memory: No host buffers are allocated,
 *    and the gather is performed on device
 * 2. Dataset is stored in host memory: Two pinned buffers are allocated
 *    in host for fast DtoH copies of cluster ids, and fast HtoD copy of the
 *    cluster matrix, while amortizing the cost of allocating pinned memory.
 *    The gather is performed on cpu, overlapping with GPU compute. Copies are
 * 	 allocated on the provided stream, allowing for overlapping with
 *		 other work on other streams.
 */
template <typename T, typename LabelT>
class cluster_loader {
 private:
  raft::pinned_matrix<T, int64_t> cluster_buf_;
  raft::pinned_vector<LabelT, int64_t> cluster_ids_buf_;
  raft::device_matrix<T, int64_t> d_cluster_buf_;
  raft::device_matrix<T, int64_t> d_cluster_copy_buf_;
  const T* dataset_ptr_;
  raft::host_vector_view<const LabelT, int64_t> h_cluster_offsets_;
  raft::device_vector_view<const LabelT, int64_t> cluster_ids_;
  cudaStream_t stream_;
  int64_t dim_;
  int64_t n_rows_;
  bool needs_copy_;

  int64_t cur_idx_  = -1;
  int64_t copy_idx_ = -1;

  size_t cluster_size(LabelT idx)
  {
    if (idx + 1 < h_cluster_offsets_.extent(0)) {
      return h_cluster_offsets_(idx + 1) - h_cluster_offsets_(idx);
    }
    return n_rows_ - h_cluster_offsets_(idx);
  }

  cluster_loader(raft::resources const& res,
                 const T* dataset_ptr,
                 int64_t dim,
                 int64_t n_rows,
                 int64_t max_cluster_size,
                 int64_t h_buf_size,
                 raft::host_vector_view<LabelT, int64_t> h_cluster_offsets,
                 raft::device_vector_view<LabelT, int64_t> cluster_ids,
                 bool needs_copy,
                 cudaStream_t stream)
    : dim_(dim),
      n_rows_(n_rows),
      dataset_ptr_(dataset_ptr),
      cluster_buf_(raft::make_pinned_matrix<T, int64_t>(res, h_buf_size, dim)),
      cluster_ids_buf_(raft::make_pinned_vector<LabelT, int64_t>(res, h_buf_size)),
      d_cluster_buf_(raft::make_device_matrix<T, int64_t>(res, max_cluster_size, dim)),
      d_cluster_copy_buf_(raft::make_device_matrix<T, int64_t>(res, max_cluster_size, dim)),
      h_cluster_offsets_(h_cluster_offsets),
      cluster_ids_(cluster_ids),
      needs_copy_(needs_copy),
      stream_(stream)
  {
  }

 public:
  cluster_loader(raft::resources const& res,
                 raft::device_matrix_view<const T, int64_t> dataset_view,
                 raft::host_vector_view<LabelT, int64_t> h_cluster_offsets,
                 raft::device_vector_view<LabelT, int64_t> cluster_ids,
                 int64_t max_cluster_size,
                 cudaStream_t stream)
    : cluster_loader(res,
                     dataset_view.data_handle(),
                     dataset_view.extent(1),
                     dataset_view.extent(0),
                     max_cluster_size,
                     0,
                     h_cluster_offsets,
                     cluster_ids,
                     false,
                     stream)

  {
  }

  cluster_loader(raft::resources const& res,
                 raft::host_matrix_view<const T, int64_t> dataset_view,
                 raft::host_vector_view<LabelT, int64_t> h_cluster_offsets,
                 raft::device_vector_view<LabelT, int64_t> cluster_ids,
                 int64_t max_cluster_size,
                 cudaStream_t stream)
    : cluster_loader(res,
                     dataset_view.data_handle(),
                     dataset_view.extent(1),
                     dataset_view.extent(0),
                     max_cluster_size,
                     max_cluster_size,
                     h_cluster_offsets,
                     cluster_ids,
                     true,
                     stream)

  {
  }

  /**
   * @brief load and return a view of the provided cluster
   *
   * @param res: the raft resources;
   * @param cluster_idx: the index of the cluster to be loaded
   * @return device_matrix_view of the cluster vectors
   */
  raft::device_matrix_view<T, int64_t> load_cluster(raft::resources const& res, LabelT cluster_idx)
  {
    size_t size = cluster_size(cluster_idx);

    // Check if cluster is already loaded
    if (cur_idx_ != cluster_idx) {
      // If not, load the cluster
      if (copy_idx_ != cluster_idx) { prefetch_cluster(res, cluster_idx); }

      // swap buffers
      std::swap(d_cluster_buf_, d_cluster_copy_buf_);
      std::swap(cur_idx_, copy_idx_);
    }

    return raft::make_device_matrix_view<T, int64_t>(d_cluster_buf_.data_handle(), size, dim_);
  }

  /** @brief Perform gather operation on stream_
   *
   * @param res: the raft resources
   * @param cluster_idx: the index of the cluster
   */
  void prefetch_cluster(raft::resources const& res, LabelT cluster_idx)
  {
    if (cluster_idx >= h_cluster_offsets_.extent(0)) { return; }

    size_t size = cluster_size(cluster_idx);

    auto cluster_ids = raft::make_device_vector_view<const LabelT, int64_t>(
      cluster_ids_.data_handle() + h_cluster_offsets_(cluster_idx), size);

    auto cluster_vectors =
      raft::make_device_matrix_view<float, int64_t>(d_cluster_copy_buf_.data_handle(), size, dim_);

    if (needs_copy_) {
      // For prefetching to overlap with other gpu work
      // we need to schedule copies on the provided copy stream stream_
      auto copy_res = raft::resources(res);
      raft::resource::set_cuda_stream(copy_res, stream_);

      // htod
      auto h_cluster_ids =
        raft::make_pinned_vector_view<LabelT, int64_t>(cluster_ids_buf_.data_handle(), size);

      raft::copy(copy_res, h_cluster_ids, cluster_ids);
      raft::resource::sync_stream(res, stream_);

      auto pinned_cluster = raft::make_pinned_matrix_view<T, int64_t>(
        cluster_buf_.data_handle(), cluster_vectors.extent(0), cluster_vectors.extent(1));

      int n_threads = std::min<int>(cuvs::core::omp::get_max_threads(), 32);
#pragma omp parallel for num_threads(n_threads)
      for (int i = 0; i < h_cluster_ids.extent(0); i++) {
        memcpy(pinned_cluster.data_handle() + i * pinned_cluster.extent(1),
               dataset_ptr_ + h_cluster_ids(i) * dim_,
               sizeof(T) * dim_);
      }

      raft::copy(copy_res, cluster_vectors, raft::make_const_mdspan(pinned_cluster));
      raft::resource::sync_stream(res, stream_);
    } else {
      // dtod
      auto dataset_view =
        raft::make_device_matrix_view<const T, int64_t>(dataset_ptr_, n_rows_, dim_);

      raft::matrix::gather(res, dataset_view, cluster_ids, cluster_vectors);
    }

    copy_idx_ = cluster_idx;
  }
};

/**
 * @brief Perform AVQ adjustment on cluster centers
 *
 * Apply Anisotropic Vector Quantization to recompute cluster centers
 * via Theorem 4.2 in https://arxiv.org/abs/1908.10396 with
 *  h_i_parallel = eta * || x_i || ^ (eta - 1)
 * h_i_orthogonal = ||x _i || ^ (eta -1)
 * as taken from the open source ScaNN implementation
 *
 * @tparam T
 * @tparam IdxT
 * @tparam LabelT
 * @tparam Accessor
 * @param res raft resources
 * @param dataset the dataset (host or device), size [n_rows, dim]
 * @param centroids_view cluster centers, size [n_clusters, dim]
 * @param labels_view nearest cluster idx for each dataset vector, size [n_rows]
 * @param eta the weight for the parallel component of the residual in the avq update
 */
template <typename T,
          typename IdxT   = int64_t,
          typename LabelT = uint32_t,
          typename Accessor =
            raft::host_device_accessor<cuda::std::default_accessor<T>, raft::memory_type::host>>
void apply_avq(raft::resources const& res,
               raft::mdspan<const T, raft::matrix_extent<IdxT>, raft::row_major, Accessor> dataset,
               raft::device_matrix_view<T, IdxT> centroids_view,
               raft::device_vector_view<const LabelT, IdxT> labels_view,
               float eta,
               cudaStream_t copy_stream)
{
  // Compute clusters

  cudaStream_t stream  = raft::resource::get_cuda_stream(res);
  auto cluster_offsets = raft::make_device_vector<uint32_t, int64_t>(res, centroids_view.extent(0));
  auto clusters        = raft::make_device_vector<uint32_t, int64_t>(res, dataset.extent(0));
  int64_t max_cluster_size = 0;

  compute_cluster_offsets(res, labels_view, cluster_offsets.view(), max_cluster_size);
  auto h_cluster_offsets = raft::make_host_vector<uint32_t, int64_t>(cluster_offsets.extent(0));

  raft::copy(res, h_cluster_offsets.view(), raft::make_const_mdspan(cluster_offsets.view()));

  dim3 block(32, 1, 1);
  dim3 grid((dataset.extent(0) + block.x - 1) / block.x, 1, 1);

  build_clusters<uint32_t, uint32_t><<<grid, block>>>(labels_view.data_handle(),
                                                      clusters.view().data_handle(),
                                                      cluster_offsets.view().data_handle(),
                                                      dataset.extent(0),
                                                      labels_view.extent(0));
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  auto rescale_num   = raft::make_device_vector<float, int64_t>(res, centroids_view.extent(0));
  auto rescale_denom = raft::make_device_vector<float, int64_t>(res, centroids_view.extent(0));

  cluster_loader<T, LabelT> loader(
    res, dataset, h_cluster_offsets.view(), clusters.view(), max_cluster_size, copy_stream);
  raft::resource::sync_stream(res);

  RAFT_LOG_DEBUG("Compute AVQ centroids\n");

  for (int i = 0; i < h_cluster_offsets.extent(0); i++) {
    auto cluster_vectors = loader.load_cluster(res, i);

    auto avq_centroid = raft::make_device_vector_view<float, int64_t>(
      centroids_view.data_handle() + i * dataset.extent(1), dataset.extent(1));
    auto rescale_num_view   = raft::make_device_scalar_view<float>(rescale_num.data_handle() + i);
    auto rescale_denom_view = raft::make_device_scalar_view<float>(rescale_denom.data_handle() + i);

    compute_avq_centroid(
      res, cluster_vectors, avq_centroid, rescale_num_view, rescale_denom_view, eta);

    loader.prefetch_cluster(res, i + 1);

    // make sure work is done before swapping buffers in cluster_loader
    raft::resource::sync_stream(res);
  }

  rescale_avq_centroids(res,
                        centroids_view,
                        rescale_num.view(),
                        rescale_denom.view(),
                        cluster_offsets.view(),
                        dataset.extent(0));

  raft::resource::sync_stream(res);
}
}  // namespace cuvs::neighbors::experimental::scann::detail
