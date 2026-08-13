/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cublas_v2.h>
#include <cuvs/core/export.hpp>
#include <cuvs/distance/distance.hpp>
#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

namespace CUVS_EXPORT cuvs {
namespace distance {
namespace kernels {

template <typename math_t>
using dense_input_matrix_view_t = raft::device_matrix_view<const math_t, int, raft::layout_stride>;
template <typename math_t>
using dense_output_matrix_view_t = raft::device_matrix_view<math_t, int, raft::layout_stride>;
template <typename math_t>
using csr_input_matrix_view_t = raft::device_csr_matrix_view<const math_t, int, int, int>;

/**
 * Base class for general Gram matrices
 * A Gram matrix is the Hermitian matrix of inner probucts G_ik = <x_i, x_k>
 * Here, the  inner product is evaluated for all elements from vectors sets X1,
 * and X2.
 *
 * To be more precise, on exit the output buffer will store:
 * - if is_row_major == true: out[j+k*n1] = <x1_j, x2_k>,
 * - if is_row_major == false: out[j*n2 + k] = <x1_j, x2_k>,
 * where x1_j is the j-th vector from the x1 set and x2_k is the k-th vector
 * from the x2 set.
 */
template <typename math_t>
class GramMatrixBase {
 protected:
  cublasHandle_t cublas_handle;
  bool legacy_interface;

 public:
  GramMatrixBase() : legacy_interface(false) {};
  [[deprecated]] GramMatrixBase(cublasHandle_t cublas_handle)
    : cublas_handle(cublas_handle), legacy_interface(true) {};

  virtual ~GramMatrixBase() {};

  /** Convenience function to evaluate the Gram matrix for two vector sets.
   *  Vector sets are provided in Matrix format
   *
   * @param [in] handle raft handle
   * @param [in] x1 dense device matrix view, size [n1*n_cols]
   * @param [in] x2 dense device matrix view, size [n2*n_cols]
   * @param [out] out dense device matrix view for the Gram matrix, size [n1*n2]
   * @param norm_x1 optional L2-norm of x1's rows for computation within RBF.
   * @param norm_x2 optional L2-norm of x2's rows for computation within RBF.
   */
  void operator()(raft::resources const& handle,
                  dense_input_matrix_view_t<math_t> x1,
                  dense_input_matrix_view_t<math_t> x2,
                  dense_output_matrix_view_t<math_t> out,
                  math_t* norm_x1 = nullptr,
                  math_t* norm_x2 = nullptr);

  /** Convenience function to evaluate the Gram matrix for two vector sets.
   *  Vector sets are provided in Matrix format
   *
   * @param [in] handle raft handle
   * @param [in] x1 csr device matrix view, size [n1*n_cols]
   * @param [in] x2 dense device matrix view, size [n2*n_cols]
   * @param [out] out dense device matrix view for the Gram matrix, size [n1*n2]
   * @param norm_x1 optional L2-norm of x1's rows for computation within RBF.
   * @param norm_x2 optional L2-norm of x2's rows for computation within RBF.
   */
  void operator()(raft::resources const& handle,
                  csr_input_matrix_view_t<math_t> x1,
                  dense_input_matrix_view_t<math_t> x2,
                  dense_output_matrix_view_t<math_t> out,
                  math_t* norm_x1 = nullptr,
                  math_t* norm_x2 = nullptr);

  /** Convenience function to evaluate the Gram matrix for two vector sets.
   *  Vector sets are provided in Matrix format
   *
   * @param [in] handle raft handle
   * @param [in] x1 csr device matrix view, size [n1*n_cols]
   * @param [in] x2 csr device matrix view, size [n2*n_cols]
   * @param [out] out dense device matrix view for the Gram matrix, size [n1*n2]
   * @param norm_x1 optional L2-norm of x1's rows for computation within RBF.
   * @param norm_x2 optional L2-norm of x2's rows for computation within RBF.
   */
  void operator()(raft::resources const& handle,
                  csr_input_matrix_view_t<math_t> x1,
                  csr_input_matrix_view_t<math_t> x2,
                  dense_output_matrix_view_t<math_t> out,
                  math_t* norm_x1 = nullptr,
                  math_t* norm_x2 = nullptr);

  // unfortunately, 'evaluate' cannot be templatized as it needs to be virtual

  /** Evaluate the Gram matrix for two vector sets using simple dot product.
   *
   * @param [in] handle raft handle
   * @param [in] x1 dense device matrix view, size [n1*n_cols]
   * @param [in] x2 dense device matrix view, size [n2*n_cols]
   * @param [out] out dense device matrix view for the Gram matrix, size [n1*n2]
   * @param norm_x1 unused.
   * @param norm_x2 unused.
   */
  virtual void evaluate(raft::resources const& handle,
                        dense_input_matrix_view_t<math_t> x1,
                        dense_input_matrix_view_t<math_t> x2,
                        dense_output_matrix_view_t<math_t> out,
                        math_t* norm_x1,
                        math_t* norm_x2);

  /** Evaluate the Gram matrix for two vector sets using simple dot product.
   *
   * @param [in] handle raft handle
   * @param [in] x1 csr device matrix view, size [n1*n_cols]
   * @param [in] x2 dense device matrix view, size [n2*n_cols]
   * @param [out] out dense device matrix view for the Gram matrix, size [n1*n2]
   * @param norm_x1 unused.
   * @param norm_x2 unused.
   */
  virtual void evaluate(raft::resources const& handle,
                        csr_input_matrix_view_t<math_t> x1,
                        dense_input_matrix_view_t<math_t> x2,
                        dense_output_matrix_view_t<math_t> out,
                        math_t* norm_x1,
                        math_t* norm_x2);

  /** Evaluate the Gram matrix for two vector sets using simple dot product.
   *
   * @param [in] handle raft handle
   * @param [in] x1 csr device matrix view, size [n1*n_cols]
   * @param [in] x2 csr device matrix view, size [n2*n_cols]
   * @param [out] out dense device matrix view for the Gram matrix, size [n1*n2]
   * @param norm_x1 unused.
   * @param norm_x2 unused.
   */
  virtual void evaluate(raft::resources const& handle,
                        csr_input_matrix_view_t<math_t> x1,
                        csr_input_matrix_view_t<math_t> x2,
                        dense_output_matrix_view_t<math_t> out,
                        math_t* norm_x1,
                        math_t* norm_x2);

  /** Evaluate the Gram matrix for two vector sets using simple dot product.
   *
   * @param [in] x1 device array of vectors, size [n1*n_cols]
   * @param [in] n1 number vectors in x1
   * @param [in] n_cols number of columns (features) in x1 and x2
   * @param [in] x2 device array of vectors, size [n2*n_cols]
   * @param [in] n2 number vectors in x2
   * @param [out] out device buffer to store the Gram matrix, size [n1*n2]
   * @param [in] is_row_major whether the input and output matrices are in row
   *        major format
   * @param [in] stream cuda stream
   * @param ld1 leading dimension of x1 (usually it is n1)
   * @param ld2 leading dimension of x2 (usually it is n2)
   * @param ld_out leading dimension of out (usually it is n1)
   */
  [[deprecated]] virtual void evaluate(const math_t* x1,
                                       int n1,
                                       int n_cols,
                                       const math_t* x2,
                                       int n2,
                                       math_t* out,
                                       bool is_row_major,
                                       cudaStream_t stream,
                                       int ld1,
                                       int ld2,
                                       int ld_out);

  /** Convenience function to evaluate the Gram matrix for two vector sets.
   *
   * @param [in] x1 device array of vectors, size [n1*n_cols]
   * @param [in] n1 number vectors in x1
   * @param [in] n_cols number of columns (features) in x1 and x2
   * @param [in] x2 device array of vectors, size [n2*n_cols]
   * @param [in] n2 number vectors in x2
   * @param [out] out device buffer to store the Gram matrix, size [n1*n2]
   * @param [in] is_row_major whether the input and output matrices are in row
   *        major format
   * @param [in] stream cuda stream
   * @param ld1 leading dimension of x1
   * @param ld2 leading dimension of x2
   * @param ld_out leading dimension of out
   */
  [[deprecated]] void operator()(const math_t* x1,
                                 int n1,
                                 int n_cols,
                                 const math_t* x2,
                                 int n2,
                                 math_t* out,
                                 bool is_row_major,
                                 cudaStream_t stream,
                                 int ld1    = 0,
                                 int ld2    = 0,
                                 int ld_out = 0);

 protected:
  /** Calculates the Gram matrix using simple dot product between vector sets.
   *
   * out = x1 * x2
   *
   * Can be used as a building block for more complex kernel functions.
   *
   * @param [in] x1 device array of vectors, size [n1*n_cols]
   * @param [in] n1 number vectors in x1
   * @param [in] n_cols number of columns (features) in x1 and x2
   * @param [in] x2 device array of vectors, size [n2*n_cols]
   * @param [in] n2 number vectors in x2
   * @param [out] out device buffer to store the Gram matrix, size [n1*n2]
   * @param [in] is_row_major whether the input and output matrices are in row
   *        major format
   * @param [in] stream cuda stream
   * @param ld1 leading dimension of x1
   * @param ld2 leading dimension of x2
   * @param ld_out leading dimension of out
   */
  [[deprecated]] void linear(const math_t* x1,
                             int n1,
                             int n_cols,
                             const math_t* x2,
                             int n2,
                             math_t* out,
                             bool is_row_major,
                             cudaStream_t stream,
                             int ld1,
                             int ld2,
                             int ld_out);

 protected:
  bool get_is_row_major(dense_output_matrix_view_t<math_t> matrix);
  bool get_is_row_major(dense_input_matrix_view_t<math_t> matrix);
  bool get_is_col_major(dense_output_matrix_view_t<math_t> matrix);
  bool get_is_col_major(dense_input_matrix_view_t<math_t> matrix);

  /** Calculates the Gram matrix using simple dot product between vector sets.
   *
   * out = x1 * x2
   *
   * Can be used as a building block for more complex kernel functions.
   *
   * @param [in] handle raft handle
   * @param [in] x1 dense device matrix view, size [n1*n_cols]
   * @param [in] x2 dense device matrix view, size [n2*n_cols]
   * @param [out] out dense device matrix view for the Gram matrix, size [n1*n2]
   */
  void linear(raft::resources const& handle,
              dense_input_matrix_view_t<math_t> x1,
              dense_input_matrix_view_t<math_t> x2,
              dense_output_matrix_view_t<math_t> out);

  /** Calculates the Gram matrix using simple dot product between vector sets.
   *
   * out = x1 * x2
   *
   * Can be used as a building block for more complex kernel functions.
   *
   * @param [in] handle raft handle
   * @param [in] x1 csr device matrix view, size [n1*n_cols]
   * @param [in] x2 dense device matrix view, size [n2*n_cols]
   * @param [out] out dense device matrix view for the Gram matrix, size [n1*n2]
   */
  void linear(raft::resources const& handle,
              csr_input_matrix_view_t<math_t> x1,
              dense_input_matrix_view_t<math_t> x2,
              dense_output_matrix_view_t<math_t> out);

  /** Calculates the Gram matrix using simple dot product between vector sets.
   *
   * out = x1 * x2
   *
   * Can be used as a building block for more complex kernel functions.
   *
   * @param [in] handle raft handle
   * @param [in] x1 csr device matrix view, size [n1*n_cols]
   * @param [in] x2 csr device matrix view, size [n2*n_cols]
   * @param [out] out dense device matrix view for the Gram matrix, size [n1*n2]
   */
  void linear(raft::resources const& handle,
              csr_input_matrix_view_t<math_t> x1,
              csr_input_matrix_view_t<math_t> x2,
              dense_output_matrix_view_t<math_t> out);
};

template <typename math_t>
class KernelFactory {
 public:
  static GramMatrixBase<math_t>* create(KernelParams params);
  [[deprecated]] static GramMatrixBase<math_t>* create(KernelParams params, cublasHandle_t handle);
};

/**
 * Create a kernel matrix using polynomial kernel function.
 */
template <typename math_t, typename exp_t>
class PolynomialKernel : public GramMatrixBase<math_t> {
  exp_t exponent;
  math_t gain;
  math_t offset;

  void applyKernel(
    math_t* inout, int ld, int rows, int cols, bool is_row_major, cudaStream_t stream);

 public:
  /**
   * Constructs a polynomial kernel object.
   * It evaluates the kernel matrix using the following formula:
   * K_ij = (gain*<x1_i, x2_k> + offset)^exponent
   *
   * @tparam math_t floating point type
   * @tparam exp_t type of exponent
   * @param exponent
   * @param gain
   * @param offset
   */
  PolynomialKernel(exp_t exponent, math_t gain, math_t offset)
    : GramMatrixBase<math_t>(), exponent(exponent), gain(gain), offset(offset) {};

  [[deprecated]] PolynomialKernel(exp_t exponent, math_t gain, math_t offset, cublasHandle_t handle)
    : GramMatrixBase<math_t>(handle), exponent(exponent), gain(gain), offset(offset) {};

  /** Evaluate kernel matrix using polynomial kernel.
   *
   * output[i,k] = (gain*<x1_i, x2_k> + offset)^exponent,
   * where x1_i is the i-th vector from the x1 set, and x2_k is k-th vector
   * in the x2 set, and < , > denotes dot product.
   *
   * @param [in] handle raft handle
   * @param [in] x1 dense device matrix view, size [n1*n_cols]
   * @param [in] x2 dense device matrix view, size [n2*n_cols]
   * @param [out] out dense device matrix view for the Gram matrix, size [n1*n2]
   * @param norm_x1 unused.
   * @param norm_x2 unused.
   */
  void evaluate(raft::resources const& handle,
                dense_input_matrix_view_t<math_t> x1,
                dense_input_matrix_view_t<math_t> x2,
                dense_output_matrix_view_t<math_t> out,
                math_t* norm_x1,
                math_t* norm_x2);

  /** Evaluate kernel matrix using polynomial kernel.
   *
   * output[i,k] = (gain*<x1_i, x2_k> + offset)^exponent,
   * where x1_i is the i-th vector from the x1 set, and x2_k is k-th vector
   * in the x2 set, and < , > denotes dot product.
   *
   * @param [in] handle raft handle
   * @param [in] x1 csr device matrix view, size [n1*n_cols]
   * @param [in] x2 dense device matrix view, size [n2*n_cols]
   * @param [out] out dense device matrix view for the Gram matrix, size [n1*n2]
   * @param norm_x1 unused.
   * @param norm_x2 unused.
   */
  void evaluate(raft::resources const& handle,
                csr_input_matrix_view_t<math_t> x1,
                dense_input_matrix_view_t<math_t> x2,
                dense_output_matrix_view_t<math_t> out,
                math_t* norm_x1,
                math_t* norm_x2);

  /** Evaluate kernel matrix using polynomial kernel.
   *
   * output[i,k] = (gain*<x1_i, x2_k> + offset)^exponent,
   * where x1_i is the i-th vector from the x1 set, and x2_k is k-th vector
   * in the x2 set, and < , > denotes dot product.
   *
   * @param [in] handle raft handle
   * @param [in] x1 csr device matrix view, size [n1*n_cols]
   * @param [in] x2 csr device matrix view, size [n2*n_cols]
   * @param [out] out dense device matrix view for the Gram matrix, size [n1*n2]
   * @param norm_x1 unused.
   * @param norm_x2 unused.
   */
  void evaluate(raft::resources const& handle,
                csr_input_matrix_view_t<math_t> x1,
                csr_input_matrix_view_t<math_t> x2,
                dense_output_matrix_view_t<math_t> out,
                math_t* norm_x1,
                math_t* norm_x2);

  /** Evaluate the Gram matrix using the legacy interface.
   *
   * @param [in] x1 device array of vectors, size [n1*n_cols]
   * @param [in] n1 number vectors in x1
   * @param [in] n_cols number of columns (features) in x1 and x2
   * @param [in] x2 device array of vectors, size [n2*n_cols]
   * @param [in] n2 number vectors in x2
   * @param [out] out device buffer to store the Gram matrix, size [n1*n2]
   * @param [in] is_row_major whether the input and output matrices are in row
   *        major format
   * @param [in] stream cuda stream
   * @param ld1 leading dimension of x1 (usually it is n1)
   * @param ld2 leading dimension of x2 (usually it is n2)
   * @param ld_out leading dimension of out (usually it is n1)
   */
  [[deprecated]] void evaluate(const math_t* x1,
                               int n1,
                               int n_cols,
                               const math_t* x2,
                               int n2,
                               math_t* out,
                               bool is_row_major,
                               cudaStream_t stream,
                               int ld1,
                               int ld2,
                               int ld_out);
};

/**
 * Create a kernel matrix using tanh kernel function.
 */
template <typename math_t>
class TanhKernel : public GramMatrixBase<math_t> {
  math_t gain, offset;

  void applyKernel(
    math_t* inout, int ld, int rows, int cols, bool is_row_major, cudaStream_t stream);

 public:
  /**
   * Constructs a tanh kernel object.
   * It evaluates the kernel matrix using the following formula:
   * K_ij = tanh(gain*<x1_i, x2_k> + offset)
   *
   * @tparam math_t floating point type
   * @param gain
   * @param offset
   */
  TanhKernel(math_t gain, math_t offset) : GramMatrixBase<math_t>(), gain(gain), offset(offset) {}

  [[deprecated]] TanhKernel(math_t gain, math_t offset, cublasHandle_t handle)
    : GramMatrixBase<math_t>(handle), gain(gain), offset(offset) {};

  /** Evaluate kernel matrix using tanh kernel.
   *
   * output_[i + k*n1] = (gain*<x1_i, x2_k> + offset)^exponent,
   * where x1_i is the i-th vector from the x1 set, and x2_k is k-th vector
   * in the x2 set, and < , > denotes dot product.
   *
   * @param [in] handle raft handle
   * @param [in] x1 dense device matrix view, size [n1*n_cols]
   * @param [in] x2 dense device matrix view, size [n2*n_cols]
   * @param [out] out dense device matrix view for the Gram matrix, size [n1*n2]
   * @param norm_x1 unused.
   * @param norm_x2 unused.
   */
  void evaluate(raft::resources const& handle,
                dense_input_matrix_view_t<math_t> x1,
                dense_input_matrix_view_t<math_t> x2,
                dense_output_matrix_view_t<math_t> out,
                math_t* norm_x1,
                math_t* norm_x2);

  /** Evaluate kernel matrix using tanh kernel.
   *
   * output_[i + k*n1] = (gain*<x1_i, x2_k> + offset)^exponent,
   * where x1_i is the i-th vector from the x1 set, and x2_k is k-th vector
   * in the x2 set, and < , > denotes dot product.
   *
   * @param [in] handle raft handle
   * @param [in] x1 csr device matrix view, size [n1*n_cols]
   * @param [in] x2 dense device matrix view, size [n2*n_cols]
   * @param [out] out dense device matrix view for the Gram matrix, size [n1*n2]
   * @param norm_x1 unused.
   * @param norm_x2 unused.
   */
  void evaluate(raft::resources const& handle,
                csr_input_matrix_view_t<math_t> x1,
                dense_input_matrix_view_t<math_t> x2,
                dense_output_matrix_view_t<math_t> out,
                math_t* norm_x1,
                math_t* norm_x2);

  /** Evaluate kernel matrix using tanh kernel.
   *
   * output_[i + k*n1] = (gain*<x1_i, x2_k> + offset)^exponent,
   * where x1_i is the i-th vector from the x1 set, and x2_k is k-th vector
   * in the x2 set, and < , > denotes dot product.
   *
   * @param [in] handle raft handle
   * @param [in] x1 csr device matrix view, size [n1*n_cols]
   * @param [in] x2 csr device matrix view, size [n2*n_cols]
   * @param [out] out dense device matrix view for the Gram matrix, size [n1*n2]
   * @param norm_x1 unused.
   * @param norm_x2 unused.
   */
  void evaluate(raft::resources const& handle,
                csr_input_matrix_view_t<math_t> x1,
                csr_input_matrix_view_t<math_t> x2,
                dense_output_matrix_view_t<math_t> out,
                math_t* norm_x1,
                math_t* norm_x2);

  /** Evaluate the Gram matrix using the legacy interface.
   *
   * @param [in] x1 device array of vectors, size [n1*n_cols]
   * @param [in] n1 number vectors in x1
   * @param [in] n_cols number of columns (features) in x1 and x2
   * @param [in] x2 device array of vectors, size [n2*n_cols]
   * @param [in] n2 number vectors in x2
   * @param [out] out device buffer to store the Gram matrix, size [n1*n2]
   * @param [in] is_row_major whether the input and output matrices are in row
   *        major format
   * @param [in] stream cuda stream
   * @param ld1 leading dimension of x1 (usually it is n1)
   * @param ld2 leading dimension of x2 (usually it is n2)
   * @param ld_out leading dimension of out (usually it is n1)
   */
  [[deprecated]] void evaluate(const math_t* x1,
                               int n1,
                               int n_cols,
                               const math_t* x2,
                               int n2,
                               math_t* out,
                               bool is_row_major,
                               cudaStream_t stream,
                               int ld1,
                               int ld2,
                               int ld_out);
};

/**
 * Create a kernel matrix using RBF kernel function.
 */
template <typename math_t>
class RBFKernel : public GramMatrixBase<math_t> {
  math_t gain;

  void applyKernel(math_t* inout,
                   int ld,
                   int rows,
                   int cols,
                   math_t* norm_x1,
                   math_t* norm_x2,
                   bool is_row_major,
                   cudaStream_t stream);

 public:
  /**
   * Constructs a RBF kernel object.
   * It evaluates the kernel matrix using the following formula:
   * K_ij = exp(-gain*|x1_i- x2_k|^2)
   *
   * @tparam math_t floating point type
   * @param gain
   */
  RBFKernel(math_t gain) : GramMatrixBase<math_t>(), gain(gain) {};

  [[deprecated]] RBFKernel(math_t gain, cublasHandle_t handle)
    : GramMatrixBase<math_t>(handle), gain(gain) {};

  void matrixRowNormL2(raft::resources const& handle,
                       dense_input_matrix_view_t<math_t> matrix,
                       math_t* target);

  void matrixRowNormL2(raft::resources const& handle,
                       csr_input_matrix_view_t<math_t> matrix,
                       math_t* target);

  /** Evaluate kernel matrix using RBF kernel.
   *
   * output_[i + k*n1] = exp(-gain*|x1_i - x2_k|^2),
   * where x1_i is the i-th vector from the x1 set, and x2_k is k-th vector
   * in the x2 set, and | | euclidean distance.
   *
   * @param [in] handle raft handle
   * @param [in] x1 dense device matrix view, size [n1*n_cols]
   * @param [in] x2 dense device matrix view, size [n2*n_cols]
   * @param [out] out dense device matrix view for the Gram matrix, size [n1*n2]
   * @param norm_x1 optional L2-norm of x1's rows for computation within RBF.
   * @param norm_x2 optional L2-norm of x2's rows for computation within RBF.
   */
  void evaluate(raft::resources const& handle,
                dense_input_matrix_view_t<math_t> x1,
                dense_input_matrix_view_t<math_t> x2,
                dense_output_matrix_view_t<math_t> out,
                math_t* norm_x1,
                math_t* norm_x2);

  /** Evaluate kernel matrix using RBF kernel.
   *
   * output_[i + k*n1] = exp(-gain*|x1_i - x2_k|^2),
   * where x1_i is the i-th vector from the x1 set, and x2_k is k-th vector
   * in the x2 set, and | | euclidean distance.
   *
   * @param [in] handle raft handle
   * @param [in] x1 csr device matrix view, size [n1*n_cols]
   * @param [in] x2 dense device matrix view, size [n2*n_cols]
   * @param [out] out dense device matrix view for the Gram matrix, size [n1*n2]
   * @param norm_x1 optional L2-norm of x1's rows for computation within RBF.
   * @param norm_x2 optional L2-norm of x2's rows for computation within RBF.
   */
  void evaluate(raft::resources const& handle,
                csr_input_matrix_view_t<math_t> x1,
                dense_input_matrix_view_t<math_t> x2,
                dense_output_matrix_view_t<math_t> out,
                math_t* norm_x1,
                math_t* norm_x2);

  /** Evaluate kernel matrix using RBF kernel.
   *
   * output_[i + k*n1] = exp(-gain*|x1_i - x2_k|^2),
   * where x1_i is the i-th vector from the x1 set, and x2_k is k-th vector
   * in the x2 set, and | | euclidean distance.
   *
   * @param [in] handle raft handle
   * @param [in] x1 csr device matrix view, size [n1*n_cols]
   * @param [in] x2 csr device matrix view, size [n2*n_cols]
   * @param [out] out dense device matrix view for the Gram matrix, size [n1*n2]
   * @param norm_x1 optional L2-norm of x1's rows for computation within RBF.
   * @param norm_x2 optional L2-norm of x2's rows for computation within RBF.
   */
  void evaluate(raft::resources const& handle,
                csr_input_matrix_view_t<math_t> x1,
                csr_input_matrix_view_t<math_t> x2,
                dense_output_matrix_view_t<math_t> out,
                math_t* norm_x1,
                math_t* norm_x2);

  /** Evaluate the Gram matrix using the legacy interface.
   *
   * @param [in] x1 device array of vectors, size [n1*n_cols]
   * @param [in] n1 number vectors in x1
   * @param [in] n_cols number of columns (features) in x1 and x2
   * @param [in] x2 device array of vectors, size [n2*n_cols]
   * @param [in] n2 number vectors in x2
   * @param [out] out device buffer to store the Gram matrix, size [n1*n2]
   * @param [in] is_row_major whether the input and output matrices are in row
   *        major format
   * @param [in] stream cuda stream
   * @param ld1 leading dimension of x1 (usually it is n1)
   * @param ld2 leading dimension of x2 (usually it is n2)
   * @param ld_out leading dimension of out (usually it is n1)
   */
  [[deprecated]] void evaluate(const math_t* x1,
                               int n1,
                               int n_cols,
                               const math_t* x2,
                               int n2,
                               math_t* out,
                               bool is_row_major,
                               cudaStream_t stream,
                               int ld1,
                               int ld2,
                               int ld_out);
};
}  // namespace kernels
}  // namespace distance
}  // namespace CUVS_EXPORT cuvs
