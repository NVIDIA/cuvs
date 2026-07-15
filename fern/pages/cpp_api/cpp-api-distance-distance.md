---
slug: api-reference/cpp-api-distance-distance
---

# Distance

_Source header: `cuvs/distance/distance.hpp`_

## Types

<a id="distance-distancetype"></a>
### distance::DistanceType

enum to tell how to compute distance

```cpp
enum class DistanceType : int {
  L2Expanded = 0,
  CosineExpanded = 2,
  L1 = 3,
  L2Unexpanded = 4,
  InnerProduct = 6,
  Linf = 7,
  Canberra = 8,
  LpUnexpanded = 9,
  CorrelationExpanded = 10,
  JaccardExpanded = 11,
  HellingerExpanded = 12,
  Haversine = 13,
  BrayCurtis = 14,
  JensenShannon = 15,
  HammingUnexpanded = 16,
  KLDivergence = 17,
  RusselRaoExpanded = 18,
  DiceExpanded = 19,
  BitwiseHamming = 20,
  Precomputed = 100,
  CustomUDF = 101
};
```

**Values**

| Name | Value |
| --- | --- |
| `L2Expanded` | `0` |
| `CosineExpanded` | `2` |
| `L1` | `3` |
| `L2Unexpanded` | `4` |
| `InnerProduct` | `6` |
| `Linf` | `7` |
| `Canberra` | `8` |
| `LpUnexpanded` | `9` |
| `CorrelationExpanded` | `10` |
| `JaccardExpanded` | `11` |
| `HellingerExpanded` | `12` |
| `Haversine` | `13` |
| `BrayCurtis` | `14` |
| `JensenShannon` | `15` |
| `HammingUnexpanded` | `16` |
| `KLDivergence` | `17` |
| `RusselRaoExpanded` | `18` |
| `DiceExpanded` | `19` |
| `BitwiseHamming` | `20` |
| `Precomputed` | `100` |
| `CustomUDF` | `101` |

<a id="distance-densitykerneltype"></a>
### distance::DensityKernelType

Density kernel type for Kernel Density Estimation.

These are the smoothing kernels used in KDE — distinct from the dot-product kernels (RBF, Polynomial, etc.) in cuvs::distance::kernels used by SVMs.

```cpp
enum class DensityKernelType : int {
  Gaussian = 0,
  Tophat = 1,
  Epanechnikov = 2,
  Exponential = 3,
  Linear = 4,
  Cosine = 5
};
```

**Values**

| Name | Value |
| --- | --- |
| `Gaussian` | `0` |
| `Tophat` | `1` |
| `Epanechnikov` | `2` |
| `Exponential` | `3` |
| `Linear` | `4` |
| `Cosine` | `5` |

<a id="distance-kernels-kernelparams"></a>
### distance::kernels::KernelParams

Parameters for kernel matrices.

The following kernels are implemented:

- LINEAR $K(x_1,x_2) = \langle x_1,x_2 \rangle,$ where $\langle , \rangle$ is the dot product
- POLYNOMIAL $K(x_1, x_2) = (\gamma \langle x_1,x_2 \rangle + \mathrm{coef0})^\mathrm{degree}$
- RBF $K(x_1, x_2) = \exp(- \gamma \lVert x_1-x_2 \rVert^2)$
- TANH $K(x_1, x_2) = \tanh(\gamma \langle x_1,x_2 \rangle + \mathrm{coef0})$

```cpp
struct KernelParams {
  KernelType kernel;
  int degree;
  double gamma;
  double coef0;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `kernel` | `KernelType` |  |
| `degree` | `int` |  |
| `gamma` | `double` |  |
| `coef0` | `double` |  |

## Pairwise Distances API

<a id="distance-pairwise-distance"></a>
### distance::pairwise_distance

Compute pairwise distances for two matrices

```cpp
void pairwise_distance(
raft::resources const& handle,
raft::device_matrix_view<const float, std::int64_t, raft::layout_c_contiguous> const x,
raft::device_matrix_view<const float, std::int64_t, raft::layout_c_contiguous> const y,
raft::device_matrix_view<float, std::int64_t, raft::layout_c_contiguous> dist,
cuvs::distance::DistanceType metric,
float metric_arg = 2.0f);
```

Note: Only contiguous row- or column-major layouts supported currently.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft handle for managing expensive resources |
| `x` | in | `raft::device_matrix_view<const float, std::int64_t, raft::layout_c_contiguous> const` | first set of points (size n*k) |
| `y` | in | `raft::device_matrix_view<const float, std::int64_t, raft::layout_c_contiguous> const` | second set of points (size m*k) |
| `dist` | out | `raft::device_matrix_view<float, std::int64_t, raft::layout_c_contiguous>` | output distance matrix (size n*m) |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | distance to evaluate |
| `metric_arg` | in | `float` | metric argument (used for Minkowski distance)<br />Default: `2.0f`. |

**Returns**

`void`

**Additional overload:** `distance::pairwise_distance`

Compute pairwise distances for two matrices

```cpp
void pairwise_distance(
raft::resources const& handle,
raft::device_matrix_view<const double, std::int64_t, raft::layout_c_contiguous> const x,
raft::device_matrix_view<const double, std::int64_t, raft::layout_c_contiguous> const y,
raft::device_matrix_view<double, std::int64_t, raft::layout_c_contiguous> dist,
cuvs::distance::DistanceType metric,
double metric_arg = 2.0f);
```

Note: Only contiguous row- or column-major layouts supported currently.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft handle for managing expensive resources |
| `x` | in | `raft::device_matrix_view<const double, std::int64_t, raft::layout_c_contiguous> const` | first set of points (size n*k) |
| `y` | in | `raft::device_matrix_view<const double, std::int64_t, raft::layout_c_contiguous> const` | second set of points (size m*k) |
| `dist` | out | `raft::device_matrix_view<double, std::int64_t, raft::layout_c_contiguous>` | output distance matrix (size n*m) |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | distance to evaluate |
| `metric_arg` | in | `double` | metric argument (used for Minkowski distance)<br />Default: `2.0f`. |

**Returns**

`void`

**Additional overload:** `distance::pairwise_distance`

Compute pairwise distances for two matrices

```cpp
void pairwise_distance(
raft::resources const& handle,
raft::device_matrix_view<const half, std::int64_t, raft::layout_c_contiguous> const x,
raft::device_matrix_view<const half, std::int64_t, raft::layout_c_contiguous> const y,
raft::device_matrix_view<float, std::int64_t, raft::layout_c_contiguous> dist,
cuvs::distance::DistanceType metric,
float metric_arg = 2.0f);
```

Note: Only contiguous row- or column-major layouts supported currently.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft handle for managing expensive resources |
| `x` | in | `raft::device_matrix_view<const half, std::int64_t, raft::layout_c_contiguous> const` | first set of points (size n*k) |
| `y` | in | `raft::device_matrix_view<const half, std::int64_t, raft::layout_c_contiguous> const` | second set of points (size m*k) |
| `dist` | out | `raft::device_matrix_view<float, std::int64_t, raft::layout_c_contiguous>` | output distance matrix (size n*m) |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | distance to evaluate |
| `metric_arg` | in | `float` | metric argument (used for Minkowski distance)<br />Default: `2.0f`. |

**Returns**

`void`

**Additional overload:** `distance::pairwise_distance`

Compute pairwise distances for two matrices

```cpp
void pairwise_distance(
raft::resources const& handle,
raft::device_matrix_view<const float, std::int64_t, raft::layout_f_contiguous> const x,
raft::device_matrix_view<const float, std::int64_t, raft::layout_f_contiguous> const y,
raft::device_matrix_view<float, std::int64_t, raft::layout_f_contiguous> dist,
cuvs::distance::DistanceType metric,
float metric_arg = 2.0f);
```

Note: Only contiguous row- or column-major layouts supported currently.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft handle for managing expensive resources |
| `x` | in | `raft::device_matrix_view<const float, std::int64_t, raft::layout_f_contiguous> const` | first set of points (size n*k) |
| `y` | in | `raft::device_matrix_view<const float, std::int64_t, raft::layout_f_contiguous> const` | second set of points (size m*k) |
| `dist` | out | `raft::device_matrix_view<float, std::int64_t, raft::layout_f_contiguous>` | output distance matrix (size n*m) |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | distance to evaluate |
| `metric_arg` | in | `float` | metric argument (used for Minkowski distance)<br />Default: `2.0f`. |

**Returns**

`void`

**Additional overload:** `distance::pairwise_distance`

Compute pairwise distances for two matrices

```cpp
void pairwise_distance(
raft::resources const& handle,
raft::device_matrix_view<const double, std::int64_t, raft::layout_f_contiguous> const x,
raft::device_matrix_view<const double, std::int64_t, raft::layout_f_contiguous> const y,
raft::device_matrix_view<double, std::int64_t, raft::layout_f_contiguous> dist,
cuvs::distance::DistanceType metric,
double metric_arg = 2.0f);
```

Note: Only contiguous row- or column-major layouts supported currently.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft handle for managing expensive resources |
| `x` | in | `raft::device_matrix_view<const double, std::int64_t, raft::layout_f_contiguous> const` | first set of points (size n*k) |
| `y` | in | `raft::device_matrix_view<const double, std::int64_t, raft::layout_f_contiguous> const` | second set of points (size m*k) |
| `dist` | out | `raft::device_matrix_view<double, std::int64_t, raft::layout_f_contiguous>` | output distance matrix (size n*m) |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | distance to evaluate |
| `metric_arg` | in | `double` | metric argument (used for Minkowski distance)<br />Default: `2.0f`. |

**Returns**

`void`

**Additional overload:** `distance::pairwise_distance`

Compute pairwise distances for two matrices

```cpp
void pairwise_distance(
raft::resources const& handle,
raft::device_matrix_view<const half, std::int64_t, raft::layout_f_contiguous> const x,
raft::device_matrix_view<const half, std::int64_t, raft::layout_f_contiguous> const y,
raft::device_matrix_view<float, std::int64_t, raft::layout_f_contiguous> dist,
cuvs::distance::DistanceType metric,
float metric_arg = 2.0f);
```

Note: Only contiguous row- or column-major layouts supported currently.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft handle for managing expensive resources |
| `x` | in | `raft::device_matrix_view<const half, std::int64_t, raft::layout_f_contiguous> const` | first set of points (size n*k) |
| `y` | in | `raft::device_matrix_view<const half, std::int64_t, raft::layout_f_contiguous> const` | second set of points (size m*k) |
| `dist` | out | `raft::device_matrix_view<float, std::int64_t, raft::layout_f_contiguous>` | output distance matrix (size n*m) |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | distance to evaluate |
| `metric_arg` | in | `float` | metric argument (used for Minkowski distance)<br />Default: `2.0f`. |

**Returns**

`void`

**Additional overload:** `distance::pairwise_distance`

Compute sparse pairwise distances between x and y, using the provided input configuration and distance function.

```cpp
void pairwise_distance(raft::resources const& handle,
raft::device_csr_matrix_view<const float, int, int, int> x,
raft::device_csr_matrix_view<const float, int, int, int> y,
raft::device_matrix_view<float, int, raft::row_major> dist,
cuvs::distance::DistanceType metric,
float metric_arg = 2.0f);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft::resources |
| `x` | in | `raft::device_csr_matrix_view<const float, int, int, int>` | raft::device_csr_matrix_view |
| `y` | in | `raft::device_csr_matrix_view<const float, int, int, int>` | raft::device_csr_matrix_view |
| `dist` | out | `raft::device_matrix_view<float, int, raft::row_major>` | raft::device_matrix_view dense matrix |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | distance metric to use |
| `metric_arg` | in | `float` | metric argument (used for Minkowski distance)<br />Default: `2.0f`. |

**Returns**

`void`

**Additional overload:** `distance::pairwise_distance`

Compute sparse pairwise distances between x and y, using the provided input configuration and distance function.

```cpp
void pairwise_distance(raft::resources const& handle,
raft::device_csr_matrix_view<const double, int, int, int> x,
raft::device_csr_matrix_view<const double, int, int, int> y,
raft::device_matrix_view<double, int, raft::row_major> dist,
cuvs::distance::DistanceType metric,
float metric_arg = 2.0f);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft::resources |
| `x` | in | `raft::device_csr_matrix_view<const double, int, int, int>` | raft::device_csr_matrix_view |
| `y` | in | `raft::device_csr_matrix_view<const double, int, int, int>` | raft::device_csr_matrix_view |
| `dist` | out | `raft::device_matrix_view<double, int, raft::row_major>` | raft::device_matrix_view dense matrix |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | distance metric to use |
| `metric_arg` | in | `float` | metric argument (used for Minkowski distance)<br />Default: `2.0f`. |

**Returns**

`void`
