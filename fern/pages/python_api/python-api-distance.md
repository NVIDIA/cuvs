---
slug: api-reference/python-api-distance
---

# Distance

_Python module: `cuvs.distance`_

## DISTANCE_NAMES

```python
DISTANCE_NAMES = {v: k for k, v in DISTANCE_TYPES.items()}
```

## DISTANCE_TYPES

```python
DISTANCE_TYPES = {
"l2": cuvsDistanceType.L2SqrtExpanded,
"sqeuclidean": cuvsDistanceType.L2Expanded,
"euclidean": cuvsDistanceType.L2SqrtExpanded,
"l1": cuvsDistanceType.L1,
"cityblock": cuvsDistanceType.L1,
"inner_product": cuvsDistanceType.InnerProduct,
"chebyshev": cuvsDistanceType.Linf,
"canberra": cuvsDistanceType.Canberra,
"cosine": cuvsDistanceType.CosineExpanded,
"lp": cuvsDistanceType.LpUnexpanded,
"correlation": cuvsDistanceType.CorrelationExpanded,
"jaccard": cuvsDistanceType.JaccardExpanded,
"hellinger": cuvsDistanceType.HellingerExpanded,
"braycurtis": cuvsDistanceType.BrayCurtis,
"jensenshannon": cuvsDistanceType.JensenShannon,
"hamming": cuvsDistanceType.HammingUnexpanded,
"kl_divergence": cuvsDistanceType.KLDivergence,
"minkowski": cuvsDistanceType.LpUnexpanded,
"russellrao": cuvsDistanceType.RusselRaoExpanded,
"dice": cuvsDistanceType.DiceExpanded,
"bitwise_hamming": cuvsDistanceType.BitwiseHamming
}
```

## pairwise_distance

`@auto_sync_resources`
`@auto_convert_output`

```python
def pairwise_distance(X, Y, out=None, metric="euclidean", p=2.0, resources=None)
```

Compute pairwise distances between X and Y

Valid values for metric:
["euclidean", "l2", "l1", "cityblock", "inner_product",
"chebyshev", "canberra", "lp", "hellinger", "jensenshannon",
"kl_divergence", "russellrao", "minkowski", "correlation",
"cosine"]

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `X` | `CUDA array interface compliant matrix shape (m, k)` |  |
| `Y` | `CUDA array interface compliant matrix shape (n, k)` |  |
| `out` | `Optional writable CUDA array interface matrix shape (m, n)` |  |
| `metric` | `string denoting the metric type (default="euclidean")` |  |
| `p` | `metric parameter (currently used only for "minkowski")` |  |
| `resources` | `cuvs.common.Resources, optional` |  |

**Examples**

```python
>>> import cupy as cp
>>> from cuvs.distance import pairwise_distance
>>> n_samples = 5000
>>> n_features = 50
>>> in1 = cp.random.random_sample((n_samples, n_features),
...                               dtype=cp.float32)
>>> in2 = cp.random.random_sample((n_samples, n_features),
...                               dtype=cp.float32)
>>> output = pairwise_distance(in1, in2, metric="euclidean")
```
