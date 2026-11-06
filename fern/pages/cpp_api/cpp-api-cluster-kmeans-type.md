---
slug: api-reference/cpp-api-cluster-kmeans-type
---

# K-Means Type

_Source header: `cuvs/cluster/kmeans_type.hpp`_

## Types

<a id="cluster-kmeans-kmeans-type"></a>
### cluster::kmeans::kmeans_type

Type of k-means algorithm.

```cpp
enum class kmeans_type {
  KMeans = 0,
  KMeansBalanced = 1
};
```

**Values**

| Name | Value |
| --- | --- |
| `KMeans` | `0` |
| `KMeansBalanced` | `1` |
