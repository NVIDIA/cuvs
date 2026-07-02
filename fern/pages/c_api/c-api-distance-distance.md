---
slug: api-reference/c-api-distance-distance
---

# Distance

_Source header: `cuvs/distance/distance.h`_

## Types

<a id="cuvsdistancetype"></a>
### cuvsDistanceType

enum to tell how to compute distance

```c
typedef enum {
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
  Precomputed = 100
} cuvsDistanceType;
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
