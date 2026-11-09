---
slug: api-reference/java-api-com-nvidia-cuvs-cagraindexparams
---

# CagraIndexParams

_Java package: `com.nvidia.cuvs`_

```java
public class CagraIndexParams
```

Supplemental parameters to build CAGRA Index.

## Public Members

### AUTO_SELECT

```java
AUTO_SELECT(0), /** * Use IVF-PQ to build all-neighbors knn graph */ IVF_PQ(1), /** * Experimental, use NN-Descent to build all-neighbors knn graph */ NN_DESCENT(2), /** * Experimental, use ACE (Augmented Core Extraction) to build graph for large datasets. * 4 to be consistent with the other interfaces. */ ACE(4)
```

Select build algorithm automatically

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:37`_

### IVF_PQ

```java
IVF_PQ(1), /** * Experimental, use NN-Descent to build all-neighbors knn graph */ NN_DESCENT(2), /** * Experimental, use ACE (Augmented Core Extraction) to build graph for large datasets. * 4 to be consistent with the other interfaces. */ ACE(4)
```

Use IVF-PQ to build all-neighbors knn graph

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:41`_

### NN_DESCENT

```java
NN_DESCENT(2), /** * Experimental, use ACE (Augmented Core Extraction) to build graph for large datasets. * 4 to be consistent with the other interfaces. */ ACE(4)
```

Experimental, use NN-Descent to build all-neighbors knn graph

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:45`_

### ACE

```java
ACE(4)
```

Experimental, use ACE (Augmented Core Extraction) to build graph for large datasets.
4 to be consistent with the other interfaces.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:50`_

### SIMILAR_SEARCH_PERFORMANCE

```java
SIMILAR_SEARCH_PERFORMANCE(0), /** * Create a graph that has the same binary size as an HNSW graph with the given parameters * (graph_degree = 2 * M) while trying to match the search performance as closely as possible. * * The reference HNSW index and the corresponding from-CAGRA generated HNSW index will NOT produce * the same recalls and QPS for the same parameter ef. The graphs are different internally. For * the same ef, the from-CAGRA index likely has a slightly higher recall and slightly lower QPS. * However, the Recall-QPS curves should be similar (i.e. the points are just shifted along the * curve). */ SAME_GRAPH_FOOTPRINT(1)
```

Create a graph that is very similar to an HNSW graph in terms of the number of nodes and
search performance.
Unlike HNSW, the maximum degree of the graph is not fixed to `2*M`, but is decided
according the internal logic (smaller than `2*M`). The edge selection criteria of the
two algorithms also differ, so there's always a difference in the performance of the two.

This function attempts to produce such a graph that the QPS and recall of the two graphs being
searched by HNSW are close for any search parameter combination. The CAGRA-produced graph tends
to have a "longer tail" on the low recall side (that is being slightly faster and less
precise).

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:87`_

### SAME_GRAPH_FOOTPRINT

```java
SAME_GRAPH_FOOTPRINT(1)
```

Create a graph that has the same binary size as an HNSW graph with the given parameters
(graph_degree = 2 * M) while trying to match the search performance as closely as possible.

The reference HNSW index and the corresponding from-CAGRA generated HNSW index will NOT produce
the same recalls and QPS for the same parameter ef. The graphs are different internally. For
the same ef, the from-CAGRA index likely has a slightly higher recall and slightly lower QPS.
However, the Recall-QPS curves should be similar (i.e. the points are just shifted along the
curve).

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:98`_

### L2Expanded

```java
L2Expanded(0), /** * same as above, but inside the epilogue, perform square root operation */ L2SqrtExpanded(1), /** * cosine distance */ CosineExpanded(2), /** * L1 distance * */ L1(3), /** * evaluate as dist_ij += (x_ik - y-jk)^2 * */ L2Unexpanded(4), /** * same as above, but inside the epilogue, perform square root operation */ L2SqrtUnexpanded(5), /** * basic inner product */ InnerProduct(6), /** * Chebyshev (Linf) distance */ Linf(7), /** * Canberra distance */ Canberra(8), /** * Generalized Minkowski distance */ LpUnexpanded(9), /** * Correlation distance */ CorrelationExpanded(10), /** * Jaccard distance */ JaccardExpanded(11), /** * Hellinger distance */ HellingerExpanded(12), /** * Haversine distance */ Haversine(13), /** * Bray-Curtis distance */ BrayCurtis(14), /** * Jensen-Shannon distance */ JensenShannon(15), /** * Hamming distance */ HammingUnexpanded(16), /** * KLDivergence */ KLDivergence(17), /** * RusselRao */ RusselRaoExpanded(18), /** * Dice-Sorensen distance */ DiceExpanded(19), /** * Precomputed (special value) */ Precomputed(100)
```

evaluate as dist_ij = sum(x_ik^2) + sum(y_ij)^2 - 2*sum(x_ik * y_jk)

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:126`_

### L2SqrtExpanded

```java
L2SqrtExpanded(1), /** * cosine distance */ CosineExpanded(2), /** * L1 distance * */ L1(3), /** * evaluate as dist_ij += (x_ik - y-jk)^2 * */ L2Unexpanded(4), /** * same as above, but inside the epilogue, perform square root operation */ L2SqrtUnexpanded(5), /** * basic inner product */ InnerProduct(6), /** * Chebyshev (Linf) distance */ Linf(7), /** * Canberra distance */ Canberra(8), /** * Generalized Minkowski distance */ LpUnexpanded(9), /** * Correlation distance */ CorrelationExpanded(10), /** * Jaccard distance */ JaccardExpanded(11), /** * Hellinger distance */ HellingerExpanded(12), /** * Haversine distance */ Haversine(13), /** * Bray-Curtis distance */ BrayCurtis(14), /** * Jensen-Shannon distance */ JensenShannon(15), /** * Hamming distance */ HammingUnexpanded(16), /** * KLDivergence */ KLDivergence(17), /** * RusselRao */ RusselRaoExpanded(18), /** * Dice-Sorensen distance */ DiceExpanded(19), /** * Precomputed (special value) */ Precomputed(100)
```

same as above, but inside the epilogue, perform square root operation

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:130`_

### CosineExpanded

```java
CosineExpanded(2), /** * L1 distance * */ L1(3), /** * evaluate as dist_ij += (x_ik - y-jk)^2 * */ L2Unexpanded(4), /** * same as above, but inside the epilogue, perform square root operation */ L2SqrtUnexpanded(5), /** * basic inner product */ InnerProduct(6), /** * Chebyshev (Linf) distance */ Linf(7), /** * Canberra distance */ Canberra(8), /** * Generalized Minkowski distance */ LpUnexpanded(9), /** * Correlation distance */ CorrelationExpanded(10), /** * Jaccard distance */ JaccardExpanded(11), /** * Hellinger distance */ HellingerExpanded(12), /** * Haversine distance */ Haversine(13), /** * Bray-Curtis distance */ BrayCurtis(14), /** * Jensen-Shannon distance */ JensenShannon(15), /** * Hamming distance */ HammingUnexpanded(16), /** * KLDivergence */ KLDivergence(17), /** * RusselRao */ RusselRaoExpanded(18), /** * Dice-Sorensen distance */ DiceExpanded(19), /** * Precomputed (special value) */ Precomputed(100)
```

cosine distance

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:134`_

### L1

```java
L1(3), /** * evaluate as dist_ij += (x_ik - y-jk)^2 * */ L2Unexpanded(4), /** * same as above, but inside the epilogue, perform square root operation */ L2SqrtUnexpanded(5), /** * basic inner product */ InnerProduct(6), /** * Chebyshev (Linf) distance */ Linf(7), /** * Canberra distance */ Canberra(8), /** * Generalized Minkowski distance */ LpUnexpanded(9), /** * Correlation distance */ CorrelationExpanded(10), /** * Jaccard distance */ JaccardExpanded(11), /** * Hellinger distance */ HellingerExpanded(12), /** * Haversine distance */ Haversine(13), /** * Bray-Curtis distance */ BrayCurtis(14), /** * Jensen-Shannon distance */ JensenShannon(15), /** * Hamming distance */ HammingUnexpanded(16), /** * KLDivergence */ KLDivergence(17), /** * RusselRao */ RusselRaoExpanded(18), /** * Dice-Sorensen distance */ DiceExpanded(19), /** * Precomputed (special value) */ Precomputed(100)
```

L1 distance *

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:138`_

### L2Unexpanded

```java
L2Unexpanded(4), /** * same as above, but inside the epilogue, perform square root operation */ L2SqrtUnexpanded(5), /** * basic inner product */ InnerProduct(6), /** * Chebyshev (Linf) distance */ Linf(7), /** * Canberra distance */ Canberra(8), /** * Generalized Minkowski distance */ LpUnexpanded(9), /** * Correlation distance */ CorrelationExpanded(10), /** * Jaccard distance */ JaccardExpanded(11), /** * Hellinger distance */ HellingerExpanded(12), /** * Haversine distance */ Haversine(13), /** * Bray-Curtis distance */ BrayCurtis(14), /** * Jensen-Shannon distance */ JensenShannon(15), /** * Hamming distance */ HammingUnexpanded(16), /** * KLDivergence */ KLDivergence(17), /** * RusselRao */ RusselRaoExpanded(18), /** * Dice-Sorensen distance */ DiceExpanded(19), /** * Precomputed (special value) */ Precomputed(100)
```

evaluate as dist_ij += (x_ik - y-jk)^2 *

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:142`_

### L2SqrtUnexpanded

```java
L2SqrtUnexpanded(5), /** * basic inner product */ InnerProduct(6), /** * Chebyshev (Linf) distance */ Linf(7), /** * Canberra distance */ Canberra(8), /** * Generalized Minkowski distance */ LpUnexpanded(9), /** * Correlation distance */ CorrelationExpanded(10), /** * Jaccard distance */ JaccardExpanded(11), /** * Hellinger distance */ HellingerExpanded(12), /** * Haversine distance */ Haversine(13), /** * Bray-Curtis distance */ BrayCurtis(14), /** * Jensen-Shannon distance */ JensenShannon(15), /** * Hamming distance */ HammingUnexpanded(16), /** * KLDivergence */ KLDivergence(17), /** * RusselRao */ RusselRaoExpanded(18), /** * Dice-Sorensen distance */ DiceExpanded(19), /** * Precomputed (special value) */ Precomputed(100)
```

same as above, but inside the epilogue, perform square root operation

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:146`_

### InnerProduct

```java
InnerProduct(6), /** * Chebyshev (Linf) distance */ Linf(7), /** * Canberra distance */ Canberra(8), /** * Generalized Minkowski distance */ LpUnexpanded(9), /** * Correlation distance */ CorrelationExpanded(10), /** * Jaccard distance */ JaccardExpanded(11), /** * Hellinger distance */ HellingerExpanded(12), /** * Haversine distance */ Haversine(13), /** * Bray-Curtis distance */ BrayCurtis(14), /** * Jensen-Shannon distance */ JensenShannon(15), /** * Hamming distance */ HammingUnexpanded(16), /** * KLDivergence */ KLDivergence(17), /** * RusselRao */ RusselRaoExpanded(18), /** * Dice-Sorensen distance */ DiceExpanded(19), /** * Precomputed (special value) */ Precomputed(100)
```

basic inner product

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:150`_

### Linf

```java
Linf(7), /** * Canberra distance */ Canberra(8), /** * Generalized Minkowski distance */ LpUnexpanded(9), /** * Correlation distance */ CorrelationExpanded(10), /** * Jaccard distance */ JaccardExpanded(11), /** * Hellinger distance */ HellingerExpanded(12), /** * Haversine distance */ Haversine(13), /** * Bray-Curtis distance */ BrayCurtis(14), /** * Jensen-Shannon distance */ JensenShannon(15), /** * Hamming distance */ HammingUnexpanded(16), /** * KLDivergence */ KLDivergence(17), /** * RusselRao */ RusselRaoExpanded(18), /** * Dice-Sorensen distance */ DiceExpanded(19), /** * Precomputed (special value) */ Precomputed(100)
```

Chebyshev (Linf) distance

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:154`_

### Canberra

```java
Canberra(8), /** * Generalized Minkowski distance */ LpUnexpanded(9), /** * Correlation distance */ CorrelationExpanded(10), /** * Jaccard distance */ JaccardExpanded(11), /** * Hellinger distance */ HellingerExpanded(12), /** * Haversine distance */ Haversine(13), /** * Bray-Curtis distance */ BrayCurtis(14), /** * Jensen-Shannon distance */ JensenShannon(15), /** * Hamming distance */ HammingUnexpanded(16), /** * KLDivergence */ KLDivergence(17), /** * RusselRao */ RusselRaoExpanded(18), /** * Dice-Sorensen distance */ DiceExpanded(19), /** * Precomputed (special value) */ Precomputed(100)
```

Canberra distance

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:158`_

### LpUnexpanded

```java
LpUnexpanded(9), /** * Correlation distance */ CorrelationExpanded(10), /** * Jaccard distance */ JaccardExpanded(11), /** * Hellinger distance */ HellingerExpanded(12), /** * Haversine distance */ Haversine(13), /** * Bray-Curtis distance */ BrayCurtis(14), /** * Jensen-Shannon distance */ JensenShannon(15), /** * Hamming distance */ HammingUnexpanded(16), /** * KLDivergence */ KLDivergence(17), /** * RusselRao */ RusselRaoExpanded(18), /** * Dice-Sorensen distance */ DiceExpanded(19), /** * Precomputed (special value) */ Precomputed(100)
```

Generalized Minkowski distance

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:162`_

### CorrelationExpanded

```java
CorrelationExpanded(10), /** * Jaccard distance */ JaccardExpanded(11), /** * Hellinger distance */ HellingerExpanded(12), /** * Haversine distance */ Haversine(13), /** * Bray-Curtis distance */ BrayCurtis(14), /** * Jensen-Shannon distance */ JensenShannon(15), /** * Hamming distance */ HammingUnexpanded(16), /** * KLDivergence */ KLDivergence(17), /** * RusselRao */ RusselRaoExpanded(18), /** * Dice-Sorensen distance */ DiceExpanded(19), /** * Precomputed (special value) */ Precomputed(100)
```

Correlation distance

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:166`_

### JaccardExpanded

```java
JaccardExpanded(11), /** * Hellinger distance */ HellingerExpanded(12), /** * Haversine distance */ Haversine(13), /** * Bray-Curtis distance */ BrayCurtis(14), /** * Jensen-Shannon distance */ JensenShannon(15), /** * Hamming distance */ HammingUnexpanded(16), /** * KLDivergence */ KLDivergence(17), /** * RusselRao */ RusselRaoExpanded(18), /** * Dice-Sorensen distance */ DiceExpanded(19), /** * Precomputed (special value) */ Precomputed(100)
```

Jaccard distance

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:170`_

### HellingerExpanded

```java
HellingerExpanded(12), /** * Haversine distance */ Haversine(13), /** * Bray-Curtis distance */ BrayCurtis(14), /** * Jensen-Shannon distance */ JensenShannon(15), /** * Hamming distance */ HammingUnexpanded(16), /** * KLDivergence */ KLDivergence(17), /** * RusselRao */ RusselRaoExpanded(18), /** * Dice-Sorensen distance */ DiceExpanded(19), /** * Precomputed (special value) */ Precomputed(100)
```

Hellinger distance

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:174`_

### Haversine

```java
Haversine(13), /** * Bray-Curtis distance */ BrayCurtis(14), /** * Jensen-Shannon distance */ JensenShannon(15), /** * Hamming distance */ HammingUnexpanded(16), /** * KLDivergence */ KLDivergence(17), /** * RusselRao */ RusselRaoExpanded(18), /** * Dice-Sorensen distance */ DiceExpanded(19), /** * Precomputed (special value) */ Precomputed(100)
```

Haversine distance

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:178`_

### BrayCurtis

```java
BrayCurtis(14), /** * Jensen-Shannon distance */ JensenShannon(15), /** * Hamming distance */ HammingUnexpanded(16), /** * KLDivergence */ KLDivergence(17), /** * RusselRao */ RusselRaoExpanded(18), /** * Dice-Sorensen distance */ DiceExpanded(19), /** * Precomputed (special value) */ Precomputed(100)
```

Bray-Curtis distance

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:182`_

### JensenShannon

```java
JensenShannon(15), /** * Hamming distance */ HammingUnexpanded(16), /** * KLDivergence */ KLDivergence(17), /** * RusselRao */ RusselRaoExpanded(18), /** * Dice-Sorensen distance */ DiceExpanded(19), /** * Precomputed (special value) */ Precomputed(100)
```

Jensen-Shannon distance

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:186`_

### HammingUnexpanded

```java
HammingUnexpanded(16), /** * KLDivergence */ KLDivergence(17), /** * RusselRao */ RusselRaoExpanded(18), /** * Dice-Sorensen distance */ DiceExpanded(19), /** * Precomputed (special value) */ Precomputed(100)
```

Hamming distance

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:190`_

### KLDivergence

```java
KLDivergence(17), /** * RusselRao */ RusselRaoExpanded(18), /** * Dice-Sorensen distance */ DiceExpanded(19), /** * Precomputed (special value) */ Precomputed(100)
```

KLDivergence

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:194`_

### RusselRaoExpanded

```java
RusselRaoExpanded(18), /** * Dice-Sorensen distance */ DiceExpanded(19), /** * Precomputed (special value) */ Precomputed(100)
```

RusselRao

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:198`_

### DiceExpanded

```java
DiceExpanded(19), /** * Precomputed (special value) */ Precomputed(100)
```

Dice-Sorensen distance

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:202`_

### Precomputed

```java
Precomputed(100)
```

Precomputed (special value)

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:206`_

### getIntermediateGraphDegree

```java
public long getIntermediateGraphDegree()
```

Gets the degree of input graph for pruning.

**Returns**

the degree of input graph

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:376`_

### getGraphDegree

```java
public long getGraphDegree()
```

Gets the degree of output graph.

**Returns**

the degree of output graph

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:385`_

### getVariableGraphDegreeFraction

```java
public double getVariableGraphDegreeFraction()
```

Gets the fraction of the output graph degree that defines the minimum output graph degree.
The default value of 1.0 means the graph has a constant degree.

**Returns**

the variable graph degree fraction, in [0, 1]

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:396`_

### isGuaranteeConnectivity

```java
public boolean isGuaranteeConnectivity()
```

Gets whether the graph build guarantees the graph is connected.

**Returns**

whether MST optimization is used to guarantee graph connectivity

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:406`_

### getCagraGraphBuildAlgo

```java
public CagraGraphBuildAlgo getCagraGraphBuildAlgo()
```

Gets the `CagraGraphBuildAlgo` used to build the index.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:413`_

### getNNDescentNumIterations

```java
public long getNNDescentNumIterations()
```

Gets the number of iterations to run if building with
`CagraGraphBuildAlgo#NN_DESCENT`

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:421`_

### getCuvsDistanceType

```java
public CuvsDistanceType getCuvsDistanceType()
```

Gets the `CuvsDistanceType` used to build the index.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:428`_

### getNumWriterThreads

```java
public int getNumWriterThreads()
```

Gets the number of threads used to build the index.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:435`_

### getCuVSIvfPqParams

```java
public CuVSIvfPqParams getCuVSIvfPqParams()
```

Gets the IVF_PQ parameters.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:442`_

### getCuVSAceParams

```java
public CuVSAceParams getCuVSAceParams()
```

Gets the ACE parameters.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:449`_

### getCuvsCagraGraphBuildAlgo

```java
public CagraGraphBuildAlgo getCuvsCagraGraphBuildAlgo()
```

Gets the CAGRA build algorithm.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:456`_

### withIntermediateGraphDegree

```java
public Builder withIntermediateGraphDegree(long intermediateGraphDegree)
```

Sets the degree of input graph for pruning.

**Parameters**

| Name | Description |
| --- | --- |
| `intermediateGraphDegree` | degree of input graph for pruning |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:509`_

### withGraphDegree

```java
public Builder withGraphDegree(long graphDegree)
```

Sets the degree of output graph.

**Parameters**

| Name | Description |
| --- | --- |
| `graphDegree` | degree of output graph |

**Returns**

an instance to Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:520`_

### withVariableGraphDegreeFraction

```java
public Builder withVariableGraphDegreeFraction(double variableGraphDegreeFraction)
```

Sets the fraction of the output graph degree that defines the minimum output graph degree,
allowing variable-degree neighbor graphs.

Must be in [0, 1]. The default value of 1.0 disables the variable-degree logic (normal
CAGRA behavior). Values below 1.0 enable variable-degree graphs: the optimize step finds the
minimum detour threshold that covers at least `ceil(graphDegree * fraction)` edges per
node, then lets reverse edges expand the degree further. Unused neighbor slots are padded with
an invalid-neighbor sentinel.

This is intended for the CAGRA-to-HNSW conversion pipeline. Do not use it with CAGRA's
native GPU search.

**Parameters**

| Name | Description |
| --- | --- |
| `variableGraphDegreeFraction` | the fraction of the graph degree, in [0, 1] |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:541`_

### withGuaranteeConnectivity

```java
public Builder withGuaranteeConnectivity(boolean guaranteeConnectivity)
```

Sets whether to use MST optimization to guarantee graph connectivity.

When true, the optimize step keeps the edges of an approximate, degree-constrained minimum
spanning tree of the kNN graph in the output graph. If the kNN graph itself is disconnected,
for example for well separated clusters, nodes outside its largest component are also linked
to that component. Default: false.

**Parameters**

| Name | Description |
| --- | --- |
| `guaranteeConnectivity` | whether to guarantee graph connectivity |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:557`_

### withCagraGraphBuildAlgo

```java
public Builder withCagraGraphBuildAlgo(CagraGraphBuildAlgo cuvsCagraGraphBuildAlgo)
```

Sets the CuvsCagraGraphBuildAlgo to use.

**Parameters**

| Name | Description |
| --- | --- |
| `cuvsCagraGraphBuildAlgo` | the CuvsCagraGraphBuildAlgo to use |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:568`_

### withMetric

```java
public Builder withMetric(CuvsDistanceType cuvsDistanceType)
```

Sets the metric to use.

**Parameters**

| Name | Description |
| --- | --- |
| `cuvsDistanceType` | the `CuvsDistanceType` to use |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:579`_

### withNNDescentNumIterations

```java
public Builder withNNDescentNumIterations(long nnDescentNiter)
```

Sets the Number of Iterations to run if building with
`CagraGraphBuildAlgo#NN_DESCENT`.

**Parameters**

| Name | Description |
| --- | --- |
| `nnDescentNiter` | number of Iterations to run if building with `CagraGraphBuildAlgo#NN_DESCENT` |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:592`_

### withNumWriterThreads

```java
public Builder withNumWriterThreads(int numWriterThreads)
```

Sets the number of writer threads to use for indexing.

This sets the OpenMP thread count for the native build or merge on the calling thread, and
the previous count is restored afterwards. A value of zero or less leaves the calling
thread's OpenMP thread count unchanged. That is the OpenMP default (normally one thread per
core) unless it was set earlier on that thread.

**Parameters**

| Name | Description |
| --- | --- |
| `numWriterThreads` | number of writer threads to use, or zero or less to keep the calling thread's OpenMP thread count |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:609`_

### withCuVSIvfPqParams

```java
public Builder withCuVSIvfPqParams(CuVSIvfPqParams cuVSIvfPqParams)
```

Sets the IVF_PQ index parameters.

**Parameters**

| Name | Description |
| --- | --- |
| `cuVSIvfPqParams` | the IVF_PQ index parameters |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:620`_

### withCuVSAceParams

```java
public Builder withCuVSAceParams(CuVSAceParams cuVSAceParams)
```

Sets the ACE index parameters.

**Parameters**

| Name | Description |
| --- | --- |
| `cuVSAceParams` | the ACE index parameters |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:631`_

### build

```java
public CagraIndexParams build()
```

Builds an instance of `CagraIndexParams`.

**Returns**

an instance of `CagraIndexParams`

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:641`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:18`_
