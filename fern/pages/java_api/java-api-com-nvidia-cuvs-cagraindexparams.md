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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:35`_

### IVF_PQ

```java
IVF_PQ(1), /** * Experimental, use NN-Descent to build all-neighbors knn graph */ NN_DESCENT(2), /** * Experimental, use ACE (Augmented Core Extraction) to build graph for large datasets. * 4 to be consistent with the other interfaces. */ ACE(4)
```

Use IVF-PQ to build all-neighbors knn graph

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:39`_

### NN_DESCENT

```java
NN_DESCENT(2), /** * Experimental, use ACE (Augmented Core Extraction) to build graph for large datasets. * 4 to be consistent with the other interfaces. */ ACE(4)
```

Experimental, use NN-Descent to build all-neighbors knn graph

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:43`_

### ACE

```java
ACE(4)
```

Experimental, use ACE (Augmented Core Extraction) to build graph for large datasets.
4 to be consistent with the other interfaces.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:48`_

### SIMILAR_SEARCH_PERFORMANCE

```java
SIMILAR_SEARCH_PERFORMANCE(0), /** * Create a graph that has the same binary size as an HNSW graph with the given parameters * (graph_degree = 2 * M) while trying to match the search performance as closely as possible. * * The reference HNSW index and the corresponding from-CAGRA generated HNSW index will NOT produce * the same recalls and QPS for the same parameter ef. The graphs are different internally. For * the same ef, the from-CAGRA index likely has a slightly higher recall and slightly lower QPS. * However, the Recall-QPS curves should be similar (i.e. the points are just shifted along the * curve). */ SAME_GRAPH_FOOTPRINT(1)
```

Create a graph that is very similar to an HNSW graph in
terms of the number of nodes and search performance. Since HNSW produces a variable-degree
graph (2M being the max graph degree) and CAGRA produces a fixed-degree graph, there's always a
difference in the performance of the two.

This function attempts to produce such a graph that the QPS and recall of the two graphs being
searched by HNSW are close for any search parameter combination. The CAGRA-produced graph tends
to have a "longer tail" on the low recall side (that is being slightly faster and less
precise).

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:84`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:95`_

### L2Expanded

```java
L2Expanded(0), /** * same as above, but inside the epilogue, perform square root operation */ L2SqrtExpanded(1), /** * cosine distance */ CosineExpanded(2), /** * L1 distance * */ L1(3), /** * evaluate as dist_ij += (x_ik - y-jk)^2 * */ L2Unexpanded(4), /** * same as above, but inside the epilogue, perform square root operation */ L2SqrtUnexpanded(5), /** * basic inner product */ InnerProduct(6), /** * Chebyshev (Linf) distance */ Linf(7), /** * Canberra distance */ Canberra(8), /** * Generalized Minkowski distance */ LpUnexpanded(9), /** * Correlation distance */ CorrelationExpanded(10), /** * Jaccard distance */ JaccardExpanded(11), /** * Hellinger distance */ HellingerExpanded(12), /** * Haversine distance */ Haversine(13), /** * Bray-Curtis distance */ BrayCurtis(14), /** * Jensen-Shannon distance */ JensenShannon(15), /** * Hamming distance */ HammingUnexpanded(16), /** * KLDivergence */ KLDivergence(17), /** * RusselRao */ RusselRaoExpanded(18), /** * Dice-Sorensen distance */ DiceExpanded(19), /** * Precomputed (special value) */ Precomputed(100)
```

evaluate as dist_ij = sum(x_ik^2) + sum(y_ij)^2 - 2*sum(x_ik * y_jk)

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:123`_

### L2SqrtExpanded

```java
L2SqrtExpanded(1), /** * cosine distance */ CosineExpanded(2), /** * L1 distance * */ L1(3), /** * evaluate as dist_ij += (x_ik - y-jk)^2 * */ L2Unexpanded(4), /** * same as above, but inside the epilogue, perform square root operation */ L2SqrtUnexpanded(5), /** * basic inner product */ InnerProduct(6), /** * Chebyshev (Linf) distance */ Linf(7), /** * Canberra distance */ Canberra(8), /** * Generalized Minkowski distance */ LpUnexpanded(9), /** * Correlation distance */ CorrelationExpanded(10), /** * Jaccard distance */ JaccardExpanded(11), /** * Hellinger distance */ HellingerExpanded(12), /** * Haversine distance */ Haversine(13), /** * Bray-Curtis distance */ BrayCurtis(14), /** * Jensen-Shannon distance */ JensenShannon(15), /** * Hamming distance */ HammingUnexpanded(16), /** * KLDivergence */ KLDivergence(17), /** * RusselRao */ RusselRaoExpanded(18), /** * Dice-Sorensen distance */ DiceExpanded(19), /** * Precomputed (special value) */ Precomputed(100)
```

same as above, but inside the epilogue, perform square root operation

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:127`_

### CosineExpanded

```java
CosineExpanded(2), /** * L1 distance * */ L1(3), /** * evaluate as dist_ij += (x_ik - y-jk)^2 * */ L2Unexpanded(4), /** * same as above, but inside the epilogue, perform square root operation */ L2SqrtUnexpanded(5), /** * basic inner product */ InnerProduct(6), /** * Chebyshev (Linf) distance */ Linf(7), /** * Canberra distance */ Canberra(8), /** * Generalized Minkowski distance */ LpUnexpanded(9), /** * Correlation distance */ CorrelationExpanded(10), /** * Jaccard distance */ JaccardExpanded(11), /** * Hellinger distance */ HellingerExpanded(12), /** * Haversine distance */ Haversine(13), /** * Bray-Curtis distance */ BrayCurtis(14), /** * Jensen-Shannon distance */ JensenShannon(15), /** * Hamming distance */ HammingUnexpanded(16), /** * KLDivergence */ KLDivergence(17), /** * RusselRao */ RusselRaoExpanded(18), /** * Dice-Sorensen distance */ DiceExpanded(19), /** * Precomputed (special value) */ Precomputed(100)
```

cosine distance

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:131`_

### L1

```java
L1(3), /** * evaluate as dist_ij += (x_ik - y-jk)^2 * */ L2Unexpanded(4), /** * same as above, but inside the epilogue, perform square root operation */ L2SqrtUnexpanded(5), /** * basic inner product */ InnerProduct(6), /** * Chebyshev (Linf) distance */ Linf(7), /** * Canberra distance */ Canberra(8), /** * Generalized Minkowski distance */ LpUnexpanded(9), /** * Correlation distance */ CorrelationExpanded(10), /** * Jaccard distance */ JaccardExpanded(11), /** * Hellinger distance */ HellingerExpanded(12), /** * Haversine distance */ Haversine(13), /** * Bray-Curtis distance */ BrayCurtis(14), /** * Jensen-Shannon distance */ JensenShannon(15), /** * Hamming distance */ HammingUnexpanded(16), /** * KLDivergence */ KLDivergence(17), /** * RusselRao */ RusselRaoExpanded(18), /** * Dice-Sorensen distance */ DiceExpanded(19), /** * Precomputed (special value) */ Precomputed(100)
```

L1 distance *

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:135`_

### L2Unexpanded

```java
L2Unexpanded(4), /** * same as above, but inside the epilogue, perform square root operation */ L2SqrtUnexpanded(5), /** * basic inner product */ InnerProduct(6), /** * Chebyshev (Linf) distance */ Linf(7), /** * Canberra distance */ Canberra(8), /** * Generalized Minkowski distance */ LpUnexpanded(9), /** * Correlation distance */ CorrelationExpanded(10), /** * Jaccard distance */ JaccardExpanded(11), /** * Hellinger distance */ HellingerExpanded(12), /** * Haversine distance */ Haversine(13), /** * Bray-Curtis distance */ BrayCurtis(14), /** * Jensen-Shannon distance */ JensenShannon(15), /** * Hamming distance */ HammingUnexpanded(16), /** * KLDivergence */ KLDivergence(17), /** * RusselRao */ RusselRaoExpanded(18), /** * Dice-Sorensen distance */ DiceExpanded(19), /** * Precomputed (special value) */ Precomputed(100)
```

evaluate as dist_ij += (x_ik - y-jk)^2 *

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:139`_

### L2SqrtUnexpanded

```java
L2SqrtUnexpanded(5), /** * basic inner product */ InnerProduct(6), /** * Chebyshev (Linf) distance */ Linf(7), /** * Canberra distance */ Canberra(8), /** * Generalized Minkowski distance */ LpUnexpanded(9), /** * Correlation distance */ CorrelationExpanded(10), /** * Jaccard distance */ JaccardExpanded(11), /** * Hellinger distance */ HellingerExpanded(12), /** * Haversine distance */ Haversine(13), /** * Bray-Curtis distance */ BrayCurtis(14), /** * Jensen-Shannon distance */ JensenShannon(15), /** * Hamming distance */ HammingUnexpanded(16), /** * KLDivergence */ KLDivergence(17), /** * RusselRao */ RusselRaoExpanded(18), /** * Dice-Sorensen distance */ DiceExpanded(19), /** * Precomputed (special value) */ Precomputed(100)
```

same as above, but inside the epilogue, perform square root operation

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:143`_

### InnerProduct

```java
InnerProduct(6), /** * Chebyshev (Linf) distance */ Linf(7), /** * Canberra distance */ Canberra(8), /** * Generalized Minkowski distance */ LpUnexpanded(9), /** * Correlation distance */ CorrelationExpanded(10), /** * Jaccard distance */ JaccardExpanded(11), /** * Hellinger distance */ HellingerExpanded(12), /** * Haversine distance */ Haversine(13), /** * Bray-Curtis distance */ BrayCurtis(14), /** * Jensen-Shannon distance */ JensenShannon(15), /** * Hamming distance */ HammingUnexpanded(16), /** * KLDivergence */ KLDivergence(17), /** * RusselRao */ RusselRaoExpanded(18), /** * Dice-Sorensen distance */ DiceExpanded(19), /** * Precomputed (special value) */ Precomputed(100)
```

basic inner product

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:147`_

### Linf

```java
Linf(7), /** * Canberra distance */ Canberra(8), /** * Generalized Minkowski distance */ LpUnexpanded(9), /** * Correlation distance */ CorrelationExpanded(10), /** * Jaccard distance */ JaccardExpanded(11), /** * Hellinger distance */ HellingerExpanded(12), /** * Haversine distance */ Haversine(13), /** * Bray-Curtis distance */ BrayCurtis(14), /** * Jensen-Shannon distance */ JensenShannon(15), /** * Hamming distance */ HammingUnexpanded(16), /** * KLDivergence */ KLDivergence(17), /** * RusselRao */ RusselRaoExpanded(18), /** * Dice-Sorensen distance */ DiceExpanded(19), /** * Precomputed (special value) */ Precomputed(100)
```

Chebyshev (Linf) distance

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:151`_

### Canberra

```java
Canberra(8), /** * Generalized Minkowski distance */ LpUnexpanded(9), /** * Correlation distance */ CorrelationExpanded(10), /** * Jaccard distance */ JaccardExpanded(11), /** * Hellinger distance */ HellingerExpanded(12), /** * Haversine distance */ Haversine(13), /** * Bray-Curtis distance */ BrayCurtis(14), /** * Jensen-Shannon distance */ JensenShannon(15), /** * Hamming distance */ HammingUnexpanded(16), /** * KLDivergence */ KLDivergence(17), /** * RusselRao */ RusselRaoExpanded(18), /** * Dice-Sorensen distance */ DiceExpanded(19), /** * Precomputed (special value) */ Precomputed(100)
```

Canberra distance

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:155`_

### LpUnexpanded

```java
LpUnexpanded(9), /** * Correlation distance */ CorrelationExpanded(10), /** * Jaccard distance */ JaccardExpanded(11), /** * Hellinger distance */ HellingerExpanded(12), /** * Haversine distance */ Haversine(13), /** * Bray-Curtis distance */ BrayCurtis(14), /** * Jensen-Shannon distance */ JensenShannon(15), /** * Hamming distance */ HammingUnexpanded(16), /** * KLDivergence */ KLDivergence(17), /** * RusselRao */ RusselRaoExpanded(18), /** * Dice-Sorensen distance */ DiceExpanded(19), /** * Precomputed (special value) */ Precomputed(100)
```

Generalized Minkowski distance

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:159`_

### CorrelationExpanded

```java
CorrelationExpanded(10), /** * Jaccard distance */ JaccardExpanded(11), /** * Hellinger distance */ HellingerExpanded(12), /** * Haversine distance */ Haversine(13), /** * Bray-Curtis distance */ BrayCurtis(14), /** * Jensen-Shannon distance */ JensenShannon(15), /** * Hamming distance */ HammingUnexpanded(16), /** * KLDivergence */ KLDivergence(17), /** * RusselRao */ RusselRaoExpanded(18), /** * Dice-Sorensen distance */ DiceExpanded(19), /** * Precomputed (special value) */ Precomputed(100)
```

Correlation distance

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:163`_

### JaccardExpanded

```java
JaccardExpanded(11), /** * Hellinger distance */ HellingerExpanded(12), /** * Haversine distance */ Haversine(13), /** * Bray-Curtis distance */ BrayCurtis(14), /** * Jensen-Shannon distance */ JensenShannon(15), /** * Hamming distance */ HammingUnexpanded(16), /** * KLDivergence */ KLDivergence(17), /** * RusselRao */ RusselRaoExpanded(18), /** * Dice-Sorensen distance */ DiceExpanded(19), /** * Precomputed (special value) */ Precomputed(100)
```

Jaccard distance

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:167`_

### HellingerExpanded

```java
HellingerExpanded(12), /** * Haversine distance */ Haversine(13), /** * Bray-Curtis distance */ BrayCurtis(14), /** * Jensen-Shannon distance */ JensenShannon(15), /** * Hamming distance */ HammingUnexpanded(16), /** * KLDivergence */ KLDivergence(17), /** * RusselRao */ RusselRaoExpanded(18), /** * Dice-Sorensen distance */ DiceExpanded(19), /** * Precomputed (special value) */ Precomputed(100)
```

Hellinger distance

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:171`_

### Haversine

```java
Haversine(13), /** * Bray-Curtis distance */ BrayCurtis(14), /** * Jensen-Shannon distance */ JensenShannon(15), /** * Hamming distance */ HammingUnexpanded(16), /** * KLDivergence */ KLDivergence(17), /** * RusselRao */ RusselRaoExpanded(18), /** * Dice-Sorensen distance */ DiceExpanded(19), /** * Precomputed (special value) */ Precomputed(100)
```

Haversine distance

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:175`_

### BrayCurtis

```java
BrayCurtis(14), /** * Jensen-Shannon distance */ JensenShannon(15), /** * Hamming distance */ HammingUnexpanded(16), /** * KLDivergence */ KLDivergence(17), /** * RusselRao */ RusselRaoExpanded(18), /** * Dice-Sorensen distance */ DiceExpanded(19), /** * Precomputed (special value) */ Precomputed(100)
```

Bray-Curtis distance

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:179`_

### JensenShannon

```java
JensenShannon(15), /** * Hamming distance */ HammingUnexpanded(16), /** * KLDivergence */ KLDivergence(17), /** * RusselRao */ RusselRaoExpanded(18), /** * Dice-Sorensen distance */ DiceExpanded(19), /** * Precomputed (special value) */ Precomputed(100)
```

Jensen-Shannon distance

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:183`_

### HammingUnexpanded

```java
HammingUnexpanded(16), /** * KLDivergence */ KLDivergence(17), /** * RusselRao */ RusselRaoExpanded(18), /** * Dice-Sorensen distance */ DiceExpanded(19), /** * Precomputed (special value) */ Precomputed(100)
```

Hamming distance

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:187`_

### KLDivergence

```java
KLDivergence(17), /** * RusselRao */ RusselRaoExpanded(18), /** * Dice-Sorensen distance */ DiceExpanded(19), /** * Precomputed (special value) */ Precomputed(100)
```

KLDivergence

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:191`_

### RusselRaoExpanded

```java
RusselRaoExpanded(18), /** * Dice-Sorensen distance */ DiceExpanded(19), /** * Precomputed (special value) */ Precomputed(100)
```

RusselRao

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:195`_

### DiceExpanded

```java
DiceExpanded(19), /** * Precomputed (special value) */ Precomputed(100)
```

Dice-Sorensen distance

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:199`_

### Precomputed

```java
Precomputed(100)
```

Precomputed (special value)

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:203`_

### getIntermediateGraphDegree

```java
public long getIntermediateGraphDegree()
```

Gets the degree of input graph for pruning.

**Returns**

the degree of input graph

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:369`_

### getGraphDegree

```java
public long getGraphDegree()
```

Gets the degree of output graph.

**Returns**

the degree of output graph

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:378`_

### getCagraGraphBuildAlgo

```java
public CagraGraphBuildAlgo getCagraGraphBuildAlgo()
```

Gets the `CagraGraphBuildAlgo` used to build the index.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:385`_

### getNNDescentNumIterations

```java
public long getNNDescentNumIterations()
```

Gets the number of iterations to run if building with
`CagraGraphBuildAlgo#NN_DESCENT`

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:393`_

### getCuvsDistanceType

```java
public CuvsDistanceType getCuvsDistanceType()
```

Gets the `CuvsDistanceType` used to build the index.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:400`_

### getNumWriterThreads

```java
public int getNumWriterThreads()
```

Gets the number of threads used to build the index.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:407`_

### getCuVSIvfPqParams

```java
public CuVSIvfPqParams getCuVSIvfPqParams()
```

Gets the IVF_PQ parameters.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:414`_

### getCuVSAceParams

```java
public CuVSAceParams getCuVSAceParams()
```

Gets the ACE parameters.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:421`_

### getCuvsCagraGraphBuildAlgo

```java
public CagraGraphBuildAlgo getCuvsCagraGraphBuildAlgo()
```

Gets the CAGRA build algorithm.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:428`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:475`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:486`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:497`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:508`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:521`_

### withNumWriterThreads

```java
public Builder withNumWriterThreads(int numWriterThreads)
```

Sets the number of writer threads to use for indexing.

**Parameters**

| Name | Description |
| --- | --- |
| `numWriterThreads` | number of writer threads to use |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:532`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:543`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:554`_

### build

```java
public CagraIndexParams build()
```

Builds an instance of `CagraIndexParams`.

**Returns**

an instance of `CagraIndexParams`

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:564`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndexParams.java:18`_
