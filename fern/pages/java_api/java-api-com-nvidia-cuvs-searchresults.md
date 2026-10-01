---
slug: api-reference/java-api-com-nvidia-cuvs-searchresults
---

# SearchResults

_Java package: `com.nvidia.cuvs`_

```java
public interface SearchResults
```

## Public Members

### mappingsFromList

```java
static LongToIntFunction mappingsFromList(List<Integer> mappingAsList)
```

Creates a mapping function from a list lookup of custom user IDs

**Parameters**

| Name | Description |
| --- | --- |
| `mappingAsList` | a positional list of custom user IDs |

**Returns**

a function that maps the input ordinal to a custom user IDs, using the input as an index in the list

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/SearchResults.java:22`_

### getResults

```java
List<Map<Integer, Float>> getResults()
```

Gets a list results as a map of neighbor IDs to distances.

**Returns**

a list of results for each query as a map of neighbor IDs to distance

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/SearchResults.java:31`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/SearchResults.java:11`_
