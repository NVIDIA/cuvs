# CuVS Java API Examples

This maven project contains examples for CAGRA, HNSW, and Bruteforce algorithms.

## Prerequisites

- [CuVS libraries](https://docs.rapids.ai/api/cuvs/stable/build/#build-from-source)
- Build the CuVS-Java API

## Run Examples

### CAGRA Example

In the current directory do:

```
mvn package && java --enable-native-access=ALL-UNNAMED -cp target/cuvs-java-examples-26.10.0.jar:$HOME/.m2/repository/com/nvidia/cuvs/cuvs-java/26.10.0/cuvs-java-26.10.0.jar com.nvidia.cuvs.examples.CagraExample
```

### HNSW Example

In the current directory do:

```
mvn package && java --enable-native-access=ALL-UNNAMED -cp target/cuvs-java-examples-26.10.0.jar:$HOME/.m2/repository/com/nvidia/cuvs/cuvs-java/26.10.0/cuvs-java-26.10.0.jar com.nvidia.cuvs.examples.HnswExample
```

### Bruteforce Example

In the current directory do:

```
mvn package && java --enable-native-access=ALL-UNNAMED -cp target/cuvs-java-examples-26.10.0.jar:$HOME/.m2/repository/com/nvidia/cuvs/cuvs-java/26.10.0/cuvs-java-26.10.0.jar com.nvidia.cuvs.examples.BruteForceExample
```
