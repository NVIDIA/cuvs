# Lucene cuVS

This is a project for using [cuVS](https://github.com/rapidsai/cuvs), NVIDIA's GPU accelerated vector search library, with [Apache Lucene](https://github.com/apache/lucene).

## Overview

This library provides a new [KnnVectorFormat](https://lucene.apache.org/core/10_3_1/core/org/apache/lucene/codecs/KnnVectorsFormat.html) which can be plugged into a Lucene codec.

## Building

### Prerequisites
- [CUDA 12.0+](https://developer.nvidia.com/cuda-toolkit-archive),
- [Maven 3.9.6+](https://maven.apache.org/download.cgi),
- [JDK 22](https://jdk.java.net/archive/)

```sh
mvn clean compile package
```
The artifacts would be built and available in the target / folder.

### Running Tests
```sh
export LD_LIBRARY_PATH={ PATH TO YOUR LOCAL libcuvs_c.so }:$LD_LIBRARY_PATH && mvn clean test
```

## Contributing

> [!NOTE]
> The code style format is automatically enforced (including the missing license header, if any) using the [Spotless maven plugin](https://github.com/diffplug/spotless/tree/main/plugin-maven). This currently happens in the maven's `validate` stage.
