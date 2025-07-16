# Lucene CuVS Integration

This is a codec for connecting [CuVS](https://github.com/rapidsai/cuvs), NVIDIA's GPU accelerated vector search library, into [Apache Lucene](https://github.com/apache/lucene).

## Overview

The CuVS library is plugged in as a new KnnVectorFormat via a custom codec.

> :warning: This is not production ready yet.

## Building

Install NVIDIA drivers, CUDA 12.8, Maven 3.9.6+ and JDK 22.

    mvn clean compile package

The artifacts would be built and available in target/ folder.
