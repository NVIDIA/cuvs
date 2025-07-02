# Lucene CuVS Integration

This is an integration for [CuVS](https://github.com/rapidsai/cuvs), GPU accelerated vector search library from NVIDIA (formerly part of [Raft](https://github.com/rapidsai/raft)), into [Apache Lucene](https://github.com/apache/lucene).

## Overview

The CuVS library is plugged in as a new KnnVectorFormat via a custom codec.

![Architecture](lucene-cuvs-architecture.png "Lucene CuVS Architecture")

> :warning: This is not production ready yet.

## Building

Install NVIDIA drivers, CUDA 12.8, Maven 3.9.6+ and JDK 22.

    mvn clean compile package

The artifacts would be built and available in target/ folder.

## Contributors

* Vivek Narang, SearchScale
* Ishan Chattopadhyaya, SearchScale & Committer, Apache Lucene & Solr
* Puneet Ahuja, SearchScale
* Corey Nolet, NVIDIA
