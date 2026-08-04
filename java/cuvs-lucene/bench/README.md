# Microbenchmarks

This maven project provides microbenchmarks for `cuvs-lucene` using JMH.

To run do (if on current directory):

```sh
cd .. && mvn clean install -DskipTests && cd benchmarks && mvn clean install && java -jar target/benchmarks.jar
```
