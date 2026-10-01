# CuVS Java API benchmarks

This maven project contains JMH benchmarks for the CAGRA Java API.

## Prerequisites

- [CuVS libraries](https://docs.rapids.ai/api/cuvs/stable/build/#build-from-source)
- Build the CuVS-Java API (`./build.sh` from the parent directory)

## Run benchmarks

Build:

```shell
mvn clean verify
```

Run:

```shell
export RAFT_DEBUG_LOG_FILE=/dev/null
java -jar target/benchmarks.jar
```

The environment variable is needed to silence RAFT logging; RAFT emits some logs at INFO level when
building indices and queries, and writing them to stdout (the default) influences benchmark results.

It is possible to change the dataset size and the vectors dimension via 2 parameters:

```shell
java -jar target/benchmarks.jar -p size=4 -p dims=4
```

Use `java -jar target/benchmarks.jar -h` for details on the options to fine-tune your benchmark runs.
