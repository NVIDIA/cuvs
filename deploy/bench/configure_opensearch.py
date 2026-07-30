#!/usr/bin/env python3
"""Configure OpenSearch and write the cuvs-bench backend configuration."""

import argparse
import os

import requests
import yaml


def _optional_int(name: str) -> int | None:
    value = os.environ.get(name, "").strip()
    return int(value) if value else None


def create_backend_config() -> dict:
    remote_index_build = (
        os.environ.get("REMOTE_INDEX_BUILD", "false").lower() == "true"
    )
    number_of_shards = int(os.environ.get("NUMBER_OF_SHARDS", "1"))
    if number_of_shards < 1:
        raise ValueError("NUMBER_OF_SHARDS must be at least 1")

    approximate_threshold = _optional_int("APPROXIMATE_THRESHOLD")
    if approximate_threshold is not None and approximate_threshold < -1:
        raise ValueError("APPROXIMATE_THRESHOLD must be -1 or greater")

    config = {
        "backend": "opensearch",
        "host": os.environ.get("OPENSEARCH_HOST", "opensearch"),
        "port": int(os.environ.get("OPENSEARCH_PORT", "9200")),
        "use_ssl": False,
        "verify_certs": False,
        "number_of_shards": number_of_shards,
        "remote_index_build": remote_index_build,
    }
    if approximate_threshold is not None:
        config["approximate_threshold"] = approximate_threshold

    build_batch_size = _optional_int("BUILD_BATCH_SIZE")
    if build_batch_size is not None:
        config["build_batch_size"] = build_batch_size

    refresh_interval = os.environ.get("REFRESH_INTERVAL", "").strip()
    if refresh_interval:
        config["refresh_interval"] = refresh_interval

    if remote_index_build:
        config["remote_build_timeout"] = int(
            os.environ.get("REMOTE_BUILD_TIMEOUT", "1800")
        )
        remote_build_size_min = os.environ.get(
            "REMOTE_BUILD_SIZE_MIN", ""
        ).strip()
        if remote_build_size_min:
            config["remote_build_size_min"] = remote_build_size_min

    return config


def configure_cluster() -> None:
    opensearch_url = os.environ.get(
        "OPENSEARCH_URL", "http://opensearch:9200"
    )
    remote_index_build = (
        os.environ.get("REMOTE_INDEX_BUILD", "false").lower() == "true"
    )
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})

    if remote_index_build:
        bucket = os.environ.get("S3_BUCKET", "").strip()
        if not bucket:
            raise ValueError(
                "S3_BUCKET must be set when REMOTE_INDEX_BUILD=true"
            )
        repository = (
            os.environ.get("REMOTE_VECTOR_REPOSITORY", "vector-repo").strip()
            or "vector-repo"
        )
        response = session.put(
            f"{opensearch_url}/_snapshot/{repository}",
            json={
                "type": "s3",
                "settings": {
                    "bucket": bucket,
                    "base_path": (
                        os.environ.get("S3_PREFIX", "knn-indexes").strip()
                        or "knn-indexes"
                    ),
                    "region": os.environ.get(
                        "AWS_DEFAULT_REGION", "us-west-2"
                    ),
                },
            },
        )
        response.raise_for_status()
        settings = {
            "knn.remote_index_build.enabled": True,
            "knn.remote_index_build.repository": repository,
            "knn.remote_index_build.service.endpoint": os.environ.get(
                "BUILDER_URL", "http://remote-index-builder:1025"
            ),
        }
    else:
        settings = {"knn.remote_index_build.enabled": False}

    response = session.put(
        f"{opensearch_url}/_cluster/settings",
        json={"persistent": settings},
    )
    response.raise_for_status()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    args = parser.parse_args()

    configure_cluster()
    with open(args.output, "w") as file:
        yaml.safe_dump(create_backend_config(), file, sort_keys=False)


if __name__ == "__main__":
    main()
