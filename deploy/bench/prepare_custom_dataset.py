#!/usr/bin/env python3
"""Download the custom cuvs-bench dataset described by its S3 YAML file."""

import os
from pathlib import Path, PurePosixPath

import boto3
import yaml
from botocore.exceptions import ClientError


DATASET_NAME = "miracl-en-5m-1024d-fp32"
S3_BUCKET = "opensearch-cuvs-bench"
S3_PREFIX = "miracl-en-5m-1024d-fp32"


def download_if_needed(s3, bucket: str, key: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"Using existing {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading s3://{bucket}/{key} -> {destination}")
    try:
        s3.download_file(bucket, key, str(temporary))
    except ClientError as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download s3://{bucket}/{key}. The dataset YAML "
            "references this file, so it is required by the benchmark."
        ) from error
    temporary.replace(destination)


def main() -> None:
    dataset_root = Path(os.environ.get("DATASET_PATH", "/data/datasets"))
    region = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
    s3 = boto3.client("s3", region_name=region)

    config_path = dataset_root / DATASET_NAME / "config.yaml"
    download_if_needed(
        s3, S3_BUCKET, f"{S3_PREFIX}/config.yaml", config_path
    )

    with config_path.open() as config_file:
        configs = yaml.safe_load(config_file)
    if not isinstance(configs, list):
        raise ValueError(f"Expected a list in {config_path}")

    try:
        config = next(item for item in configs if item["name"] == DATASET_NAME)
    except (KeyError, StopIteration) as error:
        raise ValueError(
            f"Dataset {DATASET_NAME!r} was not found in {config_path}"
        ) from error

    file_fields = (
        "base_file",
        "query_file",
        "groundtruth_neighbors_file",
        "groundtruth_distances_file",
    )
    for field in file_fields:
        relative_path = config.get(field)
        if not relative_path:
            continue
        relative_path = PurePosixPath(relative_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"Unsafe {field} path in {config_path}: {relative_path}"
            )
        download_if_needed(
            s3,
            S3_BUCKET,
            f"{S3_PREFIX}/{relative_path.name}",
            dataset_root.joinpath(*relative_path.parts),
        )

    print(f"Custom dataset is ready: {DATASET_NAME}")
    print(f"Dataset configuration: {config_path}")


if __name__ == "__main__":
    main()
