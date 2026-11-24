# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tested runtime defaults and separately maintained saved-index identities.

Only PyLucene 10.2.0 is supported. This module does not discover installations
or infer compatibility with other versions. Codec generations and persisted
format identifiers are not interchangeable with Lucene release numbers.
"""

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class LuceneCompatibility:
    """The runtime and writing codecs validated together for one release."""

    pylucene_version: str
    cpu_hnsw_codec: str
    accelerated_hnsw_codec: str
    cagra_codec: str

    @property
    def module_directory(self) -> str:
        minor_version = self.pylucene_version.rsplit(".", 1)[0]
        return f"lucene-{minor_version}"

    @property
    def artifact_id(self) -> str:
        return f"cuvs-{self.module_directory}"


LUCENE_COMPATIBILITY = LuceneCompatibility(
    pylucene_version="10.2.0",
    cpu_hnsw_codec="Lucene101",
    accelerated_hnsw_codec="Lucene101AcceleratedHNSWCodec",
    cagra_codec="CuVS2510GPUSearchCodec",
)

# Saved identities are independent of the default writer. A future writer
# change must not silently rename existing indexes or promise new read support.
PERSISTED_CODEC_KINDS = MappingProxyType(
    {
        "Lucene101": "cpu_hnsw",
        "Lucene101AcceleratedHNSWCodec": "hnsw",
        "CuVS2510GPUSearchCodec": "gpu_cagra",
    }
)
SAVED_CUVS_LUCENE_ARTIFACT_IDS = (
    "cuvs-lucene",
    "cuvs-lucene-10.2",
)
