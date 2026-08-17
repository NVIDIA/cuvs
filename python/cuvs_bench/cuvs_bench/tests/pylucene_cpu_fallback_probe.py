#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Fresh-process probe for the cuVS-Lucene HNSW CPU fallback."""

import os
import re
import tempfile
from pathlib import Path

import lucene
import numpy as np

from cuvs_bench.backends.pylucene import (
    _BuildCodec,
    _PyLuceneRuntime,
    _validate_pylucene_version,
)

_CPU_HNSW_WRITER = "org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsWriter"
_HNSW_CODEC = "Lucene101AcceleratedHNSWCodec"
_WRITER_SELECTION_CODEC = "com.nvidia.cuvs.bench.PyLuceneWriterSelectionCodec"


def _writer_diagnostics(java_codec):
    diagnostics = str(java_codec.knnVectorsFormat())
    match = re.search(r"writerClass=([^,)]+), fieldsWriterCalls=(\d+)", diagnostics)
    if match is None:
        raise AssertionError(f"Unexpected writer diagnostics: {diagnostics}")
    return match.group(1), int(match.group(2))


def main():
    _validate_pylucene_version(lucene)
    lucene.CLASSPATH = os.pathsep.join(
        (os.environ["PYLUCENE_WRITER_SELECTION_CLASSES"], lucene.CLASSPATH)
    )
    runtime = _PyLuceneRuntime.create(
        {
            "cuvs_java_jar": os.environ["CUVS_LUCENE_CUVS_JAVA_JAR"],
            "cuvs_lucene_jar": os.environ["CUVS_LUCENE_JAR"],
            "java_library_path": os.environ["JAVA_LIBRARY_PATH"],
        }
    )
    reflected_codec = runtime.Class.forName(_WRITER_SELECTION_CODEC).newInstance()
    java_codec = runtime.Codec.cast_(reflected_codec)
    vectors = np.random.default_rng(174).standard_normal((4, 32)).astype(np.float32)

    with tempfile.TemporaryDirectory(prefix="pylucene-cpu-fallback-") as temp:
        index_path = Path(temp)
        runtime.build_index(
            index_path,
            vectors,
            _BuildCodec(
                codec_name=_HNSW_CODEC,
                java_codec=java_codec,
                writer_policy="gpu-with-cpu-fallback",
            ),
        )
        writer_class, writer_calls = _writer_diagnostics(java_codec)
        if writer_class != _CPU_HNSW_WRITER:
            raise AssertionError(f"Expected {_CPU_HNSW_WRITER}, found {writer_class}")

        search = runtime.search_index(
            index_path,
            vectors[[2]],
            k=1,
            batch_size=1,
        )
        first_hit = search.hits[0][0].document_id

    if first_hit != 2:
        raise AssertionError(f"Expected document 2, found {first_hit}")
    print(
        f"writerClass={writer_class} fieldsWriterCalls={writer_calls} "
        f"firstHit={first_hit}"
    )


if __name__ == "__main__":
    main()
