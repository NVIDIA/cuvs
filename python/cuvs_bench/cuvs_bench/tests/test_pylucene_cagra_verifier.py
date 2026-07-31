#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Persisted CAGRA verifier unit tests."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

import cuvs_bench.backends.pylucene as pylucene_backend


class _FakeCagraMetadataInput:
    def __init__(self, integers, variable_longs):
        self._integers = iter(integers)
        self._variable_longs = iter(variable_longs)

    def readInt(self):
        return next(self._integers)

    def readVLong(self):
        return next(self._variable_longs)


class _FakeChecksumInput(_FakeCagraMetadataInput):
    def __init__(self, integers, variable_longs):
        super().__init__(integers, variable_longs)
        self.closed = False

    def close(self):
        self.closed = True


class _FakeIndexInput:
    def __init__(self, payload_start=57, payload_length=1000):
        self.payload_start = payload_start
        self.file_length = payload_start + payload_length + 16
        self.closed = False

    def getFilePointer(self):
        return self.payload_start

    def length(self):
        return self.file_length

    def close(self):
        self.closed = True


class _FakeCodecUtil:
    def __init__(self, error_at=None):
        self.error_at = error_at
        self.header_calls = []
        self.footer_calls = []
        self.checksum_calls = []
        self.retrieved_checksum_calls = []

    def checkIndexHeader(self, *args):
        self.header_calls.append(args)
        if self.error_at == "header":
            raise RuntimeError("unsupported metadata version")
        return 0

    def checkFooter(self, metadata_input):
        self.footer_calls.append(metadata_input)
        if self.error_at == "footer":
            raise RuntimeError("invalid checksum footer")

    def footerLength(self):
        return 16

    def checksumEntireFile(self, data_input):
        self.checksum_calls.append(data_input)
        if self.error_at == "data-checksum":
            raise RuntimeError("invalid data checksum")
        return 0

    def retrieveChecksum(self, data_input):
        self.retrieved_checksum_calls.append(data_input)
        return 0


class _FakeFieldInfos:
    def __init__(self, field_infos):
        self._field_infos = {
            field_info.number: field_info for field_info in field_infos
        }

    def fieldInfo(self, field_number):
        return self._field_infos.get(field_number)

    def __iter__(self):
        return iter(self._field_infos.values())


def _fake_cagra_index_verifier(
    metadata_input,
    *,
    metadata_files=("_0.vemc",),
    codec_util=None,
    data_payload_length=1000,
    field_name="vector",
    field_dimensions=32,
    field_updates=False,
    max_documents=4,
    deletion_count=0,
    soft_deletion_count=0,
    extra_vector_field=False,
    data_error=None,
):
    data_input = _FakeIndexInput(payload_length=data_payload_length)
    vector_encoding = object()
    vector_similarity = object()
    field_info = SimpleNamespace(
        number=1,
        getName=lambda: field_name,
        getVectorDimension=lambda: field_dimensions,
        getVectorEncoding=lambda: vector_encoding,
        getVectorSimilarityFunction=lambda: vector_similarity,
    )
    all_field_infos = [field_info]
    if extra_vector_field:
        all_field_infos.append(
            SimpleNamespace(
                number=2,
                getName=lambda: "extra-vector",
                getVectorDimension=lambda: field_dimensions,
                getVectorEncoding=lambda: vector_encoding,
                getVectorSimilarityFunction=lambda: vector_similarity,
            )
        )
    field_infos = _FakeFieldInfos(all_field_infos)
    field_infos_format = SimpleNamespace(
        read=lambda _directory, _info, _suffix, _context: field_infos
    )

    def open_data(_file_name, _context):
        if data_error is not None:
            raise data_error
        return data_input

    directory = SimpleNamespace(
        openChecksumInput=lambda _file_name: metadata_input,
        openInput=open_data,
        close=lambda: None,
    )
    segment_info = SimpleNamespace(
        info=SimpleNamespace(
            name="_0",
            getId=lambda: b"segment-id",
            getCodec=lambda: SimpleNamespace(
                fieldInfosFormat=lambda: field_infos_format
            ),
            maxDoc=lambda: max_documents,
        ),
        files=lambda: metadata_files,
        hasFieldUpdates=lambda: field_updates,
        hasDeletions=lambda: deletion_count > 0,
        getDelCount=lambda: deletion_count,
        getSoftDelCount=lambda: soft_deletion_count,
    )
    verifier = pylucene_backend._CagraIndexVerifier(
        attach_current_thread=lambda: None,
        paths=SimpleNamespace(get=lambda path: path),
        codec_util=codec_util or _FakeCodecUtil(),
        field_info=SimpleNamespace(
            cast_=lambda raw_field_info: raw_field_info
        ),
        segment_commit_info=SimpleNamespace(
            cast_=lambda raw_segment_info: raw_segment_info
        ),
        segment_infos=SimpleNamespace(
            readLatestCommit=lambda _directory: [segment_info]
        ),
        vector_encoding=SimpleNamespace(FLOAT32=vector_encoding),
        vector_similarity_function=SimpleNamespace(
            EUCLIDEAN=vector_similarity
        ),
        fs_directory=SimpleNamespace(open=lambda _path: directory),
        io_context=SimpleNamespace(READONCE=object()),
    )
    verifier._test_data_input = data_input
    verifier._test_directory = directory
    verifier._test_segment_info = segment_info
    return verifier


def _cagra_data_context(verifier, index_path):
    return pylucene_backend._CagraDataFileContext(
        index_path=index_path,
        directory=verifier._test_directory,
        segment_info=verifier._test_segment_info,
        suffix="",
        metadata_file="_0.vemc",
    )


@pytest.mark.parametrize(
    ("metadata_file", "expected_suffix"),
    [("_0.vemc", ""), ("_0_CuVS_0.vemc", "CuVS_0")],
)
def test_cagra_segment_suffix(metadata_file, expected_suffix):
    assert (
        pylucene_backend._CagraIndexVerifier._segment_suffix(
            "_0", metadata_file
        )
        == expected_suffix
    )


def test_cagra_segment_suffix_rejects_unrelated_metadata_file():
    with pytest.raises(RuntimeError, match="does not match segment"):
        pylucene_backend._CagraIndexVerifier._segment_suffix("_0", "_1.vemc")


def test_read_cagra_fields_accepts_only_persisted_cagra_data():
    metadata_input = _FakeCagraMetadataInput(
        integers=[
            1,
            1,
            0,
            32,
            4,
            2,
            1,
            0,
            32,
            0,
            -1,
        ],
        variable_longs=[57, 1000, 1057, 0, 0, 0, 0, 0],
    )

    assert pylucene_backend._CagraIndexVerifier._read_cagra_fields(
        metadata_input, "_0.vemc"
    ) == [
        pylucene_backend._CagraFieldMetadata(
            field_number=1,
            dimensions=32,
            vector_count=4,
            cagra_offset=57,
            cagra_length=1000,
        )
    ]


@pytest.mark.parametrize(
    ("cagra_length", "brute_force_length", "error"),
    [
        (0, 512, "persisted brute-force"),
        (1000, 512, "persisted brute-force"),
    ],
)
def test_read_cagra_fields_rejects_non_cagra_only_data(
    cagra_length, brute_force_length, error
):
    metadata_input = _FakeCagraMetadataInput(
        integers=[1, 1, 0, 32, 4, -1],
        variable_longs=[
            57,
            cagra_length,
            57 + cagra_length,
            brute_force_length,
        ],
    )

    with pytest.raises(RuntimeError, match=error):
        pylucene_backend._CagraIndexVerifier._read_cagra_fields(
            metadata_input, "_0.vemc"
        )


def test_read_cagra_fields_rejects_data_for_empty_field():
    metadata_input = _FakeCagraMetadataInput(
        integers=[1, 1, 0, 32, 0, -1],
        variable_longs=[57, 1, 58, 0],
    )

    with pytest.raises(RuntimeError, match="empty field"):
        pylucene_backend._CagraIndexVerifier._read_cagra_fields(
            metadata_input, "_0.vemc"
        )


@pytest.mark.parametrize(
    ("encoding", "similarity", "dimensions", "error"),
    [
        (0, 0, 32, "encoding ordinal 0"),
        (1, 1, 32, "similarity ordinal 1"),
        (1, 0, 0, "invalid field metadata"),
        (1, 0, 4097, "invalid field metadata"),
    ],
)
def test_read_cagra_fields_rejects_unsupported_vector_semantics(
    encoding, similarity, dimensions, error
):
    metadata_input = _FakeCagraMetadataInput(
        integers=[1, encoding, similarity, dimensions, 4, -1],
        variable_longs=[57, 1000, 1057, 0],
    )

    with pytest.raises(RuntimeError, match=error):
        pylucene_backend._CagraIndexVerifier._read_cagra_fields(
            metadata_input, "_0.vemc"
        )


def test_verify_cagra_index_validates_header_footer_and_expected_count(
    tmp_path,
):
    metadata_input = _FakeChecksumInput(
        integers=[1, 1, 0, 32, 4, -1],
        variable_longs=[57, 1000, 1057, 0],
    )
    codec_util = _FakeCodecUtil()
    verifier = _fake_cagra_index_verifier(
        metadata_input, codec_util=codec_util
    )

    verification = verifier.verify_index(
        tmp_path, expected_vector_count=4, expected_dimensions=32
    )

    assert verification.to_metadata() == {
        "status": "cagra-only",
        "segment_count": 1,
        "field_count": 1,
        "vector_count": 4,
        "dimensions": 32,
    }
    assert codec_util.header_calls == [
        (
            metadata_input,
            "Lucene102CuVSVectorsFormatMeta",
            0,
            0,
            b"segment-id",
            "",
        ),
        (
            verifier._test_data_input,
            "Lucene102CuVSVectorsFormatIndex",
            0,
            0,
            b"segment-id",
            "",
        ),
    ]
    assert codec_util.footer_calls == [metadata_input]
    assert codec_util.checksum_calls == [verifier._test_data_input]
    assert metadata_input.closed is True
    assert verifier._test_data_input.closed is True


def test_verify_cagra_index_traverses_segments_and_closes_each_input(
    tmp_path,
):
    events = []
    metadata_inputs = {
        name: _FakeChecksumInput(
            integers=[1, 1, 0, 32, 4, -1],
            variable_longs=[57, 1000, 1057, 0],
        )
        for name in ("_0.vemc", "_1.vemc")
    }
    data_inputs = {name: _FakeIndexInput() for name in ("_0.vcag", "_1.vcag")}
    verifier = _fake_cagra_index_verifier(metadata_inputs["_0.vemc"])

    def segment(name):
        field_info = SimpleNamespace(
            number=1,
            getName=lambda: "vector",
            getVectorDimension=lambda: 32,
            getVectorEncoding=lambda: verifier.VectorEncoding.FLOAT32,
            getVectorSimilarityFunction=(
                lambda: verifier.VectorSimilarityFunction.EUCLIDEAN
            ),
        )
        field_infos_format = SimpleNamespace(
            read=lambda _directory, _info, _suffix, _context: (
                _FakeFieldInfos([field_info])
            )
        )
        return SimpleNamespace(
            info=SimpleNamespace(
                name=name,
                getId=lambda: name.encode(),
                getCodec=lambda: SimpleNamespace(
                    fieldInfosFormat=lambda: field_infos_format
                ),
                maxDoc=lambda: 4,
            ),
            files=lambda: (f"{name}.vemc",),
            hasFieldUpdates=lambda: False,
            hasDeletions=lambda: False,
            getDelCount=lambda: 0,
            getSoftDelCount=lambda: 0,
        )

    def open_metadata(file_name):
        events.append(f"open {file_name}")
        metadata_input = metadata_inputs[file_name]

        def close_metadata():
            metadata_input.closed = True
            events.append(f"close {file_name}")

        metadata_input.close = close_metadata
        return metadata_input

    def open_data(file_name, _context):
        events.append(f"open {file_name}")
        data_input = data_inputs[file_name]

        def close_data():
            data_input.closed = True
            events.append(f"close {file_name}")

        data_input.close = close_data
        return data_input

    verifier.SegmentInfos.readLatestCommit = lambda _directory: [
        segment("_0"),
        segment("_1"),
    ]
    verifier._test_directory.openChecksumInput = open_metadata
    verifier._test_directory.openInput = open_data
    verifier._test_directory.close = lambda: events.append("close directory")

    verification = verifier.verify_index(tmp_path)

    assert verification == pylucene_backend._CagraIndexVerification(
        segment_count=2,
        field_count=2,
        vector_count=8,
        dimensions=32,
    )
    assert events == [
        "open _0.vemc",
        "close _0.vemc",
        "open _0.vcag",
        "close _0.vcag",
        "open _1.vemc",
        "close _1.vemc",
        "open _1.vcag",
        "close _1.vcag",
        "close directory",
    ]
    assert all(item.closed for item in metadata_inputs.values())
    assert all(item.closed for item in data_inputs.values())


@pytest.mark.parametrize("error_at", ["header", "footer"])
def test_verify_cagra_index_fails_closed_on_invalid_format(error_at, tmp_path):
    metadata_input = _FakeChecksumInput(
        integers=[1, 1, 0, 32, 4, -1],
        variable_longs=[57, 1000, 1057, 0],
    )
    verifier = _fake_cagra_index_verifier(
        metadata_input, codec_util=_FakeCodecUtil(error_at=error_at)
    )

    with pytest.raises(RuntimeError, match="metadata format v0"):
        verifier.verify_index(tmp_path)

    assert metadata_input.closed is True


def test_verify_cagra_index_rejects_missing_segment_metadata(tmp_path):
    verifier = _fake_cagra_index_verifier(
        _FakeChecksumInput([], []), metadata_files=()
    )

    with pytest.raises(RuntimeError, match="no .vemc metadata"):
        verifier.verify_index(tmp_path)


def test_verify_cagra_index_rejects_only_empty_fields(tmp_path):
    verifier = _fake_cagra_index_verifier(
        _FakeChecksumInput(
            integers=[1, 1, 0, 32, 0, -1],
            variable_longs=[0, 0, 0, 0],
        ),
        data_payload_length=0,
    )

    with pytest.raises(RuntimeError, match="without matching CAGRA metadata"):
        verifier.verify_index(tmp_path)


def test_verify_cagra_index_rejects_vector_count_mismatch(tmp_path):
    verifier = _fake_cagra_index_verifier(
        _FakeChecksumInput(
            integers=[1, 1, 0, 32, 4, -1],
            variable_longs=[57, 1000, 1057, 0],
        )
    )

    with pytest.raises(RuntimeError, match="4 vectors; expected 5"):
        verifier.verify_index(tmp_path, expected_vector_count=5)


def test_verify_cagra_index_rejects_dimension_mismatch(tmp_path):
    verifier = _fake_cagra_index_verifier(
        _FakeChecksumInput(
            integers=[1, 1, 0, 32, 4, -1],
            variable_longs=[57, 1000, 1057, 0],
        )
    )

    with pytest.raises(RuntimeError, match="32 dimensions; expected 16"):
        verifier.verify_index(tmp_path, expected_dimensions=16)


@pytest.mark.parametrize(
    ("runtime_kwargs", "expected_counts"),
    [
        ({"deletion_count": 1}, "deleted=1, soft_deleted=0"),
        ({"soft_deletion_count": 1}, "deleted=0, soft_deleted=1"),
    ],
    ids=["hard-deletion", "soft-deletion"],
)
def test_verify_cagra_index_rejects_committed_deletions(
    tmp_path, runtime_kwargs, expected_counts
):
    verifier = _fake_cagra_index_verifier(
        _FakeChecksumInput(
            integers=[1, 1, 0, 32, 4, -1],
            variable_longs=[57, 1000, 1057, 0],
        ),
        **runtime_kwargs,
    )

    with pytest.raises(RuntimeError, match=expected_counts):
        verifier.verify_index(tmp_path)


def test_verify_cagra_index_rejects_segment_document_count_mismatch(tmp_path):
    verifier = _fake_cagra_index_verifier(
        _FakeChecksumInput(
            integers=[1, 1, 0, 32, 4, -1],
            variable_longs=[57, 1000, 1057, 0],
        ),
        max_documents=5,
    )

    with pytest.raises(RuntimeError, match="4 vectors for 5 documents"):
        verifier.verify_index(tmp_path)


def test_verify_cagra_index_rejects_unaccounted_vector_field(tmp_path):
    verifier = _fake_cagra_index_verifier(
        _FakeChecksumInput(
            integers=[1, 1, 0, 32, 4, -1],
            variable_longs=[57, 1000, 1057, 0],
        ),
        extra_vector_field=True,
    )

    with pytest.raises(RuntimeError, match=r"metadata=\[1\], Lucene=\[1, 2\]"):
        verifier.verify_index(tmp_path)


def test_verify_cagra_index_rejects_duplicate_field_across_metadata_files(
    tmp_path,
):
    metadata_inputs = {
        name: _FakeChecksumInput(
            integers=[1, 1, 0, 32, 4, -1],
            variable_longs=[57, 1000, 1057, 0],
        )
        for name in ("_0.vemc", "_0_CuVS_0.vemc")
    }
    verifier = _fake_cagra_index_verifier(
        metadata_inputs["_0.vemc"],
        metadata_files=tuple(metadata_inputs),
    )
    verifier._test_directory.openChecksumInput = metadata_inputs.__getitem__
    verifier._test_directory.openInput = (
        lambda _file_name, _context: _FakeIndexInput()
    )

    with pytest.raises(RuntimeError, match="duplicate field 1"):
        verifier.verify_index(tmp_path)


def test_verify_cagra_index_rejects_inconsistent_dimensions_across_segments(
    tmp_path,
):
    metadata_inputs = {
        "_0.vemc": _FakeChecksumInput(
            integers=[1, 1, 0, 32, 4, -1],
            variable_longs=[57, 1000, 1057, 0],
        ),
        "_1.vemc": _FakeChecksumInput(
            integers=[1, 1, 0, 16, 4, -1],
            variable_longs=[57, 1000, 1057, 0],
        ),
    }
    verifier = _fake_cagra_index_verifier(metadata_inputs["_0.vemc"])

    def segment(name, dimensions):
        field_info = SimpleNamespace(
            number=1,
            getName=lambda: "vector",
            getVectorDimension=lambda: dimensions,
            getVectorEncoding=lambda: verifier.VectorEncoding.FLOAT32,
            getVectorSimilarityFunction=(
                lambda: verifier.VectorSimilarityFunction.EUCLIDEAN
            ),
        )
        field_infos = _FakeFieldInfos([field_info])
        field_infos_format = SimpleNamespace(
            read=lambda _directory, _info, _suffix, _context: field_infos
        )
        return SimpleNamespace(
            info=SimpleNamespace(
                name=name,
                getId=lambda: name.encode(),
                getCodec=lambda: SimpleNamespace(
                    fieldInfosFormat=lambda: field_infos_format
                ),
                maxDoc=lambda: 4,
            ),
            files=lambda: (f"{name}.vemc",),
            hasFieldUpdates=lambda: False,
            hasDeletions=lambda: False,
            getDelCount=lambda: 0,
            getSoftDelCount=lambda: 0,
        )

    segments = [segment("_0", 32), segment("_1", 16)]
    verifier.SegmentInfos.readLatestCommit = lambda _directory: segments
    verifier._test_directory.openChecksumInput = metadata_inputs.__getitem__
    verifier._test_directory.openInput = (
        lambda _file_name, _context: _FakeIndexInput()
    )

    with pytest.raises(RuntimeError, match=r"dimensions: \[16, 32\]"):
        verifier.verify_index(tmp_path)


@pytest.mark.parametrize(
    ("runtime_kwargs", "error"),
    [
        ({"field_name": "other"}, "unexpected field"),
        ({"field_dimensions": 16}, "field metadata"),
        ({"field_updates": True}, "field-info updates"),
        ({"data_payload_length": 999}, "do not exactly cover"),
    ],
)
def test_verify_cagra_index_rejects_foreign_or_inconsistent_index(
    tmp_path, runtime_kwargs, error
):
    verifier = _fake_cagra_index_verifier(
        _FakeChecksumInput(
            integers=[1, 1, 0, 32, 4, -1],
            variable_longs=[57, 1000, 1057, 0],
        ),
        **runtime_kwargs,
    )

    with pytest.raises(RuntimeError, match=error):
        verifier.verify_index(tmp_path)


def test_verify_cagra_index_rejects_corrupt_data_file(tmp_path):
    verifier = _fake_cagra_index_verifier(
        _FakeChecksumInput(
            integers=[1, 1, 0, 32, 4, -1],
            variable_longs=[57, 1000, 1057, 0],
        ),
        codec_util=_FakeCodecUtil(error_at="data-checksum"),
    )

    with pytest.raises(RuntimeError, match="invalid data checksum"):
        verifier.verify_index(tmp_path)

    assert verifier._test_data_input.closed is True


def test_verify_cagra_index_rejects_missing_data_file(tmp_path):
    verifier = _fake_cagra_index_verifier(
        _FakeChecksumInput(
            integers=[1, 1, 0, 32, 4, -1],
            variable_longs=[57, 1000, 1057, 0],
        ),
        data_error=FileNotFoundError("_0.vcag"),
    )

    with pytest.raises(RuntimeError, match="cannot read '_0.vcag'"):
        verifier.verify_index(tmp_path)


def test_file_signature_exposes_named_stat_fields(tmp_path):
    data_path = tmp_path / "data.vcag"
    data_path.write_bytes(b"data")
    file_stat = data_path.stat()

    signature = pylucene_backend._file_signature(data_path)

    assert signature == pylucene_backend._FileSignature(
        resolved_path=str(data_path.resolve()),
        device=file_stat.st_dev,
        inode=file_stat.st_ino,
        size=file_stat.st_size,
        modified_at_ns=file_stat.st_mtime_ns,
        changed_at_ns=file_stat.st_ctime_ns,
    )


def test_cagra_data_checksum_is_cached_for_unchanged_file(
    tmp_path, monkeypatch
):
    codec_util = _FakeCodecUtil()
    verifier = _fake_cagra_index_verifier(
        _FakeChecksumInput([], []), codec_util=codec_util
    )
    (tmp_path / "_0.vcag").write_bytes(b"data")
    data_ctime_ns = (tmp_path / "_0.vcag").stat().st_ctime_ns
    monkeypatch.setattr(
        pylucene_backend.time,
        "time_ns",
        lambda: (
            data_ctime_ns + pylucene_backend._CAGRA_CACHE_MIN_FILE_AGE_NS + 1
        ),
    )
    fields = [
        pylucene_backend._CagraFieldMetadata(
            field_number=1,
            dimensions=32,
            vector_count=4,
            cagra_offset=57,
            cagra_length=1000,
        )
    ]
    context = _cagra_data_context(verifier, tmp_path)

    for _ in range(2):
        verifier._verify_cagra_data_file(context, fields)

    assert codec_util.checksum_calls == [verifier._test_data_input]
    assert codec_util.retrieved_checksum_calls == [verifier._test_data_input]


def test_cagra_data_checksum_cache_invalidates_on_size_change(tmp_path):
    codec_util = _FakeCodecUtil()
    verifier = _fake_cagra_index_verifier(
        _FakeChecksumInput([], []), codec_util=codec_util
    )
    data_path = tmp_path / "_0.vcag"
    data_path.write_bytes(b"data")
    fields = [
        pylucene_backend._CagraFieldMetadata(
            field_number=1,
            dimensions=32,
            vector_count=4,
            cagra_offset=57,
            cagra_length=1000,
        )
    ]
    context = _cagra_data_context(verifier, tmp_path)

    verifier._verify_cagra_data_file(context, fields)
    data_path.write_bytes(b"changed-size")
    verifier._verify_cagra_data_file(context, fields)

    assert codec_util.checksum_calls == [
        verifier._test_data_input,
        verifier._test_data_input,
    ]
    assert codec_util.retrieved_checksum_calls == []


def test_cagra_data_checksum_cache_invalidates_same_size_restored_mtime(
    tmp_path,
):
    codec_util = _FakeCodecUtil()
    verifier = _fake_cagra_index_verifier(
        _FakeChecksumInput([], []), codec_util=codec_util
    )
    data_path = tmp_path / "_0.vcag"
    data_path.write_bytes(b"data")
    fields = [
        pylucene_backend._CagraFieldMetadata(
            field_number=1,
            dimensions=32,
            vector_count=4,
            cagra_offset=57,
            cagra_length=1000,
        )
    ]
    context = _cagra_data_context(verifier, tmp_path)

    verifier._verify_cagra_data_file(context, fields)
    original_stat = data_path.stat()
    data_path.write_bytes(b"evil")
    os.utime(
        data_path,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    assert data_path.stat().st_mtime_ns == original_stat.st_mtime_ns

    verifier._verify_cagra_data_file(context, fields)

    assert codec_util.checksum_calls == [
        verifier._test_data_input,
        verifier._test_data_input,
    ]
    assert codec_util.retrieved_checksum_calls == []
