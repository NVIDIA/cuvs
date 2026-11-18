#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The sourcing builder consumes these declarations.
# shellcheck disable=SC2034

# Pinned inputs for the only supported custom PyLucene build. New Lucene
# releases require their own reviewed sources, patch, toolchain and smoke tests.
readonly PYLUCENE_VERSION="10.2.0"
readonly PYLUCENE_SCAFFOLD_VERSION="10.0.0"
readonly PYLUCENE_ARCHIVE_SHA256="100c3d61d6799ac16b7b8c1826cddf07fb1715141ebdb0d7b8119cdd96b24574"
readonly LUCENE_ARCHIVE_SHA512="2e8ad344631031416d96277abb844ca4d4c266086041f6b9339aade1ffb015875bdb34eeb3dfba2563648ce56f8fdafc52839b75ecf716036a3feaf1873fd02f"
readonly PYLUCENE_URL="https://archive.apache.org/dist/lucene/pylucene/pylucene-${PYLUCENE_SCAFFOLD_VERSION}-src.tar.gz"
readonly LUCENE_URL="https://archive.apache.org/dist/lucene/java/${PYLUCENE_VERSION}/lucene-${PYLUCENE_VERSION}-src.tgz"
readonly SOURCE_RECIPE_REVISION="1"
readonly NUM_GENERATED_FILES="18"
readonly UPSTREAM_MAKEFILE_SHA256="0bb9ce7d8e473eeebab264aa6dd9515149de67569c29aed2a67954a63c398f65"
readonly UPSTREAM_SETTINGS_GRADLE_SHA256="a799dbfac9fdda839d468fa8edaf2ea483bfa6957b7e29b3711b7f16d4fc7f04"
readonly UPSTREAM_GRADLE_WRAPPER_SHA256="2caec011a749b18ab9a7dea68bf7639a179a3b8316b0e35655d0a62a1d7390fd"
readonly PREPARED_MAKEFILE_SHA256="d9a20059f349f1d2eb346ce63f44b796f7c6b1f688d6c0dc61004ee5fecf45cf"
readonly PREPARED_SETTINGS_GRADLE_SHA256="bc0cbce02a431bca779901af419d62f058f5836d1b2fb2c9dede42f7bb8f3bca"
readonly PREPARED_GRADLE_WRAPPER_SHA256="f0b2060d483a15e115abfa1f99bc271a9a5e09be5c33f2714eda59a3d2e9d7cf"
readonly JCC_VERSION="3.15"
readonly JDK_VERSION="22"
readonly COMPATIBILITY_PATCH="pylucene-${PYLUCENE_VERSION}.patch"
readonly PYLUCENE_SMOKE_CLASSES=(
  "org.apache.lucene.codecs.lucene101.Lucene101Codec"
  "org.apache.lucene.codecs.lucene102.Lucene102HnswBinaryQuantizedVectorsFormat"
)
