#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly PYLUCENE_VERSION="10.2.0"
readonly PYLUCENE_SCAFFOLD_VERSION="10.0.0"
readonly PYLUCENE_ARCHIVE_SHA256="100c3d61d6799ac16b7b8c1826cddf07fb1715141ebdb0d7b8119cdd96b24574"
readonly LUCENE_ARCHIVE_SHA512="2e8ad344631031416d96277abb844ca4d4c266086041f6b9339aade1ffb015875bdb34eeb3dfba2563648ce56f8fdafc52839b75ecf716036a3feaf1873fd02f"
readonly PYLUCENE_URL="https://archive.apache.org/dist/lucene/pylucene/pylucene-${PYLUCENE_SCAFFOLD_VERSION}-src.tar.gz"
readonly LUCENE_URL="https://archive.apache.org/dist/lucene/java/${PYLUCENE_VERSION}/lucene-${PYLUCENE_VERSION}-src.tgz"
readonly SOURCE_RECIPE_REVISION="1"
readonly NUM_GENERATED_FILES="18"

readonly SETUPTOOLS_VERSION="80.9.0"
readonly BUILD_VERSION="1.5.0"
readonly WHEEL_VERSION="0.47.0"
readonly PACKAGING_VERSION="26.3"
readonly PYPROJECT_HOOKS_VERSION="1.2.0"
readonly PYTEST_VERSION="9.1.1"
readonly INICONFIG_VERSION="2.3.0"
readonly PLUGGY_VERSION="1.6.0"
readonly PYGMENTS_VERSION="2.20.0"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
patch_file="$script_dir/pylucene-10.2.0.patch"
user_home="${HOME:-}"
if [[ -n "${XDG_CACHE_HOME:-}" ]]; then
  default_cache_root="$XDG_CACHE_HOME"
elif [[ -n "$user_home" ]]; then
  default_cache_root="$user_home/.cache"
else
  default_cache_root=""
fi
build_root="${CUVS_BENCH_PYLUCENE_BUILD_ROOT:-${default_cache_root:+${default_cache_root}/cuvs-bench/pylucene-${PYLUCENE_VERSION}}}"
python_command="${PYTHON:-python3}"
prepare_only=false

usage() {
  cat <<EOF
Usage: $0 [--build-root PATH] [--python PATH] [--prepare-only]

Build an isolated PyLucene ${PYLUCENE_VERSION} environment from the official
PyLucene ${PYLUCENE_SCAFFOLD_VERSION} and Lucene ${PYLUCENE_VERSION} source archives.

PATH must be a stable absolute path because JCC records native-library paths
in the generated extension. A full build requires Linux, CPython 3.11-3.14,
and JDK 22. --prepare-only only downloads, verifies, and patches the sources.
EOF
}

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

validate_build_root() {
  local candidate="$1"
  [[ "$candidate" == /* ]] || fail "--build-root must be an absolute path: $candidate"
  if [[ ! "$candidate" =~ ^/[A-Za-z0-9_./-]+$ ]]; then
    fail "--build-root may contain only letters, digits, '/', '.', '_', and '-' because JCC and Make interpolate it into shell recipes"
  fi
}

while (($# > 0)); do
  case "$1" in
    --build-root)
      (($# >= 2)) || fail "--build-root requires a path"
      build_root="$2"
      shift 2
      ;;
    --python)
      (($# >= 2)) || fail "--python requires an executable"
      python_command="$2"
      shift 2
      ;;
    --prepare-only)
      prepare_only=true
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      fail "unknown argument: $1"
      ;;
  esac
done

[[ "$(uname -s)" == "Linux" ]] || fail "this helper currently supports Linux only"
[[ -f "$patch_file" ]] || fail "compatibility patch is missing: $patch_file"
[[ -n "$build_root" ]] ||
  fail "--build-root is required when HOME and XDG_CACHE_HOME are unset"
validate_build_root "$build_root"

for command_name in awk cmp cp curl flock gzip mktemp mv patch rm sha256sum sha512sum tar uname; do
  command -v "$command_name" >/dev/null ||
    fail "required command is missing: $command_name"
done

mkdir -p -- "$build_root"
build_root="$(cd -- "$build_root" && pwd -P)"
validate_build_root "$build_root"
if [[ "$build_root" == "/" || (-n "$user_home" && "$build_root" == "$user_home") ]]; then
  fail "refusing unsafe build root: $build_root"
fi

exec {build_lock_fd}>"$build_root/.build.lock"
flock -n "$build_lock_fd" || fail "another PyLucene build is using $build_root"

downloads_dir="$build_root/downloads"
source_dir="$build_root/src/pylucene-${PYLUCENE_VERSION}"
source_manifest="$source_dir/.cuvs-pylucene-source-manifest"
venv_dir="$build_root/venv"
build_manifest="$build_root/.cuvs-pylucene-build-manifest"
complete_marker="$build_root/.complete"
activation_file="$build_root/activate.sh"
mkdir -p -- "$downloads_dir" "$(dirname -- "$source_dir")"

source_manifest_temp=""
build_manifest_temp=""
activation_temp=""
complete_temp=""
staging_dir=""
download_temp=""

cleanup() {
  [[ -z "$source_manifest_temp" ]] || rm -f -- "$source_manifest_temp"
  [[ -z "$build_manifest_temp" ]] || rm -f -- "$build_manifest_temp"
  [[ -z "$activation_temp" ]] || rm -f -- "$activation_temp"
  [[ -z "$complete_temp" ]] || rm -f -- "$complete_temp"
  [[ -z "$download_temp" ]] || rm -f -- "$download_temp"
  [[ -z "$staging_dir" ]] || rm -rf -- "$staging_dir"
}
trap cleanup EXIT

verify_checksum() {
  local algorithm="$1"
  local expected="$2"
  local path="$3"
  printf '%s  %s\n' "$expected" "$path" |
    "$algorithm" --check --status -
}

download_verified() {
  local url="$1"
  local destination="$2"
  local algorithm="$3"
  local expected="$4"

  if [[ -f "$destination" ]]; then
    verify_checksum "$algorithm" "$expected" "$destination" ||
      fail "cached download has the wrong checksum; remove $destination or choose a new --build-root"
    return
  fi

  download_temp="$(mktemp "${destination}.part.XXXXXX")"
  printf 'Downloading %s\n' "$url"
  if ! curl --fail --location --show-error --proto '=https' --proto-redir '=https' \
    --retry 3 --output "$download_temp" "$url"; then
    fail "download failed: $url"
  fi
  verify_checksum "$algorithm" "$expected" "$download_temp" ||
    fail "downloaded archive checksum mismatch: $url"
  mv -- "$download_temp" "$destination"
  download_temp=""
}

pylucene_archive="$downloads_dir/pylucene-${PYLUCENE_SCAFFOLD_VERSION}-src.tar.gz"
lucene_archive="$downloads_dir/lucene-${PYLUCENE_VERSION}-src.tgz"
download_verified \
  "$PYLUCENE_URL" "$pylucene_archive" sha256sum "$PYLUCENE_ARCHIVE_SHA256"
download_verified \
  "$LUCENE_URL" "$lucene_archive" sha512sum "$LUCENE_ARCHIVE_SHA512"

patch_sha256="$(sha256sum "$patch_file" | awk '{print $1}')"
source_manifest_temp="$(mktemp "$build_root/source-manifest.XXXXXX")"
cat >"$source_manifest_temp" <<EOF
format=1
source_recipe_revision=${SOURCE_RECIPE_REVISION}
pylucene_version=${PYLUCENE_VERSION}
pylucene_scaffold_version=${PYLUCENE_SCAFFOLD_VERSION}
pylucene_url=${PYLUCENE_URL}
pylucene_archive_sha256=${PYLUCENE_ARCHIVE_SHA256}
lucene_url=${LUCENE_URL}
lucene_archive_sha512=${LUCENE_ARCHIVE_SHA512}
patch_sha256=${patch_sha256}
num_generated_files=${NUM_GENERATED_FILES}
makefile_sha256=d9a20059f349f1d2eb346ce63f44b796f7c6b1f688d6c0dc61004ee5fecf45cf
settings_gradle_sha256=bc0cbce02a431bca779901af419d62f058f5836d1b2fb2c9dede42f7bb8f3bca
gradle_wrapper_properties_sha256=f0b2060d483a15e115abfa1f99bc271a9a5e09be5c33f2714eda59a3d2e9d7cf
EOF

verify_prepared_source() {
  [[ -f "$source_manifest" ]] ||
    fail "prepared-source manifest is missing: $source_manifest"
  cmp -s "$source_manifest_temp" "$source_manifest" ||
    fail "prepared sources use different inputs; choose a new --build-root or remove $build_root"
  verify_checksum sha256sum \
    d9a20059f349f1d2eb346ce63f44b796f7c6b1f688d6c0dc61004ee5fecf45cf \
    "$source_dir/Makefile" ||
    fail "prepared PyLucene Makefile changed under $source_dir"
  verify_checksum sha256sum \
    bc0cbce02a431bca779901af419d62f058f5836d1b2fb2c9dede42f7bb8f3bca \
    "$source_dir/lucene-java-${PYLUCENE_VERSION}/settings.gradle" ||
    fail "prepared Lucene settings.gradle changed under $source_dir"
  verify_checksum sha256sum \
    f0b2060d483a15e115abfa1f99bc271a9a5e09be5c33f2714eda59a3d2e9d7cf \
    "$source_dir/lucene-java-${PYLUCENE_VERSION}/gradle/wrapper/gradle-wrapper.properties" ||
    fail "prepared Gradle wrapper properties changed under $source_dir"
}

if [[ -e "$source_dir" ]]; then
  [[ -d "$source_dir" ]] || fail "source path is not a directory: $source_dir"
  verify_prepared_source
else
  staging_dir="$(mktemp -d "$build_root/prepare.XXXXXX")"
  tar -xzf "$pylucene_archive" -C "$staging_dir" \
    --exclude="pylucene-${PYLUCENE_SCAFFOLD_VERSION}/lucene-java-${PYLUCENE_SCAFFOLD_VERSION}"
  staged_source="$staging_dir/pylucene-${PYLUCENE_VERSION}"
  mv -- "$staging_dir/pylucene-${PYLUCENE_SCAFFOLD_VERSION}" "$staged_source"

  mkdir -p -- "$staged_source/lucene-java-${PYLUCENE_VERSION}"
  tar -xzf "$lucene_archive" \
    -C "$staged_source/lucene-java-${PYLUCENE_VERSION}" --strip-components=1

  verify_checksum sha256sum \
    0bb9ce7d8e473eeebab264aa6dd9515149de67569c29aed2a67954a63c398f65 \
    "$staged_source/Makefile" ||
    fail "unexpected PyLucene ${PYLUCENE_SCAFFOLD_VERSION} Makefile"
  verify_checksum sha256sum \
    a799dbfac9fdda839d468fa8edaf2ea483bfa6957b7e29b3711b7f16d4fc7f04 \
    "$staged_source/lucene-java-${PYLUCENE_VERSION}/settings.gradle" ||
    fail "unexpected Lucene ${PYLUCENE_VERSION} settings.gradle"
  verify_checksum sha256sum \
    2caec011a749b18ab9a7dea68bf7639a179a3b8316b0e35655d0a62a1d7390fd \
    "$staged_source/lucene-java-${PYLUCENE_VERSION}/gradle/wrapper/gradle-wrapper.properties" ||
    fail "unexpected Lucene ${PYLUCENE_VERSION} Gradle wrapper properties"

  cp -a -- "$staged_source/extensions" \
    "$staged_source/lucene-java-${PYLUCENE_VERSION}/lucene/extensions"
  patch --batch --fuzz=0 --directory="$staged_source" --strip=1 <"$patch_file"

  verify_checksum sha256sum \
    d9a20059f349f1d2eb346ce63f44b796f7c6b1f688d6c0dc61004ee5fecf45cf \
    "$staged_source/Makefile" ||
    fail "patched PyLucene Makefile has an unexpected checksum"
  verify_checksum sha256sum \
    bc0cbce02a431bca779901af419d62f058f5836d1b2fb2c9dede42f7bb8f3bca \
    "$staged_source/lucene-java-${PYLUCENE_VERSION}/settings.gradle" ||
    fail "patched Lucene settings.gradle has an unexpected checksum"
  verify_checksum sha256sum \
    f0b2060d483a15e115abfa1f99bc271a9a5e09be5c33f2714eda59a3d2e9d7cf \
    "$staged_source/lucene-java-${PYLUCENE_VERSION}/gradle/wrapper/gradle-wrapper.properties" ||
    fail "patched Gradle wrapper properties have an unexpected checksum"

  cp -- "$source_manifest_temp" \
    "$staged_source/.cuvs-pylucene-source-manifest"
  mv -- "$staged_source" "$source_dir"
  rm -rf -- "$staging_dir"
  staging_dir=""
  verify_prepared_source
fi

printf 'Prepared custom PyLucene source at %s\n' "$source_dir"
if [[ "$prepare_only" == true ]]; then
  exit 0
fi

for command_name in find make readlink xargs; do
  command -v "$command_name" >/dev/null ||
    fail "required command is missing: $command_name"
done

python_path="$(command -v "$python_command" 2>/dev/null || true)"
[[ -n "$python_path" ]] ||
  fail "Python executable is unavailable: $python_command"
python_path="$(readlink -f "$python_path")"

if ! python_details="$("$python_path" - <<'PY'
import platform
import struct
import sys
import sysconfig

if sys.implementation.name != "cpython":
    raise SystemExit("PyLucene requires CPython for this build")
if not ((3, 11) <= sys.version_info[:2] <= (3, 14)):
    raise SystemExit(
        "this helper supports CPython 3.11-3.14; "
        f"found {platform.python_version()}"
    )
if struct.calcsize("P") != 8:
    raise SystemExit("this helper requires a 64-bit CPython")
print(f"python_implementation={sys.implementation.name}")
print(f"python_version={platform.python_version()}")
print(f"python_cache_tag={sys.implementation.cache_tag}")
print(f"python_soabi={sysconfig.get_config_var('SOABI')}")
PY
)"; then
  fail "the selected Python interpreter is incompatible: $python_path"
fi

"$python_path" - <<'PY'
import pathlib
import sysconfig

header = pathlib.Path(sysconfig.get_paths()["include"]) / "Python.h"
if not header.is_file():
    raise SystemExit(f"Python development header is missing: {header}")
PY

java_home="${JAVA_HOME:-}"
if [[ -z "$java_home" ]]; then
  javac_path="$(command -v javac 2>/dev/null || true)"
  [[ -n "$javac_path" ]] ||
    fail "JAVA_HOME is unset and javac is unavailable; install JDK 22"
  java_home="$(dirname -- "$(dirname -- "$(readlink -f "$javac_path")")")"
fi
[[ -d "$java_home" ]] || fail "JAVA_HOME is not a directory: $java_home"
java_home="$(cd -- "$java_home" && pwd -P)"

for required_path in \
  "$java_home/bin/java" \
  "$java_home/bin/javac" \
  "$java_home/bin/javadoc" \
  "$java_home/include/jni.h" \
  "$java_home/include/linux/jni_md.h" \
  "$java_home/lib/libjava.so" \
  "$java_home/lib/server/libjvm.so" \
  "$java_home/release"; do
  [[ -e "$required_path" ]] ||
    fail "JDK 22 component is missing: $required_path"
done

java_properties="$("$java_home/bin/java" -XshowSettings:properties -version 2>&1)"
java_specification_version="$(awk -F'= ' '/java.specification.version/{print $2; exit}' <<<"$java_properties")"
java_runtime_version="$(awk -F'= ' '/java.runtime.version/{print $2; exit}' <<<"$java_properties")"
[[ "$java_specification_version" == "22" ]] ||
  fail "JDK 22 is required; found Java ${java_specification_version:-unknown} at $java_home"

cc_command="${CC:-cc}"
cxx_command="${CXX:-c++}"
if [[ "$cc_command" =~ [[:space:]] || "$cxx_command" =~ [[:space:]] ]]; then
  fail "CC and CXX must name executables without command-line arguments"
fi
cc_path="$(command -v "$cc_command" 2>/dev/null || true)"
cxx_path="$(command -v "$cxx_command" 2>/dev/null || true)"
[[ -n "$cc_path" ]] || fail "C compiler is unavailable: $cc_command"
[[ -n "$cxx_path" ]] || fail "C++ compiler is unavailable: $cxx_command"
cc_path="$(readlink -f "$cc_path")"
cxx_path="$(readlink -f "$cxx_path")"
cc_version="$("$cc_path" --version | awk 'NR == 1 {print; exit}')"
cxx_version="$("$cxx_path" --version | awk 'NR == 1 {print; exit}')"

for unsupported_path in "$python_path" "$java_home" "$cc_path" "$cxx_path"; do
  if [[ "$unsupported_path" =~ [[:space:]|] ]]; then
    fail "toolchain paths cannot contain whitespace or '|': $unsupported_path"
  fi
done

script_sha256="$(sha256sum "$script_dir/build_pylucene_10_2.sh" | awk '{print $1}')"
source_manifest_sha256="$(sha256sum "$source_manifest" | awk '{print $1}')"
java_release_sha256="$(sha256sum "$java_home/release" | awk '{print $1}')"
make_version="$(make --version | awk 'NR == 1 {print; exit}')"

build_manifest_temp="$(mktemp "$build_root/build-manifest.XXXXXX")"
cat >"$build_manifest_temp" <<EOF
format=1
source_manifest_sha256=${source_manifest_sha256}
build_script_sha256=${script_sha256}
python_path=${python_path}
${python_details}
java_home=${java_home}
java_runtime_version=${java_runtime_version}
java_release_sha256=${java_release_sha256}
operating_system=$(uname -s)
machine=$(uname -m)
cc_path=${cc_path}
cc_version=${cc_version}
cxx_path=${cxx_path}
cxx_version=${cxx_version}
make_version=${make_version}
jcc_version=3.15
setuptools_version=${SETUPTOOLS_VERSION}
build_version=${BUILD_VERSION}
wheel_version=${WHEEL_VERSION}
packaging_version=${PACKAGING_VERSION}
pyproject_hooks_version=${PYPROJECT_HOOKS_VERSION}
pytest_version=${PYTEST_VERSION}
iniconfig_version=${INICONFIG_VERSION}
pluggy_version=${PLUGGY_VERSION}
pygments_version=${PYGMENTS_VERSION}
num_generated_files=${NUM_GENERATED_FILES}
EOF
build_manifest_sha256="$(sha256sum "$build_manifest_temp" | awk '{print $1}')"

if [[ -f "$build_manifest" ]]; then
  cmp -s "$build_manifest_temp" "$build_manifest" ||
    fail "build root was created for different inputs; choose a new --build-root or remove $build_root"
  rm -f -- "$build_manifest_temp"
  build_manifest_temp=""
else
  if [[ -e "$venv_dir" || -e "$complete_marker" || -e "$activation_file" ]]; then
    fail "untracked build state exists under $build_root; choose a new --build-root or remove this root"
  fi
  mv -- "$build_manifest_temp" "$build_manifest"
  build_manifest_temp=""
fi

completed_before=false
if [[ -e "$complete_marker" ]]; then
  [[ -f "$complete_marker" ]] ||
    fail "completion marker is not a file: $complete_marker"
  [[ "$(<"$complete_marker")" == "$build_manifest_sha256" ]] ||
    fail "completion marker does not match the build manifest under $build_root"
  [[ -x "$venv_dir/bin/python" ]] ||
    fail "completed build is missing its Python environment: $venv_dir"
  completed_before=true
fi

export JAVA_HOME="$java_home"
export PATH="$JAVA_HOME/bin:$PATH"
export CC="$cc_path"
export CXX="$cxx_path"
export PYTHONNOUSERSITE=1
export PIP_NO_INPUT=1
export JCC_ARGSEP='|'
export JCC_JDK="$JAVA_HOME"
export JCC_INCLUDES="$JAVA_HOME/include|$JAVA_HOME/include/linux"
export JCC_CFLAGS='-fno-strict-aliasing|-Wno-write-strings'
export JCC_DEBUG_CFLAGS='-O0|-g|-DDEBUG'
export JCC_LFLAGS="-L$JAVA_HOME/lib|-ljava|-L$JAVA_HOME/lib/server|-ljvm|-Wl,-rpath,$JAVA_HOME/lib|-Wl,-rpath,$JAVA_HOME/lib/server"
export JCC_JAVAC="$JAVA_HOME/bin/javac"
export JCC_JAVADOC="$JAVA_HOME/bin/javadoc"
export GRADLE_USER_HOME="$build_root/gradle-home"
unset PYTHONPATH

if [[ "$completed_before" == false ]]; then
  if [[ ! -x "$venv_dir/bin/python" ]]; then
    "$python_path" -m venv --help >/dev/null 2>&1 ||
      fail "Python venv support is unavailable for $python_path"
    "$python_path" -m venv "$venv_dir" ||
      fail "could not create a virtual environment; install venv support for $python_path"
  fi

  "$venv_dir/bin/python" -m pip install --disable-pip-version-check --no-input \
    "setuptools==$SETUPTOOLS_VERSION" \
    "build==$BUILD_VERSION" \
    "wheel==$WHEEL_VERSION" \
    "packaging==$PACKAGING_VERSION" \
    "pyproject-hooks==$PYPROJECT_HOOKS_VERSION" \
    "pytest==$PYTEST_VERSION" \
    "iniconfig==$INICONFIG_VERSION" \
    "pluggy==$PLUGGY_VERSION" \
    "Pygments==$PYGMENTS_VERSION"

  if ! "$venv_dir/bin/python" -c \
    'import importlib.metadata as metadata; assert metadata.version("JCC") == "3.15"' \
    2>/dev/null; then
    (
      cd "$source_dir/jcc"
      "$venv_dir/bin/python" -m build --wheel --no-isolation
    )
    shopt -s nullglob
    jcc_wheels=("$source_dir"/jcc/dist/[Jj][Cc][Cc]-3.15-*.whl)
    shopt -u nullglob
    ((${#jcc_wheels[@]} == 1)) ||
      fail "expected exactly one JCC 3.15 wheel"
    "$venv_dir/bin/python" -m pip install --no-deps --force-reinstall \
      "${jcc_wheels[0]}"
  fi

  make_variables=(
    "PYTHON=$venv_dir/bin/python"
    "JCC=$venv_dir/bin/python -m jcc --shared"
    "NUM_FILES=$NUM_GENERATED_FILES"
    "MODERN_PACKAGING=true"
  )
  make -C "$source_dir" "${make_variables[@]}" all
  make -C "$source_dir" "${make_variables[@]}" test
fi

activation_temp="$(mktemp "$build_root/activate.XXXXXX")"
{
  printf '# Generated by %q. Source from Bash; do not execute.\n' "$0"
  printf 'source %q\n' "$venv_dir/bin/activate"
  printf 'export JAVA_HOME=%q\n' "$java_home"
  # shellcheck disable=SC2016 # Expansion is intentionally deferred until source time.
  printf 'export PATH=%q${PATH:+":$PATH"}\n' "$java_home/bin"
} >"$activation_temp"
mv -- "$activation_temp" "$activation_file"
activation_temp=""

CUVS_PYLUCENE_EXPECTED_ROOT="$build_root" "$venv_dir/bin/python" -I - <<'PY'
import importlib.metadata as metadata
import os
from pathlib import Path

import jcc
import lucene

root = Path(os.environ["CUVS_PYLUCENE_EXPECTED_ROOT"]).resolve()
assert lucene.VERSION == "10.2.0", lucene.VERSION
assert metadata.version("JCC") == "3.15"
assert Path(lucene.__file__).resolve().is_relative_to(root)
assert Path(jcc.__file__).resolve().is_relative_to(root)
lucene.initVM(
    vmargs=[
        "--add-modules=jdk.incubator.vector",
        "--enable-native-access=ALL-UNNAMED",
    ]
)
from java.lang import Class

Class.forName("org.apache.lucene.codecs.lucene101.Lucene101Codec")
Class.forName(
    "org.apache.lucene.codecs.lucene102.Lucene102HnswBinaryQuantizedVectorsFormat"
)
print(f"PyLucene {lucene.VERSION} JVM smoke test passed")
PY
"$venv_dir/bin/python" -m pip check

shopt -s nullglob
pylucene_wheels=("${source_dir}/dist/lucene-${PYLUCENE_VERSION}-"*.whl)
jcc_wheels=("$source_dir"/jcc/dist/[Jj][Cc][Cc]-3.15-*.whl)
shopt -u nullglob
((${#pylucene_wheels[@]} == 1)) ||
  fail "expected exactly one PyLucene ${PYLUCENE_VERSION} wheel"
((${#jcc_wheels[@]} == 1)) ||
  fail "expected exactly one JCC 3.15 wheel"
printf 'PyLucene wheel: %s\n' "${pylucene_wheels[0]}"
sha256sum "${pylucene_wheels[0]}"
printf 'JCC wheel: %s\n' "${jcc_wheels[0]}"
sha256sum "${jcc_wheels[0]}"

if [[ "$completed_before" == false ]]; then
  complete_temp="$(mktemp "$build_root/complete.XXXXXX")"
  printf '%s\n' "$build_manifest_sha256" >"$complete_temp"
  mv -- "$complete_temp" "$complete_marker"
  complete_temp=""
fi

printf 'PyLucene %s is ready. Run: source %s\n' \
  "$PYLUCENE_VERSION" "$activation_file"
