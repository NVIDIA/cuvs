#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import gzip
import msgspec
import pathlib
import sys

from check_c_abi.abi import analyze_c_abi, Abi

try:
    from termcolor import colored
except ImportError:

    def colored(text, *args, **kwargs):
        return text


def main_cli():
    # figure out the default header c path - by scanning recursively for the
    # `cuvs/core/all.h` header
    current_path = pathlib.Path(".").resolve()
    while current_path:
        if (
            current_path / "c" / "include" / "cuvs" / "core" / "all.h"
        ).is_file():
            default_c_header_path = current_path / "c" / "include"
            break
        if current_path.parent == current_path:
            default_c_header_path = None
            break
        current_path = current_path.parent

    parent_parser = argparse.ArgumentParser(
        add_help=False, description="Common arguments for all subcommands"
    )
    parent_parser.add_argument(
        "--header-path",
        help="Path of C headers to analyze (default: %(default)s)",
        default=str(default_c_header_path),
    )
    parent_parser.add_argument(
        "--include-file",
        help="root header file to examine (default: %(default)s)",
        default="cuvs/core/all.h",
    )
    parent_parser.add_argument(
        "--dlpack-include-path", help="path of dlpack header include"
    )

    parser = argparse.ArgumentParser(
        description="Analyze C headers for breaking ABI changes"
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Available subcommands"
    )

    parser_extract = subparsers.add_parser(
        "extract",
        parents=[parent_parser],
        help="Extract the ABI from a set of header files",
    )
    parser_extract.add_argument(
        "--output-file",
        type=str,
        help="The file to output the ABI into (default: %(default)s)",
        default="c_abi.json.gz",
    )

    parser_analyze = subparsers.add_parser(
        "analyze",
        parents=[parent_parser],
        help="Analyze a set of header files for breaking changes",
    )
    parser_analyze.add_argument(
        "--abi-file",
        type=str,
        help="The extracted ABI file to compare against (default: %(default)s)",
        default="c_abi.json.gz",
    )

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    header_path = pathlib.Path(args.header_path)

    if args.dlpack_include_path:
        dlpack_include_path = pathlib.Path(args.dlpack_include_path).resolve()

    else:
        # try getting from the cmake build directory dependencies if we
        # haven't specified the include directory
        dlpack_include_path = (
            header_path.parent.parent
            / "cpp"
            / "build"
            / "_deps"
            / "dlpack-src"
            / "include"
        )

    if not dlpack_include_path.is_dir():
        raise ValueError(
            f"dlpack header path '{dlpack_include_path}' not found"
        )

    if not (dlpack_include_path / "dlpack" / "dlpack.h").is_file():
        raise ValueError(
            f"dlpack header 'dlpack/dlpack.h' not found in '{dlpack_include_path}'"
        )

    print(f"using dlpack from {dlpack_include_path}")

    extra_clang_args = [f"-I{str(dlpack_include_path)}"]

    if args.command == "extract":
        abi = Abi.from_include_path(
            header_path, args.include_file, extra_clang_args
        )
        with open(args.output_file, "wb") as o:
            o.write(gzip.compress(msgspec.json.encode(abi)))
        print(f"wrote abi to {args.output_file}")
    elif args.command == "analyze":
        old_abi = msgspec.json.decode(
            gzip.decompress(open(args.abi_file, "rb").read()), type=Abi
        )
        new_abi = Abi.from_include_path(
            header_path, args.include_file, extra_clang_args
        )
        errors = analyze_c_abi(old_abi, new_abi)
        for error in errors:
            print(
                f"Error: {colored(error.error, attrs=['bold'])}. Symbol {colored(error.symbol, 'red')} from {error.location.filename}:{error.location.line}"
            )

        if errors:
            sys.exit(1)
        else:
            print("no breaking abi changes detected")


if __name__ == "__main__":
    main_cli()
