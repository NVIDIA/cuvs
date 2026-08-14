#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import pathlib
import re
import subprocess

import clang_format


parser = argparse.ArgumentParser()
parser.add_argument("filenames", metavar="file", nargs="*")
args = parser.parse_args()

for filename in args.filenames:
    with open(filename) as f:
        contents = f.read()

    placeholder_counter = 0
    placeholder_forward_replacements: dict[str, str] = {}
    placeholder_reverse_replacements: dict[str, str] = {}
    replaced_contents = ""
    start = 0

    for placeholder_match in re.finditer(
        r"@(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)@", contents
    ):
        try:
            replacement = placeholder_forward_replacements[
                placeholder_match.group("name")
            ]
        except KeyError:
            while contents.find(f"##__{placeholder_counter}##") != -1:
                placeholder_counter += 1
            replacement = placeholder_forward_replacements[
                placeholder_match.group("name")
            ] = f"##__{placeholder_counter}##"
            placeholder_reverse_replacements[replacement] = (
                f"@{placeholder_match.group('name')}@"
            )
            placeholder_counter += 1
        replaced_contents += (
            contents[start : placeholder_match.start()] + replacement
        )
        start = placeholder_match.end()

    replaced_contents += contents[start:]

    clang_format_exe = (
        pathlib.Path(clang_format.__file__).parent / "data/bin/clang-format"
    )
    style_file = (
        pathlib.Path(__file__).parent.parent.parent / "cpp/.clang-format"
    )

    result = subprocess.run(
        [clang_format_exe, f"--style=file:{style_file}"],
        input=replaced_contents,
        capture_output=True,
        text=True,
    )
    formatted_contents = result.stdout

    for placeholder, replacement in placeholder_reverse_replacements.items():
        formatted_contents = formatted_contents.replace(
            placeholder, replacement
        )

    with open(filename, "w") as f:
        f.write(formatted_contents)
