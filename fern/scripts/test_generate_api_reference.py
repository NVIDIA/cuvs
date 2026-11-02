# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for elaborated C field types in the native API reference."""

import importlib.util
from pathlib import Path
import sys
import unittest

spec = importlib.util.spec_from_file_location(
    "fern_api_reference_under_test", Path(__file__).with_name("generate_api_reference.py")
)
api = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = api
spec.loader.exec_module(api)


class StructFieldDocumentationTest(unittest.TestCase):
    def test_elaborated_field_is_a_member(self):
        self.assertEqual(api.parse_doxygen_kind("struct Options tiered;"), "member")
        self.assertEqual(api.parse_doxygen_kind("struct Options* tiered;"), "member")

    def test_structure_declarations_keep_their_kind(self):
        self.assertEqual(api.parse_doxygen_kind("struct Options;"), "struct")
        self.assertEqual(api.parse_doxygen_kind("struct Options { int value; };"), "struct")

    def test_documented_field_keeps_its_type_and_description(self):
        entry = api.parse_doxygen_entry(
            "Parameters", "struct Params { /** Tiered settings. */ struct Options tiered; };",
            "test.h", 1
        )
        members = api.parse_struct_members(entry)
        self.assertEqual([m.name for m in members], ["tiered"])
        self.assertEqual(api.member_c_type(members[0]), "struct Options")
        self.assertIn("Tiered settings", api.member_description(members[0]))
        self.assertIn("struct Options tiered;", api.render_compound_signature(entry, members, []))

    def test_plain_field_keeps_its_type(self):
        entry = api.parse_doxygen_entry(
            "Parameters", "struct Params { struct Options tiered; };", "test.h", 1
        )
        members = api.parse_struct_members(entry)
        self.assertEqual([m.name for m in members], ["tiered"])
        self.assertEqual(api.member_c_type(members[0]), "struct Options")


if __name__ == "__main__":
    unittest.main()
