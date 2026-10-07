# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from fern.scripts.generate_api_reference import parse_java_members, parse_javadoc


class GenerateApiReferenceTest(unittest.TestCase):
    def test_java_member_name_skips_leading_annotations(self):
        source = """
public class Example {
  /** Retains the old API. */
  @Deprecated(since = "26.12", forRemoval = false)
  public void retainedMethod() {}
}
"""

        members = parse_java_members(source, "Example")

        self.assertEqual(
            [member.name for member in members], ["retainedMethod"]
        )

    def test_multiline_javadoc_method_link_is_rendered_as_code(self):
        doc = parse_javadoc(
            """
            /**
             * Use {@link
             * #replacement(int, int)}.
             */
            """
        )

        self.assertEqual(doc.summary, "Use `#replacement(int, int)`.")


if __name__ == "__main__":
    unittest.main()
