# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from generate_api_reference import parse_doxygen_entry, render_native_function


class FencedExamplesTest(unittest.TestCase):
    def test_example_preserves_code_and_following_parameters(self):
        comment = """@brief Build an index.

Usage example:
```cpp
auto data = make_matrix<float>(handle, rows, dim);
if (rows > 0) {
  build(data);

  // @note is literal code here.
}
```

@param rows the number of vectors
"""
        entry = parse_doxygen_entry(
            comment, "void build(int rows);", "example.hpp", 1
        )
        rendered = "\n".join(render_native_function(entry, "cpp"))
        example = comment.split("```cpp\n")[1].split("\n```")[0]
        self.assertIn(f"Usage example:\n\n```cpp\n{example}\n```", rendered)
        self.assertIn("the number of vectors", rendered)
        self.assertLess(
            rendered.index("```cpp\nauto data"),
            rendered.index("**Parameters**"),
        )


if __name__ == "__main__":
    unittest.main()
