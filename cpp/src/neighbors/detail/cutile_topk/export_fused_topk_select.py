# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select-stage entrypoint for the shared cuTile CMake exporter."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_fused_topk import main

if __name__ == "__main__":
    raise SystemExit(main("select"))
