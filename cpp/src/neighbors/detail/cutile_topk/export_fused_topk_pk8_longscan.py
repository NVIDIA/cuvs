# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export the separately named, SM120-only long-scan PK8 variant."""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from export_fused_topk import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(whole_tile_guard=True))
