# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_fused_topk_pk16_stages import main

if __name__ == "__main__":
    raise SystemExit(main("select_witness"))
