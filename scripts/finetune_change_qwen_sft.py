#!/usr/bin/env python3
"""Backward-compatible ChangeAgent profile wrapper.

New model-family-agnostic runs should use ``scripts/finetune_multimodal_sft.py``
with ``--data-profile change_agent``.  This file remains as the stable legacy
CLI for existing Phase2 checkpoints and tests.
"""

from __future__ import annotations

import sys
from typing import Sequence

from scripts.finetune_qwen3vl_phase2 import main as _shared_main


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--data_profile" in args or "--data-profile" in args:
        raise ValueError("finetune_change_qwen_sft fixes --data_profile=change_agent")
    if "--repeat_group_key" not in args and "--repeat-group-key" not in args:
        args = ["--repeat_group_key", "task", *args]
    _shared_main(["--data_profile", "change_agent", *args])


if __name__ == "__main__":
    main()
