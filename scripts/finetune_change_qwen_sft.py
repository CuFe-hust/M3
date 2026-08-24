#!/usr/bin/env python3
"""Thin ChangeAgent profile wrapper around the shared Phase2 trainer."""

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
