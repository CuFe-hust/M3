from __future__ import annotations

import tomllib
from pathlib import Path


def test_setuptools_discovers_only_the_runtime_package() -> None:
    project_file = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(project_file.read_text(encoding="utf-8"))

    assert "Pillow>=10.0.0" in project["project"]["dependencies"]
    assert project["tool"]["setuptools"]["packages"]["find"]["include"] == [
        "spacers_agent*"
    ]
