"""Central Prompt Catalog — maps logical keys to versioned prompt files.
集中 Prompt 目录 — 将逻辑键映射到版本化 Prompt 文件。

Every prompt asset is a separate, versioned file in ``prompts/``.
Agents request prompts by logical key; the catalog resolves the path
and records the active version. Hard-coding long prompts in Python is prohibited.
每个 Prompt 资源都是 ``prompts/`` 下独立的版本化文件。
Agent 通过逻辑键请求 Prompt；Catalog 解析路径并记录活跃版本。
禁止在 Python 中硬编码长 Prompt。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True)
class PromptAsset:
    """One immutable logical Prompt binding. / 一个不可变的逻辑 Prompt 绑定。"""

    key: str
    path: Path
    version: str
    text: str


class PromptCatalog:
    """Read-only mapping from logical keys to versioned prompt files.
    从逻辑键到版本化 Prompt 文件的只读映射。

    Usage / 用法::

        catalog = PromptCatalog(PROJECT_ROOT / "prompts")
        count_prompt = catalog["count_tile"]          # -> "count_tile_v4.md" text
        version      = catalog.version("count_tile")   # -> "count-point-v4"
    """

    # ── logical → filename mapping / 逻辑键 → 文件名映射 ──────────────────
    # Bind versions by logical use because one file can serve multiple contracts.
    # 按逻辑用途绑定版本，因为同一文件可以服务于多个请求契约。
    _SPECS: ClassVar[dict[str, tuple[str, str]]] = {
        "count": ("count_tile_v4.md", "count-point-v4"),
        "count_tile": ("count_tile_v4.md", "count-point-v4"),
        "zero_review": ("missing_point_review_v3.md", "missing-point-review-v3"),
        "count_zero_review": ("missing_point_review_v3.md", "missing-point-review-v3"),
        "vrsbench_proposal": ("general_vqa_v1.md", "general-vqa-v1-count-proposal"),
        "count_proposal": ("general_vqa_v1.md", "general-vqa-v1-count-proposal"),
        "vrsbench_localizer": ("count_localize_v1.md", "count-localize-v1"),
        "count_localize": ("count_localize_v1.md", "count-localize-v1"),
        "target": ("target_parse_v1.md", "target-parse-v1"),
        "target_parse": ("target_parse_v1.md", "target-parse-v1"),
        "change": ("change_v1.md", "change-expert-v1"),
        "change_analysis": ("change_analysis_v2.md", "change-analysis-v2"),
        "change_verification": ("change_verification_v4.md", "change-verification-v4"),
        "spatial": ("spatial_v4.md", "spatial-v4"),
        "spatial_grid": ("spatial_v5.md", "spatial-v5"),
        "spatial_review": ("spatial_candidate_review_v2.md", "spatial-candidate-review-v2"),
        "spatial_grid_review": ("spatial_candidate_review_v3.md", "spatial-candidate-review-v3"),
        "general": ("general_vqa_v2.md", "general-vqa-v2"),
        "general_vqa": ("general_vqa_v2.md", "general-vqa-v2"),
        "grounding": ("general_vqa_v2.md", "general-vqa-v2"),
        "caption": ("caption_v1.md", "caption-v1"),
        "seam": ("seam_verify_v1.md", "seam-verify-v1"),
        "seam_verify": ("seam_verify_v1.md", "seam-verify-v1"),
        "router": ("router_v1.md", "router-v1"),
        "count_judge": ("deepseek_judge_v1.md", "deepseek-judge-v1"),
        "deepseek_judge": ("deepseek_judge_v1.md", "deepseek-judge-v1"),
        "vqa_judge": ("deepseek_vqa_judge_v1.md", "deepseek-vqa-judge-v1"),
        "deepseek_vqa_judge": ("deepseek_vqa_judge_v1.md", "deepseek-vqa-judge-v1"),
        "json_repair": ("json_repair_v1.md", "json-repair-v1"),
    }

    def __init__(self, prompts_root: Path) -> None:
        if not prompts_root.is_dir():
            raise FileNotFoundError(f"Prompts directory not found: {prompts_root}")
        self._root = prompts_root
        self._cache: dict[str, PromptAsset] = {}

    def __getitem__(self, key: str) -> str:
        """Return prompt text for a logical key. / 返回逻辑键对应的 Prompt 文本。"""
        return self.asset(key).text

    def asset(self, key: str) -> PromptAsset:
        """Return the immutable binding for a logical key.
        返回逻辑键对应的不可变绑定。
        """

        if key in self._cache:
            return self._cache[key]
        spec = self._SPECS.get(key)
        if spec is None:
            raise KeyError(f"Unknown prompt key: {key!r}. Available: {sorted(self._SPECS)}")
        filename, version = spec
        path = self._root / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Prompt file missing for key {key!r}: {path}"
            )
        asset = PromptAsset(key=key, path=path, version=version, text=path.read_text(encoding="utf-8"))
        self._cache[key] = asset
        return asset

    def version(self, key: str) -> str:
        """Return the declared version for a logical key. / 返回逻辑键的声明版本。"""
        return self.asset(key).version

    def all_keys(self) -> list[str]:
        """Return all registered logical keys. / 返回所有已注册的逻辑键。"""
        return sorted(self._SPECS)

    def all_paths(self) -> list[Path]:
        """Return all prompt file paths. / 返回所有 Prompt 文件路径。"""
        return sorted({self._root / filename for filename, _ in self._SPECS.values()})

    def snapshot(self, destination: Path) -> dict[str, str]:
        """Copy all prompt files to a snapshot directory; return content hashes.
        将所有 Prompt 文件复制到快照目录；返回内容哈希。
        """
        import hashlib
        import shutil

        destination.mkdir(parents=True, exist_ok=True)
        hashes: dict[str, str] = {}
        for key in self.all_keys():
            src = self.asset(key).path
            if not src.is_file():
                raise FileNotFoundError(f"Prompt missing for snapshot: {src}")
            dst = destination / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            hashes[key] = hashlib.sha256(src.read_bytes()).hexdigest()
        return hashes
