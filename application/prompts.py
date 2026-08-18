"""Versioned prompt catalog: explicit bindings, clear missing errors, texts
loaded once at construction — never re-read per sample.
版本化 Prompt 目录：显式绑定、缺失明确报错、文本构造时一次性加载——绝不
逐样本重读文件。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Explicit logical key → (filename, version) bindings for the active prompt
# set. Files not bound here are historical assets, intentionally unused.
# 显式逻辑键 →（文件名、版本）绑定，覆盖现役 Prompt 集。未绑定的文件是
# 历史资产，有意不使用。
_BINDINGS: dict[str, tuple[str, str]] = {
    "count_tile": ("count_tile_v4.md", "v4"),
    "count_localize": ("count_localize_v1.md", "v1"),
    "zero_review": ("missing_point_review_v3.md", "v3"),
    "change": ("change_dual_path_v3.md", "v3"),
    "general": ("general_vqa_v3.md", "v3"),
    "grounding": ("general_vqa_v3.md", "v3"),
    "caption": ("caption_v1.md", "v1"),
    "seam": ("seam_review_v2.md", "v2"),
    "visual_task_plan": ("visual_task_plan_v5.md", "v5"),
    "count_judge": ("deepseek_judge_v1.md", "v1"),
    "vqa_judge": ("deepseek_vqa_judge_v2.md", "v2"),
    "json_repair": ("json_repair_v1.md", "v1"),
}


class PromptNotFoundError(KeyError):
    """Stable error for a missing prompt file; the message carries the key and
    filename only, never raw file content. Prompt 文件缺失的稳定错误；消息只
    携带 key 与文件名，绝不携带原始文件内容。"""

    def __init__(self, key: str, filename: str) -> None:
        super().__init__(f"prompt {key!r} missing: {filename}")
        self.key = key
        self.filename = filename


@dataclass(frozen=True)
class PromptAsset:
    """One loaded prompt: logical key, source path, version, and text.
    一条已加载 Prompt：逻辑键、源路径、版本与文本。"""

    key: str
    path: Path
    version: str
    text: str


class PromptCatalog:
    """Load every bound prompt once at construction; accesses never touch the
    filesystem again. 构造时一次性加载全部绑定 Prompt；此后访问绝不触碰
    文件系统。"""

    def __init__(self, prompts_root: Path) -> None:
        self._prompts_root = prompts_root
        self._assets: dict[str, PromptAsset] = {}
        for key, (filename, version) in _BINDINGS.items():
            path = prompts_root / filename
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise PromptNotFoundError(key, filename) from exc
            self._assets[key] = PromptAsset(
                key=key, path=path, version=version, text=text
            )

    def __getitem__(self, key: str) -> str:
        """Prompt text for a logical key. 逻辑键对应的 Prompt 文本。"""

        return self.asset(key).text

    def asset(self, key: str) -> PromptAsset:
        """Full prompt asset; unknown keys fail with a stable error.
        完整 Prompt 资产；未知键以稳定错误失败。"""

        try:
            return self._assets[key]
        except KeyError as exc:
            raise PromptNotFoundError(key, "") from exc

    def version(self, key: str) -> str:
        """Declared prompt version for a logical key. 逻辑键的声明版本。"""

        return self.asset(key).version

    def all_keys(self) -> tuple[str, ...]:
        """All bound keys in stable declaration order. 按稳定声明顺序的全部
        绑定键。"""

        return tuple(self._assets)

    def snapshot_paths(self) -> list[Path]:
        """Source paths of every loaded prompt, deduplicated in stable order,
        for the run manifest prompt snapshot (two keys may share one file).
        全部已加载 Prompt 的源路径（去重、稳定顺序），供 run manifest 的
        prompt 快照使用（两个键可共享同一文件）。"""

        seen: set[str] = set()
        paths: list[Path] = []
        for asset in self._assets.values():
            key = str(asset.path)
            if key in seen:
                continue
            seen.add(key)
            paths.append(asset.path)
        return paths

    def texts(self) -> dict[str, str]:
        """Key → text mapping (cached; no re-reads). key → 文本映射（已缓存；
        不重读）。"""

        return {key: asset.text for key, asset in self._assets.items()}
