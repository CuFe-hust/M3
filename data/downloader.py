"""Explicit dataset download utilities.

显式数据集下载工具。唯一自动下载路径：绝不隐式下载——run-dataset/ask/
serve/adapters/loader 均不得调用本模块。下载使用惰性 huggingface_hub
（缺失时稳定错误，导入本模块无副作用、不联网）；zip 提取使用安全成员
校验（拒绝 ..、绝对、drive、UNC 成员路径，杜绝 zip-slip）。
"""

from __future__ import annotations

import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

# Official Hugging Face dataset targets. / 官方 Hugging Face 数据集目标。
OFFICIAL_DOWNLOAD_TARGETS: dict[str, str] = {
    "vrsbench": "xiang709/VRSBench",
    "mme_realworld": "yifanzhang114/MME-RealWorld",
    "xlrs_caption": "initiacms/XLRS-Bench_caption_en",
    "xlrs_grounding": "initiacms/XLRS-Bench_visual_grounding_en",
    "xlrs_lite": "initiacms/XLRS-Bench-lite",
    "levir_cc": "lcybuaa/LEVIR-CC",
}

# Zip member names that are always rejected before extraction: dot segments,
# absolute paths, drive/UNC prefixes, and control characters.
# 提取前一律拒绝的 zip 成员名：dot 段、绝对路径、drive/UNC 前缀与控制字符。
_WINDOWS_RESERVED_STEMS = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)


def dataset_download_target(dataset: str) -> str:
    """Return the official repository id for one dataset key; unknown keys
    fail stably. 返回一个数据集键的官方仓库 id；未知键稳定失败。"""

    target = OFFICIAL_DOWNLOAD_TARGETS.get(dataset)
    if target is None:
        raise ValueError(
            "unknown download dataset; expected one of "
            f"{sorted(OFFICIAL_DOWNLOAD_TARGETS)}"
        )
    return target


def download_dataset(
    dataset: str,
    *,
    root: Path,
    hf_token: str | None = None,
) -> Path:
    """Download one official dataset snapshot into ``root/<dataset>`` and
    extract any archives it contains. Network happens only here, on explicit
    invocation. 将一份官方数据集快照下载到 ``root/<dataset>`` 并提取其中
    的归档。网络只发生在这里、显式调用时。"""

    target = dataset_download_target(dataset)
    destination = root.expanduser().resolve() / dataset
    destination.mkdir(parents=True, exist_ok=True)
    hub = _import_hub()
    snapshot = hub.snapshot_download(
        repo_id=target,
        local_dir=destination,
        token=hf_token,
    )
    snapshot_path = Path(snapshot).resolve()
    extract_archives(snapshot_path)
    return snapshot_path


def extract_archives(directory: Path) -> list[Path]:
    """Extract every ``*.zip`` directly inside the directory with strict
    member validation; unsafe members fail the whole archive stably instead
    of being silently skipped. 以严格成员校验提取目录内的每个 ``*.zip``；
    不安全成员使整个归档稳定失败，绝不静默跳过。"""

    directory = directory.expanduser().resolve()
    extracted: list[Path] = []
    for archive_path in sorted(directory.glob("*.zip")):
        _extract_one(archive_path, directory)
        extracted.append(archive_path)
    return extracted


def _extract_one(archive_path: Path, directory: Path) -> None:
    """Extract one archive into the directory; member paths must stay inside
    it (reject dot segments, absolute, drive, UNC, and control characters).
    将一份归档提取到目录内；成员路径必须留在目录内（拒绝 dot 段、绝对、
    drive、UNC 与控制字符）。"""

    try:
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                if not _safe_zip_member(member.filename):
                    raise ValueError("unsafe archive member path rejected")
            archive.extractall(directory)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("archive extraction failed") from exc


def _safe_zip_member(name: str) -> bool:
    """A member path is safe only when it is a relative, non-escaping path
    without drive/UNC prefixes or control characters.
    成员路径只有在相对、无逃逸段、无 drive/UNC 前缀且无控制字符时安全。"""

    if not name or "\x00" in name or "\r" in name or "\n" in name:
        return False
    normalized = name.replace("\\", "/")
    segments = normalized.split("/")
    if any(segment in {"", ".", ".."} for segment in segments):
        return False
    if _is_absolute_like(normalized):
        return False
    basename = segments[-1]
    stem = basename.split(".")[0].casefold()
    if stem.upper() in _WINDOWS_RESERVED_STEMS:
        return False
    return True


def _is_absolute_like(value: str) -> bool:
    """Detect absolute paths on both Windows and POSIX spellings.
    同时识别 Windows 与 POSIX 写法的绝对路径。"""

    if value.startswith(("/", "\\")):
        return True
    if len(value) >= 3 and value[1] == ":" and value[2] in "/\\":
        return True
    if value.startswith("\\\\"):
        return True
    return False


def _import_hub() -> Any:
    """Lazily import huggingface_hub; a missing dependency fails stably and
    importing this module never touches the network.
    惰性导入 huggingface_hub；缺失依赖稳定失败，导入本模块绝不联网。"""

    try:
        import huggingface_hub
    except ImportError as exc:
        raise ValueError(
            "huggingface_hub is required for dataset download"
        ) from exc
    return huggingface_hub
