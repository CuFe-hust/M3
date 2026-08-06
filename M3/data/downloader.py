"""Official dataset downloaders, organized as one function per dataset.
官方数据集下载器，每个数据集对应一个下载函数。
"""

from __future__ import annotations

import argparse
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Callable


# Default local data root shared by download, load, and validation.
# 下载、读取与校验共用的默认本地数据根目录。
DEFAULT_DATA_ROOT = Path("/data")

# Official Hugging Face dataset repository IDs.
# Hugging Face 官方数据集仓库 ID。
DATASET_REPOS = {
    "vrsbench": "xiang709/VRSBench",
    "mme_real_rs": "yifanzhang114/MME-RealWorld",
    "xlrs_caption_en": "initiacms/XLRS-Bench_caption_en",
    "xlrs_grounding_en": "initiacms/XLRS-Bench_visual_grounding_en",
    "xlrs_vqa_lite": "initiacms/XLRS-Bench-lite",
    "levir_cc": "lcybuaa/LEVIR-CC",
}


def _snapshot_download(repo_id: str, target: Path) -> None:
    """Download one official Hugging Face dataset repository into target.
    将单个 Hugging Face 官方数据集仓库下载到目标目录。
    """

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("Install requirements.txt before downloading datasets.") from error
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=target)


def _extract_archives(root: Path) -> None:
    """Extract dataset zip archives in place when needed.
    按需就地解压数据集压缩包。
    """

    for archive in root.rglob("*.zip"):
        destination = archive.with_suffix("")
        if destination.exists():
            continue
        with zipfile.ZipFile(archive) as zip_file:
            zip_file.extractall(destination)


def download_vrsbench(data_root: Path) -> Path:
    """Download the official VRSBench repository.
    下载官方 VRSBench 数据集。
    """

    target = data_root / "vrsbench"
    _snapshot_download(DATASET_REPOS["vrsbench"], target)
    _extract_archives(target)
    return target


def download_xlrs_bench(data_root: Path) -> Path:
    """Download the official XLRS-Bench releases under one root.
    将官方 XLRS-Bench 各发布下载到同一根目录。
    """

    target = data_root / "xlrs_bench"
    for name in ("xlrs_caption_en", "xlrs_grounding_en", "xlrs_vqa_lite"):
        _snapshot_download(DATASET_REPOS[name], target / DATASET_REPOS[name].split("/")[-1])
    return target


def download_levir_cc(data_root: Path) -> Path:
    """Download the official LEVIR-CC repository.
    下载官方 LEVIR-CC 数据集。
    """

    target = data_root / "levir_cc"
    _snapshot_download(DATASET_REPOS["levir_cc"], target)
    _extract_archives(target)
    return target


def download_mme_real_rs(data_root: Path) -> Path:
    """Download the official MME-RealWorld repository.
    下载官方 MME-RealWorld 数据集。
    """

    target = data_root / "mme_real_rs"
    _snapshot_download(DATASET_REPOS["mme_real_rs"], target)
    return target


# Extension point: add a new dataset by adding one function and registering it here.
# 扩展点：新增数据集时添加一个下载函数并在此注册。
DATASET_DOWNLOADERS: dict[str, Callable[[Path], Path]] = {
    "vrsbench": download_vrsbench,
    "xlrs_bench": download_xlrs_bench,
    "levir_cc": download_levir_cc,
    "mme_real_rs": download_mme_real_rs,
}


def download_datasets(names: Iterable[str], data_root: Path) -> dict[str, Path]:
    """Download named datasets through the per-dataset dispatch table.
    通过单数据集分发表下载指定数据集。
    """

    root = Path(data_root)
    downloaded: dict[str, Path] = {}
    for name in names:
        if name in DATASET_DOWNLOADERS:
            downloaded[name] = DATASET_DOWNLOADERS[name](root)
        elif name in DATASET_REPOS:
            # Legacy flat names (for example xlrs_caption_en) keep their old target path.
            # 旧式平铺名称（如 xlrs_caption_en）保持原有目标路径。
            target = root / name
            _snapshot_download(DATASET_REPOS[name], target)
            downloaded[name] = target
        else:
            raise ValueError(f"Unsupported dataset download target: {name}")
    return downloaded


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download official dataset releases into a data root.")
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA_ROOT, help="Target data root (default: /data).")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted({*DATASET_DOWNLOADERS, *DATASET_REPOS}),
        default=sorted(DATASET_DOWNLOADERS),
        help="Dataset names to download.",
    )
    args = parser.parse_args(argv)
    downloaded = download_datasets(args.datasets, args.root)
    for name, path in downloaded.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    _main()
