"""Backward-compatible exports; the unified implementation lives in data/loader.py.
兼容转发层：统一实现位于 data/loader.py。
"""

from data.downloader import DATASET_REPOS, download_datasets
from data.loader import _caption_texts, _load_vrsbench, _normalize_box, load_samples

__all__ = [
    "DATASET_REPOS",
    "download_datasets",
    "load_samples",
    "_caption_texts",
    "_load_vrsbench",
    "_normalize_box",
]
