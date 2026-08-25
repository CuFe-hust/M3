#!/usr/bin/env python3
"""Phase 2 dataset pipeline: episodes -> augmentation -> rendering -> encoding.

Consumes only the canonical episode JSONL produced by
scripts/prepare_qwen3vl_phase2_sft.py (docs/train/01) and provides:

- safe image path resolution against explicit image-source roots;
- online, task-aware geometry augmentation (rotate90 / affine / perspective)
  with synchronized input/target box transforms and quality gates;
- coordinate-preserving imaging degradation simulation (contrast, brightness,
  vignette, defocus/motion blur, sensor noise, JPEG);
- unified Qwen chat rendering (grounding JSON boxes, boxed/unboxed VQA,
  GeoChat refer/identify/conversation);
- processor encoding with chat-template assistant masks, delta alignment for
  image token expansion, and turn-pair truncation;
- right-padded collation of mixed-length sequences and mixed visual token
  counts.

It never parses VRSBench/GeoChat raw annotations, never loads the main model,
never attaches LoRA, never creates an optimizer and never saves checkpoints.

本模块只消费第一轮生成的 canonical Episode JSONL：安全解析图片路径、在线任务
感知几何增强（同步变换框）、坐标保持的恶劣成像质量模拟、统一 Qwen 对话渲染、
processor 编码与 labels 构造、batch collate。不解析原始标注、不加载主模型。

Processor contract (verified against transformers 5.14.1, the pinned version
in the M3 conda env; / 对照 M3 环境钉死的 transformers 5.14.1 验证):
- apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
  renders the conversation text with a single <|image_pad|> placeholder;
- apply_chat_template(..., tokenize=True, return_dict=True,
  return_assistant_tokens_mask=True) returns "assistant_masks" (4.x used
  "assistant_tokens_mask"; both are accepted);
- processor(text=[text], images=[image], return_tensors="pt") returns
  input_ids / attention_mask / pixel_values / image_grid_thw /
  mm_token_type_ids (default-on for Qwen3-VL); per-image pixel_values is
  (grid_h*grid_w, C*patch*patch) so batches concatenate along dim 0.

Design contracts / 关键设计契约:
- group seed = sha256(seed | epoch | parent_episode_id); paired boxed/unboxed
  VQA views share the seed so one epoch shows the identical image;
- geometry and degradation use independent random substreams; each
  degradation step draws parameters from its own substream so adding one
  degradation never changes another step's parameters;
- any required box failing the quality gates falls the whole episode back to
  the identity geometry transform (never partial boxes);
- degradation failure falls back to the geometry-stage output with a stable
  code; degradations never touch boxes, sizes or pixel grids;
- labels come from the chat-template assistant mask verbatim (fixed policy),
  aligned across image-token expansion by the total length delta;
- truncation only at complete user/assistant turn pairs, keeping the image
  turn; a single over-long pair raises EpisodeTooLongError.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import numpy as np
    import cv2
    import torch
    from PIL import Image
    from torch.utils.data import Dataset
except ImportError as exc:  # pragma: no cover - dependency gate
    raise ImportError(
        "scripts/qwen3vl_phase2_data.py requires numpy, opencv-python, torch "
        "and Pillow (declared project dependencies); refusing to run with a "
        "silently disabled pipeline. 数据管线要求 numpy/opencv-python/torch/"
        "Pillow，缺失时明确失败而不是静默关闭。"
    ) from exc

# Loss ignore index; label padding uses the same value.
# 损失忽略索引；label padding 也使用该值。
IGNORE_INDEX = -100

# Canonical degradation pipeline order (fixed; only selected steps run).
# 恶劣成像质量模拟的固定执行顺序（只执行被选中的步骤）。
DEGRADATION_ORDER = (
    "low_contrast",
    "brightness",
    "vignette",
    "blur",
    "sensor_noise",
    "jpeg",
)
BLUR_KINDS = ("defocus", "motion")

_IMAGE_TOKEN_LITERAL = "<image>"
_P_TAG = re.compile(r"</?p>")

_GEOMETRY_FALLBACK_BOX_QUALITY = "geometry_fallback:box_quality"


# ---------------------------------------------------------------------------
# Stable error types (messages never carry machine absolute paths).
# 稳定错误类型（错误信息不携带机器绝对路径）。
# ---------------------------------------------------------------------------


class Phase2DataError(Exception):
    """Base class for pipeline errors. / 数据管线错误基类。"""


class ImagePathError(Phase2DataError):
    """Image path or image load failure with a stable code.

    The message carries the relative image path only; the resolved absolute
    path is never part of the message.
    """

    def __init__(self, code: str, relative_path: str, detail: str = "") -> None:
        super().__init__(f"image error {code}: {relative_path}")
        self.code = code
        self.relative_path = relative_path
        self.detail = detail


class EpisodeTooLongError(Phase2DataError):
    """A single turn pair exceeds max_seq_length; no all-(-100) labels are
    ever produced. / 单个 turn pair 超过长度限制；绝不产生全 -100 labels。"""

    def __init__(self, episode_id: str) -> None:
        super().__init__(f"episode_too_long: {episode_id}")
        self.episode_id = episode_id


class FeatureError(Phase2DataError):
    """Encoding/label construction failure with a stable code.
    编码/label 构造失败，带稳定错误码。"""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"feature error {code}")
        self.code = code
        self.detail = detail


class CollatorError(Phase2DataError):
    """Collation failure. / batch 拼接失败。"""


# ---------------------------------------------------------------------------
# DatasetRootConfig: image source -> root resolution with path safety.
# 图片源 root 映射与路径安全解析。
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetRootConfig:
    """Mapping of image_source to the dataset root directory it resolves
    against. Training CLI provides this explicitly (e.g. vrsbench=/data/
    VRSBench-full, geochat=/data/GeoChat/images).
    image_source -> 数据集 root 目录映射，由训练 CLI 显式提供。"""

    roots: Mapping[str, str]

    def image_path(self, image_source: str, relative_path: str) -> Path:
        """Resolve a relative episode image path inside its declared root.

        Rejects absolute paths, Windows drives, UNC, '.', '..', backslashes
        and nested escapes; after resolve() the file must still live inside
        the declared root (symlink escapes included).
        """
        if image_source not in self.roots:
            raise ImagePathError("unknown_image_source", relative_path, image_source)
        root = Path(self.roots[image_source])
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise ImagePathError("unsafe_image_path", str(relative_path))
        if "\\" in relative_path or "\x00" in relative_path:
            raise ImagePathError("unsafe_image_path", relative_path)
        if re.match(r"^[A-Za-z]:", relative_path) or relative_path.startswith("//"):
            raise ImagePathError("unsafe_image_path", relative_path)
        rel = Path(relative_path)
        if rel.is_absolute() or rel.drive:
            raise ImagePathError("unsafe_image_path", relative_path)
        # Path.parts collapses '.' segments, so check the raw segments.
        # Path.parts 会折叠 '.' 段，因此用原始字符串分段检查。
        segments = relative_path.split("/")
        if not segments or any(s in (".", "..") for s in segments):
            raise ImagePathError("unsafe_image_path", relative_path)
        resolved = (root / rel).resolve()
        if not resolved.is_relative_to(root.resolve()):
            raise ImagePathError("unsafe_image_path", relative_path)
        return resolved


# ---------------------------------------------------------------------------
# AugmentationConfig: all probabilities and magnitudes live here; no magic
# numbers inside __getitem__.
# 增强配置：所有概率与幅度都集中在此，__getitem__ 内不散落魔法数字。
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AugmentationConfig:
    """Online augmentation configuration (train only; validation is always
    identity). 在线增强配置（仅 train；validation 恒为 identity）。"""

    enabled: bool = True
    # --- geometry ---------------------------------------------------------
    rotate90_prob: float = 0.5
    affine_rotation_prob: float = 0.3
    affine_rotation_degrees: tuple[float, float] = (-5.0, 5.0)
    scale_prob: float = 0.3
    scale_range: tuple[float, float] = (0.95, 1.05)
    translate_prob: float = 0.2
    translate_max_fraction: float = 0.02
    perspective_prob: float = 0.2
    perspective_frac_range: tuple[float, float] = (0.01, 0.02)
    # --- box quality gates ------------------------------------------------
    box_visible_ratio_min: float = 0.5
    box_aabb_inflation_max: float = 2.0
    # --- degradations -----------------------------------------------------
    degradation_probability: float = 0.45
    min_degradations: int = 1
    max_degradations: int = 3
    brightness_weight: float = 1.0
    brightness_factor_min: float = 0.55
    brightness_factor_max: float = 0.90
    contrast_weight: float = 0.6
    contrast_factor_min: float = 0.55
    contrast_factor_max: float = 0.90
    defocus_weight: float = 0.6
    defocus_kernel_sizes: tuple[int, ...] = (3, 5)
    defocus_sigma_min: float = 0.4
    defocus_sigma_max: float = 1.2
    motion_blur_weight: float = 0.4
    motion_blur_kernel_sizes: tuple[int, ...] = (3, 5, 7)
    motion_blur_angle_min: float = 0.0
    motion_blur_angle_max: float = 180.0
    sensor_noise_weight: float = 0.7
    noise_sigma_min: float = 0.005
    noise_sigma_max: float = 0.03
    jpeg_weight: float = 0.5
    jpeg_quality_min: int = 55
    jpeg_quality_max: int = 90
    vignette_weight: float = 0.4
    vignette_edge_factor_min: float = 0.55
    vignette_edge_factor_max: float = 0.85
    vignette_power_min: float = 1.5
    vignette_power_max: float = 3.0

    def __post_init__(self) -> None:
        def check_prob(name: str, v: float) -> None:
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {v}")

        for name, v in (
            ("rotate90_prob", self.rotate90_prob),
            ("affine_rotation_prob", self.affine_rotation_prob),
            ("scale_prob", self.scale_prob),
            ("translate_prob", self.translate_prob),
            ("perspective_prob", self.perspective_prob),
            ("degradation_probability", self.degradation_probability),
        ):
            check_prob(name, v)
        if not (0.0 < self.min_degradations <= self.max_degradations):
            raise ValueError(
                f"degradation count range invalid: "
                f"{self.min_degradations}..{self.max_degradations}"
            )
        if self.max_degradations > len(DEGRADATION_ORDER):
            raise ValueError(
                f"max_degradations {self.max_degradations} exceeds the "
                f"{len(DEGRADATION_ORDER)} available degradation steps"
            )
        if any(k % 2 == 0 for k in self.defocus_kernel_sizes) or any(
            k % 2 == 0 for k in self.motion_blur_kernel_sizes
        ):
            raise ValueError("blur kernel sizes must be odd")
        if self.brightness_factor_min > self.brightness_factor_max:
            raise ValueError("brightness factor range invalid")
        if self.brightness_factor_max > 1.0:
            raise ValueError("brightness factor must never exceed 1.0")
        if self.contrast_factor_min > self.contrast_factor_max:
            raise ValueError("contrast factor range invalid")
        if self.contrast_factor_max > 1.0:
            raise ValueError("contrast factor must never exceed 1.0")
        for lo, hi, name in (
            (self.noise_sigma_min, self.noise_sigma_max, "noise sigma"),
            (self.jpeg_quality_min, self.jpeg_quality_max, "jpeg quality"),
            (self.vignette_edge_factor_min, self.vignette_edge_factor_max, "vignette edge"),
            (self.vignette_power_min, self.vignette_power_max, "vignette power"),
        ):
            if lo > hi:
                raise ValueError(f"{name} range invalid")


# ---------------------------------------------------------------------------
# Lazy episode store: byte-offset index, one JSON line parsed per access.
# 惰性 episode 存储：字节偏移索引，每次访问只解析一行。
# ---------------------------------------------------------------------------


class LazyJsonLines:
    """Line-based lazy JSON reader with a byte-offset index.

    Each access opens the file itself (safe across forked DataLoader workers
    sharing no file-descriptor state) and parses exactly one line.
    基于行偏移的惰性 JSON 读取；每次访问自行打开文件并解析一行。
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._offsets: list[int] = []
        with open(self._path, "r", encoding="utf-8") as fh:
            while True:
                pos = fh.tell()
                line = fh.readline()
                if not line:
                    break
                if line.strip():
                    self._offsets.append(pos)
        if not self._offsets:
            raise ValueError(f"empty episode file: {self._path.name}")

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, index: int) -> dict:
        with open(self._path, "r", encoding="utf-8") as fh:
            fh.seek(self._offsets[index])
            line = fh.readline()
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:  # pragma: no cover - corrupt input
            raise ValueError(
                f"corrupt episode line {index} in {self._path.name}"
            ) from exc


# ---------------------------------------------------------------------------
# Deterministic seeds
# 确定性种子
# ---------------------------------------------------------------------------


def group_seed_hex(seed: str, epoch: int, group_id: str) -> str:
    """Group seed: sha256(seed | epoch | group_id).

    Paired boxed/unboxed VQA views share parent_episode_id, so one epoch
    shows them the identical augmented image. No Python hash(), no runtime
    randomness, no worker dependence.
    """
    return hashlib.sha256(
        f"{seed}|{epoch}|{group_id}".encode("utf-8")
    ).hexdigest()


def _sub_seed(group_seed: str, label: str) -> str:
    return hashlib.sha256(
        (group_seed + "|" + label).encode("utf-8")
    ).hexdigest()


def _rng(group_seed: str, label: str) -> random.Random:
    return random.Random(int(_sub_seed(group_seed, label)[:16], 16))


# ---------------------------------------------------------------------------
# Image loading (file handles closed promptly; PIL objects never persisted).
# 图片加载（及时关闭文件句柄；PIL 对象绝不持久化）。
# ---------------------------------------------------------------------------


def load_image_rgb(path: Path, relative_path: str) -> Image.Image:
    """Open + decode + convert to RGB uint8 PIL image.

    RGBA/grayscale/palette inputs are converted with the fixed rule: PIL
    convert("RGB") (alpha channel dropped, not composited). Decode failures
    raise ImagePathError with a stable code; the message carries only the
    relative path.
    """
    try:
        with Image.open(path) as im:
            im.load()
            width, height = im.size
            if width <= 0 or height <= 0:
                raise ImagePathError("image_decode_error", relative_path)
            if im.mode != "RGB":
                im = im.convert("RGB")
            return im
    except FileNotFoundError:
        raise ImagePathError("image_missing", relative_path)
    except (OSError, ValueError, Image.DecompressionBombError):
        raise ImagePathError("image_decode_error", relative_path)


# ---------------------------------------------------------------------------
# Box geometry: xyxy_999 <-> pixels, transforms, quality gates.
# 框几何：xyxy_999 与像素互转、变换、质量门禁。
# ---------------------------------------------------------------------------


def box_999_to_pixels(xyxy_999: Sequence[float], w: int, h: int) -> list[float]:
    """0..999 xyxy -> float pixel xyxy (box_999 was derived from 0..1
    normalized coords times 999, so pixel = v / 999 * size).
    0..999 xyxy -> 浮点像素 xyxy（box_999 由 0..1 归一化坐标乘 999 而来）。"""
    return [
        xyxy_999[0] / 999.0 * w,
        xyxy_999[1] / 999.0 * h,
        xyxy_999[2] / 999.0 * w,
        xyxy_999[3] / 999.0 * h,
    ]


def pixels_to_box_999(xyxy: Sequence[float], w: int, h: int) -> list[int]:
    """Float pixel xyxy (clipped to canvas) -> rounded 0..999 xyxy.
    浮点像素 xyxy（裁剪到画布）-> 四舍五入的 0..999 xyxy。"""
    x1 = int(np.clip(round(xyxy[0] / w * 999.0), 0, 999))
    y1 = int(np.clip(round(xyxy[1] / h * 999.0), 0, 999))
    x2 = int(np.clip(round(xyxy[2] / w * 999.0), 0, 999))
    y2 = int(np.clip(round(xyxy[3] / h * 999.0), 0, 999))
    return [x1, y1, x2, y2]


def _quad_from_xyxy(xyxy: Sequence[float]) -> np.ndarray:
    """Four corners of an axis-aligned box. / 轴对齐框的四个角点。"""
    x1, y1, x2, y2 = xyxy
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)


def _quad_area(quad: np.ndarray) -> float:
    """Signed shoelace area (abs) of a quad. / 四边形鞋带面积（绝对值）。"""
    x = quad[:, 0]
    y = quad[:, 1]
    return float(abs(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))))


def _clip_polygon_to_rect(quad: np.ndarray, w: int, h: int) -> np.ndarray:
    """Sutherland-Hodgman clip of a convex-ish quad against the canvas rect.
    将四边形裁剪到画布矩形（Sutherland-Hodgman）。"""
    poly = quad
    for edge in range(4):
        if len(poly) == 0:
            return poly
        if edge == 0:  # x >= 0
            inside = lambda p: p[0] >= 0  # noqa: E731
        elif edge == 1:  # x <= w
            inside = lambda p: p[0] <= w  # noqa: E731
        elif edge == 2:  # y >= 0
            inside = lambda p: p[1] >= 0  # noqa: E731
        else:  # y <= h
            inside = lambda p: p[1] <= h  # noqa: E731
        out: list[np.ndarray] = []
        n = len(poly)
        for i in range(n):
            cur = poly[i]
            nxt = poly[(i + 1) % n]
            cur_in = inside(cur)
            nxt_in = inside(nxt)
            if cur_in:
                out.append(cur)
            if cur_in != nxt_in:
                # intersect segment with the edge line
                t = 0.0
                if edge == 0:
                    t = (0 - cur[0]) / (nxt[0] - cur[0]) if nxt[0] != cur[0] else 0.0
                elif edge == 1:
                    t = (w - cur[0]) / (nxt[0] - cur[0]) if nxt[0] != cur[0] else 0.0
                elif edge == 2:
                    t = (0 - cur[1]) / (nxt[1] - cur[1]) if nxt[1] != cur[1] else 0.0
                else:
                    t = (h - cur[1]) / (nxt[1] - cur[1]) if nxt[1] != cur[1] else 0.0
                out.append(cur + t * (nxt - cur))
        poly = np.array(out) if out else np.zeros((0, 2))
    return poly


def _transform_quad_rot90(quad: np.ndarray, k: int, w: int, h: int) -> np.ndarray:
    """Apply np.rot90 point mapping to a quad.

    k=1 (clockwise 90): (x, y) -> (y, w-1-x); k=2 (180): -> (w-1-x, h-1-y);
    k=3 (ccw 90): -> (w-1-y, x). Matches np.rot90(image, k) exactly.
    """
    x = quad[:, 0]
    y = quad[:, 1]
    if k == 1:
        return np.stack([y, w - 1.0 - x], axis=1)
    if k == 2:
        return np.stack([w - 1.0 - x, h - 1.0 - y], axis=1)
    if k == 3:
        return np.stack([w - 1.0 - y, x], axis=1)
    return quad


def _transform_quad_affine(quad: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine matrix to the quad. / 对四边形应用 2x3 仿射矩阵。"""
    ones = np.ones((len(quad), 1), dtype=np.float64)
    pts = np.concatenate([quad, ones], axis=1)
    return pts @ matrix.T


def _transform_quad_perspective(quad: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to the quad. / 对四边形应用 3x3 单应矩阵。"""
    pts = quad.astype(np.float64).reshape(1, -1, 2)
    return cv2.perspectiveTransform(pts, matrix.astype(np.float64))[0]


def _aabb(quad: np.ndarray) -> list[float]:
    return [float(quad[:, 0].min()), float(quad[:, 1].min()),
            float(quad[:, 0].max()), float(quad[:, 1].max())]


def transform_box_quad(
    quad: np.ndarray,
    ops: Sequence[tuple[str, Any]],
    w: int,
    h: int,
) -> np.ndarray:
    """Apply the recorded transform op chain to a quad.

    Ops are ("rot90", k), ("affine", M2x3), ("perspective", H3x3) in the
    same order applied to the image; the canvas size used by each op is the
    current (possibly swapped) size.
    """
    out = quad
    cur_w, cur_h = w, h
    for op, arg in ops:
        if op == "rot90":
            out = _transform_quad_rot90(out, arg, cur_w, cur_h)
            if arg % 2 == 1:
                cur_w, cur_h = cur_h, cur_w
        elif op == "affine":
            out = _transform_quad_affine(out, arg)
        elif op == "perspective":
            out = _transform_quad_perspective(out, arg)
        else:  # pragma: no cover - internal invariant
            raise FeatureError(f"unknown geometry op: {op}")
    return out


def check_box_quality(
    orig_quad: np.ndarray,
    transformed_quad: np.ndarray,
    w: int,
    h: int,
    cfg: AugmentationConfig,
) -> bool:
    """Quality gates for one transformed box (finite / non-degenerate /
    non-empty intersection / visible ratio / AABB inflation).
    单个变换框的质量门禁（有限、非退化、非空交集、可见比例、AABB 膨胀）。"""
    if not np.all(np.isfinite(transformed_quad)):
        return False
    orig_area = _quad_area(orig_quad)
    new_area = _quad_area(transformed_quad)
    if orig_area <= 0.0 or new_area <= 0.0:
        return False
    clipped = _clip_polygon_to_rect(transformed_quad, w, h)
    if len(clipped) < 3:
        return False
    visible = _quad_area(clipped)
    if visible / new_area < cfg.box_visible_ratio_min:
        return False
    aabb = _aabb(transformed_quad)
    aabb_area = max(0.0, (aabb[2] - aabb[0])) * max(0.0, (aabb[3] - aabb[1]))
    if aabb_area / orig_area > cfg.box_aabb_inflation_max:
        return False
    return True


# ---------------------------------------------------------------------------
# Geometry augmentation (geometry_safe only; orientation_locked -> identity).
# 几何增强（仅 geometry_safe；orientation_locked 恒为 identity）。
# ---------------------------------------------------------------------------

_GEOMETRY_INTERP = cv2.INTER_LINEAR
_GEOMETRY_BORDER = cv2.BORDER_REFLECT_101


def apply_geometry(
    image: np.ndarray,
    boxes_px: list[list[float]],
    cfg: AugmentationConfig,
    group_seed: str,
) -> tuple[np.ndarray, list[list[float]], list[dict], str | None]:
    """Apply the sampled geometry chain to image and all boxes together.

    Ops in fixed order: rotate90 (canvas may swap), affine chain (small
    rotation + scale + translate about the canvas center, one matrix, same
    canvas), mild perspective (same canvas). If any required box fails the
    quality gates, the whole episode falls back to the identity transform
    (original image and original boxes) with a stable fallback code.
    """
    rng = _rng(group_seed, "geometry")
    ops: list[dict] = []
    out_img = image
    w, h = image.shape[1], image.shape[0]
    quads = [_quad_from_xyxy(b) for b in boxes_px]
    orig_quads = [q.copy() for q in quads]
    op_chain: list[tuple[str, Any]] = []

    if rng.random() < cfg.rotate90_prob:
        k = int(rng.choice([1, 2, 3]))
        out_img = np.rot90(out_img, k)
        op_chain.append(("rot90", k))
        ops.append({"kind": "rotate90", "k": k})
        if k % 2 == 1:
            w, h = h, w

    angle = (
        rng.uniform(*cfg.affine_rotation_degrees)
        if rng.random() < cfg.affine_rotation_prob
        else 0.0
    )
    scale = rng.uniform(*cfg.scale_range) if rng.random() < cfg.scale_prob else 1.0
    tx = ty = 0.0
    if rng.random() < cfg.translate_prob:
        tx = rng.uniform(-1.0, 1.0) * cfg.translate_max_fraction * w
        ty = rng.uniform(-1.0, 1.0) * cfg.translate_max_fraction * h
    if angle != 0.0 or scale != 1.0 or tx != 0.0 or ty != 0.0:
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
        matrix[0, 2] += tx
        matrix[1, 2] += ty
        out_img = cv2.warpAffine(
            out_img, matrix, (w, h),
            flags=_GEOMETRY_INTERP, borderMode=_GEOMETRY_BORDER,
        )
        op_chain.append(("affine", matrix))
        ops.append({"kind": "affine", "matrix": matrix.tolist()})

    if rng.random() < cfg.perspective_prob:
        frac_w = rng.uniform(*cfg.perspective_frac_range)
        frac_h = rng.uniform(*cfg.perspective_frac_range)
        src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
        dst = src + np.array([
            [rng.uniform(-frac_w * w, frac_w * w), rng.uniform(-frac_h * h, frac_h * h)],
            [rng.uniform(-frac_w * w, frac_w * w), rng.uniform(-frac_h * h, frac_h * h)],
            [rng.uniform(-frac_w * w, frac_w * w), rng.uniform(-frac_h * h, frac_h * h)],
            [rng.uniform(-frac_w * w, frac_w * w), rng.uniform(-frac_h * h, frac_h * h)],
        ], np.float32)
        matrix = cv2.getPerspectiveTransform(src, dst)
        out_img = cv2.warpPerspective(
            out_img, matrix, (w, h),
            flags=_GEOMETRY_INTERP, borderMode=_GEOMETRY_BORDER,
        )
        op_chain.append(("perspective", matrix))
        ops.append({"kind": "perspective", "matrix": matrix.tolist()})

    if not op_chain:
        return image, boxes_px, [], None

    # Transform all boxes through the same op chain, then gate.
    # 所有框走同一 op 链变换，再统一过质量门禁。
    new_quads = [
        transform_box_quad(q, op_chain, image.shape[1], image.shape[0])
        for q in quads
    ]
    new_w, new_h = out_img.shape[1], out_img.shape[0]
    for orig_q, new_q in zip(orig_quads, new_quads):
        if not check_box_quality(orig_q, new_q, new_w, new_h, cfg):
            # Whole-episode identity fallback: never drop single boxes.
            # 整条 episode 回退 identity：绝不只丢弃单个框。
            return image, boxes_px, [], _GEOMETRY_FALLBACK_BOX_QUALITY

    final_boxes: list[list[int]] = []
    for q in new_quads:
        aabb = _aabb(q)
        # Clip the enclosing AABB to the augmented canvas.
        # 将包围 AABB 裁剪到增强后画布。
        aabb = [
            min(max(aabb[0], 0.0), new_w),
            min(max(aabb[1], 0.0), new_h),
            min(max(aabb[2], 0.0), new_w),
            min(max(aabb[3], 0.0), new_h),
        ]
        xyxy = pixels_to_box_999(aabb, new_w, new_h)
        if not (xyxy[0] < xyxy[2] and xyxy[1] < xyxy[3]):
            return image, boxes_px, [], _GEOMETRY_FALLBACK_BOX_QUALITY
        final_boxes.append(xyxy)
    return out_img, final_boxes, ops, None


# ---------------------------------------------------------------------------
# Coordinate-preserving imaging degradation simulation.
# 坐标保持的恶劣成像质量模拟。
# ---------------------------------------------------------------------------


def apply_brightness(img: np.ndarray, factor: float) -> np.ndarray:
    """I_out = clamp(I_in * factor); RGB channels share one factor.
    亮度减弱：RGB 三通道乘同一系数，只减暗不提高。"""
    return np.clip(img * factor, 0.0, 1.0)


def apply_low_contrast(img: np.ndarray, factor: float) -> np.ndarray:
    """Per-channel mean contraction: I = mean_c + f * (I - mean_c).
    低对比度：围绕逐通道均值收缩动态范围（固定均值定义）。"""
    mean = img.mean(axis=(0, 1), keepdims=True)
    return np.clip(mean + factor * (img - mean), 0.0, 1.0)


def apply_vignette(
    img: np.ndarray, edge_factor: float, power: float
) -> np.ndarray:
    """Radial brightness mask centered on the image: mask = 1 at the center,
    monotonically decreasing to edge_factor at the corners. No pixel moves.
    暗角：以图像中心为中心的径向亮度 mask，中心为 1 向边缘单调下降。"""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    r = np.sqrt((xx - (w - 1) / 2.0) ** 2 + (yy - (h - 1) / 2.0) ** 2)
    r_max = float(r.max())
    if r_max > 0.0:
        r = r / r_max
    mask = 1.0 - (1.0 - edge_factor) * np.power(r, power)
    return np.clip(img * mask[..., None], 0.0, 1.0)


def apply_defocus_blur(img: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    """Mild Gaussian blur (odd kernel, capped sigma). / 轻度高斯失焦模糊。"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)


def motion_blur_kernel(kernel_size: int, angle_degrees: float) -> np.ndarray:
    """Center-aligned, normalized 1D linear convolution kernel.

    A horizontal line through the kernel center is rotated about the center
    by angle_degrees, so the output grid never shifts; the kernel sums to 1.
    中心对齐、归一化的一维线性卷积核：直线过核中心绕中心旋转，不平移输出网格。
    """
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0
    matrix = cv2.getRotationMatrix2D(
        (kernel_size / 2.0, kernel_size / 2.0), angle_degrees, 1.0
    )
    kernel = cv2.warpAffine(
        kernel, matrix, (kernel_size, kernel_size),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )
    total = float(kernel.sum())
    if total <= 0.0:
        return kernel
    return kernel / total


def apply_motion_blur(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """1D linear convolution with reflect padding; size unchanged.
    一维线性卷积，reflect padding；尺寸不变。"""
    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT)


def apply_sensor_noise(img: np.ndarray, sigma: float, group_seed: str) -> np.ndarray:
    """Zero-mean Gaussian noise with its own deterministic sub-seed.
    零均值高斯噪声，使用独立确定性子种子。"""
    gen = torch.Generator().manual_seed(int(_sub_seed(group_seed, "noise_values")[:16], 16))
    noise = torch.randn(img.shape, generator=gen, dtype=torch.float32).numpy()
    return np.clip(img + noise * sigma, 0.0, 1.0)


def apply_jpeg_compression(img: np.ndarray, quality: int) -> np.ndarray:
    """One JPEG encode/decode round through an in-memory buffer; no temp
    files, RGB preserved, dimensions unchanged.
    内存 JPEG 编解码一次；不写临时文件、保持 RGB 与尺寸。"""
    buf = io.BytesIO()
    Image.fromarray((np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)).save(
        buf, format="JPEG", quality=int(quality)
    )
    buf.seek(0)
    with Image.open(buf) as jpeg:
        jpeg.load()
        back = np.asarray(jpeg.convert("RGB"), dtype=np.float32) / 255.0
    return back


def _weighted_sample(
    rng: random.Random, items: Sequence[tuple[str, float]], count: int
) -> list[str]:
    """Weighted sampling without replacement (deterministic via rng).
    按权重不重复抽样（由 rng 决定，确定性）。"""
    pool = list(items)
    chosen: list[str] = []
    for _ in range(count):
        total = sum(w for _, w in pool)
        r = rng.random() * total
        acc = 0.0
        picked = -1
        for i, (name, w) in enumerate(pool):
            acc += w
            if r <= acc:
                picked = i
                chosen.append(name)
                break
        if picked < 0:  # pragma: no cover - floating point edge
            picked = len(pool) - 1
            chosen.append(pool[picked][0])
        pool.pop(picked)
    return chosen


def apply_degradations(
    image: Image.Image,
    cfg: AugmentationConfig,
    group_seed: str,
) -> tuple[Image.Image, dict]:
    """Select 1..max degradation steps (fixed order, no repeats) and apply
    them to the float [0,1] RGB copy. Never touches boxes or sizes; on any
    step failure the whole degradation pipeline falls back to the input
    image with a stable code.
    """
    rng_sel = _rng(group_seed, "degradation_select")
    metadata: dict[str, Any] = {"selected": [], "fallback_code": None}
    if rng_sel.random() >= cfg.degradation_probability:
        return image, metadata

    candidates = [
        ("low_contrast", cfg.contrast_weight),
        ("brightness", cfg.brightness_weight),
        ("vignette", cfg.vignette_weight),
        ("blur", cfg.defocus_weight + cfg.motion_blur_weight),
        ("sensor_noise", cfg.sensor_noise_weight),
        ("jpeg", cfg.jpeg_weight),
    ]
    count = rng_sel.randint(cfg.min_degradations, cfg.max_degradations)
    selected = _weighted_sample(rng_sel, candidates, count)
    # Fixed canonical order; only selected steps execute.
    # 固定规范顺序；只执行被选中的步骤。
    selected.sort(key=lambda name: DEGRADATION_ORDER.index(name))

    img = np.asarray(image, dtype=np.float32) / 255.0
    applied: list[dict[str, Any]] = []
    for name in selected:
        rng = _rng(group_seed, f"degradation_{name}")
        try:
            if name == "brightness":
                factor = rng.uniform(cfg.brightness_factor_min, cfg.brightness_factor_max)
                img = apply_brightness(img, factor)
                applied.append({"name": name, "factor": round(factor, 6)})
            elif name == "low_contrast":
                factor = rng.uniform(cfg.contrast_factor_min, cfg.contrast_factor_max)
                img = apply_low_contrast(img, factor)
                applied.append({"name": name, "factor": round(factor, 6)})
            elif name == "vignette":
                edge = rng.uniform(cfg.vignette_edge_factor_min, cfg.vignette_edge_factor_max)
                power = rng.uniform(cfg.vignette_power_min, cfg.vignette_power_max)
                img = apply_vignette(img, edge, power)
                applied.append({"name": name, "edge_factor": round(edge, 6), "power": round(power, 6)})
            elif name == "blur":
                # defocus and motion blur are mutually exclusive.
                # 失焦与运动模糊二选一，不会同时叠加。
                kind_rng = _rng(group_seed, "degradation_blur")
                total = cfg.defocus_weight + cfg.motion_blur_weight
                kind = "defocus" if kind_rng.random() < cfg.defocus_weight / total else "motion"
                if kind == "defocus":
                    params = dict(
                        kernel_size=int(kind_rng.choice(cfg.defocus_kernel_sizes)),
                        sigma=kind_rng.uniform(cfg.defocus_sigma_min, cfg.defocus_sigma_max),
                    )
                    img = apply_defocus_blur(img, params["kernel_size"], params["sigma"])
                else:
                    params = dict(
                        kernel_size=int(kind_rng.choice(cfg.motion_blur_kernel_sizes)),
                        angle=kind_rng.uniform(cfg.motion_blur_angle_min, cfg.motion_blur_angle_max),
                    )
                    kernel = motion_blur_kernel(params["kernel_size"], params["angle"])
                    img = apply_motion_blur(img, kernel)
                applied.append({"name": "blur", "kind": kind, **{k: v for k, v in params.items() if k != "kernel_size"}, "kernel_size": params["kernel_size"]})
            elif name == "sensor_noise":
                sigma = rng.uniform(cfg.noise_sigma_min, cfg.noise_sigma_max)
                img = apply_sensor_noise(img, sigma, group_seed)
                applied.append({"name": name, "sigma": round(sigma, 6)})
            elif name == "jpeg":
                quality = int(rng.randint(cfg.jpeg_quality_min, cfg.jpeg_quality_max))
                img = apply_jpeg_compression(img, quality)
                applied.append({"name": name, "quality": quality})
        except Exception:  # noqa: BLE001 - stable fallback, never partial state
            # Whole degradation pipeline falls back to the geometry-stage
            # output; boxes and sample type stay untouched.
            # 整条退化管线回退到几何阶段输出；框与样本类型不变。
            metadata["fallback_code"] = f"degradation_fallback:{name}"
            return image, metadata

    if not np.all(np.isfinite(img)):
        metadata["fallback_code"] = "degradation_fallback:non_finite"
        return image, metadata
    out = Image.fromarray(
        (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    )
    metadata["selected"] = applied
    return out, metadata


# ---------------------------------------------------------------------------
# Augmentation orchestrator (geometry + degradation, one group seed).
# 增强编排（几何 + 退化，共享 group seed）。
# ---------------------------------------------------------------------------


def apply_augmentation(
    image: Image.Image,
    box_entries: Sequence[dict],
    policy_geometry: str,
    cfg: AugmentationConfig,
    group_seed: str,
) -> tuple[Image.Image, dict]:
    """Run the full train augmentation for one episode.

    orientation_locked episodes skip geometry but still get imaging
    degradation. Box entries are updated in place with the synchronized
    xyxy_999 values. Returns (augmented RGB image, light metadata).
    """
    w, h = image.size
    metadata: dict[str, Any] = {
        "group_seed": group_seed[:16],
        "geometry": {"kind": "identity", "ops": [], "fallback_code": None, "reason": policy_geometry},
        "degradations": {"selected": [], "fallback_code": None},
    }
    if policy_geometry != "geometry_safe":
        # orientation_locked: identity geometry, imaging degradation still
        # allowed (it preserves coordinates and orientation semantics).
        # orientation_locked：几何阶段恒为 identity，恶劣成像质量模拟仍允许。
        out_image, deg_meta = apply_degradations(image, cfg, group_seed)
        metadata["degradations"] = deg_meta
        return out_image, metadata

    boxes_px = [box_999_to_pixels(b["xyxy_999"], w, h) for b in box_entries]
    image_np = np.asarray(image, dtype=np.uint8)
    out_np, out_boxes_px, ops, fallback = apply_geometry(image_np, boxes_px, cfg, group_seed)
    if fallback is not None:
        metadata["geometry"] = {
            "kind": "identity", "ops": [], "fallback_code": fallback, "reason": policy_geometry,
        }
        out_np = image_np
        out_boxes_px = boxes_px
    else:
        metadata["geometry"] = {
            "kind": "combined" if ops else "identity",
            "ops": ops,
            "fallback_code": None,
            "reason": policy_geometry,
        }
    out_w, out_h = out_np.shape[1], out_np.shape[0]
    for entry, box in zip(box_entries, out_boxes_px):
        if isinstance(box, list) and box and isinstance(box[0], float):
            # identity path kept float pixels; convert to 0..999 ints.
            # identity 路径保留浮点像素；统一转回 0..999 整数。
            box = pixels_to_box_999(box, out_w, out_h)
        entry["xyxy_999"] = [int(v) for v in box]

    out_image = Image.fromarray(out_np)
    out_image, deg_meta = apply_degradations(out_image, cfg, group_seed)
    metadata["degradations"] = deg_meta
    return out_image, metadata


# ---------------------------------------------------------------------------
# Chat rendering protocol.
# 对话渲染协议。
# ---------------------------------------------------------------------------


def _strip_markup(text: str) -> str:
    return _P_TAG.sub("", text).strip()


def _region_line(entry: dict, fallback_label: str = "region") -> str:
    label = entry.get("label") or fallback_label
    desc = entry.get("description") or ""
    xy = entry["xyxy_999"]
    line = f"- {label}: [{xy[0]}, {xy[1]}, {xy[2]}, {xy[3]}]"
    if desc:
        line += f" — {desc}"
    return line


def _json_boxes(box_entries: Sequence[dict]) -> str:
    return json.dumps(
        {"boxes": [{"xyxy": [int(v) for v in b["xyxy_999"]]} for b in box_entries]},
        separators=(",", ":"),
        ensure_ascii=False,
    )


def render_messages(episode: dict) -> tuple[list[dict], list[dict]]:
    """Render one episode into chat messages plus the per-turn texts.

    The image is the first content item of the first user turn; any literal
    <image> left in the text is removed (the chat template emits the
    placeholder). GeoChat <p> markup is stripped. Boxes are rendered from
    the (already augmented) structured entries; the old GeoChat box syntax
    can never leak into prompts.
    """
    kind = episode["task_kind"]
    turns_out: list[dict] = []
    for turn in episode["turns"]:
        user = turn["user_text"].replace(_IMAGE_TOKEN_LITERAL, "").strip()
        user = _strip_markup(user)
        assistant = turn["assistant_text"]
        if kind == "vrsbench_grounding":
            user = (
                "Locate the region described below.\n"
                f"Description: {user}\n"
                "Return the bounding boxes as JSON in 0..999 xyxy coordinates."
            )
            assistant = _json_boxes(turn["target_boxes"])
        elif kind == "geochat_refer":
            # Assistant answer is re-rendered from the structured boxes.
            # refer 的 assistant 由结构化框重新渲染。
            assistant = _json_boxes(turn["target_boxes"])
        elif kind == "vqa_box_assisted":
            regions = "\n".join(_region_line(b) for b in turn["input_boxes"])
            user = f"Question: {user}\nAvailable annotated regions:\n{regions}"
        elif kind in ("vqa_self_attention", "vqa_naturally_unboxed"):
            user = f"Question: {user}"
        elif kind == "geochat_identify":
            regions = "\n".join(_region_line(b) for b in turn["input_boxes"])
            user = f"{user}\nAvailable annotated regions:\n{regions}"
        # geochat_conversation: texts stay as-is (markup stripped).
        assistant = _strip_markup(assistant)
        turns_out.append({"user_text": user, "assistant_text": assistant})

    messages: list[dict] = []
    for i, t in enumerate(turns_out):
        content: list[dict] = [{"type": "text", "text": t["user_text"]}]
        if i == 0:
            content.insert(0, {"type": "image"})
        messages.append({"role": "user", "content": content})
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": t["assistant_text"]}]}
        )
    return messages, turns_out


# ---------------------------------------------------------------------------
# Processor encoding, assistant masks, alignment, truncation.
# Processor 编码、assistant mask、图像 token 对齐、截断。
# ---------------------------------------------------------------------------


def _flat_ids(ids: Any) -> list:
    if ids and isinstance(ids[0], list):
        return ids[0]
    return list(ids)


def _assistant_mask(processor: Any, messages: list[dict]) -> tuple[list[int], list[int]]:
    """Return (text_ids, assistant_mask) over the text-only tokenization.

    Primary path: the chat-template assistant mask (transformers 5.14.1 key
    "assistant_masks"; 4.x "assistant_tokens_mask") used verbatim — the
    fixed supervision policy. Fallback (flag unsupported): per-turn boundary
    encoding (prefix with add_generation_prompt=True vs full turn), marking
    assistant content + end token; requires prefix-stable template rendering.
    """
    try:
        result = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
    except (ValueError, TypeError):
        result = None
    if result is not None and isinstance(result, dict):
        mask = result.get("assistant_masks")
        if mask is None:
            mask = result.get("assistant_tokens_mask")
        if mask is not None:
            ids = _flat_ids(result["input_ids"])
            mask = _flat_ids(mask)
            if len(mask) != len(ids):
                raise FeatureError("assistant mask length mismatch")
            if any(mask):
                return ids, mask
    return _assistant_mask_by_turns(processor, messages)


def _assistant_mask_by_turns(processor: Any, messages: list[dict]) -> tuple[list[int], list[int]]:
    """Turn-boundary fallback mask. / 逐 turn 边界编码的回退 mask。"""
    full = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_dict=True
    )
    ids_full = _flat_ids(full["input_ids"])
    mask = [0] * len(ids_full)
    for j, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        prefix = processor.apply_chat_template(
            messages[:j], tokenize=True, add_generation_prompt=True, return_dict=True
        )
        turn_full = processor.apply_chat_template(
            messages[: j + 1], tokenize=True, add_generation_prompt=False, return_dict=True
        )
        p = _flat_ids(prefix["input_ids"])
        t = _flat_ids(turn_full["input_ids"])
        if len(t) < len(p) or len(t) > len(ids_full):
            raise FeatureError("assistant span malformed")
        for pos in range(len(p), len(t)):
            mask[pos] = 1
    return ids_full, mask


def _align_labels(
    text_ids: Sequence[int], mask: Sequence[int], input_ids: torch.Tensor
) -> torch.Tensor:
    """Align the text-only assistant mask to the with-image tokenization.

    The processor replaces the single image placeholder with N image tokens,
    so the with-image sequence is the text sequence with one span expanded;
    every assistant span lies after that span, so each masked position
    shifts by the total length delta.
    """
    delta = int(input_ids.shape[0]) - len(text_ids)
    ids_list = input_ids.tolist()
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    for pos, m in enumerate(mask):
        if not m:
            continue
        target = pos + delta
        if 0 <= target < len(ids_list):
            labels[target] = int(ids_list[target])
    return labels


def _validate_image_placeholder_gate(messages: list[dict], image_count: int) -> None:
    """Ensure every image placeholder is present before assistant supervision.

    Qwen expands each placeholder into a variable visual-token span.  Keeping
    all placeholders before the first assistant span makes label alignment a
    single safe suffix shift, including multi-image episodes.
    保证所有图像占位符都位于首个 assistant span 之前，便于安全对齐多图标签。
    """
    placeholders = 0
    first_assistant = None
    for index, message in enumerate(messages):
        if message.get("role") == "assistant" and first_assistant is None:
            first_assistant = index
        content = message.get("content", [])
        items = content if isinstance(content, list) else [{"type": "text", "text": content}]
        placeholders += sum(1 for item in items if item.get("type") == "image")
        if message.get("role") == "assistant" and any(item.get("type") == "image" for item in items):
            raise FeatureError("image placeholder inside assistant span")
    if placeholders != image_count:
        raise FeatureError(
            f"image_placeholder_count_mismatch expected={image_count} got={placeholders}"
        )
    if first_assistant is not None:
        for message in messages[first_assistant:]:
            content = message.get("content", [])
            items = content if isinstance(content, list) else [{"type": "text", "text": content}]
            if any(item.get("type") == "image" for item in items):
                raise FeatureError("image placeholder after assistant span")


def encode_multimodal_episode(
    processor: Any,
    images: Sequence[Image.Image],
    messages: list[dict],
    max_seq_length: int,
    episode_id: str,
    truncate: bool = True,
) -> dict:
    """Encode a one-or-more image conversation with assistant-only labels.

    Raises EpisodeTooLongError when a single turn pair exceeds
    max_seq_length; never returns all-(-100) labels.
    """
    if not images:
        raise FeatureError("episode has no images", episode_id)
    _validate_image_placeholder_gate(messages, len(images))
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    text_ids, mask = _assistant_mask(processor, messages)
    enc = processor(text=[text], images=list(images), return_tensors="pt")
    if "mm_token_type_ids" not in enc:
        raise FeatureError("mm_token_type_ids missing from processor output")
    input_ids = enc["input_ids"][0]
    labels = _align_labels(text_ids, mask, input_ids)
    feature = {
        "input_ids": input_ids,
        "attention_mask": enc["attention_mask"][0],
        "labels": labels,
        "pixel_values": enc["pixel_values"],
        "image_grid_thw": enc["image_grid_thw"],
        "mm_token_type_ids": enc["mm_token_type_ids"][0],
    }
    n = int(input_ids.shape[0])
    if truncate and n > max_seq_length:
        feature = _truncate_feature(
            processor, images, messages, episode_id, max_seq_length,
            delta=n - len(text_ids),
        )
    if not (feature["labels"] != IGNORE_INDEX).any():
        raise FeatureError("empty_supervision", episode_id)
    return feature


def encode_episode(
    processor: Any,
    image: Image.Image,
    messages: list[dict],
    max_seq_length: int,
    episode_id: str,
    truncate: bool = True,
) -> dict:
    """Backward-compatible single-image wrapper. / 兼容旧单图调用。"""
    return encode_multimodal_episode(
        processor, [image], messages, max_seq_length, episode_id, truncate
    )


def _truncate_feature(
    processor: Any,
    images: Sequence[Image.Image],
    messages: list[dict],
    episode_id: str,
    max_seq_length: int,
    delta: int,
) -> dict:
    """Turn-pair truncation: keep the image turn and drop complete trailing
    turn pairs; the image token span is never cut; a single over-long pair
    raises EpisodeTooLongError instead of producing all-(-100) labels.
    """
    n_turns = len(messages) // 2
    if n_turns <= 1:
        raise EpisodeTooLongError(episode_id)
    for k in range(n_turns - 1, 0, -1):
        probe = processor.apply_chat_template(
            messages[: 2 * k], tokenize=True, add_generation_prompt=False, return_dict=True
        )
        probe_len = len(_flat_ids(probe["input_ids"]))
        # with-image length = text-only length + constant image delta
        if probe_len + delta <= max_seq_length:
            return encode_multimodal_episode(
                processor, images, messages[: 2 * k], max_seq_length, episode_id
            )
    raise EpisodeTooLongError(episode_id)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class Phase2EpisodeDataset(Dataset):
    """Map-style dataset over one canonical episode JSONL.

    - lazy episode loading (byte offsets, one JSON line per access);
    - lazy image loading (opened per item, closed promptly);
    - online augmentation keyed by sha256(seed | epoch | parent_episode_id);
    - processor encoding with assistant-mask labels and turn truncation;
    - set_epoch(epoch) must be called by the training loop each epoch so the
      online augmentation seed is epoch-driven (no global random state).
    """

    def __init__(
        self,
        episode_jsonl: str | Path,
        roots: DatasetRootConfig | Mapping[str, str],
        processor: Any,
        aug_config: AugmentationConfig,
        max_seq_length: int,
        seed: str,
        split: str = "train",
        start_epoch: int = 0,
    ) -> None:
        if split not in ("train", "validation"):
            raise ValueError(f"split must be 'train' or 'validation', got {split!r}")
        if max_seq_length <= 0:
            raise ValueError(f"max_seq_length must be positive, got {max_seq_length}")
        if not seed:
            raise ValueError("seed must not be empty")
        self._store = LazyJsonLines(Path(episode_jsonl))
        self._roots = (
            roots if isinstance(roots, DatasetRootConfig)
            else DatasetRootConfig(dict(roots))
        )
        self._processor = processor
        self._aug = aug_config
        self._max_seq_length = int(max_seq_length)
        self._seed = str(seed)
        self._split = split
        self._epoch = int(start_epoch)

    # -- training-loop interface -------------------------------------------
    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch; online augmentation seeds derive from it.
        设置当前 epoch；在线增强种子由它派生。"""
        self._epoch = int(epoch)

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._store)

    def _augment_enabled(self) -> bool:
        return self._split == "train" and self._aug.enabled

    def __getitem__(self, index: int) -> dict:
        episode = self._store[index]
        relative_image = episode["image"]
        image_path = self._roots.image_path(episode["image_source"], relative_image)
        image = load_image_rgb(image_path, relative_image)
        group_seed = group_seed_hex(
            self._seed, self._epoch, episode["parent_episode_id"]
        )
        box_entries = [
            b
            for turn in episode["turns"]
            for b in turn["input_boxes"] + turn["target_boxes"]
        ]
        if self._augment_enabled():
            image, aug_meta = apply_augmentation(
                image,
                box_entries,
                episode["augmentation_policy"]["geometry"],
                self._aug,
                group_seed,
            )
        else:
            aug_meta = {
                "group_seed": group_seed[:16],
                "geometry": {
                    "kind": "identity", "ops": [], "fallback_code": None,
                    "reason": "validation_identity" if self._split == "validation" else "augmentation_disabled",
                },
                "degradations": {"selected": [], "fallback_code": None},
            }
        messages, _ = render_messages(episode)
        feature = encode_episode(
            self._processor, image, messages, self._max_seq_length, episode["episode_id"]
        )
        feature["episode_id"] = episode["episode_id"]
        feature["augmentation"] = aug_meta
        return feature

    def preflight(self, limit: int | None = None) -> dict:
        """Startup preflight: count over-long episodes without augmentation.

        Encodes every (or up to limit) episode with the identity transform
        and reports: too_long (turn-truncatable), episode_too_long (single
        pair exceeds the limit; hard failure), image_errors, other_errors.
        启动预检：不增强地编码统计过长 Episode 数量。
        """
        counts = {
            "checked": 0,
            "too_long": 0,
            "episode_too_long": 0,
            "image_errors": 0,
            "other_errors": 0,
        }
        n_total = len(self)
        limit = n_total if limit is None else min(int(limit), n_total)
        for i in range(limit):
            counts["checked"] += 1
            try:
                episode = self._store[i]
                relative_image = episode["image"]
                image_path = self._roots.image_path(
                    episode["image_source"], relative_image
                )
                image = load_image_rgb(image_path, relative_image)
                messages, _ = render_messages(episode)
                feature = encode_episode(
                    self._processor, image, messages, self._max_seq_length,
                    episode["episode_id"], truncate=False,
                )
                if int(feature["input_ids"].shape[0]) > self._max_seq_length:
                    if len(messages) // 2 <= 1:
                        counts["episode_too_long"] += 1
                    else:
                        counts["too_long"] += 1
            except EpisodeTooLongError:
                counts["episode_too_long"] += 1
            except ImagePathError:
                counts["image_errors"] += 1
            except Exception:  # noqa: BLE001 - preflight must not abort
                counts["other_errors"] += 1
        return counts


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------


class Phase2DataCollator:
    """Right-padded collation for Phase2EpisodeDataset features.

    Pads input_ids / attention_mask / mm_token_type_ids to the right (0),
    labels with IGNORE_INDEX, concatenates pixel_values and image_grid_thw
    along dim 0 (batch images differ in token counts), and strips episode
    metadata out of the batch: returns (batch, meta) where meta is one
    light dict per sample (episode_id + augmentation) for error location.
    """

    REQUIRED_KEYS = (
        "input_ids",
        "attention_mask",
        "labels",
        "pixel_values",
        "image_grid_thw",
        "mm_token_type_ids",
    )

    def __call__(self, features: Sequence[dict]) -> tuple[dict, list[dict]]:
        if not features:
            raise CollatorError("empty batch")
        for f in features:
            missing = [k for k in self.REQUIRED_KEYS if k not in f]
            if missing:
                raise CollatorError(f"feature missing required keys: {missing}")
        meta = [
            {
                "episode_id": f.get("episode_id"),
                "augmentation": f.get("augmentation"),
            }
            for f in features
        ]
        input_ids = [f["input_ids"] for f in features]
        max_len = max(int(t.shape[0]) for t in input_ids)
        pad_right = lambda t: torch.nn.functional.pad(t, (0, max_len - int(t.shape[0])))  # noqa: E731
        batch = {
            "input_ids": torch.stack([pad_right(t) for t in input_ids]),
            "attention_mask": torch.stack(
                [pad_right(f["attention_mask"]) for f in features]
            ),
            "labels": torch.stack(
                [
                    torch.nn.functional.pad(
                        f["labels"], (0, max_len - int(f["labels"].shape[0])),
                        value=IGNORE_INDEX,
                    )
                    for f in features
                ]
            ),
            "mm_token_type_ids": torch.stack(
                [pad_right(f["mm_token_type_ids"]) for f in features]
            ),
            "pixel_values": torch.cat([f["pixel_values"] for f in features], dim=0),
            "image_grid_thw": torch.cat(
                [f["image_grid_thw"] for f in features], dim=0
            ),
        }
        return batch, meta
