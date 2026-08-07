"""Translate VRSBench train/val caption and VQA annotations into Chinese.
使用 DeepSeek API 将 VRSBench train/val 的 caption 与 VQA 标注翻译为中文。

Reads the four English source files (train caption uses the cleaned file),
translates every unique caption/question/answer string once through the
DeepSeek OpenAI-compatible API, caches the results, and writes one Chinese
JSONL file per source. Caption records keep a single fixed Chinese instruction
that is replaced directly without an API call.

API key, model, base URL, and concurrency are read from the environment or an
ignored .env file (DEEPSEEK_API_KEY, DEEPSEEK_MODEL, DEEPSEEK_BASE_URL,
DEEPSEEK_MAX_CONCURRENCY); explicit CLI flags take precedence.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import ssl
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# DeepSeek OpenAI-compatible chat completions endpoint.
# DeepSeek OpenAI 兼容模式的对话补全端点。
DEFAULT_BASE_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-v4-flash"

# Default concurrency for translation batches; override via env or CLI.
# 翻译批次的默认并发数；可通过环境变量或 CLI 覆盖。
DEFAULT_CONCURRENCY = 4

# Persist the whole cache after this many completed batches; keeps disk writes
# cheap at high concurrency while limiting loss on interrupt.
# 每完成这么多批后落盘一次全量缓存；高并发下降低磁盘写压力，中断时损失有限。
CACHE_SAVE_INTERVAL = 25

# Fixed Chinese instruction shared by every caption record; no API call needed.
# 所有 caption 记录共用的固定中文指令，直接替换，无需调用 API。
ZH_INSTRUCTION = "请描述这张图片的内容。"

# System prompt version for reproducible translation behavior.
# 用于保证翻译行为可复现的系统提示词版本。
SYSTEM_PROMPT = (
    "You are a professional translator for remote sensing image datasets. "
    "Translate each English text into natural simplified Chinese. "
    "Rules: keep numbers and symbols unchanged; translate technical terms and "
    "classification labels accurately (for example 'expressway-toll-station' "
    "becomes '高速公路收费站'); preserve the exact meaning without adding or "
    "omitting information; respond with only a JSON object whose keys exactly "
    "match the input keys and whose values are the Chinese translations."
)
SYSTEM_PROMPT_VERSION = "translate-v1"

# (source, output, fields to translate, replace instruction with ZH_INSTRUCTION)
# (源文件, 输出文件, 需要翻译的字段, 是否用固定中文指令替换 instruction)
JOBS = (
    {
        "source": "VRSBench_train_caption_cleaned.jsonl",
        "output": "VRSBench_train_caption_cleaned_zh.jsonl",
        "fields": ("caption",),
        "replace_instruction": True,
    },
    {
        "source": "VRSBench_val_caption.jsonl",
        "output": "VRSBench_val_caption_zh.jsonl",
        "fields": ("caption",),
        "replace_instruction": True,
    },
    {
        "source": "VRSBench_train_vqa.jsonl",
        "output": "VRSBench_train_vqa_zh.jsonl",
        "fields": ("question", "answer"),
        "replace_instruction": False,
    },
    {
        "source": "VRSBench_val_vqa.jsonl",
        "output": "VRSBench_val_vqa_zh.jsonl",
        "fields": ("question", "answer"),
        "replace_instruction": False,
    },
)


def build_parser() -> argparse.ArgumentParser:
    """Build the translation CLI. / 构建翻译 CLI。"""
    parser = argparse.ArgumentParser(
        description=(
            "Translate VRSBench train/val caption and VQA annotations into "
            "Chinese using the DeepSeek API."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory containing the VRSBench *_caption/*_vqa JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to --root.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="DeepSeek model name; overrides DEEPSEEK_MODEL.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="DeepSeek OpenAI-compatible chat completions URL; overrides DEEPSEEK_BASE_URL.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Dotenv file to load; defaults to ./.env.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Max concurrent API batches; overrides DEEPSEEK_MAX_CONCURRENCY.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Unique strings per API call; default 20.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Retries per failed batch; default 4.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP timeout in seconds; default 120.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Translation cache JSON path; defaults to <output-dir>/vrsbench_zh_translations.json.",
    )
    return parser


def _load_dotenv(env_file: Path) -> None:
    """Load KEY=VALUE pairs from a dotenv file without overriding process env.
    从 dotenv 文件加载 KEY=VALUE 对，不覆盖已存在的进程环境变量。
    """
    if not env_file.is_file():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _resolve_concurrency(value: str | None) -> int:
    """Parse and validate a concurrency value. / 解析并校验并发数。"""
    if not value:
        return DEFAULT_CONCURRENCY
    try:
        concurrency = int(value)
    except (TypeError, ValueError) as error:
        raise SystemExit(f"Invalid concurrency value: {value!r}") from error
    if concurrency < 1:
        raise SystemExit(f"Concurrency must be >= 1, got {concurrency}")
    return concurrency


def _chat_completions_url(base_url: str) -> str:
    """Normalize a DeepSeek base URL into the full chat completions URL.
    将 DeepSeek base URL 规范化为完整的 chat completions URL。
    """
    base = base_url.strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/chat/completions"


def _read_rows(path: Path) -> list[dict[str, Any]]:
    """Read one JSONL file into parsed records in source order.
    按源顺序读取一个 JSONL 文件为解析后的记录。
    """
    if not path.is_file():
        raise SystemExit(f"Input file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}: {error}") from error
    return rows


def _write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    """Atomically write records as compact JSONL. / 以紧凑 JSONL 原子写出记录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for record in rows:
                f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _write_cache(cache: dict[str, str], path: Path) -> None:
    """Atomically persist the translation cache as JSON.
    将翻译缓存以 JSON 形式原子化持久化。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _parse_json_object(content: str) -> dict[str, Any]:
    """Parse a JSON object from a model response, tolerating extra text.
    从模型响应中解析 JSON 对象，容忍前后多余文本。
    """
    content = content.strip()
    try:
        value = json.loads(content)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"Model response contains no JSON object: {content[:200]!r}")
    value = json.loads(content[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError(f"Model response JSON is not an object: {content[:200]!r}")
    return value


def _default_ssl_context() -> ssl.SSLContext:
    """Build an SSL context, preferring certifi when the system CA bundle is absent.
    构建 SSL 上下文；系统 CA 缺失时优先使用 certifi 的 CA 包。
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        # Fall back to the platform default context when certifi is unavailable.
        # certifi 不可用时回退到平台默认 SSL 上下文。
        return ssl.create_default_context()


def _translate_batch(
    api_key: str,
    base_url: str,
    model: str,
    batch: dict[str, str],
    max_retries: int,
    timeout: int,
    ssl_context: ssl.SSLContext,
) -> dict[str, str]:
    """Translate one batch of strings through the DeepSeek API.
    通过 DeepSeek API 翻译一批字符串。
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(batch, ensure_ascii=False)},
        ],
        "temperature": 0.1,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    request = urllib.request.Request(base_url, data=data, headers=headers, method="POST")
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(
                request, timeout=timeout, context=ssl_context
            ) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            parsed = _parse_json_object(content)
            translated: dict[str, str] = {}
            for key, expected in batch.items():
                value = parsed.get(key)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"Missing translation for key {key!r}: {parsed}")
                translated[key] = value.strip()
            return translated
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, ValueError) as error:
            last_error = error
            if attempt + 1 < max_retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"Translation batch failed after {max_retries} attempts: {last_error}")


def _collect_unique(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[str]:
    """Return unique non-empty field values in first-seen order.
    按首次出现顺序返回字段的非空唯一值。
    """
    unique: list[str] = []
    seen: set[str] = set()
    for record in rows:
        for field in fields:
            value = str(record.get(field, "")).strip()
            if value and value not in seen:
                seen.add(value)
                unique.append(value)
    return unique


def _translate_missing(
    unique_strings: list[str],
    cache: dict[str, str],
    *,
    api_key: str,
    base_url: str,
    model: str,
    batch_size: int,
    max_retries: int,
    timeout: int,
    ssl_context: ssl.SSLContext,
    cache_path: Path,
    concurrency: int,
) -> int:
    """Translate missing strings concurrently, saving the cache after each batch.
    并发翻译缓存中缺失的字符串，每批完成后保存缓存。
    """
    pending = [text for text in unique_strings if text not in cache]
    if not pending:
        return 0
    batches = [
        {
            str(index): text
            for index, text in enumerate(pending[start : start + batch_size])
        }
        for start in range(0, len(pending), batch_size)
    ]
    cache_lock = threading.Lock()
    done = 0
    last_save = 0
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=concurrency, thread_name_prefix="dashscope"
    ) as executor:
        futures = {
            executor.submit(
                _translate_batch,
                api_key,
                base_url,
                model,
                batch,
                max_retries,
                timeout,
                ssl_context,
            ): batch
            for batch in batches
        }
        try:
            for future in concurrent.futures.as_completed(futures):
                batch = futures[future]
                # A failed batch raises here after retries are exhausted; the
                # cache already contains earlier batches, so rerunning resumes.
                # 重试耗尽后失败批在此抛出；缓存已包含此前批次，重跑可干净续传。
                translated = future.result()
                with cache_lock:
                    for key, text in batch.items():
                        cache[text] = translated[key]
                done += len(batch)
                # Coalesce disk writes; saving the full cache after every batch
                # becomes a bottleneck at high concurrency.
                # 合并磁盘写入；高并发下每批都写全量缓存会成为瓶颈。
                if done - last_save >= CACHE_SAVE_INTERVAL or done >= len(pending):
                    with cache_lock:
                        _write_cache(cache, cache_path)
                    last_save = done
                print(f"translated {done}/{len(pending)} pending strings", flush=True)
        finally:
            # Always persist completed translations before exiting.
            # 退出前始终持久化已完成的翻译。
            with cache_lock:
                _write_cache(cache, cache_path)
    return len(pending)


def main() -> int:
    args = build_parser().parse_args()
    _load_dotenv(args.env_file)
    # Values come from .env or the process environment; CLI flags win.
    # 值来自 .env 或进程环境；显式 CLI 参数优先。
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "DEEPSEEK_API_KEY is not set in .env or the process environment; "
            "add it to .env or export it first."
        )
    model = args.model or os.environ.get("DEEPSEEK_MODEL", "").strip() or DEFAULT_MODEL
    base_url = _chat_completions_url(
        args.base_url
        or os.environ.get("DEEPSEEK_BASE_URL", "").strip()
        or DEFAULT_BASE_URL
    )
    concurrency = (
        _resolve_concurrency(str(args.workers))
        if args.workers is not None
        else _resolve_concurrency(os.environ.get("DEEPSEEK_MAX_CONCURRENCY"))
    )
    out_root = args.output_dir if args.output_dir is not None else args.root
    out_root.mkdir(parents=True, exist_ok=True)
    cache_path = args.cache if args.cache is not None else out_root / "vrsbench_zh_translations.json"
    cache: dict[str, str] = {}
    if cache_path.is_file():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    # Reuse one SSL context for every request instead of rebuilding it per call.
    # 复用同一个 SSL 上下文，避免每个请求都重建。
    ssl_context = _default_ssl_context()

    # Collect every unique string that needs translation across all jobs.
    # 汇总所有任务中需要翻译的唯一字符串。
    all_unique: list[str] = []
    seen: set[str] = set()
    job_rows: list[tuple[dict[str, Any], list[dict[str, Any]], list[str]]] = []
    for job in JOBS:
        # Sources are always read from --root; only outputs go to --output-dir.
        # 源文件始终从 --root 读取；只有输出写入 --output-dir。
        source = args.root / job["source"]
        rows = _read_rows(source)
        unique = _collect_unique(rows, job["fields"])
        for text in unique:
            if text not in seen:
                seen.add(text)
                all_unique.append(text)
        job_rows.append((job, rows, unique))

    new_count = _translate_missing(
        all_unique,
        cache,
        api_key=api_key,
        base_url=base_url,
        model=model,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        timeout=args.timeout,
        ssl_context=ssl_context,
        cache_path=cache_path,
        concurrency=concurrency,
    )

    # Rebuild every Chinese output from the source rows and the cache.
    # 根据源记录与缓存重建每一份中文输出。
    summary: dict[str, Any] = {}
    for job, rows, unique in job_rows:
        missing = [text for text in unique if text not in cache]
        if missing:
            raise RuntimeError(
                f"Cache missing {len(missing)} translations for {job['source']}; rerun the script."
            )
        output_rows: list[dict[str, Any]] = []
        for record in rows:
            translated_record = dict(record)
            for field in job["fields"]:
                value = str(record.get(field, "")).strip()
                if value:
                    translated_record[field] = cache[value]
            if job["replace_instruction"]:
                translated_record["instruction"] = ZH_INSTRUCTION
            output_rows.append(translated_record)
        output_path = out_root / job["output"]
        _write_rows(output_rows, output_path)
        summary[job["output"]] = {
            "rows": len(output_rows),
        }
        print(f"wrote {output_path} ({len(output_rows)} rows)")

    meta = {
        "model": model,
        "base_url": base_url,
        "max_concurrency": concurrency,
        "system_prompt_version": SYSTEM_PROMPT_VERSION,
        "batch_size": args.batch_size,
        "cache_file": str(cache_path),
        "instruction_zh": ZH_INSTRUCTION,
        "new_translation_count": new_count,
        "jobs": summary,
    }
    meta_path = out_root / "vrsbench_zh_translation_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
