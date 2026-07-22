"""Generate visual audit artifacts beside baseline result files.
在基线结果文件旁生成可视化审计产物。
"""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import re
from pathlib import Path
from typing import Any, TextIO

from PIL import Image

from data.schema import CanonicalPrediction, CanonicalSample


class AuditReportWriter:
    """Persist a bounded visual subset while inference still owns image objects.
    在推理仍持有图像对象时保存有上限的可视化子集。
    """

    def __init__(self, result_path: Path, max_samples: int) -> None:
        if max_samples < 1:
            raise ValueError("report.max_samples must be at least 1.")
        self.report_dir = report_dir_for_result(result_path)
        self.images_dir = self.report_dir / "images"
        self.samples_path = self.report_dir / "samples.jsonl"
        self.max_samples = max_samples
        self.captured_samples = 0
        self._file: TextIO | None = None

    def __enter__(self) -> AuditReportWriter:
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._file = self.samples_path.open("w", encoding="utf-8")
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def capture(
        self,
        sample: CanonicalSample,
        prediction: CanonicalPrediction,
        inference_seconds: float,
    ) -> None:
        if self._file is None:
            raise RuntimeError("AuditReportWriter must be used as a context manager.")
        if self.captured_samples >= self.max_samples:
            return
        image_files = [self._save_image(image) for image in sample.images]
        artifact = {
            "sample": sample.serializable(),
            "prediction": prediction.serializable(),
            "image_files": image_files,
            "inference_seconds": round(inference_seconds, 6),
        }
        self._file.write(json.dumps(artifact, ensure_ascii=False) + "\n")
        self._file.flush()
        self.captured_samples += 1

    def _save_image(self, value: Any) -> str:
        buffer = io.BytesIO()
        if isinstance(value, Image.Image):
            image = value
            image.convert("RGB").save(buffer, format="PNG")
        else:
            with Image.open(Path(str(value))) as opened:
                opened.convert("RGB").save(buffer, format="PNG")
        content = buffer.getvalue()
        filename = f"{hashlib.sha256(content).hexdigest()[:20]}.png"
        destination = self.images_dir / filename
        if not destination.exists():
            destination.write_bytes(content)
        return f"images/{filename}"


def report_dir_for_result(result_path: Path) -> Path:
    """Return the stable report directory for one prediction file.
    返回一个预测文件对应的稳定报告目录。
    """

    return result_path.with_suffix(".report")


def write_deepseek_audit(path: Path, records: list[dict[str, Any]]) -> None:
    """Persist one auditable DeepSeek record per evaluated sample.
    为每条已评测样本保存一条可审计的 DeepSeek 记录。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_audit_report(
    result_path: Path,
    metrics_path: Path | None = None,
    deepseek_audit_path: Path | None = None,
) -> Path | None:
    """Build HTML and CSV from persisted inference and optional evaluation artifacts.
    根据已保存的推理产物和可选评测产物生成 HTML 与 CSV。
    """

    report_dir = report_dir_for_result(result_path)
    samples_path = report_dir / "samples.jsonl"
    if not samples_path.exists():
        return None
    displayed = _read_jsonl(samples_path)
    if not displayed:
        return None
    metadata_path = result_path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path and metrics_path.exists() else {}
    deepseek_records = _read_jsonl(deepseek_audit_path) if deepseek_audit_path and deepseek_audit_path.exists() else []
    deepseek_by_id = {str(record["sample_id"]): record for record in deepseek_records}

    rows: list[dict[str, Any]] = []
    cards: list[str] = []
    displayed_exact = 0
    displayed_semantic = 0
    semantic_evaluated = 0
    for index, artifact in enumerate(displayed, start=1):
        sample = artifact["sample"]
        prediction = artifact["prediction"]
        sample_id = str(sample["id"])
        candidate = prediction.get("answer") or prediction.get("text", "")
        references = sample.get("answers", [])
        exact = _normalize(candidate) in {_normalize(answer) for answer in references}
        displayed_exact += int(exact)
        deepseek = deepseek_by_id.get(sample_id)
        semantic = None if not deepseek or deepseek.get("score") is None else float(deepseek["score"]) == 1.0
        if semantic is not None:
            semantic_evaluated += 1
            displayed_semantic += int(semantic)
        usage = deepseek.get("token_usage") if deepseek else None
        rows.append(
            {
                "index": index,
                "sample_id": sample_id,
                "question": sample.get("meta", {}).get("question", sample.get("prompt", "")),
                "qwen_raw_text": prediction.get("meta", {}).get("raw_text", prediction.get("text", "")),
                "qwen_final_answer": candidate,
                "reference_answers": " | ".join(map(str, references)),
                "exact_match": exact,
                "deepseek_score": "" if semantic is None else int(semantic),
                "inference_seconds": artifact.get("inference_seconds", ""),
                "deepseek_seconds": "" if not deepseek else deepseek.get("duration_seconds", ""),
            }
        )
        cards.append(_sample_card(index, artifact, exact, deepseek, usage))

    csv_path = report_dir / "samples.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    completed = int(metadata.get("completed_samples", len(displayed)))
    metric_summary = _metric_summary(metrics, displayed_exact, len(displayed), displayed_semantic, semantic_evaluated)
    model = metadata.get("model", {})
    html_path = report_dir / "report.html"
    html_path.write_text(
        _page(
            result_path=result_path,
            completed=completed,
            displayed=len(displayed),
            metric_summary=metric_summary,
            model=model,
            metadata=metadata,
            has_deepseek=bool(deepseek_records),
            cards=cards,
        ),
        encoding="utf-8",
    )
    return html_path.resolve()


def _sample_card(
    index: int,
    artifact: dict[str, Any],
    exact: bool,
    deepseek: dict[str, Any] | None,
    usage: Any,
) -> str:
    sample = artifact["sample"]
    prediction = artifact["prediction"]
    candidate = prediction.get("answer") or prediction.get("text", "")
    raw_text = prediction.get("meta", {}).get("raw_text", prediction.get("text", ""))
    references = " | ".join(map(str, sample.get("answers", []))) or "—"
    semantic = None if not deepseek or deepseek.get("score") is None else float(deepseek["score"]) == 1.0
    semantic_badge = "未评判" if semantic is None else ("正确" if semantic else "错误")
    semantic_class = "neutral" if semantic is None else ("ok" if semantic else "bad")
    images = "".join(
        f'<img src="{_escape(path)}" alt="sample {_escape(sample["id"])} image">'
        for path in artifact.get("image_files", [])
    )
    deepseek_section = "<dt>DeepSeek 审查</dt><dd>尚未运行 DeepSeek 代理评测。</dd>"
    details = ""
    if deepseek:
        error = deepseek.get("error")
        content = deepseek.get("raw_content") or (f"ERROR: {error}" if error else "")
        deepseek_section = (
            f"<dt>DeepSeek 原始审查内容</dt><dd><code>{_escape(content)}</code></dd>"
            f"<dt>DeepSeek 调用</dt><dd>{_escape(deepseek.get('duration_seconds', ''))} 秒 · "
            f"{_escape(deepseek.get('attempts', ''))} 次尝试 · token {_escape(json.dumps(usage, ensure_ascii=False))}</dd>"
        )
        raw_response = deepseek.get("raw_api_response")
        if raw_response is not None:
            details = (
                "<details><summary>查看 DeepSeek 完整原始 API 响应</summary><pre>"
                f"{_escape(json.dumps(raw_response, ensure_ascii=False, indent=2))}</pre></details>"
            )
    return f"""
<article class="card">
  <div class="card-head"><h2>样本 {index:03d} · ID {_escape(sample['id'])}</h2>
    <span class="badge {'ok' if exact else 'bad'}">严格匹配：{'正确' if exact else '错误'}</span>
    <span class="badge {semantic_class}">DeepSeek：{semantic_badge}</span>
  </div>
  <div class="grid"><figure>{images}</figure><section><dl>
    <dt>实际问题</dt><dd>{_escape(sample.get('meta', {}).get('question', sample.get('prompt', '')))}</dd>
    <dt>送入 Qwen 的提示</dt><dd>{_escape(sample.get('prompt', ''))}</dd>
    <dt>Qwen 原始回复</dt><dd class="answer">{_escape(raw_text)}</dd>
    <dt>Qwen 最终答案</dt><dd class="answer">{_escape(candidate)}</dd>
    <dt>标准答案</dt><dd class="reference">{_escape(references)}</dd>
    <dt>Qwen 单条推理耗时</dt><dd>{_escape(artifact.get('inference_seconds', ''))} 秒</dd>
    {deepseek_section}
  </dl>{details}</section></div>
</article>"""


def _metric_summary(
    metrics: dict[str, Any],
    displayed_exact: int,
    displayed_total: int,
    displayed_semantic: int,
    semantic_evaluated: int,
) -> str:
    if metrics.get("metric") == "exact_match_accuracy":
        exact_text = f"{metrics.get('correct', 0)}/{metrics.get('total', 0)}（{float(metrics.get('score', 0)):.1%}）"
    else:
        exact_text = f"{displayed_exact}/{displayed_total}（展示子集）"
    proxy = metrics.get("deepseek_proxy")
    if proxy:
        semantic_text = f"{float(proxy.get('score', 0)):.1%}，成功 {proxy.get('evaluated', 0)} 条"
    elif semantic_evaluated:
        semantic_text = f"{displayed_semantic}/{semantic_evaluated}（展示子集）"
    else:
        semantic_text = "尚未运行"
    return f"严格匹配：{exact_text}；DeepSeek 语义代理：{semantic_text}。"


def _page(
    result_path: Path,
    completed: int,
    displayed: int,
    metric_summary: str,
    model: dict[str, Any],
    metadata: dict[str, Any],
    has_deepseek: bool,
    cards: list[str],
) -> str:
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen3-VL 逐样本审计报告</title><style>
:root{{--bg:#f4f6f8;--card:#fff;--text:#17202a;--muted:#5d6d7e;--line:#dfe6e9;--ok:#18794e;--okbg:#e8f7ef;--bad:#b42318;--badbg:#ffebe9;--accent:#2257a6}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1180px;margin:auto;padding:32px 20px 80px}}h1{{font-size:30px;margin:.2em 0}}h2{{font-size:19px;margin:0}}.lead{{color:var(--muted)}}.summary,.process,.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;box-shadow:0 2px 8px #0000000a}}.summary,.process{{padding:20px;margin:20px 0}}.card{{padding:20px;margin:18px 0}}.card-head{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:15px}}.badge{{padding:3px 9px;border-radius:99px;font-size:13px}}.ok{{background:var(--okbg);color:var(--ok)}}.bad{{background:var(--badbg);color:var(--bad)}}.neutral{{background:#eef2f7;color:var(--muted)}}.grid{{display:grid;grid-template-columns:minmax(280px,42%) 1fr;gap:22px}}figure{{margin:0}}img{{width:100%;max-height:520px;object-fit:contain;background:#111;border-radius:10px;margin-bottom:8px}}dl{{margin:0}}dt{{font-weight:700;color:var(--muted);margin-top:8px}}dd{{margin:2px 0 8px}}.answer,.reference{{font-size:17px}}.reference{{color:var(--ok)}}pre{{white-space:pre-wrap;word-break:break-word;background:#111827;color:#e5e7eb;padding:12px;border-radius:8px;font-size:12px}}code{{background:#eef2f7;padding:2px 5px;border-radius:4px}}a{{color:var(--accent)}}@media(max-width:760px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Qwen3-VL 逐样本审计报告</h1>
<p class="lead">结果文件：{_escape(result_path)}。共完成 {completed} 条，报告展示 {displayed} 条。{_escape(metric_summary)}</p>
<section class="summary"><h2>运行信息</h2><p>模型：{_escape(model.get('id', ''))}；dtype：{_escape(model.get('dtype', ''))}；max_new_tokens：{_escape(model.get('max_new_tokens', ''))}；local_files_only：{_escape(model.get('local_files_only', False))}。</p><p>模型加载耗时：{_escape(metadata.get('model_load_seconds', ''))} 秒；推理总耗时：{_escape(metadata.get('inference_seconds', ''))} 秒。</p></section>
<section class="process"><h2>可审计运行过程</h2><ol><li>加载配置指定的 Qwen3-VL 权重和 processor。</li><li>逐条校验规范样本，将图片与提示交给模型确定性生成。</li><li>保存 Qwen 原始回复、最终答案、标准答案、图片和单条耗时。</li><li>评测命令计算原有指标；启用 DeepSeek 时，它只接收问题、参考答案和候选答案，不查看图片。</li></ol><p>本报告不保存、生成或推测隐藏思维链。DeepSeek 逐条审查：{'已保存' if has_deepseek else '未运行'}。<a href="samples.csv">打开 CSV 对照表</a></p></section>
{''.join(cards)}
</main></body></html>"""


def _read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower().strip(".,;:!"))


def _escape(value: Any) -> str:
    return html.escape(str(value))
