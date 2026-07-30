"""Build a portable LEVIR-CC baseline-versus-Agent audit report.
生成可携带的 LEVIR-CC 基线与 Agent 审计对比报告。
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
from pathlib import Path
from typing import Any


METRIC_NAMES = ("BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4", "METEOR", "ROUGE_L", "CIDEr")
NO_CHANGE_PHRASES = (
    "no difference",
    "identical",
    "same as before",
    "no change",
    "nothing has changed",
    "no significant land-cover change",
    "no verifiable land-cover change",
)


def build_change_comparison_report(
    baseline_path: Path,
    run_dir: Path,
    output_path: Path,
    metrics_path: Path,
    manifest_path: Path,
) -> tuple[Path, Path]:
    """Render saved results without invoking a model or changing evaluation.
    仅渲染已保存结果，不调用模型，也不改变评测方法。
    """

    baseline_records = _read_jsonl(baseline_path)
    metrics = _read_json(metrics_path)
    manifest_by_id = _manifest_records(manifest_path)
    baseline_by_id = {str(item["sample"]["id"]): item for item in baseline_records}
    sample_dirs = sorted(
        (path for path in (run_dir / "samples").iterdir() if path.is_dir()),
        key=lambda path: _natural_key(path.name),
    )
    if not sample_dirs:
        raise ValueError(f"No sample artifacts found below {run_dir / 'samples'}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    images_dir = output_path.parent / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    cards: list[str] = []
    for index, sample_dir in enumerate(sample_dirs, start=1):
        sample = _read_json(sample_dir / "sample.json")
        sample_id = str(sample["sample_id"])
        if sample_id not in baseline_by_id:
            raise ValueError(f"Baseline result missing sample {sample_id}")
        baseline = baseline_by_id[sample_id]
        analysis = _read_json(sample_dir / "change_expert" / "analysis" / "parsed.json")
        verification = _read_optional_json(
            sample_dir / "change_expert" / "parsed.json"
        )
        final = _read_json(sample_dir / "expert_result.json")
        trace = _read_json(sample_dir / "agent_trace.json")
        flag = _change_flag(sample, manifest_by_id)
        baseline_answer = str(baseline["prediction"].get("text", ""))
        final_answer = str(final.get("answer", ""))
        image_files = _copy_images(sample, images_dir, sample_id)
        row = {
            "index": index,
            "sample_id": sample_id,
            "changeflag": flag,
            "truth": _truth_label(flag),
            "baseline_answer": baseline_answer,
            "baseline_correct": _binary_correct(baseline_answer, flag),
            "analysis_answer": str(analysis.get("answer", "")),
            "verification_answer": (
                str(verification.get("answer", ""))
                if verification is not None
                else "未触发条件核验。"
            ),
            "final_answer": final_answer,
            "agent_correct": _binary_correct(final_answer, flag),
            "selected_stage": trace.get("selected_stage", "unknown"),
            "verification_guard": trace.get("verification_guard"),
            "verification_triggered": bool(
                trace.get(
                    "verification_triggered",
                    verification is not None,
                )
            ),
            "verification_reasons": list(
                trace.get("verification_reasons", [])
            ),
            "inference_seconds": float(trace.get("inference_seconds", 0.0) or 0.0),
            "references": list((sample.get("ground_truth") or {}).get("answers", [])),
            "analysis_evidence": list(analysis.get("evidence", [])),
            "verification_evidence": (
                list(verification.get("evidence", []))
                if verification is not None
                else []
            ),
            "image_files": image_files,
        }
        rows.append(row)
        cards.append(_sample_card(row))

    summary = _summary(rows, metrics)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_path.write_text(_page(summary, cards), encoding="utf-8")
    return output_path.resolve(), summary_path.resolve()


def _copy_images(sample: dict[str, Any], images_dir: Path, sample_id: str) -> list[dict[str, str]]:
    copied: list[dict[str, str]] = []
    for image in sample.get("images", []):
        source = Path(str(image["path"]))
        if not source.is_file():
            raise FileNotFoundError(source)
        role = str(image.get("role", "image"))
        suffix = source.suffix.lower() or ".png"
        destination = images_dir / f"{_safe_name(sample_id)}_{_safe_name(role)}{suffix}"
        shutil.copy2(source, destination)
        copied.append({"role": role, "path": f"images/{destination.name}"})
    return copied


def _change_flag(sample: dict[str, Any], manifest_by_id: dict[str, dict[str, Any]]) -> int:
    raw = (sample.get("ground_truth") or {}).get("raw") or {}
    sample_id = str(sample.get("sample_id", ""))
    value = raw.get("changeflag")
    if value is None:
        record = manifest_by_id.get(sample_id)
        if record is None:
            raise ValueError(f"Manifest missing sample {sample_id}")
        value = record.get("changeflag")
    if value not in (0, 1, "0", "1"):
        raise ValueError(f"Sample {sample_id} has invalid changeflag {value!r}")
    return int(value)


def _manifest_records(path: Path) -> dict[str, dict[str, Any]]:
    """Index the report-only derived manifest without changing an adapter.
    仅为报告索引派生清单，不改变数据适配器。
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("samples", []) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"Manifest must contain a sample list: {path}")
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"Manifest contains a non-object record: {path}")
        sample_id = str(record.get("sample_id", record.get("id", "")))
        if not sample_id:
            raise ValueError(f"Manifest record is missing an ID: {path}")
        if sample_id in indexed:
            raise ValueError(f"Manifest contains duplicate sample ID {sample_id}")
        indexed[sample_id] = record
    return indexed


def _predicts_no_change(text: str) -> bool:
    normalized = " ".join(str(text).casefold().split())
    return any(phrase in normalized for phrase in NO_CHANGE_PHRASES)


def _binary_correct(text: str, flag: int) -> bool:
    predicts_changed = not _predicts_no_change(text)
    return predicts_changed == bool(flag)


def _truth_label(flag: int) -> str:
    return "有变化" if flag else "无变化"


def _summary(rows: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    baseline_correct = sum(int(row["baseline_correct"]) for row in rows)
    agent_correct = sum(int(row["agent_correct"]) for row in rows)
    metric_deltas = metrics.get("metric_deltas") or {
        name: metrics["agent_metrics"][name] - metrics["baseline_metrics"][name]
        for name in METRIC_NAMES
    }
    return {
        "dataset": "LEVIR-CC",
        "scope": "Fixed saved comparison set; no inference performed by report generation.",
        "sample_count": len(rows),
        "truth_distribution": {
            "no_change": sum(row["changeflag"] == 0 for row in rows),
            "changed": sum(row["changeflag"] == 1 for row in rows),
        },
        "auxiliary_changeflag_accuracy": {
            "baseline": baseline_correct / len(rows),
            "agent": agent_correct / len(rows),
            "baseline_correct": baseline_correct,
            "agent_correct": agent_correct,
        },
        "caption_metrics": {
            "baseline": metrics["baseline_metrics"],
            "agent": metrics["agent_metrics"],
            "deltas": metric_deltas,
        },
        "selected_stage_distribution": _counts(rows, "selected_stage"),
        "verification_trigger_distribution": _counts(
            rows,
            "verification_triggered",
        ),
        "verification_guard_distribution": _counts(rows, "verification_guard"),
        "total_inference_seconds": round(sum(row["inference_seconds"] for row in rows), 6),
        "notes": [
            "Auxiliary changeflag accuracy is not an official LEVIR-CC caption metric.",
            "Per-sample strict full-text equality is intentionally not presented as accuracy.",
            "Caption metrics are copied from the separately persisted repository evaluation.",
        ],
    }


def _counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for row in rows:
        value = "None" if row[key] is None else str(row[key])
        values[value] = values.get(value, 0) + 1
    return values


def _sample_card(row: dict[str, Any]) -> str:
    badges = "".join(
        (
            _badge(f"官方标签：{row['truth']}", "neutral"),
            _badge(f"基线辅助判断：{'正确' if row['baseline_correct'] else '错误'}", "ok" if row["baseline_correct"] else "bad"),
            _badge(f"Agent 辅助判断：{'正确' if row['agent_correct'] else '错误'}", "ok" if row["agent_correct"] else "bad"),
        )
    )
    images = "".join(
        f'<figure><img src="{_escape(item["path"])}" alt="{_escape(item["role"])}">'
        f'<figcaption>{_escape(item["role"].upper())}</figcaption></figure>'
        for item in row["image_files"]
    )
    references = "".join(f"<li>{_escape(value)}</li>" for value in row["references"])
    guard = row["verification_guard"] or "未触发"
    verification_state = (
        "已触发：" + ", ".join(row["verification_reasons"])
        if row["verification_triggered"]
        else "未触发（沿用第一阶段）"
    )
    return f"""
<article class="card">
  <h2>样本 {row['index']:02d} · ID {_escape(row['sample_id'])}</h2>
  <div class="badges">{badges}</div>
  <div class="images">{images}</div>
  <div class="grid">
    <section><h3>原始基线回答</h3><p>{_escape(row['baseline_answer'])}</p></section>
    <section><h3>第一阶段：变化证据分析</h3><p>{_escape(row['analysis_answer'])}</p>
      {_evidence(row['analysis_evidence'])}</section>
    <section><h3>第二阶段：独立核验</h3><p>{_escape(row['verification_answer'])}</p>
      <p class="muted">{_escape(verification_state)}</p>
      {_evidence(row['verification_evidence'])}</section>
    <section><h3>Agent 最终回答</h3><p>{_escape(row['final_answer'])}</p>
      <dl><dt>选中阶段</dt><dd>{_escape(row['selected_stage'])}</dd>
      <dt>证据门控</dt><dd>{_escape(guard)}</dd>
      <dt>总推理耗时</dt><dd>{row['inference_seconds']:.3f} 秒</dd></dl></section>
  </div>
  <details><summary>查看 5 条官方参考描述</summary><ol>{references}</ol></details>
</article>"""


def _evidence(items: list[str]) -> str:
    if not items:
        return "<p class=\"muted\">未返回文字证据。</p>"
    return "<ul>" + "".join(f"<li>{_escape(item)}</li>" for item in items) + "</ul>"


def _badge(label: str, kind: str) -> str:
    return f'<span class="badge {kind}">{_escape(label)}</span>'


def _page(summary: dict[str, Any], cards: list[str]) -> str:
    metrics = summary["caption_metrics"]
    metric_rows = "".join(
        f"<tr><td>{name}</td><td>{metrics['baseline'][name]:.6f}</td>"
        f"<td>{metrics['agent'][name]:.6f}</td><td>{metrics['deltas'][name]:+.6f}</td></tr>"
        for name in METRIC_NAMES
    )
    accuracy = summary["auxiliary_changeflag_accuracy"]
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LEVIR-CC 两阶段 Agent 对比报告</title>
<style>
body{{margin:0;background:#f4f6f8;color:#18212f;font:16px/1.55 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1240px;margin:auto;padding:28px}} .hero,.card{{background:white;border-radius:18px;padding:24px;margin-bottom:22px;box-shadow:0 5px 24px #17203312}}
h1,h2,h3{{margin-top:0}} table{{border-collapse:collapse;width:100%}} th,td{{padding:9px 12px;border-bottom:1px solid #e5e9ef;text-align:right}} th:first-child,td:first-child{{text-align:left}}
.badges{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:18px}} .badge{{padding:5px 11px;border-radius:999px;background:#edf1f5}} .badge.ok{{background:#daf6e5;color:#08763b}} .badge.bad{{background:#ffe4e1;color:#b3261e}}
.images{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin-bottom:20px}} figure{{margin:0}} img{{display:block;width:100%;border-radius:12px}} figcaption{{text-align:center;color:#647184;margin-top:5px}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}} section{{background:#f8fafc;border:1px solid #e6ebf0;border-radius:12px;padding:16px}} p{{white-space:pre-wrap}} .muted{{color:#687588}} dl{{display:grid;grid-template-columns:max-content 1fr;gap:5px 12px}} dt{{font-weight:700}} dd{{margin:0}}
details{{margin-top:16px}} @media(max-width:760px){{main{{padding:12px}}.images,.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<section class="hero"><h1>LEVIR-CC 两阶段变化 Agent 对比报告</h1>
<p>固定 {summary['sample_count']} 条已保存样本；本报告未重新调用模型。逐样本“正确/错误”仅指基于官方 changeflag 的辅助有变化/无变化判断，不是官方 caption 准确率。</p>
<p><strong>辅助判断：</strong>基线 {accuracy['baseline_correct']}/{summary['sample_count']} ({accuracy['baseline']:.0%}) → Agent {accuracy['agent_correct']}/{summary['sample_count']} ({accuracy['agent']:.0%})</p>
<table><thead><tr><th>正式文本指标</th><th>基线</th><th>Agent</th><th>变化</th></tr></thead><tbody>{metric_rows}</tbody></table>
<p class="muted">全文严格一致不是 LEVIR-CC 的官方准确率，因此不作为逐样本成败标签。</p></section>
{''.join(cards)}
</main></body></html>"""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    """Return a persisted stage artifact when that optional stage ran.
    当可选阶段实际运行时返回其持久化产物。
    """

    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _natural_key(value: str) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value))


def _escape(value: Any) -> str:
    return html.escape(str(value))


def main() -> None:
    """Parse report-only paths and render saved artifacts.
    解析仅用于报告的路径并渲染已保存产物。
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    html_path, summary_path = build_change_comparison_report(
        args.baseline,
        args.run_dir,
        args.output,
        args.metrics,
        args.manifest,
    )
    print(f"HTML report: {html_path}")
    print(f"JSON summary: {summary_path}")


if __name__ == "__main__":
    main()
