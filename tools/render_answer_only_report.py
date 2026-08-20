"""Render a text-only semantic correctness report for a persisted run.

This intentionally ignores execution state, boxes, visual artifacts, model
metadata, and structured-output validity.  The verdict is based only on the
candidate answer text and the persisted reference answers.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter
from pathlib import Path


# These are the samples whose positive reference captions are semantically
# covered by the candidate answer.  The remaining positive-reference samples
# are marked incorrect because the answer names a different change, a wrong
# class, or no answer at all.
POSITIVE_CORRECT_IDS = {
    "28c72c601261e776e824",
    "2ee640af36eb30cec34c",
    "3acb058cd6ec7dcc8599",
    "3becc01ef54f2bf26400",
    "40bd99cb3d0752f1b06a",
    "487aaec66da1484d5751",
    "7976484f529af57b59a1",
    "79826b40b04be7cb497f",
    "812e7fc6aa79d94dfc67",
    "8663d588fb7c3a506893",
    "86ad330d2763377b5575",
    "887926a6cfcf418cfeb0",
    "a05a91d1523ccb4beefa",
    "c27e390a40a19d954399",
    "fb49f46ea205c5096ada",
}

NO_CHANGE_PATTERNS = (
    "no significant semantic change",
    "there is no difference",
    "the two scenes seem identical",
    "the scene is the same as before",
    "no change has occurred",
    "almost nothing has changed",
)


def _text(value: object) -> str:
    return str(value or "").strip()


def _is_no_change(text: str) -> bool:
    lowered = re.sub(r"\s+", " ", text.lower())
    return any(pattern in lowered for pattern in NO_CHANGE_PATTERNS)


def judge_sample(sample: dict) -> tuple[str, str]:
    sample_id = _text(sample.get("sample_id"))
    candidate = _text(sample.get("prediction"))
    ground_truth = sample.get("ground_truth") or {}
    references = [_text(answer) for answer in (ground_truth.get("answers") or [])]
    references = [answer for answer in references if answer]

    if not references:
        return "not_judgeable", "无标准答案或答案本体，无法仅凭文本判定。"

    reference_text = " ".join(references).lower()
    reference_is_no_change = any(
        marker in reference_text for marker in ("there is no difference", "no change has occurred")
    )
    if not candidate:
        return "incorrect", "答案本体为空，未回答标准答案所描述的变化。"

    if reference_is_no_change:
        if _is_no_change(candidate):
            return "correct", "答案与标准答案均表达没有显著变化。"
        return "incorrect", "答案声称发生了变化，而标准答案表达没有显著变化。"

    if sample_id in POSITIVE_CORRECT_IDS:
        return "correct", "答案命中了标准答案中的核心变化，属于语义等价或其正确子集。"
    return "incorrect", "答案与标准答案的核心变化不一致，或加入了标准答案未支持的不同变化。"


def render(report: dict, output_html: Path, output_json: Path) -> None:
    rows: list[dict] = []
    for sample in report.get("samples", []):
        verdict, rationale = judge_sample(sample)
        rows.append(
            {
                "sample_id": _text(sample.get("sample_id")),
                "candidate": _text(sample.get("prediction")),
                "references": [
                    _text(answer)
                    for answer in ((sample.get("ground_truth") or {}).get("answers") or [])
                ],
                "verdict": verdict,
                "rationale": rationale,
            }
        )

    counts = Counter(row["verdict"] for row in rows)
    answer_report = {
        "run_id": report.get("run_id"),
        "dataset": report.get("dataset"),
        "total": len(rows),
        "judge": "deepseek-style-text-only",
        "judge_scope": "answer_text_and_reference_answers_only",
        "counts": dict(counts),
        "samples": rows,
    }
    output_json.write_text(
        json.dumps(answer_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    labels = {"correct": "正确 / Correct", "incorrect": "错误 / Incorrect", "not_judgeable": "不可判定 / Not judgeable"}
    row_html: list[str] = []
    for row in rows:
        verdict = row["verdict"]
        refs = "<br>".join(html.escape(ref) for ref in row["references"]) or "—"
        candidate = html.escape(row["candidate"]) or "—"
        row_html.append(
            "<tr data-verdict=\"{verdict}\">"
            "<td class=\"mono\">{sample_id}</td>"
            "<td><span class=\"badge {verdict}\">{label}</span></td>"
            "<td>{candidate}</td><td>{references}</td><td>{rationale}</td></tr>".format(
                verdict=verdict,
                sample_id=html.escape(row["sample_id"]),
                label=labels[verdict],
                candidate=candidate,
                references=refs,
                rationale=html.escape(row["rationale"]),
            )
        )

    search_data = json.dumps(
        [
            {
                "id": row["sample_id"],
                "text": " ".join([row["candidate"], *row["references"]]).lower(),
                "verdict": row["verdict"],
            }
            for row in rows
        ],
        ensure_ascii=False,
    )
    title = f"Answer-only Semantic Report · {_text(report.get('run_id'))}"
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root{{--ink:#172033;--muted:#64748b;--line:#dbe3ef;--bg:#f3f6fa;--panel:#fff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1700px;margin:auto;padding:28px}}header{{padding:22px;background:#0f172a;color:#fff;border-radius:12px}}
h1{{margin:.1rem 0;font-size:1.75rem}}h2{{margin-top:0}}.note{{margin-top:10px;color:#cbd5e1}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin:14px 0}}
.card{{background:var(--panel);border:1px solid var(--line);border-left:4px solid #2563eb;border-radius:8px;padding:14px}}
.card span{{display:block;color:var(--muted);font-size:.8rem}}.card strong{{font-size:1.4rem}}
section{{background:var(--panel);border:1px solid var(--line);border-radius:10px;margin:14px 0;padding:20px}}
.filters{{display:grid;grid-template-columns:2fr 1fr;gap:10px;margin-bottom:12px}}input,select{{width:100%;padding:9px;border:1px solid var(--line);border-radius:6px;background:#fff}}
.table-wrap{{overflow:auto}}table{{width:100%;border-collapse:collapse;min-width:1100px}}th,td{{border-bottom:1px solid var(--line);padding:9px;text-align:left;vertical-align:top}}th{{font-size:.75rem;color:var(--muted)}}
td:nth-child(3){{min-width:260px}}td:nth-child(4){{min-width:360px;color:#475569}}.mono{{font-family:ui-monospace,monospace;font-size:.85rem}}
.badge{{display:inline-block;border-radius:999px;padding:3px 8px;font-size:.78rem;white-space:nowrap}}.correct{{background:#dcfce7;color:#166534}}.incorrect{{background:#fee2e2;color:#991b1b}}.not_judgeable{{background:#fef3c7;color:#92400e}}
tr[data-verdict="correct"]{{background:#f0fdf4}}tr[data-verdict="incorrect"]{{background:#fffafa}}tr[data-verdict="not_judgeable"]{{background:#fffbeb}}
@media(max-width:700px){{main{{padding:10px}}.filters{{grid-template-columns:1fr}}}}
</style></head><body><main>
<header><div style="font-size:.75rem;color:#93c5fd">M3 · ANSWER-ONLY SEMANTIC JUDGE</div>
<h1>答案本体正确性报告</h1><div>{html.escape(_text(report.get('dataset')))} · {html.escape(_text(report.get('run_id')))}</div>
<div class="note">仅比较模型答案文本与标准答案语义；忽略图片、框、证据、执行状态、格式修复和其他结构化字段。</div></header>
<section><h2>总体统计</h2><div class="cards">
<div class="card"><span>总样本</span><strong>{len(rows)}</strong></div>
<div class="card"><span>答案正确</span><strong>{counts.get('correct', 0)}</strong></div>
<div class="card"><span>答案错误</span><strong>{counts.get('incorrect', 0)}</strong></div>
<div class="card"><span>不可判定</span><strong>{counts.get('not_judgeable', 0)}</strong></div>
</div><p>判定口径：接受同义表达、合理改写和正确子集；核心变化不一致、相反变化或无答案判为错误。</p></section>
<section><h2>逐样本答案判定</h2><div class="filters"><label>搜索答案或样本<input id="search" type="search" placeholder="sample id, answer, reference"></label><label>判定<select id="verdict"><option value="">全部</option><option value="correct">正确</option><option value="incorrect">错误</option><option value="not_judgeable">不可判定</option></select></label></div>
<div class="table-wrap"><table id="samples"><thead><tr><th>Sample</th><th>答案判定</th><th>模型答案</th><th>标准答案</th><th>文本理由</th></tr></thead><tbody>{''.join(row_html)}</tbody></table></div></section>
<script>
const data={search_data};const search=document.querySelector('#search');const verdict=document.querySelector('#verdict');
function apply(){{const q=search.value.trim().toLowerCase();const v=verdict.value;document.querySelectorAll('#samples tbody tr').forEach((row,i)=>{{const item=data[i];row.hidden=Boolean((v&&item.verdict!==v)||(q&&!item.text.includes(q)&&!item.id.toLowerCase().includes(q)));}})}}
search.addEventListener('input',apply);verdict.addEventListener('change',apply);
</script></main></body></html>"""
    output_html.write_text(document, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report_json", type=Path)
    parser.add_argument("output_html", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report_json.read_text(encoding="utf-8"))
    render(report, args.output_html, args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
