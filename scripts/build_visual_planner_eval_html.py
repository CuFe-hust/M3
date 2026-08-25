from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.result_dir.resolve()
    output = (args.output or root / "visual_planner_results.html").resolve()

    results = read_jsonl(root / "results.jsonl")
    selection = read_jsonl(root / "selection.jsonl")
    selection_by_question = {
        str(item.get("question_id")): item for item in selection
    }
    records: list[dict[str, Any]] = []
    for row in results:
        index = int(row["index"])
        sample_id = str(row["sample_id"])
        sample_dir = root / "samples" / f"{index:03d}-{sample_id}"
        call_dir = sample_dir / "visual_task_plan"
        selected = selection_by_question.get(sample_id.removeprefix("lrs-vqa-"), {})
        parsed = read_json(call_dir / "parsed.json") if (call_dir / "parsed.json").is_file() else None
        roi_request = (parsed or {}).get("region_request") or {}
        roi_preview = root / "roi_previews" / f"{index:03d}-{sample_id}.jpg"
        record = {
            "index": index,
            "sample_id": sample_id,
            "status": row.get("status"),
            "question_type": selected.get("source_category", "unknown"),
            "predicted_task": (parsed or {}).get("task"),
            "image": row.get("image"),
            "image_sha256": row.get("image_sha256"),
            "preview": (
                f"roi_previews/{index:03d}-{sample_id}.jpg"
                if roi_preview.is_file()
                else f"image_previews/{row.get('image_sha256')}.jpg"
            ),
            "roi_explicit": bool(roi_request.get("explicit")),
            "roi_xyxy": roi_request.get("roi_xyxy"),
            "question": row.get("question"),
            "raw_input": read_json(sample_dir / "raw_input.json"),
            "system_input": read_json(call_dir / "request.json"),
            "raw_output": (call_dir / "raw_response.txt").read_text(encoding="utf-8"),
            "parsed_output": parsed,
            "validation": read_json(call_dir / "validation.json"),
            "error_type": row.get("error_type"),
        }
        records.append(record)

    summary = read_json(root / "summary.json")
    payload = {
        "summary": summary,
        "question_types": dict(sorted(Counter(item["question_type"] for item in records).items())),
        "predicted_tasks": dict(sorted(Counter(str(item["predicted_task"]) for item in records).items())),
        "records": records,
    }
    encoded = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    document = TEMPLATE.replace("__REPORT_DATA__", encoded)
    output.write_text(document, encoding="utf-8")
    print(json.dumps({"html": str(output), "records": len(records), "bytes": output.stat().st_size}))


TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Visual Planner LoRA · LRS-VQA 审计</title>
<style>
:root{--paper:#eef2f3;--surface:#f9fbfb;--ink:#142433;--muted:#60727d;--line:#c7d2d7;--blue:#174f78;--blue2:#dcebf2;--orange:#c9582a;--ok:#21705a;--shadow:0 14px 38px rgba(23,50,65,.09)}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font-family:"Avenir Next","Segoe UI",sans-serif}.shell{max-width:1540px;margin:auto;padding:34px}.hero{position:relative;overflow:hidden;background:var(--blue);color:white;padding:42px 48px;border-radius:4px;box-shadow:var(--shadow)}.hero:after{content:"";position:absolute;right:-90px;top:-210px;width:510px;height:510px;border:1px solid rgba(255,255,255,.25);border-radius:50%;box-shadow:0 0 0 52px rgba(255,255,255,.035),0 0 0 104px rgba(255,255,255,.025)}.eyebrow,.label{font:700 11px/1.3 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.13em;text-transform:uppercase}.hero h1{font-size:clamp(32px,5vw,67px);line-height:.98;letter-spacing:-.045em;max-width:850px;margin:18px 0}.hero p{max-width:720px;color:#d9e8ef;font-size:16px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:rgba(255,255,255,.25);margin-top:34px;position:relative;z-index:1}.stat{background:rgba(10,49,74,.75);padding:18px}.stat b{display:block;font-size:28px}.stat span{font-size:12px;color:#c7dce6}.toolbar{position:sticky;top:0;z-index:20;display:grid;grid-template-columns:1.5fr repeat(3,minmax(150px,.5fr));gap:10px;padding:15px 0;background:rgba(238,242,243,.94);backdrop-filter:blur(10px)}input,select{width:100%;border:1px solid var(--line);background:white;padding:12px 14px;color:var(--ink);font:inherit;border-radius:3px}.distribution{display:flex;flex-wrap:wrap;gap:8px;margin:9px 0 24px}.chip{font:700 11px ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--blue2);color:var(--blue);padding:7px 9px;border-radius:2px}.countline{color:var(--muted);margin:8px 0 18px}.records{display:grid;gap:18px}.record{display:grid;grid-template-columns:58px minmax(260px,380px) 1fr;background:var(--surface);border:1px solid var(--line);box-shadow:var(--shadow);border-radius:4px;overflow:hidden}.rail{background:#e1e8eb;border-right:1px solid var(--line);padding:20px 12px;text-align:center;font:700 13px ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--blue);writing-mode:vertical-rl}.visual{padding:20px;border-right:1px solid var(--line)}.image-zoom{display:block;width:100%;padding:0;border:0;background:transparent;cursor:zoom-in}.image-zoom:focus-visible{outline:4px solid var(--orange);outline-offset:3px}.visual img{display:block;width:100%;max-height:360px;object-fit:contain;background:#dfe6e8;border:1px solid var(--line)}.meta{display:flex;flex-wrap:wrap;gap:7px;margin:12px 0}.badge{font:700 10px ui-monospace,SFMono-Regular,Menlo,monospace;padding:5px 7px;border:1px solid var(--line);background:white}.badge.ok{color:var(--ok);border-color:#8fb9ab}.badge.failed{color:var(--orange);border-color:#d8a18b}.question{font-size:17px;line-height:1.55;margin:16px 0 0}.audit{padding:20px;min-width:0}.audit h2{font-size:20px;margin:0 0 4px}.path{color:var(--muted);font:11px ui-monospace,SFMono-Regular,Menlo,monospace;overflow-wrap:anywhere}.outputs{display:grid;gap:9px;margin-top:16px}details{border:1px solid var(--line);background:white;border-radius:3px}summary{cursor:pointer;padding:12px 14px;font-weight:700;color:var(--blue)}summary:focus-visible{outline:3px solid #e0a17d;outline-offset:2px}pre{margin:0;padding:15px;border-top:1px solid var(--line);background:#102633;color:#dce9ee;white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.58 ui-monospace,SFMono-Regular,Menlo,monospace;max-height:620px;overflow:auto}.raw{color:#ffe1bf}.empty{padding:60px;text-align:center;color:var(--muted)}.lightbox[hidden]{display:none}.lightbox{position:fixed;inset:0;z-index:100;background:rgba(6,18,27,.92);display:grid;grid-template-rows:auto 1fr;overflow:auto;padding:18px}.lightbox-bar{position:sticky;top:0;z-index:2;display:flex;align-items:center;justify-content:space-between;gap:20px;color:white;background:rgba(6,18,27,.92);padding:8px 4px 15px;font:12px ui-monospace,SFMono-Regular,Menlo,monospace}.lightbox-close{border:1px solid rgba(255,255,255,.55);background:transparent;color:white;padding:9px 13px;font:700 13px inherit;cursor:pointer}.lightbox-close:focus-visible{outline:3px solid #ff8c58;outline-offset:3px}.lightbox-stage{display:grid;place-items:center;min-height:0}.lightbox-image{display:block;max-width:none;width:auto;height:auto;min-width:min(100%,960px);object-fit:contain;box-shadow:0 20px 70px rgba(0,0,0,.48)}body.lightbox-open{overflow:hidden}@media(max-width:900px){.shell{padding:16px}.hero{padding:30px 24px}.stats{grid-template-columns:1fr 1fr}.toolbar{grid-template-columns:1fr 1fr}.record{grid-template-columns:38px 1fr}.visual{border-right:0}.audit{grid-column:2}.rail{grid-row:1/3}.lightbox{padding:10px}.lightbox-image{min-width:100%;max-width:100%}}@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
</style>
</head>
<body><main class="shell"><section class="hero"><div class="eyebrow">Remote-sensing inference audit / 2026-08-24</div><h1>Visual Planner<br>LoRA 现场记录</h1><p>50 张未与 refined-v4 重叠的 LRS-VQA 图像，每图最多 10 问。页面保留原始问题、系统输入、模型原始输出和 schema 解析结果；点击预览图可放大查看 ROI。</p><div class="stats" id="stats"></div></section><section class="toolbar"><input id="search" placeholder="搜索问题、样本 ID 或图像路径"><select id="qtype"><option value="">全部问题类型</option></select><select id="task"><option value="">全部预测任务</option></select><select id="status"><option value="">全部状态</option><option>succeeded</option><option>failed</option></select></section><div class="distribution" id="distribution"></div><div class="countline" id="countline"></div><section class="records" id="records"></section></main><div class="lightbox" id="lightbox" role="dialog" aria-modal="true" aria-label="图像放大预览" hidden><div class="lightbox-bar"><span id="lightbox-caption"></span><button class="lightbox-close" id="lightbox-close" type="button">关闭 ×</button></div><div class="lightbox-stage" id="lightbox-stage"><img class="lightbox-image" id="lightbox-image" alt=""></div></div>
<script id="report-data" type="application/json">__REPORT_DATA__</script>
<script>
const data=JSON.parse(document.getElementById('report-data').textContent);const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const pretty=v=>JSON.stringify(v,null,2);const stats=document.getElementById('stats');stats.innerHTML=[['总计',data.summary.total],['成功',data.summary.succeeded],['失败',data.summary.failed],['图像',new Set(data.records.map(x=>x.image_sha256)).size]].map(([a,b])=>`<div class="stat"><b>${b}</b><span>${a}</span></div>`).join('');
const qtype=document.getElementById('qtype'),task=document.getElementById('task');Object.keys(data.question_types).forEach(x=>qtype.insertAdjacentHTML('beforeend',`<option>${esc(x)}</option>`));Object.keys(data.predicted_tasks).forEach(x=>task.insertAdjacentHTML('beforeend',`<option>${esc(x)}</option>`));document.getElementById('distribution').innerHTML=Object.entries(data.predicted_tasks).map(([k,v])=>`<span class="chip">${esc(k)} · ${v}</span>`).join('');
const controls=['search','qtype','task','status'].map(id=>document.getElementById(id));controls.forEach(x=>x.addEventListener('input',render));function render(){const [search,qt,ta,st]=controls.map(x=>x.value.trim().toLowerCase());const rows=data.records.filter(r=>(!search||[r.question,r.sample_id,r.image].join(' ').toLowerCase().includes(search))&&(!qt||r.question_type.toLowerCase()===qt)&&(!ta||String(r.predicted_task).toLowerCase()===ta)&&(!st||r.status.toLowerCase()===st));document.getElementById('countline').textContent=`显示 ${rows.length} / ${data.records.length} 条`;document.getElementById('records').innerHTML=rows.length?rows.map(card).join(''):'<div class="empty">没有符合筛选条件的记录。</div>'}
function card(r){const roi=r.roi_explicit?`ROI · ${r.roi_xyxy.join(', ')}`:'ROI · implicit';return `<article class="record"><div class="rail">SAMPLE ${String(r.index).padStart(3,'0')}</div><div class="visual"><button class="image-zoom" type="button" data-preview="${esc(r.preview)}" data-caption="${esc(r.sample_id)} · ${esc(roi)}" aria-label="放大预览 ${esc(r.sample_id)}"><img loading="lazy" src="${esc(r.preview)}" alt="${esc(r.sample_id)}"></button><div class="meta"><span class="badge ${esc(r.status)}">${esc(r.status)}</span><span class="badge">${esc(r.question_type)}</span><span class="badge">${esc(r.predicted_task)}</span><span class="badge">${esc(roi)}</span></div><div class="label">原始问题</div><p class="question">${esc(r.question)}</p></div><div class="audit"><h2>${esc(r.sample_id)}</h2><div class="path">${esc(r.image)} · ${esc(r.image_sha256)}</div><div class="outputs"><details><summary>原始输入 raw_input.json</summary><pre>${esc(pretty(r.raw_input))}</pre></details><details><summary>系统输入 request.json</summary><pre>${esc(pretty(r.system_input))}</pre></details><details open><summary>系统原始输出 raw_response.txt</summary><pre class="raw">${esc(r.raw_output)}</pre></details><details open><summary>解析输出 parsed.json</summary><pre>${esc(pretty(r.parsed_output))}</pre></details><details><summary>校验记录 validation.json</summary><pre>${esc(pretty(r.validation))}</pre></details></div></div></article>`}
const lightbox=document.getElementById('lightbox'),lightboxImage=document.getElementById('lightbox-image'),lightboxCaption=document.getElementById('lightbox-caption'),lightboxClose=document.getElementById('lightbox-close');let previousFocus=null;function openLightbox(button){previousFocus=button;lightboxImage.src=button.dataset.preview;lightboxImage.alt=button.dataset.caption;lightboxCaption.textContent=button.dataset.caption;lightbox.hidden=false;document.body.classList.add('lightbox-open');lightboxClose.focus()}function closeLightbox(){lightbox.hidden=true;lightboxImage.removeAttribute('src');document.body.classList.remove('lightbox-open');if(previousFocus)previousFocus.focus()}document.getElementById('records').addEventListener('click',event=>{const button=event.target.closest('.image-zoom');if(button)openLightbox(button)});lightboxClose.addEventListener('click',closeLightbox);lightbox.addEventListener('click',event=>{if(event.target===lightbox||event.target.id==='lightbox-stage')closeLightbox()});document.addEventListener('keydown',event=>{if(event.key==='Escape'&&!lightbox.hidden)closeLightbox()});render();
</script></body></html>'''


if __name__ == "__main__":
    main()
