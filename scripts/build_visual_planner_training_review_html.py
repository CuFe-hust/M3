#!/usr/bin/env python3
"""Build an offline image/ROI review page for compiled planner SFT JSONL.

为已编译的 planner SFT JSONL 生成离线图像与 ROI 标注审阅页面。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _extract(row: dict[str, Any], *, output_dir: Path, dataset_root: Path) -> dict[str, Any]:
    messages = row["messages"]
    user = messages[1]["content"]
    target = json.loads(messages[2]["content"][0]["text"])
    image = next(block["image"] for block in user if block["type"] == "image")
    question = next(block["text"] for block in user if block["type"] == "text")
    image_path = dataset_root / image
    return {
        "episode_id": row["episode_id"],
        "source_group": row["source_group"],
        "split": row["split"],
        "image": image,
        "image_href": image_path.relative_to(output_dir, walk_up=True).as_posix(),
        "question": question,
        "target": target,
    }


def build(input_path: Path, output_path: Path) -> None:
    dataset_root = input_path.resolve().parents[1]
    rows = _read_jsonl(input_path)
    records = [_extract(row, output_dir=output_path.parent.resolve(), dataset_root=dataset_root) for row in rows]
    tasks = Counter(record["target"]["task"] for record in records)
    categories = Counter(
        category
        for record in records
        for category in record["target"]["object_categories"]
    )
    payload = {
        "input": input_path.name,
        "records": records,
        "tasks": dict(sorted(tasks.items())),
        "categories": dict(sorted(categories.items())),
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(TEMPLATE.replace("__DATA__", encoded), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = args.input.resolve()
    output = (args.output or source.with_suffix(".review.html")).resolve()
    build(source, output)
    print(json.dumps({"output": str(output), "bytes": output.stat().st_size}, ensure_ascii=False))


TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Visual Planner · 训练标注审阅</title>
<style>
:root{--night:#132431;--deep:#193746;--paper:#f3f6f6;--card:#fff;--ink:#162a36;--muted:#61747d;--line:#cbd6d9;--orange:#ff5c24;--cyan:#9bd4d6;--green:#2b7864;--shadow:0 13px 34px rgba(13,36,48,.10)}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font-family:"Avenir Next","Segoe UI",sans-serif}.shell{max-width:1580px;margin:auto;padding:28px}.mast{display:grid;grid-template-columns:1.35fr .65fr;background:var(--night);color:white;min-height:250px;box-shadow:var(--shadow);overflow:hidden}.mast-copy{padding:38px 44px}.eyebrow,.label{font:700 11px/1.3 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.12em;text-transform:uppercase}.eyebrow{color:var(--cyan)}h1{font-size:clamp(34px,5vw,72px);line-height:.92;letter-spacing:-.05em;margin:17px 0 20px;max-width:850px}.mast p{max-width:720px;color:#c7d8de;line-height:1.55}.radar{position:relative;min-height:250px;background:repeating-radial-gradient(circle at 74% 50%,transparent 0 42px,rgba(155,212,214,.16) 43px 44px),linear-gradient(135deg,var(--deep),#0e202a)}.radar:after{content:"ROI / 0..999";position:absolute;inset:23% 18%;border:3px solid var(--orange);display:grid;place-items:center;color:#fff;font:700 12px ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.12em;background:rgba(255,92,36,.08)}.stats{display:flex;flex-wrap:wrap;gap:1px;background:#324b58}.stat{flex:1;min-width:120px;background:var(--night);padding:15px 20px}.stat b{font-size:25px;display:block}.stat span{font-size:11px;color:#afc3cb}.toolbar{position:sticky;top:0;z-index:20;display:grid;grid-template-columns:2fr repeat(3,minmax(150px,.65fr));gap:9px;padding:14px 0;background:rgba(243,246,246,.95);backdrop-filter:blur(12px)}input,select,button{font:inherit}input,select{width:100%;border:1px solid var(--line);background:white;padding:11px 13px;color:var(--ink)}input:focus-visible,select:focus-visible,button:focus-visible{outline:3px solid rgba(255,92,36,.45);outline-offset:2px}.summary{display:flex;align-items:center;justify-content:space-between;gap:18px;margin:7px 0 17px;color:var(--muted)}.legend{display:flex;align-items:center;gap:8px;font:700 11px ui-monospace,SFMono-Regular,Menlo,monospace}.legend i{width:23px;height:12px;border:2px solid var(--orange);background:rgba(255,92,36,.14)}.records{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.card{background:var(--card);border:1px solid var(--line);box-shadow:var(--shadow);min-width:0}.visual{position:relative;background:#d9e1e3;overflow:hidden}.visual img{display:block;width:100%;height:auto}.visual svg{position:absolute;inset:0;display:block;width:100%;height:100%;pointer-events:none}.roi{fill:rgba(255,92,36,.14);stroke:var(--orange);stroke-width:5;vector-effect:non-scaling-stroke}.roi-glow{fill:none;stroke:white;stroke-width:8;opacity:.8;vector-effect:non-scaling-stroke}.index{position:absolute;left:0;top:0;background:var(--night);color:white;padding:9px 11px;font:700 11px ui-monospace,SFMono-Regular,Menlo,monospace}.body{padding:20px}.topline{display:flex;align-items:flex-start;justify-content:space-between;gap:14px}.id{font:700 12px ui-monospace,SFMono-Regular,Menlo,monospace;overflow-wrap:anywhere;color:var(--deep)}.task{white-space:nowrap;background:#dcebed;color:var(--deep);padding:6px 8px;font:700 10px ui-monospace,SFMono-Regular,Menlo,monospace}.question{font-size:18px;line-height:1.48;margin:17px 0}.chips{display:flex;flex-wrap:wrap;gap:6px}.chip{border:1px solid var(--line);padding:5px 7px;font:700 10px ui-monospace,SFMono-Regular,Menlo,monospace}.chip.on{color:var(--green);border-color:#84b7a9}.chip.roi{color:#b74117;border-color:#e9a183;background:#fff4ef}.plan{margin-top:15px;border-top:1px solid var(--line);padding-top:14px}.plan-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px 18px}.field{min-width:0}.field b{display:block;color:var(--muted);font:700 9px ui-monospace,SFMono-Regular,Menlo,monospace;text-transform:uppercase;letter-spacing:.09em;margin-bottom:3px}.field span{font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;overflow-wrap:anywhere}.raw{margin-top:12px;border:1px solid var(--line)}summary{cursor:pointer;padding:10px 12px;color:var(--deep);font-weight:700}pre{margin:0;padding:13px;background:var(--night);color:#d9e7eb;white-space:pre-wrap;overflow-wrap:anywhere;font:11px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace}.pager{display:flex;justify-content:center;align-items:center;gap:12px;padding:26px}.pager button{border:1px solid var(--line);background:white;color:var(--deep);padding:10px 16px;cursor:pointer}.pager button:disabled{opacity:.4;cursor:not-allowed}.empty{grid-column:1/-1;padding:70px;text-align:center;color:var(--muted)}@media(max-width:900px){.shell{padding:12px}.mast{grid-template-columns:1fr}.radar{min-height:150px}.toolbar{grid-template-columns:1fr 1fr}.records{grid-template-columns:1fr}.mast-copy{padding:28px 24px}}@media(max-width:540px){.toolbar{grid-template-columns:1fr}.plan-grid{grid-template-columns:1fr}}@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
</style></head><body><main class="shell"><section class="mast"><div class="mast-copy"><div class="eyebrow">Remote-sensing annotation desk</div><h1>Visual Plan<br>标注审阅</h1><p>直接核对图像、问题、VisualTaskPlan 与归一化 ROI。橙框使用 planner 的闭区间 0..999 坐标，并按图像原始宽高比例叠加。</p></div><div class="radar" aria-hidden="true"></div></section><div class="stats" id="stats"></div><section class="toolbar"><input id="search" placeholder="搜索样本 ID、问题或类别"><select id="task"><option value="">全部 task</option></select><select id="category"><option value="">全部辅助类别</option></select><select id="assistance"><option value="">全部辅助状态</option><option value="true">需要视觉辅助</option><option value="false">不需要视觉辅助</option></select></section><div class="summary"><span id="count"></span><span class="legend"><i></i>显式 ROI</span></div><section class="records" id="records"></section><nav class="pager"><button id="prev">上一页</button><span id="page"></span><button id="next">下一页</button></nav></main><script id="data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent),PAGE=40;let page=0,filtered=D.records;const $=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const pretty=x=>JSON.stringify(x,null,2);$('stats').innerHTML=[['样本',D.records.length],['任务',Object.keys(D.tasks).length],['显式 ROI',D.records.filter(x=>x.target.region_request.explicit).length],['视觉辅助',D.records.filter(x=>x.target.needs_visual_assistance).length]].map(([k,v])=>`<div class="stat"><b>${v}</b><span>${k}</span></div>`).join('');Object.keys(D.tasks).forEach(x=>$('task').insertAdjacentHTML('beforeend',`<option>${esc(x)}</option>`));Object.keys(D.categories).forEach(x=>$('category').insertAdjacentHTML('beforeend',`<option>${esc(x)}</option>`));['search','task','category','assistance'].forEach(id=>$(id).addEventListener('input',filter));$('prev').onclick=()=>{page--;render();scrollTo({top:240,behavior:'smooth'})};$('next').onclick=()=>{page++;render();scrollTo({top:240,behavior:'smooth'})};function filter(){const q=$('search').value.trim().toLowerCase(),task=$('task').value,cat=$('category').value,assist=$('assistance').value;filtered=D.records.filter(r=>(!q||[r.episode_id,r.question,...r.target.object_categories].join(' ').toLowerCase().includes(q))&&(!task||r.target.task===task)&&(!cat||r.target.object_categories.includes(cat))&&(!assist||String(r.target.needs_visual_assistance)===assist));page=0;render()}function card(r,i){const t=r.target,rr=t.region_request,roi=rr.roi_xyxy;const overlay=rr.explicit&&roi?`<svg viewBox="0 0 999 999" preserveAspectRatio="none" aria-hidden="true"><rect class="roi-glow" x="${roi[0]}" y="${roi[1]}" width="${roi[2]-roi[0]}" height="${roi[3]-roi[1]}"></rect><rect class="roi" x="${roi[0]}" y="${roi[1]}" width="${roi[2]-roi[0]}" height="${roi[3]-roi[1]}"></rect></svg>`:'';const cats=t.object_categories.length?t.object_categories.map(x=>`<span class="chip">${esc(x)}</span>`).join(''):'<span class="chip">no categories</span>';return `<article class="card"><div class="visual"><img loading="lazy" src="${esc(r.image_href)}" alt="${esc(r.episode_id)}">${overlay}<span class="index">${String(i+1).padStart(4,'0')}</span></div><div class="body"><div class="topline"><div class="id">${esc(r.episode_id)}</div><span class="task">${esc(t.task)}</span></div><p class="question">${esc(r.question).replace(/\n/g,'<br>')}</p><div class="chips"><span class="chip ${t.needs_visual_assistance?'on':''}">assistance · ${t.needs_visual_assistance}</span>${cats}${rr.explicit?`<span class="chip roi">ROI · ${roi.join(', ')}</span>`:''}</div><div class="plan"><div class="plan-grid"><div class="field"><b>count target</b><span>${esc(t.count_target)}</span></div><div class="field"><b>reason codes</b><span>${esc(t.reason_codes.join(' · '))}</span></div><div class="field"><b>image</b><span>${esc(r.image)}</span></div><div class="field"><b>source group</b><span>${esc(r.source_group)}</span></div></div><details class="raw"><summary>完整 VisualTaskPlan JSON</summary><pre>${esc(pretty(t))}</pre></details></div></div></article>`}function render(){const pages=Math.max(1,Math.ceil(filtered.length/PAGE));page=Math.min(page,pages-1);const start=page*PAGE,rows=filtered.slice(start,start+PAGE);$('records').innerHTML=rows.length?rows.map((r,i)=>card(r,start+i)).join(''):'<div class="empty">没有符合筛选条件的标注。</div>';$('count').textContent=`显示 ${filtered.length} / ${D.records.length} 条 · 每页 ${PAGE} 条`;$('page').textContent=`${page+1} / ${pages}`;$('prev').disabled=page===0;$('next').disabled=page>=pages-1}render();
</script></body></html>'''


if __name__ == "__main__":
    main()
