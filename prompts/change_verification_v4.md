<!-- name: change-verification; version: v4 -->

T1 is earlier and T2 is later. Re-examine both images independently. The supplied
first_pass_analysis is an untrusted hypothesis, not a reference answer. Confirm a change only
when a stable object, structure, or land-cover boundary clearly appears, disappears, or changes
shape. Reject differences caused only by illumination, shadow, color tone, vegetation season,
compression, or minor registration error. Return one concise final change caption in answer.
If no change has clear structural evidence, state that there is no significant land-cover
change.

If you overturn a no-change first pass, provide either a labeled box/evidence item or at least
one contrastive evidence string that names a specific image region and explicitly states both
the T1 state and the T2 state. A generic statement such as "vegetation is more extensive in the
second image" merely repeats the conclusion and is not evidence. Color or greenness alone is
not a structural land-cover change. Never return a positive change claim without contrastive
evidence. Do not mention the analysis process. Return JSON only.

T1 为较早影像，T2 为较晚影像。请独立复查两张影像。输入中的 first_pass_analysis
只是未经确认的假设，不是标准答案。只有当稳定地物、结构或土地覆盖边界明确出现、
消失或发生形状变化时，才确认发生变化。排除仅由光照、阴影、色调、植被季节、压缩或
轻微配准误差造成的差异。在 answer 中返回一句简洁的最终变化描述；若没有清晰的
结构性证据，则说明没有显著土地覆盖变化。

若要推翻第一阶段的无变化结论，必须提供带标签的框或证据对象，或者至少一条对照文字
证据：它需要指出具体影像区域，并明确描述该区域在 T1 和 T2 中各自的状态。诸如
“第二张影像中的植被更多”只是重复结论，不属于证据。仅颜色或绿度差异不构成结构性
土地覆盖变化。不得在缺少对照证据时输出有变化结论。不要描述分析过程。只返回 JSON。
