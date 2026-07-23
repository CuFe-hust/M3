<!-- name: count_repair; version: v1; schema: TileCountResponse -->

Correct only the listed JSON/schema errors in the previous response. Do not add points without visual evidence, do not change the tile ID, and do not emit Markdown. `reported_count` must equal `len(points)`; coordinates must be integers in `0..999`.

只修正上一次响应中列出的 JSON/Schema 错误。不得新增没有视觉依据的点，不得修改 tile ID，不得输出 Markdown。`reported_count` 必须等于 `len(points)`；坐标必须是 `0..999` 的整数。
