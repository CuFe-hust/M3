from eval.metrics import _caption_metric_text


def test_caption_metric_text_removes_line_protocol_tokens() -> None:
    raw = "First line.\n\nSecond\tline. ||| Third line.\r\n"

    normalized = _caption_metric_text(raw)

    assert normalized == "First line. Second line. Third line."
    assert "\n" not in normalized
    assert "\r" not in normalized
    assert "|||" not in normalized
