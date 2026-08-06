import zipfile

from data import downloader


def _fake_snapshot(calls, create_zip=False):
    def fake(repo_id, target, **kwargs):
        calls.append((repo_id, str(target)))
        target.mkdir(parents=True, exist_ok=True)
        if create_zip:
            archive = target / "Images_val.zip"
            with zipfile.ZipFile(archive, "w") as zip_file:
                zip_file.writestr("content.txt", "data")
    return fake


def test_download_datasets_routes_vrsbench_and_extracts_archives(tmp_path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(downloader, "_snapshot_download", _fake_snapshot(calls, create_zip=True))

    downloaded = downloader.download_datasets(["vrsbench"], tmp_path)

    assert calls == [("xiang709/VRSBench", str(tmp_path / "vrsbench"))]
    assert downloaded["vrsbench"] == tmp_path / "vrsbench"
    assert (tmp_path / "vrsbench" / "Images_val" / "content.txt").read_text(encoding="utf-8") == "data"


def test_download_xlrs_bench_downloads_all_releases_under_one_root(tmp_path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(downloader, "_snapshot_download", _fake_snapshot(calls))

    target = downloader.download_xlrs_bench(tmp_path)

    assert target == tmp_path / "xlrs_bench"
    assert calls == [
        ("initiacms/XLRS-Bench_caption_en", str(tmp_path / "xlrs_bench" / "XLRS-Bench_caption_en")),
        ("initiacms/XLRS-Bench_visual_grounding_en", str(tmp_path / "xlrs_bench" / "XLRS-Bench_visual_grounding_en")),
        ("initiacms/XLRS-Bench-lite", str(tmp_path / "xlrs_bench" / "XLRS-Bench-lite")),
    ]


def test_download_datasets_keeps_legacy_flat_names(tmp_path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(downloader, "_snapshot_download", _fake_snapshot(calls))

    downloaded = downloader.download_datasets(["xlrs_caption_en"], tmp_path)

    assert calls == [("initiacms/XLRS-Bench_caption_en", str(tmp_path / "xlrs_caption_en"))]
    assert downloaded["xlrs_caption_en"] == tmp_path / "xlrs_caption_en"


def test_download_datasets_rejects_unknown_name(tmp_path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(downloader, "_snapshot_download", _fake_snapshot(calls))

    try:
        downloader.download_datasets(["unknown"], tmp_path)
    except ValueError as error:
        assert "unknown" in str(error)
    else:
        raise AssertionError("Expected ValueError for unknown dataset name.")
    assert calls == []


def test_downloader_registers_one_function_per_dataset() -> None:
    assert set(downloader.DATASET_DOWNLOADERS) == {"vrsbench", "xlrs_bench", "levir_cc", "mme_real_rs"}
