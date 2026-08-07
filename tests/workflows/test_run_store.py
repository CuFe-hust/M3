"""Contract tests for RunStore, RunManifest, and EventWriter.

RunStore / RunManifest / EventWriter 契约测试：可复现 manifest（git
commit/dirty、config/prompt hash、dataset/split/filter）、快照不含 API
key、原子写入无 .tmp 残留、run 创建不调用模型、EventWriter 原子 JSONL
追加与敏感字段拒绝。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from workflows.events import EventWriter
from workflows.run_store import RunManifest, RunStore

REPO_ROOT = Path(__file__).resolve().parents[2]


def _config_payload() -> dict:
    return {"models": {"qwen": "Qwen/Qwen3-VL-4B-Instruct"}, "thresholds": {"top_k": 5}}


def _prompt(root: Path, name: str = "prompt_v1.md") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {name}\ncontent\n", encoding="utf-8")
    return path


def _store(tmp_path: Path) -> tuple[RunStore, Path]:
    root = tmp_path / "runs"
    return RunStore(root, tmp_path), root


def test_create_run_writes_manifest_and_snapshots(tmp_path: Path) -> None:
    store, root = _store(tmp_path)
    prompt = _prompt(tmp_path)
    manifest = store.create_run(
        config_payload=_config_payload(),
        model_ids={"qwen": "Qwen/Qwen3-VL-4B-Instruct", "deepseek": "deepseek-chat"},
        prompt_paths=[prompt],
        dataset="parity",
        split="test",
        sample_filter="subset-a",
    )
    assert isinstance(manifest, RunManifest)
    assert manifest.dataset == "parity"
    assert manifest.split == "test"
    assert manifest.sample_filter == "subset-a"
    run_dir = root / manifest.run_id
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "config.snapshot.json").is_file()
    assert (run_dir / "events.jsonl").is_file()
    snapshot = json.loads((run_dir / "config.snapshot.json").read_text(encoding="utf-8"))
    assert snapshot == _config_payload()


def test_manifest_reproducibility_fields(tmp_path: Path) -> None:
    store, _ = _store(tmp_path)
    prompt = _prompt(tmp_path)
    manifest = store.create_run(
        config_payload=_config_payload(),
        model_ids={"qwen": "qwen-model"},
        prompt_paths=[prompt],
    )
    assert len(manifest.config_hash) == 64
    assert manifest.prompt_hashes == {"prompt_v1.md": manifest.prompt_hashes["prompt_v1.md"]}
    assert len(manifest.prompt_hashes["prompt_v1.md"]) == 64
    assert manifest.model_ids == {"qwen": "qwen-model"}


def test_manifest_records_git_state_inside_repository(tmp_path: Path) -> None:
    """The real repository is a git working tree, so git_commit is a full
    SHA and git_dirty a real boolean.
    真实仓库是 git 工作树：git_commit 为完整 SHA，git_dirty 为真实布尔值。"""
    store = RunStore(tmp_path / "runs", REPO_ROOT)
    manifest = store.create_run(
        config_payload=_config_payload(),
        model_ids={"qwen": "q"},
        prompt_paths=[],
    )
    assert manifest.git_commit is not None
    assert len(manifest.git_commit) == 40
    assert isinstance(manifest.git_dirty, bool)


def test_manifest_records_none_git_outside_repository(tmp_path: Path) -> None:
    """A directory outside any Git working tree yields None git fields. The
    pytest tmp_path lives inside this repository, so use the system temp dir.
    位于任何 Git 工作树之外的目录产生 None git 字段。pytest 的 tmp_path 位于
    本仓库内部，因此改用系统临时目录。"""
    import tempfile

    with tempfile.TemporaryDirectory() as outside:
        store = RunStore(Path(outside) / "runs", Path(outside))
        manifest = store.create_run(
            config_payload=_config_payload(),
            model_ids={"qwen": "q"},
            prompt_paths=[],
        )
        assert manifest.git_commit is None
        assert manifest.git_dirty is None


def test_run_id_is_deterministic_when_given(tmp_path: Path) -> None:
    store, root = _store(tmp_path)
    manifest = store.create_run(
        config_payload=_config_payload(),
        model_ids={"qwen": "q"},
        prompt_paths=[],
        run_id="run-1",
    )
    assert manifest.run_id == "run-1"
    assert (root / "run-1").is_dir()


def test_duplicate_run_id_fails(tmp_path: Path) -> None:
    store, _ = _store(tmp_path)
    store.create_run(
        config_payload=_config_payload(), model_ids={"qwen": "q"}, prompt_paths=[], run_id="dup"
    )
    with pytest.raises(FileExistsError):
        store.create_run(
            config_payload=_config_payload(), model_ids={"qwen": "q"}, prompt_paths=[], run_id="dup"
        )


def test_missing_prompt_fails(tmp_path: Path) -> None:
    store, _ = _store(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.create_run(
            config_payload=_config_payload(),
            model_ids={"qwen": "q"},
            prompt_paths=[tmp_path / "missing.md"],
        )


def test_duplicate_prompt_filename_fails(tmp_path: Path) -> None:
    store, _ = _store(tmp_path)
    first = _prompt(tmp_path / "a", "same.md")
    second = _prompt(tmp_path / "b", "same.md")
    with pytest.raises(ValueError, match="Duplicate Prompt filename"):
        store.create_run(
            config_payload=_config_payload(),
            model_ids={"qwen": "q"},
            prompt_paths=[first, second],
        )


def test_prompt_snapshot_copied_with_content(tmp_path: Path) -> None:
    store, root = _store(tmp_path)
    prompt = _prompt(tmp_path, "p.md")
    manifest = store.create_run(
        config_payload=_config_payload(), model_ids={"qwen": "q"}, prompt_paths=[prompt]
    )
    copied = root / manifest.run_id / "prompts.snapshot" / "p.md"
    assert copied.read_text(encoding="utf-8") == prompt.read_text(encoding="utf-8")


# ── 密钥安全 / secret safety ───────────────────────────────────────────────


def test_config_secret_key_rejected(tmp_path: Path) -> None:
    store, _ = _store(tmp_path)
    with pytest.raises(ValueError, match="sensitive key"):
        store.create_run(
            config_payload={"models": {"api_key": "sk-abc"}},
            model_ids={"qwen": "q"},
            prompt_paths=[],
        )
    with pytest.raises(ValueError, match="sensitive key"):
        store.create_run(
            config_payload={"nested": {"authorization": "Bearer abc"}},
            model_ids={"qwen": "q"},
            prompt_paths=[],
        )


def test_config_secret_value_prefix_rejected(tmp_path: Path) -> None:
    store, _ = _store(tmp_path)
    for value in ("sk-test-secret", "Bearer abcdef", "data:image/png;base64,AAAA"):
        with pytest.raises(ValueError, match="sensitive value"):
            store.create_run(
                config_payload={"flag": value},
                model_ids={"qwen": "q"},
                prompt_paths=[],
            )


def test_model_ids_secret_rejected(tmp_path: Path) -> None:
    store, _ = _store(tmp_path)
    with pytest.raises(ValueError, match="sensitive value"):
        store.create_run(
            config_payload=_config_payload(),
            model_ids={"qwen": "sk-secret-model"},
            prompt_paths=[],
        )


def test_snapshot_never_contains_credentials(tmp_path: Path) -> None:
    store, root = _store(tmp_path)
    manifest = store.create_run(
        config_payload={"models": {"qwen": "Qwen/Qwen3-VL-4B-Instruct"}},
        model_ids={"qwen": "Qwen/Qwen3-VL-4B-Instruct"},
        prompt_paths=[],
    )
    run_dir = root / manifest.run_id
    for name in ("manifest.json", "config.snapshot.json", "events.jsonl"):
        content = (run_dir / name).read_text(encoding="utf-8").lower()
        assert "sk-" not in content
        assert "bearer" not in content
        assert "base64," not in content
        assert "api_key" not in content


# ── 原子写入 / atomic writes ───────────────────────────────────────────────


def test_no_temporary_files_left_after_create_run(tmp_path: Path) -> None:
    store, root = _store(tmp_path)
    prompt = _prompt(tmp_path)
    manifest = store.create_run(
        config_payload=_config_payload(), model_ids={"qwen": "q"}, prompt_paths=[prompt]
    )
    leftovers = [p for p in (root / manifest.run_id).rglob("*.tmp")]
    assert leftovers == []
    # Every JSON artifact parses. / 每个 JSON 产物均可解析。
    json.loads((root / manifest.run_id / "manifest.json").read_text(encoding="utf-8"))


def test_event_writer_appends_valid_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    writer = EventWriter(path)
    first = writer.write("SAMPLE_STARTED", sample_id="s1", details={"index": 1})
    second = writer.write(
        "SAMPLE_FAILED", sample_id="s1", error_code="PRIMARY_BACKEND_FAILED", details={"state": "x"}
    )
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert records[0]["event"] == "SAMPLE_STARTED"
    assert records[0]["details"] == {"index": 1}
    assert records[1]["error_code"] == "PRIMARY_BACKEND_FAILED"
    assert first.sample_id == "s1"
    assert second.timestamp >= first.timestamp
    # Atomic write leaves no temporary files. / 原子写入无临时文件残留。
    assert list(tmp_path.glob("*.tmp")) == []


def test_event_writer_rejects_secret_detail_keys(tmp_path: Path) -> None:
    writer = EventWriter(tmp_path / "events.jsonl")
    for key in ("api_key", "authorization", "base64", "image_data_url", "token"):
        with pytest.raises(ValueError, match="sensitive"):
            writer.write("X", details={key: "value"})


# ── 递归敏感扫描 / recursive secret scanning (33.5) ───────────────────────


def test_event_writer_rejects_nested_sensitive_keys(tmp_path: Path) -> None:
    writer = EventWriter(tmp_path / "events.jsonl")
    with pytest.raises(ValueError, match="sensitive"):
        writer.write("X", details={"nested": {"token": "sk-secret"}})
    with pytest.raises(ValueError, match="sensitive"):
        writer.write("X", details={"nested": {"api_key": "abc"}})
    with pytest.raises(ValueError, match="sensitive"):
        writer.write("X", details={"nested": {"deep": {"password": "p"}}})


def test_event_writer_rejects_sensitive_value_prefixes_recursively(tmp_path: Path) -> None:
    writer = EventWriter(tmp_path / "events.jsonl")
    for payload in (
        {"message": "Bearer abcdef"},
        {"message": "sk-test-secret"},
        {"message": "data:image/png;base64,AAAA"},
        {"message": "-----BEGIN PRIVATE KEY-----"},
        {"nested": {"message": "  Sk-uppercase"}},
    ):
        with pytest.raises(ValueError, match="sensitive value"):
            writer.write("X", details=payload)


def test_event_writer_accepts_clean_nested_payload(tmp_path: Path) -> None:
    writer = EventWriter(tmp_path / "events.jsonl")
    record = writer.write(
        "X",
        details={"nested": {"state": "running", "index": 3}, "labels": ["a", "b"]},
    )
    assert record.details["nested"]["state"] == "running"


def test_event_writer_error_never_echoes_secret(tmp_path: Path) -> None:
    writer = EventWriter(tmp_path / "events.jsonl")
    secret = "sk-very-secret-token-value"
    with pytest.raises(ValueError) as error:
        writer.write("X", details={"nested": {"token": secret}})
    assert secret not in str(error.value)


def test_event_writer_concurrent_appends_lose_no_lines(tmp_path: Path) -> None:
    """8 workers appending 100 events concurrently must keep every line.
    8 个 worker 并发追加 100 条事件必须一行不丢。"""
    import json as json_module
    from concurrent.futures import ThreadPoolExecutor

    path = tmp_path / "events.jsonl"
    writer = EventWriter(path)

    def append(index: int) -> None:
        writer.write("SAMPLE_DONE", sample_id=f"s{index}", details={"index": index})

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(append, range(100)))

    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 100
    rows = [json_module.loads(line) for line in lines]
    assert {row["details"]["index"] for row in rows} == set(range(100))
    assert {row["sample_id"] for row in rows} == {f"s{i}" for i in range(100)}
    assert list(tmp_path.glob("*.tmp")) == []


# ── 模型边界 / model boundary ──────────────────────────────────────────────


def test_create_run_never_calls_models(tmp_path: Path) -> None:
    """RunStore has no model client and its source must not reference any
    model call or client machinery.
    RunStore 不含模型客户端，其源码不得引用任何模型调用或客户端机制。"""
    source = (REPO_ROOT / "workflows" / "run_store.py").read_text(encoding="utf-8")
    for token in ("complete_json", "qwen", "deepseek", "visionlang", "import models", "import agents"):
        assert token not in source.casefold(), token


def test_run_store_has_no_application_dependency(tmp_path: Path) -> None:
    store, _ = _store(tmp_path)
    manifest = store.create_run(
        config_payload=_config_payload(), model_ids={"qwen": "q"}, prompt_paths=[]
    )
    assert manifest.run_id
    source = (REPO_ROOT / "workflows" / "run_store.py").read_text(encoding="utf-8")
    assert "import application" not in source
    assert "import application.settings" not in source
    assert "AppSettings" not in source
    assert "spacers_agent" not in source
