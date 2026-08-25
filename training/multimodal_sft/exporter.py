"""Generic checkpoint/export orchestration; merge semantics stay in adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .checkpoint import read_manifest
from .contracts import MultimodalModelAdapter


class ExportContractError(ValueError):
    """The selected adapter cannot safely export the checkpoint."""


class GenericExporter:
    def __init__(self, adapter: MultimodalModelAdapter) -> None:
        self.adapter = adapter

    def export(
        self,
        *,
        model_id: str | Path,
        checkpoint_dir: str | Path,
        output_dir: str | Path,
        local_files_only: bool = True,
        verify_forward: bool = False,
        change_fixture: str | Path | None = None,
    ) -> Mapping[str, Any]:
        manifest = read_manifest(checkpoint_dir)
        expected = str(getattr(self.adapter, "name", type(self.adapter).__name__))
        if manifest.get("adapter_name") != expected:
            raise ExportContractError(
                f"checkpoint adapter={manifest.get('adapter_name')!r} does not match {expected!r}"
            )
        export = getattr(self.adapter, "export_checkpoint", None)
        if not callable(export):
            raise ExportContractError("selected adapter does not implement export_checkpoint")
        return export(
            model_id=model_id,
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
            local_files_only=local_files_only,
            verify_forward=verify_forward,
            change_fixture=change_fixture,
        )

