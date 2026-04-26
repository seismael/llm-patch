"""Tests for ``runtime.preflight`` (no torch import path)."""

from __future__ import annotations

from llm_patch.runtime.preflight import probe


class TestPreflight:
    def test_no_torch_path(self) -> None:
        report = probe(import_torch=False)
        assert report.torch_imported is False
        assert report.cuda_available is False
        assert report.device_count == 0
        assert "CPU-only" in report.human_summary()

    def test_has_min_vram_zero(self) -> None:
        report = probe(import_torch=False)
        assert report.has_min_vram(0.0) is True
        assert report.has_min_vram(1.0) is False
