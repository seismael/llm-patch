"""Tests for :mod:`llm_patch_wiki_agent.daemon.runner`."""

from __future__ import annotations

from pathlib import Path

import llm_patch as engine
import pytest
from llm_patch_shared import ConfigurationError, IntegrationError

from llm_patch_wiki_agent.daemon import WikiCompileDaemon
from llm_patch_wiki_agent.registry import SidecarMetadataRegistry


class _FakeSource:
    name = "fake"

    def __init__(self, documents: list[engine.DocumentContext]) -> None:
        self._documents = documents
        self._subscribers: list = []

    def fetch_all(self) -> list[engine.DocumentContext]:
        return list(self._documents)

    # Optional IKnowledgeStream surface
    def subscribe(self, callback) -> None:  # pragma: no cover - exercised via stream tests
        self._subscribers.append(callback)

    def start(self) -> None:  # pragma: no cover
        pass

    def stop(self) -> None:  # pragma: no cover
        pass


class _FakePipeline:
    def __init__(self, *, fail_on: str | None = None) -> None:
        self.calls: list[str] = []
        self._fail_on = fail_on

    def process_document(self, context: engine.DocumentContext) -> engine.AdapterManifest:
        self.calls.append(context.document_id)
        if self._fail_on == context.document_id:
            raise RuntimeError("boom")
        return engine.AdapterManifest(
            adapter_id=context.document_id,
            rank=8,
            target_modules=["q_proj"],
            storage_uri=f"adapters/{context.document_id}",
        )

    def start(self) -> None:  # pragma: no cover
        pass

    def stop(self) -> None:  # pragma: no cover
        pass


def _make_daemon(
    tmp_path: Path,
    documents: list[engine.DocumentContext],
    *,
    fail_on: str | None = None,
) -> tuple[WikiCompileDaemon, SidecarMetadataRegistry, _FakePipeline]:
    registry = SidecarMetadataRegistry(tmp_path)
    pipeline = _FakePipeline(fail_on=fail_on)
    daemon = WikiCompileDaemon(
        source=_FakeSource(documents),
        pipeline=pipeline,
        metadata_registry=registry,
    )
    return daemon, registry, pipeline


def test_run_once_writes_sidecars_for_each_document(tmp_path: Path) -> None:
    documents = [
        engine.DocumentContext(
            document_id="api-v2-auth",
            content="# auth",
            metadata={
                "context_id": "api-v2-auth",
                "tags": ["api", "v2"],
                "summary": "OAuth flow",
                "source_path": "docs/api-v2-auth.md",
            },
        ),
        engine.DocumentContext(
            document_id="payment", content="# payment", metadata={}
        ),
    ]
    daemon, registry, _ = _make_daemon(tmp_path, documents)

    result = daemon.run_once()

    assert tuple(m.adapter_id for m in result.manifests) == ("api-v2-auth", "payment")
    saved = registry.list_all()
    by_id = {m.adapter_id: m for m in saved}
    assert by_id["api-v2-auth"].context_id == "api-v2-auth"
    assert by_id["api-v2-auth"].tags == ("api", "v2")
    assert by_id["api-v2-auth"].summary == "OAuth flow"
    # Falls back to document_id when frontmatter is silent.
    assert by_id["payment"].context_id == "payment"
    assert by_id["payment"].tags == ()


def test_run_once_wraps_pipeline_failures_in_integration_error(tmp_path: Path) -> None:
    documents = [
        engine.DocumentContext(document_id="ok", content="x", metadata={}),
        engine.DocumentContext(document_id="bad", content="x", metadata={}),
    ]
    daemon, _, _ = _make_daemon(tmp_path, documents, fail_on="bad")

    with pytest.raises(IntegrationError):
        daemon.run_once()


def test_start_without_stream_raises(tmp_path: Path) -> None:
    daemon, _, _ = _make_daemon(tmp_path, [])

    with pytest.raises(ConfigurationError):
        daemon.start()


def test_stop_is_a_noop_without_stream(tmp_path: Path) -> None:
    daemon, _, _ = _make_daemon(tmp_path, [])

    daemon.stop()  # must not raise


def test_from_config_requires_wiki_dir_and_checkpoint(tmp_path: Path) -> None:
    from llm_patch_wiki_agent import WikiAgentConfig

    config = WikiAgentConfig(adapter_dir=tmp_path / "adapters")

    with pytest.raises(ConfigurationError):
        WikiCompileDaemon.from_config(config)
