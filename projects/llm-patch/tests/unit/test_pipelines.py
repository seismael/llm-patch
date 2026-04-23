"""Tests for llm_patch.pipelines — CompilePipeline, WikiPipeline, UsePipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from llm_patch.core.interfaces import IAgentRuntime, IDataSource
from llm_patch.core.models import (
    AdapterManifest,
    ChatMessage,
    ChatResponse,
    ChatRole,
    DocumentContext,
    ModelHandle,
)

# ── CompilePipeline ─────────────────────────────────────────────────────


class TestCompilePipeline:
    def _make_deps(self):
        """Create mock source, generator, repository."""
        source = MagicMock()
        source.fetch_all.return_value = [
            DocumentContext(document_id="d1", content="c1"),
            DocumentContext(document_id="d2", content="c2"),
        ]
        generator = MagicMock()
        generator.generate.return_value = {"w": MagicMock()}
        generator.get_peft_config.return_value = MagicMock()
        repo = MagicMock()
        repo.save.side_effect = lambda aid, _weights, _cfg: AdapterManifest(
            adapter_id=aid,
            rank=8,
            target_modules=["q_proj"],
            storage_uri=str(Path("adapters") / aid),
        )
        return source, generator, repo

    def test_compile_all_processes_all_docs(self):
        from llm_patch.pipelines.compile import CompilePipeline

        source, gen, repo = self._make_deps()
        pipeline = CompilePipeline(source, gen, repo)
        manifests = pipeline.compile_all()

        assert len(manifests) == 2
        assert manifests[0].adapter_id == "d1"
        assert manifests[1].adapter_id == "d2"
        assert gen.generate.call_count == 2
        assert repo.save.call_count == 2

    def test_process_document(self):
        from llm_patch.pipelines.compile import CompilePipeline

        source, gen, repo = self._make_deps()
        pipeline = CompilePipeline(source, gen, repo)

        doc = DocumentContext(document_id="single", content="text")
        manifest = pipeline.process_document(doc)

        assert manifest.adapter_id == "single"
        gen.generate.assert_called_once_with(doc)

    def test_stream_callback_triggers_compile(self):
        from llm_patch.pipelines.compile import CompilePipeline

        source, gen, repo = self._make_deps()
        stream = MagicMock()
        CompilePipeline(source, gen, repo, stream=stream)

        # Verify subscription happened
        stream.subscribe.assert_called_once()
        callback = stream.subscribe.call_args[0][0]

        # Trigger the callback
        doc = DocumentContext(document_id="live", content="live text")
        callback(doc)
        gen.generate.assert_called_once_with(doc)

    def test_start_stop_delegates_to_stream(self):
        from llm_patch.pipelines.compile import CompilePipeline

        source, gen, repo = self._make_deps()
        stream = MagicMock()
        pipeline = CompilePipeline(source, gen, repo, stream=stream)

        pipeline.start()
        stream.start.assert_called_once()

        pipeline.stop()
        stream.stop.assert_called_once()

    def test_context_manager(self):
        from llm_patch.pipelines.compile import CompilePipeline

        source, gen, repo = self._make_deps()
        stream = MagicMock()

        with CompilePipeline(source, gen, repo, stream=stream):
            stream.start.assert_called_once()

        stream.stop.assert_called_once()

    def test_start_stop_no_stream_noop(self):
        from llm_patch.pipelines.compile import CompilePipeline

        source, gen, repo = self._make_deps()
        pipeline = CompilePipeline(source, gen, repo)
        # Should not raise
        pipeline.start()
        pipeline.stop()


# ── UsePipeline ──────────────────────────────────────────────────────────


class TestUsePipeline:
    def _make_deps(self):
        model_provider = MagicMock()
        adapter_loader = MagicMock()
        repo = MagicMock()

        handle = ModelHandle(
            model=MagicMock(),
            tokenizer=MagicMock(),
            base_model_id="test",
            device="cpu",
        )
        model_provider.load.return_value = handle
        adapter_loader.attach.side_effect = lambda h, m: ModelHandle(
            model=h.model,
            tokenizer=h.tokenizer,
            base_model_id=h.base_model_id,
            attached_adapters=(*h.attached_adapters, m.adapter_id),
            device=h.device,
        )
        return model_provider, adapter_loader, repo

    def test_load_and_attach_no_adapters(self):
        from llm_patch.pipelines.use import UsePipeline

        provider, loader, repo = self._make_deps()
        repo.list_adapters.return_value = []

        pipeline = UsePipeline(provider, loader, repo)
        handle = pipeline.load_and_attach("model-id")

        assert handle.base_model_id == "test"
        assert handle.attached_adapters == ()
        loader.attach.assert_not_called()

    def test_load_and_attach_specific_adapter_ids(self):
        from llm_patch.pipelines.use import UsePipeline

        provider, loader, repo = self._make_deps()
        m1 = AdapterManifest(adapter_id="a1", rank=8, target_modules=["q"], storage_uri="/a1")
        m2 = AdapterManifest(adapter_id="a2", rank=8, target_modules=["q"], storage_uri="/a2")
        repo.list_adapters.return_value = [m1, m2]

        pipeline = UsePipeline(provider, loader, repo)
        handle = pipeline.load_and_attach("model-id", adapter_ids=["a1"])

        assert handle.attached_adapters == ("a1",)
        assert loader.attach.call_count == 1

    def test_load_and_attach_all_adapters(self):
        from llm_patch.pipelines.use import UsePipeline

        provider, loader, repo = self._make_deps()
        m1 = AdapterManifest(adapter_id="a1", rank=8, target_modules=["q"], storage_uri="/a1")
        m2 = AdapterManifest(adapter_id="a2", rank=8, target_modules=["q"], storage_uri="/a2")
        repo.list_adapters.return_value = [m1, m2]

        pipeline = UsePipeline(provider, loader, repo)
        handle = pipeline.load_and_attach("model-id", adapter_ids=None)

        assert handle.attached_adapters == ("a1", "a2")
        assert loader.attach.call_count == 2

    def test_load_and_attach_warns_missing_adapter(self):
        from llm_patch.pipelines.use import UsePipeline

        provider, loader, repo = self._make_deps()
        repo.list_adapters.return_value = []

        pipeline = UsePipeline(provider, loader, repo)
        handle = pipeline.load_and_attach("model-id", adapter_ids=["missing"])

        assert handle.attached_adapters == ()
        loader.attach.assert_not_called()

    def test_build_agent(self):
        from llm_patch.pipelines.use import UsePipeline
        from llm_patch.runtime.agent import PeftAgentRuntime

        provider, loader, repo = self._make_deps()
        repo.list_adapters.return_value = []

        pipeline = UsePipeline(provider, loader, repo)
        agent = pipeline.build_agent("model-id")

        assert isinstance(agent, PeftAgentRuntime)


# ── WikiPipeline ──────────────────────────────────────────────────────────


class TestWikiPipeline:
    def _make_deps(self, tmp_path):
        from llm_patch.core.config import WikiConfig

        agent = MagicMock()
        agent.summarize_document.return_value = "summary"
        agent.extract_entities.return_value = []
        config = WikiConfig(base_dir=tmp_path / "wiki")
        return agent, config

    def test_init_creates_wiki(self, tmp_path):
        from llm_patch.pipelines.wiki import WikiPipeline

        agent, config = self._make_deps(tmp_path)
        pipeline = WikiPipeline(agent, config)
        pipeline.init()

        assert (tmp_path / "wiki").exists()

    def test_status(self, tmp_path):
        from llm_patch.pipelines.wiki import WikiPipeline

        agent, config = self._make_deps(tmp_path)
        pipeline = WikiPipeline(agent, config)
        pipeline.init()

        status = pipeline.status()
        assert isinstance(status, dict)

    def test_query_delegates(self, tmp_path):
        from llm_patch.pipelines.wiki import WikiPipeline
        from llm_patch.wiki.operations import QueryResult

        agent, config = self._make_deps(tmp_path)
        agent.answer_query.return_value = QueryResult(
            answer="answer",
            cited_pages=[],
        )
        pipeline = WikiPipeline(agent, config)
        pipeline.init()

        result = pipeline.query("What is?")
        assert result is not None
        assert result.answer == "answer"

    def test_lint_delegates(self, tmp_path):
        from llm_patch.pipelines.wiki import WikiPipeline

        agent, config = self._make_deps(tmp_path)
        pipeline = WikiPipeline(agent, config)
        pipeline.init()

        result = pipeline.lint()
        assert result is not None

    def test_trigger_compile_with_pipeline(self, tmp_path):
        """Test that _trigger_compile calls compile_all on the pipeline."""
        from llm_patch.pipelines.wiki import WikiPipeline
        from llm_patch.wiki.operations import IngestResult

        agent, config = self._make_deps(tmp_path)
        compile_pipeline = MagicMock()
        pipeline = WikiPipeline(agent, config, compile_pipeline=compile_pipeline)

        # Manually trigger compile with a fake result
        result = IngestResult(
            source_path="test.md",
            pages_created=["test"],
            pages_updated=[],
            entities_extracted=[],
        )
        pipeline._trigger_compile(result)
        compile_pipeline.compile_all.assert_called_once()

    def test_trigger_compile_handles_exception(self, tmp_path):
        from llm_patch.pipelines.wiki import WikiPipeline
        from llm_patch.wiki.operations import IngestResult

        agent, config = self._make_deps(tmp_path)
        compile_pipeline = MagicMock()
        compile_pipeline.compile_all.side_effect = RuntimeError("boom")
        pipeline = WikiPipeline(agent, config, compile_pipeline=compile_pipeline)

        result = IngestResult(
            source_path="test.md",
            pages_created=[],
            pages_updated=[],
            entities_extracted=[],
        )
        # Should not raise despite compile failure
        pipeline._trigger_compile(result)

    def test_trigger_compile_no_compile_all_method(self, tmp_path):
        from llm_patch.pipelines.wiki import WikiPipeline
        from llm_patch.wiki.operations import IngestResult

        agent, config = self._make_deps(tmp_path)
        compile_pipeline = object()  # no compile_all method
        pipeline = WikiPipeline(agent, config, compile_pipeline=compile_pipeline)

        result = IngestResult(
            source_path="test.md",
            pages_created=[],
            pages_updated=[],
            entities_extracted=[],
        )
        # Should not raise
        pipeline._trigger_compile(result)


class _SimpleDataSource(IDataSource):
    def __init__(self, documents: list[DocumentContext]) -> None:
        self._documents = documents

    @property
    def name(self) -> str:
        return "simple"

    def fetch_all(self):
        yield from self._documents


class _SimpleRuntime(IAgentRuntime):
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str, **kwargs: object) -> str:
        _ = kwargs
        self.prompts.append(prompt)
        return f"generated: {prompt}"

    def chat(self, messages: list[ChatMessage], **kwargs: object) -> ChatResponse:
        _ = messages, kwargs
        return ChatResponse(
            message=ChatMessage(role=ChatRole.ASSISTANT, content="reply")
        )


class TestCoreInterfaceDefaults:
    def test_datasource_fetch_one_returns_matching_document(self) -> None:
        source = _SimpleDataSource(
            [DocumentContext(document_id="doc-1", content="hello")]
        )

        result = source.fetch_one("doc-1")

        assert result is not None
        assert result.document_id == "doc-1"

    def test_datasource_fetch_one_returns_none_when_missing(self) -> None:
        source = _SimpleDataSource(
            [DocumentContext(document_id="doc-1", content="hello")]
        )

        assert source.fetch_one("missing") is None

    def test_runtime_stream_falls_back_to_generate(self) -> None:
        runtime = _SimpleRuntime()

        chunks = list(runtime.stream("hello world"))

        assert chunks == ["generated: hello world"]
        assert runtime.prompts == ["hello world"]
