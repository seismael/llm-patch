"""Tests for the real wiki-agent orchestration layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import llm_patch as engine
import pytest
from llm_patch_utils import ConfigurationError

from llm_patch_wiki_agent import WikiAgent, WikiAgentConfig, WikiAgentInfo

if TYPE_CHECKING:
    import torch


class FakeSource:
    name = "wiki"

    def __init__(self, documents: list[engine.DocumentContext]) -> None:
        self._documents = documents

    def fetch_all(self) -> list[engine.DocumentContext]:
        return list(self._documents)


class FakeGenerator(engine.IWeightGenerator):
    def generate(self, context: engine.DocumentContext) -> dict[str, torch.Tensor]:
        return cast("dict[str, torch.Tensor]", {context.document_id: object()})

    def get_peft_config(self) -> dict[str, object]:
        return {"r": 8, "target_modules": ["q_proj"]}


class FakeRepository(engine.IAdapterRepository):
    def __init__(
        self,
        output_dir: Path,
        manifests: list[engine.AdapterManifest] | None = None,
    ) -> None:
        self._output_dir = output_dir
        self._manifests = list(manifests or [])

    def save(
        self,
        adapter_id: str,
        weights: dict[str, torch.Tensor],
        peft_config: object,
    ) -> engine.AdapterManifest:
        _ = weights, peft_config
        manifest = engine.AdapterManifest(
            adapter_id=adapter_id,
            rank=8,
            target_modules=["q_proj"],
            storage_uri=str(self._output_dir / adapter_id),
        )
        self._manifests.append(manifest)
        return manifest

    def load(self, adapter_id: str) -> dict[str, torch.Tensor]:
        return cast("dict[str, torch.Tensor]", {adapter_id: object()})

    def exists(self, adapter_id: str) -> bool:
        return any(manifest.adapter_id == adapter_id for manifest in self._manifests)

    def list_adapters(self) -> list[engine.AdapterManifest]:
        return list(self._manifests)

    def delete(self, adapter_id: str) -> None:
        self._manifests = [m for m in self._manifests if m.adapter_id != adapter_id]


class FakeRuntime(engine.IAgentRuntime):
    def __init__(self) -> None:
        self.messages: list[engine.ChatMessage] = []

    def generate(self, prompt: str, **kwargs: object) -> str:
        _ = kwargs
        return f"generated: {prompt}"

    def chat(
        self,
        messages: list[engine.ChatMessage],
        **kwargs: object,
    ) -> engine.ChatResponse:
        _ = kwargs
        self.messages = list(messages)
        return engine.ChatResponse(
            message=engine.ChatMessage(
                role=engine.ChatRole.ASSISTANT,
                content="wiki-specialized reply",
            )
        )


class FakeUsePipeline:
    def __init__(self, runtime: FakeRuntime) -> None:
        self._runtime = runtime
        self.calls: list[dict[str, object]] = []

    def build_agent(
        self,
        model_id: str,
        adapter_ids: list[str] | None = None,
        **model_kwargs: object,
    ) -> FakeRuntime:
        self.calls.append(
            {
                "model_id": model_id,
                "adapter_ids": adapter_ids,
                "model_kwargs": model_kwargs,
            }
        )
        return self._runtime


def test_describe_reports_adapter_inventory(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapters"
    manifest_dir = adapter_dir / "wiki-transformer"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text(
        json.dumps(
            {
                "adapter_id": "wiki-transformer",
                "rank": 8,
                "target_modules": ["q_proj"],
                "storage_uri": str(manifest_dir),
            }
        ),
        encoding="utf-8",
    )
    agent = WikiAgent(WikiAgentConfig(adapter_dir=adapter_dir, wiki_dir=tmp_path / "wiki"))

    info = agent.describe()

    assert isinstance(info, WikiAgentInfo)
    assert info.adapter_count == 1
    assert info.adapter_ids == ("wiki-transformer",)


def test_compile_compiles_all_documents_with_injected_dependencies(tmp_path: Path) -> None:
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    adapter_dir = tmp_path / "adapters"

    documents = [
        engine.DocumentContext(document_id="entities/transformer", content="Transformer basics"),
        engine.DocumentContext(document_id="entities/attention", content="Attention basics"),
    ]
    repository = FakeRepository(adapter_dir)
    agent = WikiAgent(
        WikiAgentConfig(
            adapter_dir=adapter_dir,
            wiki_dir=wiki_dir,
            checkpoint_dir=checkpoint_dir,
        ),
        knowledge_source_factory=lambda _cfg, _aggregate: FakeSource(documents),
        generator_factory=lambda _cfg: FakeGenerator(),
        repository_factory=lambda _cfg: repository,
    )

    manifests = agent.compile()

    assert [manifest.adapter_id for manifest in manifests] == [
        "entities/transformer",
        "entities/attention",
    ]
    assert repository.exists("entities/transformer")
    assert repository.exists("entities/attention")


def test_chat_uses_configured_pipeline_and_returns_reply(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapters"
    manifest = engine.AdapterManifest(
        adapter_id="wiki-transformer",
        rank=8,
        target_modules=["q_proj"],
        storage_uri=str(adapter_dir / "wiki-transformer"),
    )
    repository = FakeRepository(adapter_dir, manifests=[manifest])
    runtime = FakeRuntime()
    use_pipeline = FakeUsePipeline(runtime)
    agent = WikiAgent(
        WikiAgentConfig(
            adapter_dir=adapter_dir, model_id="demo/model", system_prompt="Stay concise."
        ),
        repository_factory=lambda _cfg: repository,
        model_provider_factory=lambda: object(),
        adapter_loader_factory=lambda: object(),
        use_pipeline_factory=lambda _provider, _loader, _repo: use_pipeline,
    )

    reply = agent.chat("Summarize the wiki.")

    assert reply == "wiki-specialized reply"
    assert use_pipeline.calls[0]["model_id"] == "demo/model"
    assert use_pipeline.calls[0]["adapter_ids"] == ["wiki-transformer"]
    assert runtime.messages[0].role is engine.ChatRole.SYSTEM
    assert runtime.messages[1].role is engine.ChatRole.USER


def test_chat_requires_compiled_adapters(tmp_path: Path) -> None:
    agent = WikiAgent(
        WikiAgentConfig(adapter_dir=tmp_path / "adapters", model_id="demo/model"),
        repository_factory=lambda _cfg: FakeRepository(tmp_path / "adapters"),
        model_provider_factory=lambda: object(),
        adapter_loader_factory=lambda: object(),
        use_pipeline_factory=lambda _provider, _loader, _repo: FakeUsePipeline(FakeRuntime()),
    )

    with pytest.raises(ConfigurationError, match="No compiled adapters"):
        agent.chat("Hello")
