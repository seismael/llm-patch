"""Tests for llm_patch.server — FastAPI app and schemas."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llm_patch.core.models import (
    AdapterManifest,
    ChatMessage,
    ChatResponse,
    ChatRole,
)
from llm_patch.server.schemas import (
    AdapterInfo,
    ChatMessageSchema,
    ChatRequest,
    CompileRequest,
    CompileResponse,
    GenerateRequest,
    HealthResponse,
)


def _get_app_module():
    """Return the *module* llm_patch.server.app (not the FastAPI instance).

    ``import llm_patch.server.app`` resolves to the FastAPI instance because
    ``llm_patch.server.__init__`` rebinds the ``app`` attribute.  We go
    through ``sys.modules`` to get the actual module.
    """
    __import__("llm_patch.server.app")
    return sys.modules["llm_patch.server.app"]


# ── Schema tests ─────────────────────────────────────────────────────────


class TestSchemas:
    def test_health_response(self):
        h = HealthResponse(version="0.1.0")
        assert h.status == "ok"
        assert h.version == "0.1.0"

    def test_generate_request_defaults(self):
        r = GenerateRequest(prompt="hello")
        assert r.max_new_tokens == 256
        assert r.temperature == 0.7
        assert r.do_sample is True

    def test_compile_request(self):
        r = CompileRequest(document_id="doc1", content="text")
        assert r.metadata == {}

    def test_adapter_info(self):
        a = AdapterInfo(
            adapter_id="a1",
            rank=8,
            target_modules=["q_proj"],
            storage_uri=str(Path("adapters") / "a1"),
        )
        assert a.adapter_id == "a1"

    def test_chat_request(self):
        r = ChatRequest(messages=[ChatMessageSchema(role="user", content="hi")])
        assert len(r.messages) == 1
        assert r.max_new_tokens == 256

    def test_compile_response(self):
        r = CompileResponse(adapter_id="x", storage_uri=str(Path("adapters") / "x"))
        data = r.model_dump()
        assert data["adapter_id"] == "x"


# ── App endpoint tests ──────────────────────────────────────────────────


class TestAppEndpoints:
    @pytest.fixture(autouse=True)
    def _clear_state(self):
        """Clear module-level state before each test."""
        mod = _get_app_module()
        mod._state.clear()
        yield
        mod._state.clear()

    @pytest.fixture()
    def client(self):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("starlette not installed")

        mod = _get_app_module()
        return TestClient(mod.app, raise_server_exceptions=False)

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_list_adapters_empty(self, client):
        mod = _get_app_module()
        mock_repo = MagicMock()
        mock_repo.list_adapters.return_value = []
        mod._state["repository"] = mock_repo

        resp = client.get("/adapters")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_adapters_with_data(self, client):
        mod = _get_app_module()
        manifest = AdapterManifest(
            adapter_id="a1",
            rank=8,
            target_modules=["q_proj"],
            storage_uri=str(Path("adapters") / "a1"),
        )
        mock_repo = MagicMock()
        mock_repo.list_adapters.return_value = [manifest]
        mod._state["repository"] = mock_repo

        resp = client.get("/adapters")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["adapter_id"] == "a1"

    def test_get_adapter_found(self, client):
        mod = _get_app_module()
        manifest = AdapterManifest(
            adapter_id="a1",
            rank=8,
            target_modules=["q_proj"],
            storage_uri=str(Path("adapters") / "a1"),
        )
        mock_repo = MagicMock()
        mock_repo.list_adapters.return_value = [manifest]
        mod._state["repository"] = mock_repo

        resp = client.get("/adapters/a1")
        assert resp.status_code == 200
        assert resp.json()["adapter_id"] == "a1"

    def test_get_adapter_not_found(self, client):
        mod = _get_app_module()
        mock_repo = MagicMock()
        mock_repo.list_adapters.return_value = []
        mod._state["repository"] = mock_repo

        resp = client.get("/adapters/missing")
        assert resp.status_code == 404

    def test_delete_adapter(self, client):
        mod = _get_app_module()
        mock_repo = MagicMock()
        mock_repo.exists.return_value = True
        mod._state["repository"] = mock_repo

        resp = client.delete("/adapters/a1")
        assert resp.status_code == 200
        mock_repo.delete.assert_called_once_with("a1")

    def test_delete_adapter_not_found(self, client):
        mod = _get_app_module()
        mock_repo = MagicMock()
        mock_repo.exists.return_value = False
        mod._state["repository"] = mock_repo

        resp = client.delete("/adapters/missing")
        assert resp.status_code == 404

    def test_compile_no_generator(self, client):
        mod = _get_app_module()
        mock_repo = MagicMock()
        mod._state["repository"] = mock_repo

        resp = client.post(
            "/compile",
            json={
                "document_id": "d1",
                "content": "text",
            },
        )
        assert resp.status_code == 503

    def test_compile_with_generator(self, client):
        mod = _get_app_module()
        mock_repo = MagicMock()
        mock_repo.save.return_value = AdapterManifest(
            adapter_id="d1",
            rank=8,
            target_modules=["q_proj"],
            storage_uri=str(Path("adapters") / "d1"),
        )
        mock_gen = MagicMock()
        mock_gen.generate.return_value = {"w": MagicMock()}
        mock_gen.get_peft_config.return_value = MagicMock()
        mod._state["repository"] = mock_repo
        mod._state["generator"] = mock_gen

        resp = client.post(
            "/compile",
            json={
                "document_id": "d1",
                "content": "some text",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["adapter_id"] == "d1"

    def test_generate_no_model(self, client):
        resp = client.post("/generate", json={"prompt": "hello"})
        assert resp.status_code == 503

    def test_generate_with_runtime(self, client):
        mod = _get_app_module()
        mock_runtime = MagicMock()
        mock_runtime.generate.return_value = "generated text"
        mod._state["runtime"] = mock_runtime

        resp = client.post("/generate", json={"prompt": "hello"})
        assert resp.status_code == 200
        assert resp.json()["text"] == "generated text"

    def test_chat_no_model(self, client):
        resp = client.post(
            "/chat",
            json={
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 503

    def test_chat_with_runtime(self, client):
        mod = _get_app_module()
        mock_runtime = MagicMock()
        mock_runtime.chat.return_value = ChatResponse(
            message=ChatMessage(role=ChatRole.ASSISTANT, content="hello back")
        )
        mod._state["runtime"] = mock_runtime

        resp = client.post(
            "/chat",
            json={
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["message"]["content"] == "hello back"
        assert data["message"]["role"] == "assistant"
