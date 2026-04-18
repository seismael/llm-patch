"""Tests for llm_patch.runtime — PeftAgentRuntime and ChatSession."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llm_patch.core.interfaces import IAgentRuntime
from llm_patch.core.models import (
    ChatMessage,
    ChatResponse,
    ChatRole,
    GenerationOptions,
    ModelHandle,
)


def _make_handle():
    """Create a ModelHandle with mocked model and tokenizer."""
    model = MagicMock()
    model.device = "cpu"
    tokenizer = MagicMock()
    tokenizer.chat_template = None
    return ModelHandle(
        model=model,
        tokenizer=tokenizer,
        base_model_id="test-model",
        device="cpu",
    )


# ── PeftAgentRuntime ────────────────────────────────────────────────────


class TestPeftAgentRuntime:
    @patch("llm_patch.runtime.agent.torch", create=True)
    def test_generate_returns_decoded_text(self, mock_torch):
        from llm_patch.runtime.agent import PeftAgentRuntime

        handle = _make_handle()
        # Setup tokenizer to return input ids
        input_ids = MagicMock()
        input_ids.shape = [1, 5]  # batch=1, seq_len=5
        input_dict = {"input_ids": input_ids}
        handle.tokenizer.return_value = MagicMock(**input_dict)
        handle.tokenizer.return_value.__getitem__ = lambda self, k: input_dict.get(k, MagicMock())
        handle.tokenizer.return_value.to.return_value = input_dict

        # Mock model.generate returning extra tokens
        output_ids = MagicMock()
        output_ids.__getitem__ = lambda self, idx: MagicMock()
        handle.model.generate.return_value = output_ids

        handle.tokenizer.decode.return_value = "generated text"

        runtime = PeftAgentRuntime(handle)
        result = runtime.generate("test prompt")
        assert result == "generated text"
        handle.model.generate.assert_called_once()

    @patch("llm_patch.runtime.agent.torch", create=True)
    def test_chat_returns_chat_response(self, mock_torch):
        from llm_patch.runtime.agent import PeftAgentRuntime

        handle = _make_handle()
        input_ids = MagicMock()
        input_ids.shape = [1, 5]
        input_dict = {"input_ids": input_ids}
        handle.tokenizer.return_value = MagicMock(**input_dict)
        handle.tokenizer.return_value.to.return_value = input_dict

        output_ids = MagicMock()
        output_ids.__getitem__ = lambda self, idx: MagicMock()
        handle.model.generate.return_value = output_ids
        handle.tokenizer.decode.return_value = "assistant reply"

        runtime = PeftAgentRuntime(handle)
        messages = [ChatMessage(role=ChatRole.USER, content="hello")]
        response = runtime.chat(messages)

        assert isinstance(response, ChatResponse)
        assert response.message.role == ChatRole.ASSISTANT
        assert response.message.content == "assistant reply"

    def test_format_messages_fallback(self):
        """Without chat_template, falls back to role-prefixed concatenation."""
        from llm_patch.runtime.agent import PeftAgentRuntime

        handle = _make_handle()
        handle.tokenizer.chat_template = None

        runtime = PeftAgentRuntime(handle)
        messages = [
            ChatMessage(role=ChatRole.SYSTEM, content="Be helpful"),
            ChatMessage(role=ChatRole.USER, content="Hi"),
        ]
        result = runtime._format_messages(messages)
        assert "system: Be helpful" in result
        assert "user: Hi" in result
        assert result.endswith("assistant:")

    def test_format_messages_with_chat_template(self):
        """Uses tokenizer.apply_chat_template when available."""
        from llm_patch.runtime.agent import PeftAgentRuntime

        handle = _make_handle()
        handle.tokenizer.chat_template = "some_template"
        handle.tokenizer.apply_chat_template.return_value = "<formatted>"

        runtime = PeftAgentRuntime(handle)
        messages = [ChatMessage(role=ChatRole.USER, content="Hi")]
        result = runtime._format_messages(messages)
        assert result == "<formatted>"

    def test_resolve_opts_no_overrides(self):
        from llm_patch.runtime.agent import PeftAgentRuntime

        handle = _make_handle()
        opts = GenerationOptions(max_new_tokens=100)
        runtime = PeftAgentRuntime(handle, options=opts)
        resolved = runtime._resolve_opts({})
        assert resolved.max_new_tokens == 100

    def test_resolve_opts_with_overrides(self):
        from llm_patch.runtime.agent import PeftAgentRuntime

        handle = _make_handle()
        runtime = PeftAgentRuntime(handle)
        resolved = runtime._resolve_opts({"max_new_tokens": 500, "temperature": 0.1})
        assert resolved.max_new_tokens == 500
        assert resolved.temperature == 0.1

    def test_handle_property(self):
        from llm_patch.runtime.agent import PeftAgentRuntime

        handle = _make_handle()
        runtime = PeftAgentRuntime(handle)
        assert runtime.handle is handle


# ── ChatSession ─────────────────────────────────────────────────────────


class TestChatSession:
    def _make_runtime(self):
        """Create a mock IAgentRuntime subclass."""
        from llm_patch.runtime.agent import PeftAgentRuntime

        handle = _make_handle()
        runtime = PeftAgentRuntime.__new__(PeftAgentRuntime)
        runtime._handle = handle
        runtime._opts = GenerationOptions()
        # Mock the chat method
        runtime.chat = MagicMock(return_value=ChatResponse(
            message=ChatMessage(role=ChatRole.ASSISTANT, content="reply")
        ))
        return runtime

    def test_say_returns_reply(self):
        from llm_patch.runtime.session import ChatSession

        runtime = self._make_runtime()
        session = ChatSession(runtime)
        reply = session.say("hello")

        assert reply == "reply"
        runtime.chat.assert_called_once()

    def test_history_includes_user_and_assistant(self):
        from llm_patch.runtime.session import ChatSession

        runtime = self._make_runtime()
        session = ChatSession(runtime)
        session.say("hello")

        assert len(session.history) == 2
        assert session.history[0].role == ChatRole.USER
        assert session.history[0].content == "hello"
        assert session.history[1].role == ChatRole.ASSISTANT
        assert session.history[1].content == "reply"

    def test_system_prompt_prepended(self):
        from llm_patch.runtime.session import ChatSession

        runtime = self._make_runtime()
        session = ChatSession(runtime, system_prompt="Be concise")
        session.say("hello")

        call_args = runtime.chat.call_args[0][0]
        assert call_args[0].role == ChatRole.SYSTEM
        assert call_args[0].content == "Be concise"

    def test_max_history_trims(self):
        from llm_patch.runtime.session import ChatSession

        runtime = self._make_runtime()
        session = ChatSession(runtime, max_history=2)

        session.say("first")
        session.say("second")

        # max_history=2 keeps only last 2 messages
        assert len(session.history) == 2
        # Should be 2nd user + 2nd assistant reply
        assert session.history[0].role == ChatRole.USER
        assert session.history[0].content == "second"

    def test_clear(self):
        from llm_patch.runtime.session import ChatSession

        runtime = self._make_runtime()
        session = ChatSession(runtime, system_prompt="sys")
        session.say("hello")
        session.clear()

        assert session.history == []
        assert session.system_prompt == "sys"

    def test_add_message(self):
        from llm_patch.runtime.session import ChatSession

        runtime = self._make_runtime()
        session = ChatSession(runtime)
        session.add_message(ChatRole.USER, "injected")

        assert len(session.history) == 1
        assert session.history[0].content == "injected"

    def test_invalid_runtime_raises_type_error(self):
        from llm_patch.runtime.session import ChatSession

        with pytest.raises(TypeError, match="must implement IAgentRuntime"):
            ChatSession("not a runtime")

    def test_system_prompt_setter(self):
        from llm_patch.runtime.session import ChatSession

        runtime = self._make_runtime()
        session = ChatSession(runtime, system_prompt="old")
        session.system_prompt = "new"
        assert session.system_prompt == "new"

    def test_history_returns_copy(self):
        from llm_patch.runtime.session import ChatSession

        runtime = self._make_runtime()
        session = ChatSession(runtime)
        session.say("hello")
        h = session.history
        h.clear()
        assert len(session.history) == 2  # not affected
