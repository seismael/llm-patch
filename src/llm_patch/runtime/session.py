"""Chat session — conversation history management for agent runtimes."""

from __future__ import annotations

from llm_patch.core.models import ChatMessage, ChatResponse, ChatRole


class ChatSession:
    """Manages a stateful conversation with an ``IAgentRuntime``.

    Args:
        runtime: Any ``IAgentRuntime`` implementation.
        system_prompt: Optional system instruction prepended to every call.
        max_history: Maximum number of messages to retain (0 = unlimited).
    """

    def __init__(
        self,
        runtime: "IAgentRuntime",  # noqa: F821 — forward ref avoids circular import
        system_prompt: str | None = None,
        max_history: int = 0,
    ) -> None:
        from llm_patch.core.interfaces import IAgentRuntime

        if not isinstance(runtime, IAgentRuntime):
            msg = f"runtime must implement IAgentRuntime, got {type(runtime).__name__}"
            raise TypeError(msg)

        self._runtime = runtime
        self._system_prompt = system_prompt
        self._history: list[ChatMessage] = []
        self._max_history = max_history

    @property
    def history(self) -> list[ChatMessage]:
        """Return a copy of the conversation history."""
        return list(self._history)

    @property
    def system_prompt(self) -> str | None:
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str | None) -> None:
        self._system_prompt = value

    def say(self, text: str) -> str:
        """Send a user message and return the assistant's reply text."""
        user_msg = ChatMessage(role=ChatRole.USER, content=text)
        self._history.append(user_msg)

        messages = self._build_messages()
        response: ChatResponse = self._runtime.chat(messages)

        self._history.append(response.message)
        self._trim()
        return response.message.content

    def add_message(self, role: ChatRole, content: str) -> None:
        """Manually inject a message into the history."""
        self._history.append(ChatMessage(role=role, content=content))
        self._trim()

    def clear(self) -> None:
        """Clear the conversation history (system prompt is preserved)."""
        self._history.clear()

    def _build_messages(self) -> list[ChatMessage]:
        """Assemble full message list including system prompt."""
        msgs: list[ChatMessage] = []
        if self._system_prompt:
            msgs.append(ChatMessage(role=ChatRole.SYSTEM, content=self._system_prompt))
        msgs.extend(self._history)
        return msgs

    def _trim(self) -> None:
        """Trim history to max_history if set, keeping most recent messages."""
        if self._max_history > 0 and len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]
