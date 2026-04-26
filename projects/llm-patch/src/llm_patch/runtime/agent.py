"""Agent runtime — chat / generate / stream over a patched model."""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Any

from llm_patch.core.interfaces import IAgentRuntime, IRuntimeAdapterController
from llm_patch.core.models import (
    ChatMessage,
    ChatResponse,
    ChatRole,
    GenerationOptions,
    ModelHandle,
)

logger = logging.getLogger(__name__)


class PeftAgentRuntime(IAgentRuntime):
    """In-process agent backed by a patched HuggingFace model.

    Args:
        handle: A ``ModelHandle`` (base or with adapters attached).
        options: Default generation knobs.
        controller: Optional hot-swap hook. When provided, agent code
            (e.g. via MCP tools) can attach/detach adapters at runtime
            without reconstructing the runtime.
    """

    def __init__(
        self,
        handle: ModelHandle,
        options: GenerationOptions | None = None,
        controller: IRuntimeAdapterController | None = None,
    ) -> None:
        self._handle = handle
        self._opts = options or GenerationOptions()
        self._controller = controller

    @property
    def handle(self) -> ModelHandle:
        # If a controller is mutating the handle, prefer its view.
        if self._controller is not None and hasattr(self._controller, "handle"):
            return self._controller.handle  # type: ignore[no-any-return]
        return self._handle

    @property
    def controller(self) -> IRuntimeAdapterController | None:
        """Adapter hot-swap controller, or ``None`` if not configured."""
        return self._controller

    def generate(self, prompt: str, **kwargs: Any) -> str:
        import torch

        opts = self._resolve_opts(kwargs)
        handle = self.handle
        model = handle.model
        tokenizer = handle.tokenizer

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=opts.max_new_tokens,
                temperature=opts.temperature if opts.do_sample else 1.0,
                top_p=opts.top_p if opts.do_sample else 1.0,
                top_k=opts.top_k if opts.do_sample else 0,
                do_sample=opts.do_sample,
                repetition_penalty=opts.repetition_penalty,
            )

        # Decode only the new tokens
        new_token_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        return str(tokenizer.decode(new_token_ids, skip_special_tokens=True))

    def chat(self, messages: list[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = self._format_messages(messages)
        text = self.generate(prompt, **kwargs)
        reply = ChatMessage(role=ChatRole.ASSISTANT, content=text)
        return ChatResponse(message=reply)

    def stream(self, prompt: str, **kwargs: Any) -> Generator[str, None, None]:
        """Streaming token generator using HuggingFace TextIteratorStreamer."""
        try:
            from threading import Thread

            from transformers import TextIteratorStreamer
        except ImportError:
            yield self.generate(prompt, **kwargs)
            return

        import torch

        opts = self._resolve_opts(kwargs)
        handle = self.handle
        model = handle.model
        tokenizer = handle.tokenizer

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs = {
            **inputs,
            "max_new_tokens": opts.max_new_tokens,
            "temperature": opts.temperature if opts.do_sample else 1.0,
            "do_sample": opts.do_sample,
            "streamer": streamer,
        }

        thread = Thread(target=lambda: model.generate(**gen_kwargs))
        thread.start()

        with torch.inference_mode():
            yield from streamer

        thread.join()

    # ── Internals ─────────────────────────────────────────────────────

    def _resolve_opts(self, overrides: dict[str, Any]) -> GenerationOptions:
        if not overrides:
            return self._opts
        data = self._opts.model_dump()
        data.update(overrides)
        return GenerationOptions(**data)

    def _format_messages(self, messages: list[ChatMessage]) -> str:
        """Build a single prompt string from chat messages.

        Uses the tokenizer's chat template when available.
        """
        tokenizer = self.handle.tokenizer
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            dicts = [{"role": m.role.value, "content": m.content} for m in messages]
            return str(
                tokenizer.apply_chat_template(
                    dicts,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        # Fallback: simple role-prefixed concatenation
        parts: list[str] = []
        for msg in messages:
            parts.append(f"{msg.role.value}: {msg.content}")
        parts.append("assistant:")
        return "\n".join(parts)
