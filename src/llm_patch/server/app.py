"""FastAPI application for llm-patch HTTP API."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import llm_patch
from llm_patch.server.schemas import (
    AdapterInfo,
    ChatMessageSchema,
    ChatRequest,
    ChatResponse,
    CompileAllResponse,
    CompileRequest,
    CompileResponse,
    DocumentDetail,
    DocumentInfo,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
)

# ── Module-level state ────────────────────────────────────────────────

_state: dict[str, Any] = {}


def _get_repository():
    """Return the adapter repository (lazy init from env)."""
    if "repository" not in _state:
        from pathlib import Path

        from llm_patch.core.config import StorageConfig
        from llm_patch.storage.local_safetensors import LocalSafetensorsRepository

        adapter_dir = os.environ.get("LLM_PATCH_ADAPTER_DIR", "./adapters")
        cfg = StorageConfig(output_dir=Path(adapter_dir))
        Path(adapter_dir).mkdir(parents=True, exist_ok=True)
        _state["repository"] = LocalSafetensorsRepository(cfg)
    return _state["repository"]


def _get_runtime():
    """Return the agent runtime, or None if no model is loaded."""
    return _state.get("runtime")


# ── Lifespan ──────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Optionally pre-load a model on startup."""
    model_id = os.environ.get("LLM_PATCH_MODEL_ID")
    if model_id:
        from llm_patch.attach import HFModelProvider, PeftAdapterLoader
        from llm_patch.pipelines.use import UsePipeline
        from llm_patch.runtime.agent import PeftAgentRuntime

        repo = _get_repository()
        pipeline = UsePipeline(
            model_provider=HFModelProvider(),
            adapter_loader=PeftAdapterLoader(),
            repository=repo,
        )
        handle = pipeline.load_and_attach(model_id)
        _state["runtime"] = PeftAgentRuntime(handle)
    yield
    _state.clear()


# ── App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="llm-patch",
    description="Ingest → Compile → Attach → Use",
    version=llm_patch.__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", version=llm_patch.__version__)


# ── Adapters ──────────────────────────────────────────────────────────


@app.get("/adapters", response_model=list[AdapterInfo])
async def list_adapters():
    repo = _get_repository()
    return [
        AdapterInfo(
            adapter_id=m.adapter_id,
            rank=m.rank,
            target_modules=m.target_modules,
            storage_uri=m.storage_uri,
        )
        for m in repo.list_adapters()
    ]


@app.get("/adapters/{adapter_id}", response_model=AdapterInfo)
async def get_adapter(adapter_id: str):
    repo = _get_repository()
    for m in repo.list_adapters():
        if m.adapter_id == adapter_id:
            return AdapterInfo(
                adapter_id=m.adapter_id,
                rank=m.rank,
                target_modules=m.target_modules,
                storage_uri=m.storage_uri,
            )
    raise HTTPException(status_code=404, detail=f"Adapter not found: {adapter_id}")


@app.delete("/adapters/{adapter_id}")
async def delete_adapter(adapter_id: str):
    repo = _get_repository()
    if not repo.exists(adapter_id):
        raise HTTPException(status_code=404, detail=f"Adapter not found: {adapter_id}")
    repo.delete(adapter_id)
    return {"deleted": adapter_id}


# ── Compile ───────────────────────────────────────────────────────────


@app.post("/compile", response_model=CompileResponse)
async def compile_document(req: CompileRequest):
    """Compile a single document into an adapter."""
    repo = _get_repository()

    # Need a generator — check if one is configured
    if "generator" not in _state:
        raise HTTPException(
            status_code=503,
            detail="No weight generator configured. Set LLM_PATCH_CHECKPOINT_DIR.",
        )

    from llm_patch.core.models import DocumentContext

    doc = DocumentContext(
        document_id=req.document_id, content=req.content, metadata=req.metadata
    )
    generator = _state["generator"]
    weights = generator.generate(doc)
    peft_config = generator.get_peft_config()
    manifest = repo.save(req.document_id, weights, peft_config)
    return CompileResponse(adapter_id=manifest.adapter_id, storage_uri=manifest.storage_uri)


# ── Inference ─────────────────────────────────────────────────────────


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    runtime = _get_runtime()
    if runtime is None:
        raise HTTPException(status_code=503, detail="No model loaded. Set LLM_PATCH_MODEL_ID.")
    text = runtime.generate(
        req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        do_sample=req.do_sample,
    )
    return GenerateResponse(text=text)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    runtime = _get_runtime()
    if runtime is None:
        raise HTTPException(status_code=503, detail="No model loaded. Set LLM_PATCH_MODEL_ID.")

    from llm_patch.core.models import ChatMessage, ChatRole

    messages = [
        ChatMessage(role=ChatRole(m.role), content=m.content) for m in req.messages
    ]
    response = runtime.chat(
        messages,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        do_sample=req.do_sample,
    )
    return ChatResponse(
        message=ChatMessageSchema(
            role=response.message.role.value,
            content=response.message.content,
        )
    )
