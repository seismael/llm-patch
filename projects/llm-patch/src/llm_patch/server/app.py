"""FastAPI application for llm-patch HTTP API."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from llm_patch_shared import (
    AdapterNotFoundError,
    ChecksumMismatchError,
    IncompatibleBaseModelError,
    RegistryUnavailableError,
)

import llm_patch
from llm_patch.core.interfaces import (
    IAdapterCache,
    IAdapterRepository,
    IAgentRuntime,
    IRuntimeAdapterController,
    IWeightGenerator,
)
from llm_patch.server.schemas import (
    ActiveAdapters,
    AdapterInfo,
    AttachRequest,
    CacheStats,
    ChatMessageSchema,
    ChatRequest,
    ChatResponse,
    CompileRequest,
    CompileResponse,
    DetachRequest,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
)

# ── Module-level state ────────────────────────────────────────────────

_state: dict[str, object] = {}

# Single global lock serializing GPU-affecting attach/detach/generate
# operations. Concurrency contract is documented in
# ``docs/SERVER_ARCHITECTURE.md``.
_swap_lock = asyncio.Lock()


def _get_repository() -> IAdapterRepository:
    """Return the adapter repository (lazy init from env)."""
    if "repository" not in _state:
        from pathlib import Path

        from llm_patch.core.config import StorageConfig
        from llm_patch.storage.local_safetensors import LocalSafetensorsRepository

        adapter_dir = os.environ.get("LLM_PATCH_ADAPTER_DIR", "./adapters")
        cfg = StorageConfig(output_dir=Path(adapter_dir))
        Path(adapter_dir).mkdir(parents=True, exist_ok=True)
        _state["repository"] = LocalSafetensorsRepository(cfg)
    return cast(IAdapterRepository, _state["repository"])


def _get_runtime() -> IAgentRuntime | None:
    """Return the agent runtime, or None if no model is loaded."""
    runtime = _state.get("runtime")
    if runtime is None:
        return None
    return cast(IAgentRuntime, runtime)


def _get_generator() -> IWeightGenerator | None:
    """Return the configured weight generator, or None if unavailable."""
    generator = _state.get("generator")
    if generator is None:
        return None
    return cast(IWeightGenerator, generator)


def _get_controller() -> IRuntimeAdapterController | None:
    """Return the runtime adapter controller, or None if not configured."""
    ctrl = _state.get("controller")
    if ctrl is None:
        return None
    return cast(IRuntimeAdapterController, ctrl)


def _get_cache() -> IAdapterCache | None:
    """Return the in-memory adapter cache, or None if not configured."""
    cache = _state.get("cache")
    if cache is None:
        return None
    return cast(IAdapterCache, cache)


def _manifest_to_info(m: object) -> AdapterInfo:
    """Convert a domain :class:`AdapterManifest` to wire schema."""
    # Imported lazily to avoid a circular type dependency on startup.
    from llm_patch.core.models import AdapterManifest

    assert isinstance(m, AdapterManifest)
    return AdapterInfo(
        adapter_id=m.adapter_id,
        rank=m.rank,
        target_modules=m.target_modules,
        storage_uri=m.storage_uri,
        namespace=m.namespace,
        version=m.version,
        checksum_sha256=m.checksum_sha256,
        base_model_compatibility=list(m.base_model_compatibility),
        tags=list(m.tags),
        description=m.description,
    )


# ── Lifespan ──────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
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
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version=llm_patch.__version__)


# ── Adapters ──────────────────────────────────────────────────────────


@app.get("/adapters", response_model=list[AdapterInfo])
async def list_adapters() -> list[AdapterInfo]:
    repo = _get_repository()
    return [_manifest_to_info(m) for m in repo.list_adapters()]


@app.get("/adapters/{adapter_id}", response_model=AdapterInfo)
async def get_adapter(adapter_id: str) -> AdapterInfo:
    repo = _get_repository()
    for m in repo.list_adapters():
        if m.adapter_id == adapter_id:
            return _manifest_to_info(m)
    raise HTTPException(status_code=404, detail=f"Adapter not found: {adapter_id}")


@app.delete("/adapters/{adapter_id}")
async def delete_adapter(adapter_id: str) -> dict[str, str]:
    repo = _get_repository()
    if not repo.exists(adapter_id):
        raise HTTPException(status_code=404, detail=f"Adapter not found: {adapter_id}")
    repo.delete(adapter_id)
    cache = _get_cache()
    if cache is not None:
        cache.evict(adapter_id)
    return {"deleted": adapter_id}


# ── Hot-swap (Adapter Market) ─────────────────────────────────────────


@app.post("/adapters/attach", response_model=AdapterInfo)
async def attach_adapter(req: AttachRequest) -> AdapterInfo:
    """Attach an adapter onto the live model handle.

    Serialized via the global ``_swap_lock`` to honor PEFT's
    single-writer GPU contract. See ``docs/SERVER_ARCHITECTURE.md``.
    """
    controller = _get_controller()
    if controller is None:
        raise HTTPException(
            status_code=503,
            detail="No runtime adapter controller configured.",
        )
    from llm_patch.core.models import AdapterRef

    try:
        ref = AdapterRef.parse(req.ref)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    async with _swap_lock:
        try:
            manifest = await asyncio.to_thread(controller.attach, ref)
        except RegistryUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except AdapterNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except IncompatibleBaseModelError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ChecksumMismatchError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
    cache = _get_cache()
    if cache is not None:
        cache.put(manifest)
    return _manifest_to_info(manifest)


@app.post("/adapters/detach")
async def detach_adapter(req: DetachRequest) -> dict[str, str]:
    controller = _get_controller()
    if controller is None:
        raise HTTPException(
            status_code=503,
            detail="No runtime adapter controller configured.",
        )
    async with _swap_lock:
        await asyncio.to_thread(controller.detach, req.adapter_id)
    return {"detached": req.adapter_id}


@app.get("/adapters/active", response_model=ActiveAdapters)
async def active_adapters() -> ActiveAdapters:
    controller = _get_controller()
    if controller is None:
        return ActiveAdapters(active=[])
    return ActiveAdapters(active=controller.active())


@app.get("/cache/stats", response_model=CacheStats)
async def cache_stats() -> CacheStats:
    cache = _get_cache()
    if cache is None:
        return CacheStats(size=0, capacity=0)
    return CacheStats(size=len(cache), capacity=cache.capacity)


# ── Compile ───────────────────────────────────────────────────────────


@app.post("/compile", response_model=CompileResponse)
async def compile_document(req: CompileRequest) -> CompileResponse:
    """Compile a single document into an adapter."""
    repo = _get_repository()

    # Need a generator — check if one is configured
    generator = _get_generator()
    if generator is None:
        raise HTTPException(
            status_code=503,
            detail="No weight generator configured. Set LLM_PATCH_CHECKPOINT_DIR.",
        )

    from llm_patch.core.models import DocumentContext

    doc = DocumentContext(document_id=req.document_id, content=req.content, metadata=req.metadata)
    weights = generator.generate(doc)
    peft_config = generator.get_peft_config()
    manifest = repo.save(req.document_id, weights, peft_config)
    return CompileResponse(adapter_id=manifest.adapter_id, storage_uri=manifest.storage_uri)


# ── Inference ─────────────────────────────────────────────────────────


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
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
async def chat(req: ChatRequest) -> ChatResponse:
    runtime = _get_runtime()
    if runtime is None:
        raise HTTPException(status_code=503, detail="No model loaded. Set LLM_PATCH_MODEL_ID.")

    from llm_patch.core.models import ChatMessage, ChatRole

    messages = [ChatMessage(role=ChatRole(m.role), content=m.content) for m in req.messages]
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
