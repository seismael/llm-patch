"""Pipeline composition layer.

Orchestrates the end-to-end workflows:
- ``CompilePipeline`` — ingest → generate LoRA weights → store adapters.
- ``WikiPipeline``    — wiki lifecycle + optional compile delegation.
- ``UsePipeline``     — load model → attach adapters → build agent.
"""

from llm_patch.pipelines.compile import CompilePipeline
from llm_patch.pipelines.use import UsePipeline
from llm_patch.pipelines.wiki import WikiPipeline

__all__ = [
    "CompilePipeline",
    "UsePipeline",
    "WikiPipeline",
]
